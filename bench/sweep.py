#!/usr/bin/env python3
"""Sweep harness: strategy name in, CSV rows out.

This is the file that turns Person A's kernels into evidence. Its one job is to
produce numbers that are *the same numbers the organizers' script would print*,
for many configurations, unattended, without dying half-way.

Fidelity to the organizers' script
----------------------------------
Everything measured here comes from functions imported out of the organizers'
file (via `src.baseline`), never reimplemented:

  * accuracy uses their `generate_random_case` + `compare_outputs`, with their
    `seed + trial` scheme. We run the loop ourselves only because their
    `run_accuracy_tests` returns a bare bool and prints the rest, and the CSV
    needs `max_abs_err` / `max_rel_err` as numbers.
  * timing uses their `warmup_model` + `benchmark_once` in their alternating
    round order, with their `seed + 100000` benchmark input. Same reason:
    `benchmark_models` prints and returns None.
  * global state is set exactly as their `main()` sets it — `manual_seed`,
    `set_float32_matmul_precision`, the TF32 flags. These change results, so
    they are not optional.
  * model setup follows their order: construct -> copy weights -> to(device,
    dtype) -> eval -> compile.

If this file ever disagrees with `python bench/torch_transformer_benchmark.py`
at the same shape, this file is wrong.

Robustness, in the order things actually go wrong
-------------------------------------------------
  1. dirty working tree      -> refuse (a row nobody can reproduce is noise)
  2. config too big for VRAM -> `SKIPPED: <reason>` row, sweep continues
  3. accuracy fails          -> `FAIL:` row with the debugging fields, no timing
  4. baseline OOMs           -> time the optimized model anyway; that asymmetry
                                is the VRAM-ceiling result, not an error
  5. GPU throttles mid-run   -> cool down, retry once, then keep the row flagged

Usage
-----
    python bench/sweep.py --strategy baseline --matrix quick
    python bench/sweep.py --strategy sdpa --batch 8 --seq-len 1024 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import csv
import gc
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.baseline import (
    BaselineTransformer,
    TransformerConfig,
    benchmark_once,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    maybe_compile,
    resolve_device,
    resolve_dtype,
    warmup_model,
)
from src.memcheck import check_fits
from src.strategies import get_strategy, requires_cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.thermal import ThermalLogger, wait_until_cool  # noqa: E402

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

#: The 21 columns agreed with Person A in docs/INTERFACE.md. Order is fixed.
CSV_COLUMNS_AGREED: Tuple[str, ...] = (
    "timestamp", "git_sha", "strategy_name", "batch", "seq_len", "d_model",
    "heads", "layers", "dtype", "causal", "padding_ratio",
    "accuracy_pass", "max_abs_err", "max_rel_err",
    "baseline_median_ms", "optimized_median_ms", "speedup",
    "peak_vram_mb", "mean_sm_clock_mhz", "max_temp_c", "notes",
)

#: Appended at the END only, as the contract permits, pending A's confirmation.
#: `baseline_peak_vram_mb` cannot be recovered later (the two peaks are not
#: separable after the fact); `compile_baseline` and `ffn_dim` are free axes
#: that the agreed columns do not pin down, so a row without them is ambiguous.
#: If A vetoes them, empty this tuple -- readers tolerate either width.
CSV_COLUMNS_PROPOSED: Tuple[str, ...] = (
    "baseline_peak_vram_mb", "compile_baseline", "ffn_dim",
)

CSV_COLUMNS: Tuple[str, ...] = CSV_COLUMNS_AGREED + CSV_COLUMNS_PROPOSED

#: How each column is rendered. Everything absent is written as an empty cell.
_FORMATS: Dict[str, str] = {
    "max_abs_err": "%.6g", "max_rel_err": "%.6g",
    "baseline_median_ms": "%.4f", "optimized_median_ms": "%.4f",
    "speedup": "%.4f",
    "peak_vram_mb": "%.1f", "baseline_peak_vram_mb": "%.1f",
    "mean_sm_clock_mhz": "%.1f", "max_temp_c": "%.1f",
    "padding_ratio": "%.3g",
}

_OOM_ERRORS: Tuple[type, ...] = (
    getattr(torch.cuda, "OutOfMemoryError", RuntimeError),
    MemoryError,
)


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """One point in the sweep: a TransformerConfig plus the axes it lacks.

    `TransformerConfig` does not carry dtype or padding_ratio, but both change
    the measured result, so the sweep tracks them alongside it.
    """

    batch: int = 8
    seq_len: int = 512
    d_model: int = 512
    heads: int = 8
    ffn_dim: Optional[int] = None      # None -> 4 * d_model
    layers: int = 6
    dtype: str = "float32"
    causal: bool = False
    padding_ratio: float = 0.0

    @property
    def resolved_ffn_dim(self) -> int:
        return self.ffn_dim if self.ffn_dim else 4 * self.d_model

    def config(self) -> TransformerConfig:
        config = TransformerConfig(
            batch_size=self.batch,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.heads,
            ffn_dim=self.resolved_ffn_dim,
            num_layers=self.layers,
            causal=self.causal,
        )
        config.validate()
        return config

    def label(self) -> str:
        mask = f"pad{self.padding_ratio:g}" + ("+causal" if self.causal else "")
        return (
            f"B={self.batch} S={self.seq_len} d={self.d_model} H={self.heads} "
            f"L={self.layers} {self.dtype} {mask}"
        )


# ---------------------------------------------------------------------------
# Matrices (one factor at a time)
# ---------------------------------------------------------------------------

#: Axis varied by each named matrix, and the values it takes. `mask` varies two
#: axes together because padding and causality are not independent questions.
_OFAT_AXES: Dict[str, Tuple[str, Sequence]] = {
    "seq": ("seq_len", (128, 512, 1024, 2048)),
    "batch": ("batch", (1, 8, 32)),
    "dmodel": ("d_model", (256, 512, 1024)),
    "dtype": ("dtype", ("float32", "float16", "bfloat16")),
    # layers is swept for the accuracy-budget figure: error accumulates with
    # depth, and the figure needs a line per dtype across L.
    "layers": ("layers", (1, 2, 4, 6)),
}

_MASK_COMBINATIONS = ((0.0, False), (0.3, False), (0.0, True), (0.3, True))

#: layers x dtype, crossed. This is NOT one-factor-at-a-time, and it is the one
#: place that has to break the OFAT rule: the accuracy-budget figure plots error
#: against depth with a line per dtype, and an OFAT sweep only ever varies depth
#: at the base dtype -- every other dtype ends up a single point, which is not a
#: line. Twelve configs; run it once when there is time.
_ACCURACY_LAYERS = (1, 2, 4, 6)
_ACCURACY_DTYPES = ("float32", "float16", "bfloat16")

#: Long sequences at batch 1, where quadratic attention memory actually bites.
#:
#: The estimator says the baseline needs 2.28 GiB at S=4096 and 9.1 GiB at
#: S=8192 (B=1, d=512, H=8, L=6, fp32) against a 6 GB card — so S=8192 is where
#: the baseline is *expected to OOM while a fused path completes*, because a
#: fused kernel never materializes [B, H, S, S] at all. That asymmetry is a
#: categorical result and a stronger claim than any speedup ratio: not "ours is
#: faster" but "ours runs where the reference cannot".
#:
#: Batch 1 on purpose. At B=8 the baseline already exceeds the budget at
#: S=2048, so the interesting rows would all be SKIPPED before anything ran.
_LONG_SEQ_LENGTHS = (2048, 4096, 8192)

MATRIX_NAMES = ("default", "quick", "seq", "batch", "dmodel", "dtype", "layers",
                "mask", "accuracy", "long")


def build_matrix(name: str, base: RunSpec) -> List[RunSpec]:
    """Expand a named matrix around `base`, de-duplicated, order preserved."""
    if name in _OFAT_AXES:
        axis, values = _OFAT_AXES[name]
        specs = [replace(base, **{axis: value}) for value in values]
    elif name == "mask":
        specs = [
            replace(base, padding_ratio=pad, causal=causal)
            for pad, causal in _MASK_COMBINATIONS
        ]
    elif name == "long":
        specs = [replace(base, batch=1, seq_len=s) for s in _LONG_SEQ_LENGTHS]
    elif name == "accuracy":
        specs = [
            replace(base, layers=layers, dtype=dtype)
            for dtype in _ACCURACY_DTYPES for layers in _ACCURACY_LAYERS
        ]
    elif name == "quick":
        # The 2-minute confidence check A runs after every rung.
        specs = [replace(base, seq_len=s, dtype="float32", padding_ratio=0.0,
                         causal=False) for s in (128, 512)]
    elif name == "default":
        specs = []
        for sub in ("seq", "batch", "dmodel", "dtype", "layers", "mask"):
            specs.extend(build_matrix(sub, base))
    else:
        raise ValueError(f"unknown matrix {name!r}; choose from {', '.join(MATRIX_NAMES)}")

    seen, unique = set(), []
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            unique.append(spec)
    return unique


def varied_axes(name: str) -> Tuple[str, ...]:
    """Which CLI flags a matrix overrides, so we can warn about conflicts."""
    if name in _OFAT_AXES:
        return (_OFAT_AXES[name][0],)
    if name == "mask":
        return ("padding_ratio", "causal")
    if name == "accuracy":
        return ("layers", "dtype")
    if name == "long":
        return ("batch", "seq_len")
    if name == "quick":
        return ("seq_len", "dtype", "padding_ratio", "causal")
    if name == "default":
        axes = set()
        for sub in ("seq", "batch", "dmodel", "dtype", "layers", "mask"):
            axes.update(varied_axes(sub))
        return tuple(sorted(axes))
    return ()


# ---------------------------------------------------------------------------
# git provenance
# ---------------------------------------------------------------------------

def _git(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True, capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parents[1],
        )
        return completed.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def git_sha() -> str:
    return _git("rev-parse", "--short", "HEAD") or "nogit"


def git_is_dirty() -> bool:
    status = _git("status", "--porcelain")
    return bool(status)


# ---------------------------------------------------------------------------
# CSV append
# ---------------------------------------------------------------------------

_warned_about_dropped: set = set()


def append_row(path: Path | str, row: Dict[str, object]) -> List[str]:
    """Append one row. Never rewrites, never reorders, never truncates.

    If the file already exists we write against *its* header, not ours, so an
    older narrower results.csv keeps working after columns are appended. Any
    field that the existing header cannot hold is reported once, loudly, rather
    than silently dropped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as handle:
            header = next(csv.reader(handle), list(CSV_COLUMNS))
        header = [column.strip() for column in header]
    else:
        header = list(CSV_COLUMNS)
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(header)

    dropped = set(row) - set(header)
    if dropped and path not in _warned_about_dropped:
        _warned_about_dropped.add(path)
        print(
            f"[warning] {path} has an older header; these fields cannot be "
            f"stored and are omitted from every row: {sorted(dropped)}",
            file=sys.stderr,
        )

    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow([_render(column, row.get(column)) for column in header])
        handle.flush()
    return header


def _render(column: str, value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return _FORMATS.get(column, "%.6g") % value
    return str(value)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class AccuracyOutcome:
    passed: bool
    max_abs: float
    max_rel: float
    failed: int
    total: int
    note: str = ""


def measure_accuracy(
    baseline: torch.nn.Module,
    optimized: torch.nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    padding_ratio: float,
) -> AccuracyOutcome:
    """The organizers' accuracy check, with the numbers kept instead of printed."""
    passed = True
    max_abs = max_rel = 0.0
    failed = total = 0
    worst_note = ""

    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            x, valid_mask = generate_random_case(
                config=config, device=device, dtype=dtype,
                seed=args.seed + trial,
                padding_ratio=padding_ratio,
                input_scale=args.input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=args.rtol, atol=args.atol)

            passed &= result.passed
            max_abs = max(max_abs, result.max_abs_error)
            max_rel = max(max_rel, result.max_relative_error)
            failed += result.failed_elements
            total += result.total_elements

            if not result.passed and not worst_note:
                # Exactly the fields the sprint plan says to paste when a mask
                # branch is wrong -- kept from the FIRST failing trial, whose
                # seed is reproducible as `--seed {args.seed} + {trial}`.
                preview = result.failed_feature_dims[:8]
                worst_note = (
                    f"trial={trial} worst_index={result.worst_index} "
                    f"base={result.reference_at_worst:.8g} "
                    f"opt={result.optimized_at_worst:.8g} "
                    f"failed_feature_dims={preview}"
                )

    note = "" if passed else f"FAIL: {failed}/{total} {worst_note}"
    return AccuracyOutcome(passed, max_abs, max_rel, failed, total, note)


def _peak_vram_mb(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> Optional[float]:
    """Peak allocation for one forward pass of `model`, in MiB.

    Measured in a dedicated pass rather than during the timed loop. The timed
    loop interleaves the two models to cancel clock drift, and
    `max_memory_allocated` is global — so during that loop the two peaks are
    not separable, and the CSV wants them in separate columns.
    """
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(x, valid_mask)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


@dataclass
class TimingOutcome:
    baseline_median_ms: Optional[float] = None
    optimized_median_ms: Optional[float] = None
    speedup: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    baseline_peak_vram_mb: Optional[float] = None
    notes: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def measure_timing(
    baseline: torch.nn.Module,
    optimized: torch.nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    padding_ratio: float,
    logger: Optional[ThermalLogger] = None,
) -> TimingOutcome:
    """The organizers' alternating-round timing, per-model OOM tolerant."""
    import statistics

    outcome = TimingOutcome()

    x, valid_mask = generate_random_case(
        config=config, device=device, dtype=dtype,
        seed=args.seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=args.input_scale,
    )

    models = {"baseline": baseline, "optimized": optimized}
    alive = {"baseline": True, "optimized": True}
    samples: Dict[str, List[float]] = {"baseline": [], "optimized": []}
    peaks: Dict[str, Optional[float]] = {"baseline": None, "optimized": None}

    def _oom(which: str, exc: BaseException) -> None:
        alive[which] = False
        outcome.notes.append(f"{which} OOM during timing ({type(exc).__name__})")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Warmup, then a dedicated peak-memory pass, per model.
    for which, model in models.items():
        try:
            warmup_model(model, x, valid_mask, args.warmup, device)
            peaks[which] = _peak_vram_mb(model, x, valid_mask, device)
        except _OOM_ERRORS as exc:
            _oom(which, exc)

    if logger is not None:
        logger.mark("timing_start")

    for round_index in range(args.benchmark_rounds):
        order = ("baseline", "optimized") if round_index % 2 == 0 else ("optimized", "baseline")
        for which in order:
            if not alive[which]:
                continue
            try:
                samples[which].extend(
                    benchmark_once(models[which], x, valid_mask, args.repeats, device)
                )
            except _OOM_ERRORS as exc:
                _oom(which, exc)

    if logger is not None:
        logger.mark("timing_end")

    if samples["baseline"]:
        outcome.baseline_median_ms = statistics.median(samples["baseline"])
    if samples["optimized"]:
        outcome.optimized_median_ms = statistics.median(samples["optimized"])
    if outcome.baseline_median_ms and outcome.optimized_median_ms:
        outcome.speedup = outcome.baseline_median_ms / outcome.optimized_median_ms

    outcome.peak_vram_mb = peaks["optimized"]
    outcome.baseline_peak_vram_mb = peaks["baseline"]

    # A baseline that cannot run while ours can is a headline result, not a bug.
    if not alive["baseline"] and alive["optimized"]:
        outcome.notes.append("baseline OOM; optimized completed")

    return outcome


# ---------------------------------------------------------------------------
# One configuration, end to end
# ---------------------------------------------------------------------------

def run_config(
    spec: RunSpec,
    strategy_name: str,
    args: argparse.Namespace,
    sha: str,
    retry_allowed: bool = True,
) -> Dict[str, object]:
    """Measure one configuration and return the CSV row (already appended)."""
    config = spec.config()
    dtype = resolve_dtype(spec.dtype)
    device = resolve_device(args.device)

    row: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "strategy_name": strategy_name,
        "batch": spec.batch,
        "seq_len": spec.seq_len,
        "d_model": spec.d_model,
        "heads": spec.heads,
        "layers": spec.layers,
        "dtype": spec.dtype,
        "causal": spec.causal,
        "padding_ratio": spec.padding_ratio,
        "ffn_dim": spec.resolved_ffn_dim,
        "compile_baseline": bool(args.compile_baseline),
        "accuracy_pass": "",
    }
    # `notes` is parsed by the analysis layer on its reserved *prefixes*
    # (SKIPPED:/FAIL:/baseline OOM/DISCARD:), so status must lead. Free text and
    # provenance markers are appended after it, never before.
    status_notes: List[str] = []
    aux_notes: List[str] = []
    if args.notes:
        aux_notes.append(args.notes)
    if args.allow_dirty and git_is_dirty():
        aux_notes.append("dirty")

    # -- 2. memory pre-check -------------------------------------------------
    fits, reason = check_fits(
        config, dtype, device,
        memory_fraction=args.memory_fraction,
        hard_cap_gib=args.hard_cap_gib,
    )
    if not fits and not args.force:
        row["notes"] = "; ".join([f"SKIPPED: {reason}", *aux_notes])
        append_row(args.results, row)
        print(f"  SKIPPED  {spec.label()}  ({reason})")
        return row

    # -- 3. thermal logging --------------------------------------------------
    logger = ThermalLogger(
        log_dir=args.logs, git_sha=sha, strategy=strategy_name,
        batch=spec.batch, seq_len=spec.seq_len, d_model=spec.d_model,
        enabled=(not args.no_thermal) and device.type == "cuda",
    )

    with logger:
        # -- 4. build, in the organizers' order ------------------------------
        baseline = BaselineTransformer(config)
        optimized = get_strategy(strategy_name)(config)
        copy_model_weights(baseline, optimized, strict=True)
        baseline = baseline.to(device=device, dtype=dtype).eval()
        optimized = optimized.to(device=device, dtype=dtype).eval()
        baseline = maybe_compile(baseline, args.compile_baseline, "default")

        # -- 5/6. accuracy ---------------------------------------------------
        accuracy = measure_accuracy(
            baseline, optimized, config, device, dtype, args, spec.padding_ratio
        )
        row["accuracy_pass"] = accuracy.passed
        row["max_abs_err"] = accuracy.max_abs
        row["max_rel_err"] = accuracy.max_rel
        if accuracy.note:
            status_notes.append(accuracy.note)

        # -- 7/8. timing + peak VRAM -----------------------------------------
        timing: Optional[TimingOutcome] = None
        if accuracy.passed or args.benchmark_on_failure:
            timing = measure_timing(
                baseline, optimized, config, device, dtype, args,
                spec.padding_ratio, logger,
            )
            row["baseline_median_ms"] = timing.baseline_median_ms
            row["optimized_median_ms"] = timing.optimized_median_ms
            row["speedup"] = timing.speedup
            row["peak_vram_mb"] = timing.peak_vram_mb
            row["baseline_peak_vram_mb"] = timing.baseline_peak_vram_mb
            status_notes.extend(timing.notes)

    # -- 9. thermal verdict --------------------------------------------------
    stats = logger.summarize(window=logger.window("timing_start", "timing_end"))
    if stats:
        row["mean_sm_clock_mhz"] = stats["mean_sm_clock_mhz"]
        row["max_temp_c"] = stats["max_temp_c"]
        if stats["discard"]:
            if retry_allowed:
                print(
                    f"  throttled ({stats['mean_sm_clock_mhz']:.0f} MHz mean vs "
                    f"{stats['opening_sm_mhz']:.0f} MHz opening) — cooling down, retry once"
                )
                _cleanup(device)
                wait_until_cool(timeout_s=args.retry_cooldown)
                time.sleep(args.retry_cooldown if device.type == "cuda" else 0)
                return run_config(spec, strategy_name, args, sha, retry_allowed=False)
            status_notes.append("DISCARD:thermal (retried)")

    # -- 10. append ----------------------------------------------------------
    row["notes"] = "; ".join(note for note in (*status_notes, *aux_notes) if note)
    append_row(args.results, row)
    _print_summary(spec, strategy_name, row)

    # -- 11. cooldown --------------------------------------------------------
    _cleanup(device)
    if device.type == "cuda" and not args.no_cooldown and args.cooldown > 0:
        time.sleep(args.cooldown)
    return row


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _print_summary(spec: RunSpec, strategy_name: str, row: Dict[str, object]) -> None:
    if row.get("accuracy_pass") is False:
        verdict = "FAIL    "
    elif row.get("speedup") is None:
        verdict = "NO-TIME "
    else:
        verdict = f"{row['speedup']:.3f}x "
    max_abs = row.get("max_abs_err")
    err = f"max_abs={max_abs:.3g}" if isinstance(max_abs, float) else ""
    print(f"  {verdict} {strategy_name:<16s} {spec.label()}  {err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python bench/sweep.py",
        description="Run a strategy over one config or a predefined matrix, appending to results.csv.",
    )
    parser.add_argument("--strategy", required=True, help="a name from src.strategies.STRATEGIES")

    shape = parser.add_argument_group("shape (defaults come from the matrix base)")
    shape.add_argument("--batch", type=int, default=None)
    shape.add_argument("--seq-len", type=int, default=None)
    shape.add_argument("--d-model", type=int, default=None)
    shape.add_argument("--heads", type=int, default=None)
    shape.add_argument("--ffn-dim", type=int, default=None, help="0 or omitted means 4 * d_model")
    shape.add_argument("--layers", type=int, default=None)
    shape.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    shape.add_argument("--causal", action="store_true", default=None)
    shape.add_argument("--padding-ratio", type=float, default=None)

    parser.add_argument("--matrix", choices=MATRIX_NAMES, default=None,
                        help="run a predefined one-factor-at-a-time set instead of a single config")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--compile-baseline", action="store_true",
                        help="honesty check: compile the baseline too")

    guards = parser.add_argument_group("guards")
    guards.add_argument("--memory-fraction", type=float, default=0.75)
    guards.add_argument("--hard-cap-gib", type=float, default=None)
    guards.add_argument("--force", action="store_true", help="run even if the memory pre-check refuses")
    guards.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (noted in the row)")
    guards.add_argument("--no-thermal", action="store_true")
    guards.add_argument("--cooldown", type=float, default=30.0)
    guards.add_argument("--no-cooldown", action="store_true")
    guards.add_argument("--retry-cooldown", type=float, default=60.0)

    measurement = parser.add_argument_group("measurement (organizers' defaults)")
    measurement.add_argument("--warmup", type=int, default=20)
    measurement.add_argument("--repeats", type=int, default=100)
    measurement.add_argument("--benchmark-rounds", type=int, default=3)
    measurement.add_argument("--accuracy-trials", type=int, default=5)
    measurement.add_argument("--rtol", type=float, default=0.01)
    measurement.add_argument("--atol", type=float, default=0.001)
    measurement.add_argument("--seed", type=int, default=1234)
    measurement.add_argument("--input-scale", type=float, default=1.0)
    measurement.add_argument("--benchmark-on-failure", action="store_true")
    measurement.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    measurement.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--notes", default="", help="free text copied into every row")
    return parser


def _base_spec(args: argparse.Namespace) -> RunSpec:
    """RunSpec defaults overridden by whichever flags were actually passed."""
    overrides = {
        field: getattr(args, attr)
        for field, attr in (
            ("batch", "batch"), ("seq_len", "seq_len"), ("d_model", "d_model"),
            ("heads", "heads"), ("ffn_dim", "ffn_dim"), ("layers", "layers"),
            ("dtype", "dtype"), ("causal", "causal"), ("padding_ratio", "padding_ratio"),
        )
        if getattr(args, attr) is not None
    }
    return replace(RunSpec(), **overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # -- 1. provenance -------------------------------------------------------
    if git_is_dirty() and not args.allow_dirty:
        print(
            "refusing to run: the working tree has uncommitted changes, so these\n"
            "rows could not be reproduced from any commit. Commit first, or pass\n"
            "--allow-dirty to record them anyway (the rows are tagged 'dirty').",
            file=sys.stderr,
        )
        return 1
    sha = git_sha()

    device = resolve_device(args.device)
    if requires_cuda(args.strategy) and device.type != "cuda":
        print(
            f"strategy {args.strategy!r} declares REQUIRES_CUDA and device is {device}.",
            file=sys.stderr,
        )
        return 1

    # Global state, set exactly as the organizers' main() sets it. These change
    # the measured numbers, so they belong here and not in a config file.
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    base = _base_spec(args)
    if args.matrix:
        overridden = varied_axes(args.matrix)
        # RunSpec field names match the argparse destinations one-for-one, so a
        # varied axis that the user also passed explicitly is a direct conflict.
        conflicts = [axis for axis in overridden if getattr(args, axis, None) is not None]
        if conflicts:
            print(
                f"[note] --matrix {args.matrix} varies {', '.join(overridden)}; "
                f"your explicit {', '.join(conflicts)} is ignored for those axes.",
                file=sys.stderr,
            )
        specs = build_matrix(args.matrix, base)
    else:
        specs = [base]

    print(
        f"sweep: strategy={args.strategy} device={device} sha={sha} "
        f"configs={len(specs)} -> {args.results}"
    )

    for index, spec in enumerate(specs, start=1):
        print(f"[{index:02d}/{len(specs)}] {spec.label()}")
        run_config(spec, args.strategy, args, sha)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
