"""Generate a plausible `results.csv` so the analysis layer can be built and
tested before the GPU has produced a single real measurement.

Run:  python tests/fixtures/make_synthetic.py

Nothing here is a claim about performance. The numbers are shaped to exercise
every branch the analysis code has to handle, and they are deterministic:

  * five strategies across the full OFAT matrix, so every figure has lines;
  * speedups that grow with sequence length (attention is the quadratic term,
    so that is the shape a real sweep should show) and differ per strategy;
  * accuracy error that grows with depth and is larger in reduced precision,
    with fp16 crossing the atol line at 6 layers -- the accuracy-budget figure
    is pointless if nothing ever approaches the budget;
  * one row per strategy re-run with --compile-baseline, for the honesty check;
  * SKIPPED rows (memory pre-check refused), a baseline-OOM row (the VRAM
    ceiling), a thermally discarded row, and one genuine accuracy FAIL;
  * three different git_sha values with increasing timestamps, so
    `latest_per_config` has something to actually choose between.

If a figure looks wrong on this fixture, the figure is wrong.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.sweep import CSV_COLUMNS  # noqa: E402
from analysis.roofline import RTX_4050_LAPTOP, forward_flops  # noqa: E402

STRATEGIES = ("baseline", "sdpa", "bf16", "compiled", "fused_qkv")

BASE = dict(batch=8, seq_len=512, d_model=512, heads=8, layers=6,
            dtype="float32", causal=False, padding_ratio=0.0)

# One factor at a time, matching bench/sweep.py's --matrix default.
AXES = {
    "seq_len": (128, 512, 1024, 2048),
    "batch": (1, 8, 32),
    "d_model": (256, 512, 1024),
    "dtype": ("float32", "float16", "bfloat16"),
    "layers": (1, 2, 4, 6),
}
MASKS = ((0.0, False), (0.3, False), (0.0, True), (0.3, True))

SHAS = ("a1b2c3d", "d4e5f6a", "9f8e7d6")
START = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)

# speedup(S) = intercept + slope * log2(S / 128), clipped at 1.0.
SPEEDUP_MODEL: Dict[str, tuple] = {
    "baseline": (1.00, 0.00),
    "sdpa": (1.18, 0.46),
    "bf16": (1.36, 0.27),
    "compiled": (1.31, 0.15),
    "fused_qkv": (1.14, 0.08),
}

# max_abs_err per layer, by dtype. fp16 at 6 layers lands past atol=0.001.
ERROR_PER_LAYER = {"float32": 2.0e-6, "bfloat16": 1.1e-4, "float16": 2.6e-4}


def configs() -> List[dict]:
    """The de-duplicated OFAT set, base point included exactly once."""
    out, seen = [], set()
    for axis, values in AXES.items():
        for value in values:
            config = {**BASE, axis: value}
            key = tuple(sorted(config.items()))
            if key not in seen:
                seen.add(key)
                out.append(config)
    # layers x dtype, crossed -- see bench/sweep.py --matrix accuracy. Without
    # this the accuracy-budget figure has one line and two lone points.
    for dtype in ("float16", "bfloat16"):
        for layers in (1, 2, 4):
            config = {**BASE, "dtype": dtype, "layers": layers}
            key = tuple(sorted(config.items()))
            if key not in seen:
                seen.add(key)
                out.append(config)
    for padding, causal in MASKS:
        config = {**BASE, "padding_ratio": padding, "causal": causal}
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            out.append(config)
    return out


def speedup_for(strategy: str, config: dict, rng: random.Random) -> float:
    intercept, slope = SPEEDUP_MODEL[strategy]
    value = intercept + slope * math.log2(config["seq_len"] / 128)
    # Reduced precision helps more; large batch helps a little; masks cost a bit.
    if config["dtype"] in ("float16", "bfloat16") and strategy != "baseline":
        value *= 1.12
    if config["batch"] >= 32 and strategy != "baseline":
        value *= 1.05
    if config["causal"] and strategy != "baseline":
        value *= 0.96
    if strategy == "baseline":
        return round(rng.uniform(0.995, 1.006), 4)
    return round(max(1.0, value) * rng.uniform(0.97, 1.03), 4)


def errors_for(strategy: str, config: dict, rng: random.Random) -> tuple:
    if strategy == "baseline":
        return 0.0, 0.0
    dtype = "bfloat16" if strategy == "bf16" and config["dtype"] == "float32" else config["dtype"]
    max_abs = ERROR_PER_LAYER[dtype] * config["layers"] * rng.uniform(0.85, 1.15)
    max_rel = max_abs * rng.uniform(1.5, 3.0)
    return float(f"{max_abs:.6g}"), float(f"{max_rel:.6g}")


def baseline_ms_for(config: dict, rng: random.Random) -> float:
    """Latency derived from the analytic FLOP count and a plausible efficiency.

    Inventing a latency independently of `forward_flops` produces a fixture
    whose roofline shows fp32 running above the card's fp32 peak — physically
    impossible, and exactly the sort of thing that gets mistaken for real data
    later. Deriving it keeps the fixture self-consistent with analysis/roofline.py.

    Efficiency falls as sequence length grows: the baseline materializes
    [B,H,S,S], so at long sequences it is bandwidth-bound and achieves a smaller
    fraction of peak. That is the shape the roofline is supposed to reveal.
    """
    peak = RTX_4050_LAPTOP.peak_for(config["dtype"])
    efficiency = {128: 0.44, 512: 0.36, 1024: 0.29, 2048: 0.22}.get(config["seq_len"], 0.34)
    achieved = peak * efficiency * rng.uniform(0.94, 1.06)
    flops = forward_flops({**config, "ffn_dim": 4 * config["d_model"]})
    return round(flops / (achieved * 1e12) * 1e3, 4)


def vram_for(config: dict, strategy: str) -> tuple:
    """MiB. The baseline materializes [B,H,S,S]; fused paths do not."""
    b, s, d, h, layers = (config["batch"], config["seq_len"], config["d_model"],
                          config["heads"], config["layers"])
    element = 2 if config["dtype"] in ("float16", "bfloat16") else 4
    params = layers * (4 * d * d + 2 * d * 4 * d) + 2 * d
    weights = params * element / 2**20
    activations = 10 * b * s * d * element / 2**20
    scores = b * h * s * s * (2 * element + 8) / 2**20
    baseline_mb = weights + activations + scores
    optimized_mb = weights + activations + (0.0 if strategy != "baseline" else scores)
    return round(baseline_mb, 1), round(optimized_mb, 1)


def main() -> None:
    rng = random.Random(20260828)
    rows: List[dict] = []
    clock = START

    for index, config in enumerate(configs()):
        for strategy in STRATEGIES:
            clock += timedelta(minutes=2)
            speedup = speedup_for(strategy, config, rng)
            max_abs, max_rel = errors_for(strategy, config, rng)
            baseline_vram, optimized_vram = vram_for(config, strategy)
            baseline_ms = baseline_ms_for(config, rng)
            rows.append({
                "timestamp": clock.isoformat(timespec="seconds"),
                "git_sha": SHAS[index % len(SHAS)],
                "strategy_name": strategy,
                "batch": config["batch"], "seq_len": config["seq_len"],
                "d_model": config["d_model"], "heads": config["heads"],
                "layers": config["layers"], "dtype": config["dtype"],
                "causal": config["causal"], "padding_ratio": config["padding_ratio"],
                "accuracy_pass": True, "max_abs_err": max_abs, "max_rel_err": max_rel,
                "baseline_median_ms": baseline_ms,
                "optimized_median_ms": round(baseline_ms / speedup, 4),
                "speedup": speedup,
                "peak_vram_mb": optimized_vram,
                "baseline_peak_vram_mb": baseline_vram,
                "mean_sm_clock_mhz": round(rng.uniform(2320, 2410), 1),
                "max_temp_c": round(rng.uniform(66, 79), 1),
                "notes": "", "compile_baseline": False,
                "ffn_dim": 4 * config["d_model"],
            })

    def special(**overrides) -> dict:
        clock_local = overrides.pop("timestamp", None)
        row = {column: "" for column in CSV_COLUMNS}
        row.update({
            "timestamp": clock_local or (START + timedelta(hours=8)).isoformat(timespec="seconds"),
            "git_sha": SHAS[-1], "heads": 8, "layers": 6, "dtype": "float32",
            "causal": False, "padding_ratio": 0.0, "compile_baseline": False,
            "ffn_dim": 2048,
        })
        row.update(overrides)
        return row

    # --- the memory pre-check refusing an impossible config -----------------
    for strategy in ("baseline", "sdpa"):
        rows.append(special(
            strategy_name=strategy, batch=8, seq_len=4096, d_model=512,
            ffn_dim=2048,
            notes="SKIPPED: estimated baseline peak 17.27 GiB vs budget 4.31 GiB "
                  "(75% of 5.75 GiB free on cuda:0) — will not fit",
        ))

    # --- the VRAM ceiling: baseline cannot run, ours can --------------------
    rows.append(special(
        strategy_name="sdpa", batch=32, seq_len=2048, d_model=512, ffn_dim=2048,
        accuracy_pass=True, max_abs_err=1.4e-5, max_rel_err=3.1e-5,
        baseline_median_ms="", optimized_median_ms=486.2201, speedup="",
        peak_vram_mb=1902.4, baseline_peak_vram_mb="",
        mean_sm_clock_mhz=2371.5, max_temp_c=77.8,
        notes="baseline OOM; optimized completed",
    ))

    # --- a thermally discarded row -----------------------------------------
    rows.append(special(
        strategy_name="bf16", batch=8, seq_len=1024, d_model=512, ffn_dim=2048,
        dtype="bfloat16", accuracy_pass=True, max_abs_err=6.6e-4, max_rel_err=1.5e-3,
        baseline_median_ms=61.4402, optimized_median_ms=27.9013, speedup=2.2020,
        peak_vram_mb=612.7, baseline_peak_vram_mb=1259.5,
        mean_sm_clock_mhz=1786.0, max_temp_c=86.0,
        notes="DISCARD:thermal (retried)",
    ))

    # --- a genuine accuracy failure ----------------------------------------
    rows.append(special(
        strategy_name="fused_qkv", batch=8, seq_len=512, d_model=512, ffn_dim=2048,
        dtype="float16", accuracy_pass=False, max_abs_err=0.0412, max_rel_err=0.184,
        notes="FAIL: 3104/2097152 trial=0 worst_index=(2, 117, 41) base=-1.8823 "
              "opt=-1.9235 failed_feature_dims=[41, 42, 43]",
    ))

    # --- the --compile-baseline honesty check ------------------------------
    for strategy in STRATEGIES:
        speedup = speedup_for(strategy, BASE, rng)
        survived = 1.0 if strategy == "baseline" else max(1.0, speedup * rng.uniform(0.55, 0.78))
        compile_baseline_vram, compile_optimized_vram = vram_for(BASE, strategy)
        rows.append(special(
            timestamp=(START + timedelta(hours=9)).isoformat(timespec="seconds"),
            strategy_name=strategy, batch=8, seq_len=512, d_model=512, ffn_dim=2048,
            accuracy_pass=True, max_abs_err=errors_for(strategy, BASE, rng)[0],
            max_rel_err=errors_for(strategy, BASE, rng)[1],
            baseline_median_ms=39.8, optimized_median_ms=round(39.8 / survived, 4),
            speedup=round(survived, 4),
            peak_vram_mb=compile_optimized_vram,
            baseline_peak_vram_mb=compile_baseline_vram,
            mean_sm_clock_mhz=2388.2, max_temp_c=72.1,
            compile_baseline=True,
            notes="compile-baseline honesty check",
        ))

    out = Path(__file__).parent / "results_synthetic.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    print(f"wrote {out} — {len(rows)} rows, {len(configs())} configs, "
          f"{len(STRATEGIES)} strategies")


if __name__ == "__main__":
    main()
