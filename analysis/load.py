"""Load, clean and summarise `results.csv`.

`results.csv` is an append-only log, so it accumulates rows that are not
measurements: configs the memory pre-check refused, runs where the baseline
OOMed, runs the GPU throttled through, and outright accuracy failures. Every
one of those is worth keeping -- several are results in their own right -- but
none of them may leak into a mean. This module is where that filtering happens
once, so no figure has to re-derive it.

The central idea is `status`, derived from the reserved `notes` prefixes agreed
in docs/INTERFACE.md:

    SKIPPED       the config was never run (memory pre-check refused it)
    FAIL          the implementation was wrong; timing is meaningless
    OOM_BASELINE  the baseline could not run and ours could -- a *result*,
                  but there is no speedup ratio to average
    DISCARDED     the clock fell >15% mid-run; the timing is not trustworthy
    PASS          a usable measurement

Only PASS rows reach the summary statistics. The others are counted and
reported, because "we discarded four runs to thermal throttling" is a claim the
report should be able to make with a number behind it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

__all__ = [
    "load_results", "latest_per_config", "geometric_mean_speedup",
    "speedup_summary", "SpeedupSummary", "crossover_table", "dispatch_choices",
    "write_dispatch_table", "results_markdown_table", "status_counts",
    "CONFIG_FIELDS", "DISPATCH_FIELDS",
]

#: Everything that makes two rows different measurements of different things.
CONFIG_FIELDS: Sequence[str] = (
    "batch", "seq_len", "d_model", "heads", "layers", "dtype", "causal",
    "padding_ratio", "compile_baseline",
)

#: The projection the dispatcher actually keys on. `layers` is absent on
#: purpose: depth changes how long a forward pass takes but not which kernel is
#: the right one for a shape, so including it would fragment the table.
DISPATCH_FIELDS: Sequence[str] = (
    "batch", "seq_len", "d_model", "heads", "dtype", "causal", "padded",
)

_NUMERIC = (
    "batch", "seq_len", "d_model", "heads", "layers", "padding_ratio",
    "max_abs_err", "max_rel_err", "baseline_median_ms", "optimized_median_ms",
    "speedup", "peak_vram_mb", "mean_sm_clock_mhz", "max_temp_c",
    "baseline_peak_vram_mb", "ffn_dim",
)
_INTEGER = ("batch", "seq_len", "d_model", "heads", "layers", "ffn_dim")
_BOOLEAN = ("accuracy_pass", "causal", "compile_baseline")


def _to_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "1": True, "yes": True,
              "false": False, "0": False, "no": False})
    )


def _derive_status(row: pd.Series) -> str:
    notes = str(row.get("notes") or "")
    if notes.startswith("SKIPPED:"):
        return "SKIPPED"
    if notes.startswith("FAIL:") or row.get("accuracy_pass") is False:
        return "FAIL"
    if "baseline OOM" in notes:
        return "OOM_BASELINE"
    if "DISCARD:thermal" in notes:
        return "DISCARDED"
    return "PASS"


def load_results(path: Path | str) -> pd.DataFrame:
    """Read results.csv into a typed frame with `status` and key columns added.

    Tolerates a file written before the three proposed columns existed: any
    missing column is created empty rather than raising, so an older log still
    plots.
    """
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        frame = pd.DataFrame(columns=list(_NUMERIC) + list(_BOOLEAN) + ["notes"])

    for column in (*_NUMERIC, *_BOOLEAN, "notes", "timestamp", "git_sha", "strategy_name"):
        if column not in frame.columns:
            frame[column] = ""

    for column in _NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in _INTEGER:
        frame[column] = frame[column].astype("Int64")
    for column in _BOOLEAN:
        frame[column] = _to_bool(frame[column]).astype("boolean")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True, format="mixed")
    frame["notes"] = frame["notes"].fillna("")
    frame["status"] = (pd.Series(dtype="object") if frame.empty
                       else frame.apply(_derive_status, axis=1))
    frame["discarded"] = frame["notes"].str.contains("DISCARD:", na=False)
    frame["padded"] = frame["padding_ratio"].fillna(0) > 0

    # `DataFrame.apply(axis=1)` on an empty frame returns a DataFrame rather
    # than a Series, so assigning its result to one column raises. A results.csv
    # with a header and no rows is the normal state before the first sweep, so
    # this path has to work.
    if frame.empty:
        frame["config_key"] = pd.Series(dtype="object")
        frame["dispatch_key"] = pd.Series(dtype="object")
        return frame

    frame["config_key"] = frame.apply(
        lambda row: ",".join(str(row[field]) for field in CONFIG_FIELDS), axis=1
    )
    frame["dispatch_key"] = frame.apply(
        lambda row: ",".join(str(row[field]) for field in DISPATCH_FIELDS), axis=1
    )
    return frame


def status_counts(frame: pd.DataFrame) -> Dict[str, int]:
    """How many rows of each status. The report quotes these directly."""
    return frame["status"].value_counts().to_dict()


def usable(frame: pd.DataFrame, include_compiled_baseline: bool = False) -> pd.DataFrame:
    """PASS rows with a real speedup -- the only rows a mean may be taken over.

    Rows measured with `--compile-baseline` are excluded by default. They are a
    *different experiment*, not a newer run of the same one: the denominator of
    the ratio is a compiled baseline rather than the eager one the organizers'
    script uses. Mixing them in would silently understate every speedup, and
    because those runs happen later they would win `latest_per_config` and
    replace the eager rows outright. Only `compile_baseline_survival` wants
    both, and it asks for them.
    """
    subset = frame[(frame["status"] == "PASS") & frame["speedup"].notna()]
    if not include_compiled_baseline:
        subset = subset[subset["compile_baseline"] != True]  # noqa: E712
    return subset


def latest_per_config(
    frame: pd.DataFrame, only_pass: bool = True,
    include_compiled_baseline: bool = False,
) -> pd.DataFrame:
    """Newest row per (strategy, config), so a re-run supersedes its predecessor.

    A sweep gets re-run after a fix; without this, the old rows would still be
    averaged in and a fixed strategy would look worse than it is.
    """
    subset = usable(frame, include_compiled_baseline) if only_pass else frame
    if subset.empty:
        return subset
    ordered = subset.sort_values("timestamp")
    return (
        ordered.groupby(["strategy_name", "config_key"], as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class SpeedupSummary:
    strategy: str
    geomean: float
    minimum: float
    maximum: float
    n: int


def geometric_mean_speedup(
    frame: pd.DataFrame, strategy: str, only_pass: bool = True
) -> float:
    """Geometric mean of a strategy's speedups. NaN if it has none.

    Geometric, not arithmetic: speedups are ratios. A strategy that is 2x on one
    shape and 0.5x on another has done nothing on average, and only the
    geometric mean says so -- the arithmetic mean would call it 1.25x.
    """
    subset = usable(frame) if only_pass else frame
    values = subset.loc[subset["strategy_name"] == strategy, "speedup"].dropna()
    values = values[values > 0]
    if values.empty:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def speedup_summary(
    frame: pd.DataFrame, strategies: Optional[Sequence[str]] = None
) -> List[SpeedupSummary]:
    """Per-strategy geomean/min/max/n, best first."""
    subset = usable(frame)
    names = strategies if strategies is not None else sorted(subset["strategy_name"].unique())
    out = []
    for name in names:
        values = subset.loc[subset["strategy_name"] == name, "speedup"].dropna()
        if values.empty:
            continue
        out.append(SpeedupSummary(
            strategy=name,
            geomean=geometric_mean_speedup(frame, name),
            minimum=float(values.min()),
            maximum=float(values.max()),
            n=int(len(values)),
        ))
    return sorted(out, key=lambda s: s.geomean, reverse=True)


def crossover_table(frame: pd.DataFrame, exclude_baseline: bool = True) -> pd.DataFrame:
    """Per config: the winning strategy, its speedup, and the runner-up.

    The gap between winner and runner-up is the number that decides whether a
    dispatch table is worth having at all. If it is 2% everywhere, one strategy
    would do and the shape checks are theatre.
    """
    subset = latest_per_config(frame)
    if exclude_baseline:
        subset = subset[subset["strategy_name"] != "baseline"]
    if subset.empty:
        return pd.DataFrame(columns=[
            *CONFIG_FIELDS, "config_key", "dispatch_key", "best_strategy",
            "best_speedup", "runner_up", "runner_up_speedup", "margin",
        ])

    rows = []
    for config_key, group in subset.groupby("config_key", sort=False):
        ordered = group.sort_values("speedup", ascending=False)
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else None
        rows.append({
            **{field: best[field] for field in CONFIG_FIELDS},
            "config_key": config_key,
            "dispatch_key": best["dispatch_key"],
            "best_strategy": best["strategy_name"],
            "best_speedup": float(best["speedup"]),
            "runner_up": None if second is None else second["strategy_name"],
            "runner_up_speedup": None if second is None else float(second["speedup"]),
            "margin": None if second is None
                      else float(best["speedup"]) - float(second["speedup"]),
        })
    return pd.DataFrame(rows).sort_values(["seq_len", "batch", "d_model"]).reset_index(drop=True)


def dispatch_choices(frame: pd.DataFrame, exclude_baseline: bool = True) -> pd.DataFrame:
    """Per *dispatch* key: which strategy to pick, and how clear the win is.

    This is not `crossover_table` with a different index. `config_key` carries
    `layers`; `dispatch_key` cannot, because the dispatcher only sees the tensor
    it is handed and a shape says nothing about how deep the stack is. So
    several measured configs collapse onto one dispatch key, and the winner
    among them has to be *decided*, not taken from whichever row sorted last.

    The decision: within each dispatch key, score every strategy by the
    geometric mean of its speedups across the configs that collapsed together,
    and take the argmax. A strategy that wins narrowly at one depth and loses
    badly at another does not win the key.
    """
    subset = latest_per_config(frame)
    if exclude_baseline:
        subset = subset[subset["strategy_name"] != "baseline"]
    if subset.empty:
        return pd.DataFrame(columns=[
            "dispatch_key", "best_strategy", "best_speedup", "runner_up",
            "runner_up_speedup", "margin", "n_configs",
        ])

    rows = []
    for dispatch_key, group in subset.groupby("dispatch_key", sort=False):
        scored = (
            group.groupby("strategy_name")["speedup"]
            .apply(lambda v: math.exp(sum(math.log(x) for x in v if x > 0) / max(1, len(v))))
            .sort_values(ascending=False)
        )
        best_name = scored.index[0]
        second_name = scored.index[1] if len(scored) > 1 else None
        rows.append({
            "dispatch_key": dispatch_key,
            "best_strategy": best_name,
            "best_speedup": float(scored.iloc[0]),
            "runner_up": second_name,
            "runner_up_speedup": None if second_name is None else float(scored.iloc[1]),
            "margin": None if second_name is None
                      else float(scored.iloc[0] - scored.iloc[1]),
            "n_configs": int(group["config_key"].nunique()),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def write_dispatch_table(
    frame: pd.DataFrame,
    path: Path | str = "results/dispatch_table.json",
    capability: str = "sm_89",
) -> Path:
    """Turn the crossover table into the JSON `src/dispatch.py` consumes.

    `capability` names the GPU these measurements came from. Every key under it
    was measured on that card; a different card gets its own block, and
    `src/dispatch.py` falls back to `default` when it has no block of its own.
    Performance is only ever claimed for capabilities we actually measured.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    choices = dispatch_choices(frame)
    mapping = {
        str(row["dispatch_key"]): str(row["best_strategy"])
        for _, row in choices.iterrows()
    }
    assert len(mapping) == len(choices), "dispatch keys must be unique by construction"

    summaries = speedup_summary(frame)
    non_baseline = [s for s in summaries if s.strategy != "baseline"]
    default = non_baseline[0].strategy if non_baseline else "baseline"

    usable_rows = usable(frame)
    newest = (
        usable_rows.sort_values("timestamp").iloc[-1] if not usable_rows.empty else None
    )

    table = {
        capability: mapping,
        "default": default,
        "meta": {
            "generated_from_git_sha": None if newest is None else str(newest["git_sha"]),
            "newest_row_timestamp": None if newest is None
                                    else str(newest["timestamp"]),
            "n_dispatch_keys": len(mapping),
            "n_measured_configs": int(latest_per_config(frame)["config_key"].nunique())
                                  if not latest_per_config(frame).empty else 0,
            "geometric_mean_speedup": {
                s.strategy: round(s.geomean, 4) for s in summaries
            },
            "status_counts": status_counts(frame),
            "note": (
                "Generated by analysis/load.py from results.csv. "
                "src/dispatch.py falls back to a hard-coded table if this file "
                "is missing, so the submission never depends on a build artefact."
            ),
        },
    }
    path.write_text(json.dumps(table, indent=2, default=str) + "\n")
    return path


def results_markdown_table(frame: pd.DataFrame, index: str = "seq_len") -> str:
    """Speedup table for the README and report: rows = config, cols = strategy."""
    subset = latest_per_config(frame)
    if subset.empty:
        return "_no passing rows yet_"

    pivot = subset.pivot_table(
        index=index, columns="strategy_name", values="speedup", aggfunc="median"
    )
    if "baseline" in pivot.columns:  # control first, then best-to-worst
        others = sorted(
            (c for c in pivot.columns if c != "baseline"),
            key=lambda c: pivot[c].median(), reverse=True,
        )
        pivot = pivot[["baseline", *others]]

    header = f"| {index} | " + " | ".join(pivot.columns) + " |"
    divider = "|---" * (len(pivot.columns) + 1) + "|"
    lines = [header, divider]
    for key, row in pivot.iterrows():
        cells = ["—" if pd.isna(v) else f"{v:.2f}x" for v in row]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
