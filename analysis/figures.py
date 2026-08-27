"""Every figure in the report. One function per figure, each returning its path.

Conventions that hold across all of them, so the set reads as one system:

  * colours come from `analysis.style` in slot order and are never cycled;
  * the grid and axes are recessive and solid — a dashed rule reads as data;
  * a legend is always present for two or more series, so identity is never
    carried by colour alone;
  * no value is printed on every point; annotation is selective;
  * only PASS rows are plotted. Skipped, failed, OOM and thermally discarded
    rows are counted in `results/summary.md` instead, where they can be read as
    what they are rather than smoothed into a line.

Figures are light-only PNGs: they are embedded in Markdown, not rendered in a
themed page, so there is no viewer preference to respond to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from analysis.load import CONFIG_FIELDS, latest_per_config, speedup_summary, usable
from analysis.style import INK, STATUS, color_for, finish, new_figure

__all__ = [
    "speedup_vs_seq_len", "speedup_vs_batch", "speedup_vs_dmodel",
    "accuracy_budget", "vram_ceiling", "thermal_trace",
    "compile_baseline_survival", "ALL_FIGURES",
]

FIGURE_DIR = Path("results/figures")

#: 6 GB card. The line every VRAM figure is really about.
VRAM_LIMIT_MB = 6144

#: The tolerance we target, and the looser one the problem statement PDF quotes.
ATOL_TARGET, ATOL_PDF = 0.001, 0.002
RTOL_TARGET, RTOL_PDF = 0.01, 0.02


def _prepare(out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def _empty(fig, ax, path: Path, title: str, message: str) -> Path:
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, color=INK["muted"], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    finish(fig, ax, title)
    return _save(fig, path)


def _ofat_slice(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Rows that vary only in `axis`, with every other config field held fixed.

    Without this, `groupby(axis)` silently aggregates across *different*
    configurations that happen to share one coordinate -- so a "peak memory vs
    sequence length" line mixes fp32 and bf16 runs at different depths, and can
    come out non-monotonic. Peak memory cannot fall as the sequence grows; a
    line that says otherwise is an artefact of the grouping, not a measurement.

    The held-fixed combination is the modal one, which for a one-factor-at-a-time
    sweep is the base config by construction: it is the only combination that
    appears at every value of the swept axis.
    """
    if frame.empty:
        return frame
    others = [field for field in CONFIG_FIELDS if field != axis]
    combinations = frame[others].astype(str).agg(",".join, axis=1)
    modal = combinations.mode()
    if modal.empty:
        return frame
    return frame[combinations == modal.iloc[0]]


def _strategy_order(frame: pd.DataFrame) -> List[str]:
    """Best first, control last. Colour follows the entity, so this order is
    computed once per figure set and never changes when a filter drops a series."""
    ranked = [s.strategy for s in speedup_summary(frame) if s.strategy != "baseline"]
    present = set(frame["strategy_name"].dropna().unique())
    ordered = [s for s in ranked if s in present]
    if "baseline" in present:
        ordered.append("baseline")
    return ordered


# ---------------------------------------------------------------------------
# 1-3. speedup against each shape axis
# ---------------------------------------------------------------------------

def _speedup_vs(
    frame: pd.DataFrame,
    axis: str,
    axis_label: str,
    filename: str,
    title: str,
    out_dir: Path | str = FIGURE_DIR,
    log_x: bool = True,
) -> Path:
    out_dir = _prepare(out_dir)
    path = out_dir / filename
    fig, ax = new_figure()

    data = _ofat_slice(latest_per_config(frame), axis)
    if data.empty or data[axis].nunique() < 2:
        return _empty(fig, ax, path, title,
                      f"needs PASS rows at two or more {axis} values\n"
                      f"(run: bench/sweep.py --matrix {_MATRIX_FOR[axis]})")

    order = _strategy_order(data)
    for index, strategy in enumerate(order):
        group = data[data["strategy_name"] == strategy]
        series = group.groupby(axis)["speedup"].median().sort_index()
        if series.empty:
            continue
        is_control = strategy == "baseline"
        ax.plot(
            series.index, series.values,
            marker="o", markersize=5,
            color=INK["muted"] if is_control else color_for(index),
            linewidth=1.4 if is_control else 2.0,
            label=strategy, zorder=2 if is_control else 3,
        )

    # The reference the whole chart is read against.
    ax.axhline(1.0, color=INK["axis"], linewidth=1.0, zorder=1)
    ax.annotate("1.0x — no change", xy=(0.995, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK["muted"], ha="right", va="bottom")

    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(data[axis].dropna().unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(axis_label)
    ax.set_ylabel("speedup (baseline median / optimized median)")
    # Headroom so the legend never sits on the top series.
    top_value = float(data["speedup"].max())
    ax.set_ylim(
        bottom=min(0.95, float(data["speedup"].min()) * 0.95),
        top=top_value * 1.18,
    )
    ax.legend(loc="upper left", ncols=2)

    finish(fig, ax, title, _held_fixed(data, axis))
    return _save(fig, path)


#: Which sweep produces the rows each figure needs, for the empty-state message.
_MATRIX_FOR = {"seq_len": "seq", "batch": "batch", "d_model": "dmodel"}


def _held_fixed(frame: pd.DataFrame, axis: str) -> str:
    """One-line statement of what was held constant, for the subtitle.

    A reader cannot judge a one-factor-at-a-time line without knowing the other
    factors, so the figure states them rather than leaving them implicit.
    """
    if frame.empty:
        return ""
    row = frame.iloc[0]
    parts = []
    for field, label in (("batch", "B"), ("seq_len", "S"), ("d_model", "d"),
                         ("heads", "H"), ("layers", "L")):
        if field != axis:
            parts.append(f"{label}={row[field]}")
    parts.append(str(row["dtype"]))
    if bool(row["causal"]):
        parts.append("causal")
    if float(row["padding_ratio"] or 0) > 0:
        parts.append(f"pad={row['padding_ratio']:g}")
    return "held fixed: " + " ".join(parts) + "; PASS rows only"


def speedup_vs_seq_len(frame, out_dir=FIGURE_DIR) -> Path:
    return _speedup_vs(
        frame, "seq_len", "sequence length (tokens)", "speedup_vs_seq_len.png",
        "Speedup against sequence length", out_dir,
    )


def speedup_vs_batch(frame, out_dir=FIGURE_DIR) -> Path:
    return _speedup_vs(
        frame, "batch", "batch size", "speedup_vs_batch.png",
        "Speedup against batch size", out_dir,
    )


def speedup_vs_dmodel(frame, out_dir=FIGURE_DIR) -> Path:
    return _speedup_vs(
        frame, "d_model", "model dimension", "speedup_vs_dmodel.png",
        "Speedup against model dimension", out_dir,
    )


# ---------------------------------------------------------------------------
# 4. accuracy budget
# ---------------------------------------------------------------------------

def accuracy_budget(frame: pd.DataFrame, out_dir: Path | str = FIGURE_DIR) -> Path:
    """Error against depth, one line per dtype, with both tolerance lines drawn.

    Two panels rather than two y-axes on one plot: absolute and relative error
    are different measures on different scales, and overlaying them on twin axes
    invents a crossing point that means nothing.
    """
    out_dir = _prepare(out_dir)
    path = out_dir / "accuracy_budget.png"
    fig, axes = new_figure(7.5, 6.0, nrows=2, sharex=True)
    top, bottom = axes

    data = usable(frame)
    data = data[data["strategy_name"] != "baseline"]  # the control is exactly 0
    if data.empty or data["layers"].nunique() < 2:
        return _empty(fig, top, path, "Accuracy budget against depth",
                      "needs PASS rows at two or more layer counts\n"
                      "(run: bench/sweep.py --matrix layers)")

    dtypes = sorted(data["dtype"].dropna().unique())
    for index, dtype in enumerate(dtypes):
        group = data[data["dtype"] == dtype]
        for axis, column in ((top, "max_abs_err"), (bottom, "max_rel_err")):
            series = group.groupby("layers")[column].max().sort_index()
            if series.empty:
                continue
            axis.plot(series.index, series.values, marker="o", markersize=5,
                      color=color_for(index), label=str(dtype))

    for axis, target, pdf, target_name, pdf_name in (
        (top, ATOL_TARGET, ATOL_PDF, "atol 0.001 (we target)", "0.002 (PDF)"),
        (bottom, RTOL_TARGET, RTOL_PDF, "rtol 1% (we target)", "2% (PDF)"),
    ):
        axis.axhline(target, color=STATUS["critical"], linewidth=1.2, zorder=1)
        axis.axhline(pdf, color=STATUS["warning"], linewidth=1.2, zorder=1)
        axis.annotate(target_name, xy=(0.01, target), xycoords=("axes fraction", "data"),
                      xytext=(0, 3), textcoords="offset points",
                      fontsize=8, color=STATUS["critical"])
        axis.annotate(pdf_name, xy=(0.01, pdf), xycoords=("axes fraction", "data"),
                      xytext=(0, 3), textcoords="offset points",
                      fontsize=8, color=STATUS["warning"])
        axis.set_yscale("log")

    top.set_ylabel("max absolute error")
    bottom.set_ylabel("max relative error")
    bottom.set_xlabel("layers")
    bottom.set_xticks(sorted(data["layers"].dropna().unique()))
    # Lower right: the tolerance rules and their labels own the top of both
    # panels, and the data climbs left-to-right, so this corner is the only one
    # that is reliably empty.
    top.legend(loc="lower right", title="dtype", ncols=len(dtypes))

    finish(fig, [top], "Accuracy budget against depth",
           "worst element across runs; error compounds with layer count")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 5. VRAM ceiling
# ---------------------------------------------------------------------------

def vram_ceiling(frame: pd.DataFrame, out_dir: Path | str = FIGURE_DIR) -> Path:
    """Peak memory against sequence length, and where the baseline stops fitting.

    The interesting rows here are the ones with no speedup at all: configs where
    the baseline OOMed and the optimized path completed. They are marked at the
    top of the plot rather than dropped, because "ours runs where the reference
    cannot" is a result the speedup column has no way to express.
    """
    out_dir = _prepare(out_dir)
    path = out_dir / "vram_ceiling.png"
    fig, ax = new_figure()

    measured = usable(frame)
    if measured.empty or measured["peak_vram_mb"].isna().all():
        return _empty(fig, ax, path, "VRAM ceiling",
                      "no peak-memory data yet (CUDA-only measurement)")

    # Hold every axis but seq_len fixed: peak memory is an absolute quantity, so
    # mixing dtypes and depths into one line produces a curve that can fall as S
    # grows, which is impossible.
    measured = _ofat_slice(measured, "seq_len")
    if measured.empty:
        return _empty(fig, ax, path, "VRAM ceiling",
                      "needs peak-memory rows across sequence lengths\n"
                      "(run: bench/sweep.py --matrix seq, on the GPU)")

    baseline_line = (
        measured.groupby("seq_len")["baseline_peak_vram_mb"].max().dropna().sort_index()
    )
    best = next((s.strategy for s in speedup_summary(frame)
                 if s.strategy != "baseline"), None)
    best_line = pd.Series(dtype=float)
    if best is not None:
        best_line = (
            measured[measured["strategy_name"] == best]
            .groupby("seq_len")["peak_vram_mb"].max().dropna().sort_index()
        )

    if not baseline_line.empty:
        ax.plot(baseline_line.index, baseline_line.values, marker="o",
                color=color_for(0), label="baseline")
    if not best_line.empty:
        ax.plot(best_line.index, best_line.values, marker="o",
                color=color_for(1), label=f"{best} (best)")

    ax.axhline(VRAM_LIMIT_MB, color=INK["secondary"], linewidth=1.2, zorder=1)
    ax.annotate(f"{VRAM_LIMIT_MB} MB card", xy=(0.99, VRAM_LIMIT_MB),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=8,
                color=INK["secondary"], ha="right")

    oom = frame[frame["status"] == "OOM_BASELINE"]
    if not oom.empty:
        top_y = max(VRAM_LIMIT_MB, float(measured["baseline_peak_vram_mb"].max() or 0)) * 1.08
        ax.scatter(oom["seq_len"], [top_y] * len(oom), marker="X", s=90,
                   color=STATUS["critical"], zorder=4,
                   label="baseline OOM — optimized completed")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(measured["seq_len"].dropna().unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("peak memory (MB)")
    ax.legend(loc="upper left")

    finish(fig, ax, "VRAM ceiling",
           "the baseline materializes [B, H, S, S]; a fused path does not")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 6. thermal trace
# ---------------------------------------------------------------------------

def thermal_trace(
    log_csv: Path | str,
    out_dir: Path | str = FIGURE_DIR,
    filename: str = "thermal_trace.png",
    window: Optional[Sequence[float]] = None,
) -> Path:
    """SM clock and temperature over one run, as two stacked panels.

    Deliberately **not** a dual-axis chart. Two y-scales on one plot let the
    reader infer a relationship from where the lines happen to cross, which is
    an artefact of the two arbitrary scalings and not a fact about the GPU.
    Stacked panels sharing the time axis show the same data and support the
    same reading — clock falls, temperature rises — without inviting that
    inference.
    """
    from bench.thermal import DISCARD_CLOCK_FRACTION, parse_clock_log, summarize

    out_dir = _prepare(out_dir)
    path = out_dir / filename
    fig, axes = new_figure(7.5, 5.4, nrows=2, sharex=True)
    clock_ax, temp_ax = axes

    frame = parse_clock_log(log_csv)
    if frame.empty:
        return _empty(fig, clock_ax, path, "Clock and temperature during one run",
                      "clock log is empty")

    stats = summarize(frame, window=tuple(window) if window else None)

    clock_ax.plot(frame["t"], frame["sm_mhz"], color=color_for(0), linewidth=1.8)
    clock_ax.set_ylabel("SM clock (MHz)")

    opening = stats["opening_sm_mhz"]
    if opening:
        threshold = DISCARD_CLOCK_FRACTION * opening
        clock_ax.axhline(threshold, color=STATUS["critical"], linewidth=1.2)
        clock_ax.annotate(
            f"discard below {threshold:.0f} MHz "
            f"({DISCARD_CLOCK_FRACTION:.0%} of opening)",
            xy=(0.99, threshold), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points",
            fontsize=8, color=STATUS["critical"], ha="right",
        )
        clock_ax.axhline(opening, color=INK["axis"], linewidth=1.0, zorder=1)

    temp_ax.plot(frame["t"], frame["temp_c"], color=color_for(1), linewidth=1.8)
    temp_ax.set_ylabel("temperature (°C)")
    temp_ax.set_xlabel("seconds from start of run")

    if window:
        for axis in (clock_ax, temp_ax):
            axis.axvspan(window[0], window[1], color=INK["grid"], zorder=0)

    verdict = "DISCARDED" if stats["discard"] else "kept"
    finish(fig, [clock_ax], "Clock and temperature during one run",
           f"mean {stats['mean_sm_clock_mhz']:.0f} MHz vs opening {opening:.0f} MHz "
           f"-> {verdict}")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 7. compile-baseline honesty check
# ---------------------------------------------------------------------------

def compile_baseline_survival(frame: pd.DataFrame, out_dir: Path | str = FIGURE_DIR) -> Path:
    """How much speedup survives when the baseline is compiled too.

    Some of any speedup over an eager baseline is just `torch.compile` doing
    what it does to *any* PyTorch model. Comparing against a compiled baseline
    separates what our kernels contributed from what the compiler would have
    given anyone. Reporting only the first number is the easiest way to overstate
    a result, so the check gets its own figure.
    """
    out_dir = _prepare(out_dir)
    path = out_dir / "compile_baseline_survival.png"
    fig, ax = new_figure()

    data = usable(frame, include_compiled_baseline=True)
    data = data[data["strategy_name"] != "baseline"]
    if data.empty or data["compile_baseline"].nunique() < 2:
        return _empty(fig, ax, path, "Speedup survival against a compiled baseline",
                      "needs rows both with and without --compile-baseline\n"
                      "(run: bench/sweep.py --strategy <s> --compile-baseline)")

    eager = data[data["compile_baseline"] == False].groupby("strategy_name")["speedup"].median()  # noqa: E712
    compiled = data[data["compile_baseline"] == True].groupby("strategy_name")["speedup"].median()  # noqa: E712
    strategies = [s for s in _strategy_order(data) if s in eager.index and s in compiled.index]
    if not strategies:
        return _empty(fig, ax, path, "Speedup survival against a compiled baseline",
                      "no strategy has rows in both conditions yet")

    import numpy as np

    positions = np.arange(len(strategies))
    width = 0.38
    gap = 0.02  # a surface gap between adjacent bars, never a stroke

    ax.bar(positions - width / 2 - gap, [eager[s] for s in strategies], width,
           color=color_for(0), label="vs eager baseline")
    ax.bar(positions + width / 2 + gap, [compiled[s] for s in strategies], width,
           color=color_for(1), label="vs compiled baseline")

    ax.axhline(1.0, color=INK["axis"], linewidth=1.0, zorder=1)
    for index, strategy in enumerate(strategies):
        survived = compiled[strategy] / eager[strategy] if eager[strategy] else float("nan")
        ax.annotate(f"{survived:.0%} survives",
                    xy=(index, max(eager[strategy], compiled[strategy])),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color=INK["secondary"])

    ax.set_xticks(positions)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("speedup")
    ax.set_xlabel("")
    ax.legend(loc="upper right")

    finish(fig, ax, "Speedup survival against a compiled baseline",
           "how much of the gain is ours rather than torch.compile's")
    return _save(fig, path)


#: Figures that take only the results frame. `thermal_trace` is excluded: it
#: takes a clock log, and there may be many or none.
ALL_FIGURES = (
    speedup_vs_seq_len, speedup_vs_batch, speedup_vs_dmodel,
    accuracy_budget, vram_ceiling, compile_baseline_survival,
)
