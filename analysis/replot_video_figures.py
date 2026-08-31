#!/usr/bin/env python3
"""Video-legible replots of two figures, written alongside the originals.

Standalone on purpose: these are presentation variants, not the canonical
figures. Nothing here edits `analysis/figures.py` and nothing there imports
this, so `python -m analysis.make_all` still regenerates `vram_ceiling.png`
and `accuracy_budget.png` exactly as before. These write `*_v2.png` beside
them.

Run:

    python analysis/replot_video_figures.py

Two figures, two different problems.

1. `vram_ceiling.png` draws a 6144 MB card-capacity rule, but the measured data
   tops out at 808.5 MB -- 13% of the axis. Every real point is squashed into
   the bottom eighth of the plot and the finding is invisible. The v2 drops the
   capacity line and lets the axis follow the data.

2. `accuracy_budget.png` has a subtler problem. It sources from `usable()`,
   which keeps only PASS rows -- and for reduced precision the only PASS rows
   are the ones where the router fell back to the baseline, so they carry an
   error of exactly 0.0. Zero is unplottable on a log axis, which is why
   bfloat16 appears in the legend with no line: not a missing series, a series
   of baseline-fallback zeros. The rows that carry the actual finding --
   bf16 failing from 0.031 to 0.094, fp16 from 0.0039 to 0.0078 -- are FAIL
   rows, and `usable()` drops them.

   An accuracy-budget plot is the one figure where filtering to PASS is
   actively wrong: the failures are the result. The v2 plots every row with a
   real measured error, pass or fail, and excludes the baseline-fallback zeros
   because they are not measurements of our kernel.

   The relative-error panel is dropped entirely. `max_rel_err` divides by
   `ref.abs().clamp_min(1e-12)`, so it explodes to 1e+06 wherever the reference
   output is near zero. That is a property of the denominator, not of the
   implementation, and on camera it reads as catastrophic failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.figures import _ofat_slice  # noqa: E402  read-only import
from analysis.load import load_results, speedup_summary, usable  # noqa: E402
from analysis.roofline import (  # noqa: E402  read-only import
    RTX_4050_LAPTOP, ridge_point, roofline_frame,
)
from analysis.style import INK, STATUS, color_for, finish, new_figure  # noqa: E402

RESULTS = ROOT / "results" / "results.csv"
FIGURE_DIR = ROOT / "results" / "figures"

ATOL_TARGET = 0.001

#: Bumped for video. The house style targets a report page read at arm's length;
#: these are read off a 1080p screen recording, where 10pt axis labels turn to
#: mush after compression.
VIDEO_FONTS = {
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
}

DTYPE_ORDER = ("float32", "float16", "bfloat16")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=INK["surface"])
    plt.close(fig)
    return path


def vram_ceiling_v2(frame: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> Path:
    """Peak memory against sequence length, scaled to the data.

    No capacity rule: at 6144 MB it compresses an 808 MB dataset into the
    bottom eighth of the axis. The card limit is a real constraint but it is
    not what this figure is about, and `vram_ceiling.png` still carries it.
    """
    path = out_dir / "vram_ceiling_v2.png"

    measured = _ofat_slice(usable(frame), "seq_len")
    if measured.empty or measured["peak_vram_mb"].isna().all():
        raise SystemExit("no peak-memory rows to plot")

    baseline = (
        measured.groupby("seq_len")["baseline_peak_vram_mb"].max().dropna().sort_index()
    )
    best = next((s.strategy for s in speedup_summary(frame) if s.strategy != "baseline"), None)
    ours = (
        measured[measured["strategy_name"] == best]
        .groupby("seq_len")["peak_vram_mb"].max().dropna().sort_index()
    )
    shared = [s for s in ours.index if s in baseline.index]

    with plt.rc_context(VIDEO_FONTS):
        fig, ax = new_figure(8.0, 5.0)

        ax.plot(baseline.index, baseline.values, marker="o", markersize=8,
                linewidth=2.4, color=color_for(0),
                label="baseline — materializes [B, H, S, S]")
        ax.plot(ours.index, ours.values, marker="s", markersize=8,
                linewidth=2.4, color=color_for(1),
                label=f"fused attention ({best}) — never materializes it")

        # The finding is the *widening*, so label every ratio, not just the best.
        for seq in shared:
            ratio = float(baseline[seq]) / float(ours[seq])
            headline = seq == shared[-1]
            # Where the two lines nearly touch there is no midpoint to write in,
            # so the label goes above the pair instead of on top of them.
            if ratio < 1.10:
                anchor, offset, align = float(baseline[seq]), (0, 14), "center"
            else:
                anchor, offset, align = (
                    (float(baseline[seq]) + float(ours[seq])) / 2, (10, 0), "left",
                )
            ax.annotate(
                f"{ratio:.2f}x",
                xy=(seq, anchor), xytext=offset, textcoords="offset points",
                fontsize=15 if headline else 13,
                fontweight="bold" if headline else "normal",
                color=STATUS["good"] if headline else INK["secondary"],
                va="center", ha=align,
            )
            ax.vlines(seq, ours[seq], baseline[seq],
                      color=INK["muted"], linewidth=1.0, zorder=1)

        top = float(baseline.max())
        ax.set_ylim(0, top * 1.18)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(baseline.index))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("sequence length (tokens)")
        ax.set_ylabel("peak GPU memory (MB)")
        ax.legend(loc="upper left", frameon=False)

        finish(fig, ax, "Peak memory falls further the longer the sequence",
               "B=8, d=512, 6 layers, fp32 — the saving compounds because the "
               "score matrix grows as S²")
        return _save(fig, path)


def accuracy_budget_v2(frame: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> Path:
    """Absolute error against depth, per dtype, against the atol budget.

    Includes FAIL rows deliberately -- see the module docstring. Excludes rows
    whose error is exactly 0.0, which are baseline-fallback runs rather than
    measurements of the fused path.
    """
    path = out_dir / "accuracy_budget_v2.png"

    data = frame[
        (frame["strategy_name"] != "baseline")
        & frame["max_abs_err"].notna()
        & (frame["max_abs_err"] > 0)
        & frame["layers"].notna()
    ]
    if data.empty:
        raise SystemExit("no rows with a measured absolute error")

    series = {}
    for dtype in DTYPE_ORDER:
        group = data[data["dtype"] == dtype]
        if group.empty:
            continue
        points = group.groupby("layers")["max_abs_err"].max().sort_index()
        if points.empty:
            continue
        series[dtype] = points

    dropped = [d for d in DTYPE_ORDER if d not in series]

    with plt.rc_context(VIDEO_FONTS):
        fig, ax = new_figure(8.0, 5.0)

        for index, (dtype, points) in enumerate(series.items()):
            ax.plot(points.index, points.values, linewidth=2.4,
                    color=color_for(index), label=dtype, zorder=3)
            over = points[points > ATOL_TARGET]
            under = points[points <= ATOL_TARGET]
            # Shape carries pass/fail so the reading survives a greyscale
            # compression pass; colour is already spent on dtype.
            ax.scatter(under.index, under.values, marker="o", s=90,
                       color=color_for(index), zorder=4)
            ax.scatter(over.index, over.values, marker="X", s=130,
                       color=color_for(index), edgecolors=STATUS["critical"],
                       linewidths=1.4, zorder=4)

        ax.axhline(ATOL_TARGET, color=STATUS["critical"], linewidth=2.0, zorder=2)
        ax.annotate("atol = 0.001  (the budget)",
                    xy=(0.015, ATOL_TARGET), xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=13, color=STATUS["critical"], fontweight="bold")

        ax.set_yscale("log")
        ax.set_xticks(sorted({int(i) for p in series.values() for i in p.index}))
        ax.set_xlabel("layers")
        ax.set_ylabel("max absolute error")
        ax.legend(loc="lower right", title="dtype", frameon=False, ncols=len(series))
        ax.annotate("X = over budget", xy=(0.015, 0.93), xycoords="axes fraction",
                    fontsize=12, color=INK["secondary"])

        finish(fig, ax, "Only fp32 stays inside the accuracy budget",
               "worst element per depth, pass and fail rows alike")
        saved = _save(fig, path)

    if dropped:
        print(f"  note: no plottable rows for {', '.join(dropped)} — omitted from the legend")
    return saved


#: Strategy → marker. Shape carries strategy so colour is free for dtype.
_STRATEGY_MARKERS = {"baseline": "o", "sdpa": "^", "optimized": "s"}


def roofline_v2(frame: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> Path:
    """Log-log roofline, rebuilt for legibility at video resolution.

    Three defects in the original, two cosmetic and one substantive.

    *Ticks.* A log x-axis draws minor ticks per decade and, over a range this
    narrow, matplotlib labelled majors and minors both — producing overlapping
    strings like "2 x 10^4 10 10.0 10^1 10^2". Minor labels are suppressed and
    the majors are set explicitly.

    *Legend.* Two legends sat at upper-left, directly under the title and across
    the dotted roofs. One combined legend now sits lower-right, the only region
    the data leaves empty.

    *The missing reduced-precision points.* They were never missing. Both bf16
    rows and the fp16 six-layer row land at the same arithmetic intensity
    (227.55) and within 0.6% of the same throughput (5.70 / 5.71 / 5.73), so
    they draw on top of one another and only the last survives. But they should
    not be there at all: all three are runs where the router fell back to the
    baseline, so they measure the *reference* in bf16, not a fused kernel. What
    remains is the one genuine reduced-precision measurement — sdpa in fp16 at a
    single layer, the highest throughput anywhere in the log — which is worth
    drawing precisely because it fails the tolerance from two layers up.
    """
    path = out_dir / "roofline_v2.png"
    machine = RTX_4050_LAPTOP

    data = roofline_frame(frame)
    if data.empty:
        raise SystemExit("no passing rows to plot")

    fp32 = data[data["dtype"] == "float32"]
    # Genuine reduced-precision work only: a fallback row (speedup ~1.0) is the
    # baseline running in that dtype, not our kernel.
    reduced = data[(data["dtype"] != "float32") & (data["speedup"] > 1.05)]
    dropped = len(data) - len(fp32) - len(reduced)

    with plt.rc_context(VIDEO_FONTS):
        fig, ax = new_figure(9.0, 5.6)

        # Roofs. Widened well past the data so the ridge corners are visible and
        # nothing is crushed against a vertical rule.
        lo, hi = 20.0, 700.0
        xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
        roofs = [
            ("float32", machine.peak_fp32_tflops, "-", 2.4, True),
            ("float16", machine.peak_fp16_tflops, ":", 2.0, False),
            ("bfloat16", machine.peak_bf16_tflops, ":", 2.0, False),
        ]
        handles = []
        for index, (dtype, peak, style, width, shippable) in enumerate(roofs):
            ys = [min(machine.bandwidth_gbs * xi / 1000.0, peak) for xi in xs]
            colour = INK["primary"] if shippable else INK["muted"]
            line, = ax.plot(xs, ys, style, color=colour, linewidth=width, zorder=2)
            ridge = ridge_point(peak, machine.bandwidth_gbs)
            ax.plot([ridge], [peak], marker="|", markersize=13,
                    color=colour, zorder=3)
            # Short labels on purpose: the legend has to fit in the empty
            # right-hand quarter without covering any measured point.
            short = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[dtype]
            label = (f"{short} roof · {peak:g}" if shippable
                     else f"{short} roof · {peak:g} — unreachable")
            line.set_label(label)
            handles.append(line)

        # fp32 points, shape by strategy.
        for index, (strategy, marker) in enumerate(_STRATEGY_MARKERS.items()):
            group = fp32[fp32["strategy_name"] == strategy]
            if group.empty:
                continue
            ax.scatter(group["intensity"], group["achieved_tflops"],
                       marker=marker, s=95, color=color_for(index),
                       edgecolors="white", linewidths=0.8, zorder=5,
                       label=f"{strategy}, fp32")

        # The one real reduced-precision datapoint.
        if not reduced.empty:
            ax.scatter(reduced["intensity"], reduced["achieved_tflops"],
                       marker="*", s=320, color=STATUS["warning"],
                       edgecolors=INK["primary"], linewidths=1.0, zorder=6,
                       label="sdpa fp16, 1 layer")
            # Parked in the empty wedge above the fp32 roof and left of the
            # fp16 ridge, with a leader to the point — anywhere nearer the star
            # crosses either the roof line or the legend.
            ax.annotate(
                f"fp16 reaches {reduced['achieved_tflops'].max():.1f} TFLOP/s —\n"
                "the highest measured anywhere,\nand it fails from two layers up",
                xy=(float(reduced["intensity"].max()),
                    float(reduced["achieved_tflops"].max())),
                xycoords="data",
                xytext=(0.03, 0.90), textcoords="axes fraction",
                fontsize=12.5, color=INK["secondary"], ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK["muted"],
                                linewidth=1.0, connectionstyle="arc3,rad=-0.15"),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(2.0, 34.0)

        # Explicit majors, no minor labels. Decades alone would leave a single
        # tick across this range, so the majors are decade-anchored round
        # numbers rather than powers of ten only.
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 30]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.set_xlabel("arithmetic intensity (FLOP / byte)")
        ax.set_ylabel("achieved throughput (TFLOP/s)")
        ax.legend(loc="lower right", frameon=True, framealpha=0.96,
                  edgecolor=INK["grid"], fontsize=11.5, borderpad=0.7,
                  labelspacing=0.45, handletextpad=0.7)

        finish(fig, ax, "Two of the three roofs are unreachable",
               f"{machine.name} · {machine.bandwidth_gbs:g} GB/s measured · "
               "ridge marked per dtype")
        saved = _save(fig, path)

    if dropped:
        print(f"  note: dropped {dropped} baseline-fallback reduced-precision "
              f"point(s) — those measure the reference, not a fused kernel")
    return saved


def main() -> int:
    frame = load_results(RESULTS)
    print(f"read {len(frame)} rows from {RESULTS.relative_to(ROOT)}")

    vram = vram_ceiling_v2(frame)
    print(f"wrote {vram.relative_to(ROOT)}")

    acc = accuracy_budget_v2(frame)
    print(f"wrote {acc.relative_to(ROOT)}")

    roof = roofline_v2(frame)
    print(f"wrote {roof.relative_to(ROOT)}")

    print("\noriginals untouched — `python -m analysis.make_all` still regenerates them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
