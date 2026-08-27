"""Chrome-trace analysis: where the time actually goes, and how much of it is idle.

A speedup tells you that something got faster. A trace tells you *what the GPU
was doing*, and — more usefully — what it was doing nothing at all. That second
number is the one Rung 0 turns on:

    GPU busy % low  -> the GPU is starving between kernels. The bottleneck is
                       launch overhead, and the fix is fewer, bigger kernels
                       (fusion, CUDA graphs). Making the kernels themselves
                       faster buys almost nothing.
    GPU busy % high -> the GPU is genuinely working. Now the kernels' own
                       efficiency is what matters, and the roofline says which
                       side of the ridge to attack.

Getting that backwards costs a day, which is why this is measured rather than
guessed at.

Input format
------------
`torch.profiler`'s `export_chrome_trace` writes a JSON object with a
`traceEvents` list. Only complete events (`"ph": "X"`) carry a duration; the
file also contains metadata records, instant events, CPU-side operator records
and flow arrows, all of which must be ignored rather than counted.

Which events are GPU kernels depends on the PyTorch version: the category has
been `kernel`, `Kernel` and `gpu_op` at different times. All three are accepted,
because the alternative is a busy fraction that silently reads 0% on a version
we did not anticipate.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from analysis.style import INK, color_for, finish, new_figure

__all__ = [
    "load_trace", "gpu_busy_fraction", "gpu_active_span_us", "merged_intervals",
    "kernel_breakdown", "kernel_family", "busy_table",
    "plot_timeline", "plot_busy_vs_shape", "KERNEL_CATEGORIES",
]

#: Categories that mean "this ran on the GPU". Version-dependent, so accept all.
KERNEL_CATEGORIES = frozenset({"kernel", "gpu_op", "gpu_kernel"})

#: Categories that are GPU work but not compute; counted separately so a trace
#: dominated by copies is not mistaken for one dominated by maths.
MEMORY_CATEGORIES = frozenset({"gpu_memcpy", "gpu_memset"})

#: Kernel name -> family. Ordered: the first match wins, so the more specific
#: patterns come first.
_FAMILY_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"sgemm|hgemm|gemm|cutlass|implicit_gemm|dot_kernel|addmm", "matmul"),
    (r"softmax", "softmax"),
    (r"layer_norm|layernorm|native_norm", "layernorm"),
    (r"masked_fill|masked_scatter", "masking"),
    (r"gelu|erf|tanh", "activation"),
    (r"memcpy|memset", "memory"),
    (r"elementwise|vectorized|copy_|add_|mul_|transpose|permute|contiguous", "elementwise"),
)


def kernel_family(name: str) -> str:
    """Bucket a kernel name into a family for the breakdown table.

    Deliberately coarse. The question a reader asks of this table is "is the
    time in the maths or in the plumbing", and a hundred distinct kernel names
    answers it worse than six families do.
    """
    lowered = str(name).lower()
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, lowered):
            return family
    return "other"


def load_trace(path: Path | str) -> pd.DataFrame:
    """Read a Chrome trace into a frame of complete events.

    Columns: `name`, `cat`, `ts_us`, `dur_us`, `end_us`, `pid`, `tid`,
    `is_kernel`, `is_gpu`, `family`. Accepts `.json` and `.json.gz`.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)

    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"{path}: no traceEvents list found")

    rows = [
        {
            "name": event.get("name", ""),
            "cat": str(event.get("cat", "")),
            "ts_us": float(event["ts"]),
            "dur_us": float(event["dur"]),
            "pid": event.get("pid"),
            "tid": event.get("tid"),
        }
        # "X" is the only phase that carries a duration. Metadata ("M"), instant
        # ("i") and flow ("s"/"f") records have no extent and must not be summed.
        for event in events
        if isinstance(event, dict) and event.get("ph") == "X"
        and event.get("ts") is not None and event.get("dur") is not None
    ]

    frame = pd.DataFrame(rows, columns=["name", "cat", "ts_us", "dur_us", "pid", "tid"])
    if frame.empty:
        # Dtypes must be set explicitly here. An empty column assigned from []
        # lands as object dtype, and `frame[frame["is_kernel"]]` on an
        # object-dtype Series is read by pandas as *label* selection rather than
        # a boolean mask -- so the caller gets a KeyError instead of an empty
        # result. A trace with no GPU events is the normal case on CPU.
        return frame.astype({
            "name": "object", "cat": "object",
            "ts_us": "float64", "dur_us": "float64",
        }).assign(
            end_us=pd.Series(dtype="float64"),
            is_kernel=pd.Series(dtype="bool"),
            is_gpu=pd.Series(dtype="bool"),
            family=pd.Series(dtype="object"),
        )

    lowered = frame["cat"].str.lower()
    frame["end_us"] = frame["ts_us"] + frame["dur_us"]
    frame["is_kernel"] = lowered.isin(KERNEL_CATEGORIES)
    frame["is_gpu"] = frame["is_kernel"] | lowered.isin(MEMORY_CATEGORIES)
    frame["family"] = frame["name"].map(kernel_family)
    return frame.sort_values("ts_us").reset_index(drop=True)


def merged_intervals(
    frame: pd.DataFrame, include_memory: bool = False
) -> List[Tuple[float, float]]:
    """Non-overlapping [start, end) intervals during which the GPU was busy.

    Merging matters: kernels on different streams overlap, and summing their
    durations would report a GPU more than 100% busy. The union is the only
    quantity that means "wall-clock time during which something was running".
    """
    column = "is_gpu" if include_memory else "is_kernel"
    selected = frame[frame[column]].sort_values("ts_us")
    merged: List[Tuple[float, float]] = []
    for start, end in zip(selected["ts_us"], selected["end_us"]):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def gpu_active_span_us(frame: pd.DataFrame, include_memory: bool = False) -> float:
    """First kernel start to last kernel end."""
    intervals = merged_intervals(frame, include_memory)
    if not intervals:
        return 0.0
    return intervals[-1][1] - intervals[0][0]


def gpu_busy_fraction(
    frame: pd.DataFrame,
    window: Optional[Tuple[float, float]] = None,
    include_memory: bool = False,
) -> float:
    """Fraction of the window during which a kernel was running.

    The default window runs from the first kernel's start to the last kernel's
    end, so what it measures is the *gaps between kernels* — the launch-bound
    signal. It deliberately excludes whatever the process was doing before the
    first kernel and after the last, which is setup, not starvation.

    Pass an explicit `window` in microseconds to score a specific region, for
    instance one profiler step.
    """
    intervals = merged_intervals(frame, include_memory)
    if not intervals:
        return 0.0

    if window is None:
        start, end = intervals[0][0], intervals[-1][1]
    else:
        start, end = window
        intervals = [
            (max(a, start), min(b, end)) for a, b in intervals
            if b > start and a < end
        ]
    span = end - start
    if span <= 0:
        return 0.0
    return sum(b - a for a, b in intervals) / span


def kernel_breakdown(
    frame: pd.DataFrame, top: int = 12, by: str = "name"
) -> pd.DataFrame:
    """Where GPU time goes, by kernel name (or `by="family"`).

    Uses summed durations rather than the merged union, because the question is
    "which kernel owns this time", and attributing overlap to one of them would
    be arbitrary. The two totals differ only when streams overlap; `busy_table`
    reports the union separately.
    """
    kernels = frame[frame["is_kernel"]]
    if kernels.empty:
        return pd.DataFrame(columns=[by, "total_us", "calls", "mean_us", "share"])

    grouped = (
        kernels.groupby(by)["dur_us"]
        .agg(total_us="sum", calls="count", mean_us="mean")
        .reset_index()
        .sort_values("total_us", ascending=False)
    )
    grouped["share"] = grouped["total_us"] / grouped["total_us"].sum()
    return grouped.head(top).reset_index(drop=True)


def busy_table(traces: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per trace: busy %, span, kernel count, mean kernel duration.

    This is the table the report's Rung 0 section is built from, and the
    `verdict` column is the classification it exists to make.
    """
    rows = []
    for label, frame in traces.items():
        kernels = frame[frame["is_kernel"]]
        busy = gpu_busy_fraction(frame)
        rows.append({
            "trace": label,
            "gpu_busy": busy,
            "span_ms": gpu_active_span_us(frame) / 1e3,
            "kernels": int(len(kernels)),
            "mean_kernel_us": float(kernels["dur_us"].mean()) if len(kernels) else float("nan"),
            "idle_ms": gpu_active_span_us(frame) * (1 - busy) / 1e3,
            # The threshold is a reading aid, not a law of nature: below ~70%
            # the GPU spends most of a run waiting for work, and fusion or CUDA
            # graphs pay off more than a faster kernel does.
            "verdict": "launch-bound" if busy < 0.70 else "occupied",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _family_colors(families: Sequence[str]) -> Dict[str, str]:
    return {family: color_for(index) for index, family in enumerate(families[:8])}


def plot_timeline(
    baseline: pd.DataFrame,
    optimized: pd.DataFrame,
    out_dir: Path | str = "results/figures",
    filename: str = "trace_timeline.png",
    window_ms: float = 5.0,
    labels: Tuple[str, str] = ("baseline", "optimized"),
) -> Path:
    """Two stacked strips of kernel activity over the same time window.

    The strongest visual in the report: gaps are the GPU doing nothing, and a
    gap-riddled baseline strip beside a dense optimized one is the argument for
    fusion made without a single number. Both strips share one time axis and one
    window length, so the comparison is honest — a strip drawn to its own scale
    would make any trace look dense.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    fig, axes = new_figure(7.5, 3.6, nrows=2, sharex=True)
    window_us = window_ms * 1e3

    families = sorted(
        set(baseline.loc[baseline["is_kernel"], "family"])
        | set(optimized.loc[optimized["is_kernel"], "family"])
    )
    colors = _family_colors(families)

    for axis, frame, label in zip(axes, (baseline, optimized), labels):
        kernels = frame[frame["is_kernel"]]
        if kernels.empty:
            axis.text(0.5, 0.5, f"no kernels in {label}", ha="center", va="center",
                      transform=axis.transAxes, color=INK["muted"])
        else:
            origin = float(kernels["ts_us"].min())
            visible = kernels[kernels["ts_us"] - origin < window_us]
            for family, group in visible.groupby("family"):
                axis.broken_barh(
                    [(row.ts_us - origin, row.dur_us) for row in group.itertuples()],
                    (0.15, 0.7),
                    facecolors=colors.get(str(family), INK["muted"]),
                    label=str(family),
                )
            busy = gpu_busy_fraction(frame)
            axis.text(
                0.995, 0.5, f"{busy:.0%} busy", transform=axis.transAxes,
                ha="right", va="center", fontsize=9, color=INK["secondary"],
            )
        axis.set_ylim(0, 1)
        axis.set_yticks([0.5])
        axis.set_yticklabels([label])
        axis.set_xlim(0, window_us)
        axis.grid(False)
        axis.tick_params(axis="y", length=0)

    axes[-1].set_xlabel(f"microseconds from first kernel (window = {window_ms:g} ms)")

    from matplotlib.patches import Patch

    # Below the strips, not above them: the space above the top strip belongs
    # to the title and subtitle, and a legend there lands on both.
    axes[-1].legend(
        handles=[Patch(facecolor=colors[f], label=f) for f in families if f in colors],
        loc="upper center", bbox_to_anchor=(0.5, -0.55),
        ncols=min(len(families), 6), fontsize=8,
    )
    finish(fig, [axes[0]], "Kernel timeline", "white space is an idle GPU")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_busy_vs_shape(
    traces: Mapping[str, pd.DataFrame],
    out_dir: Path | str = "results/figures",
    filename: str = "gpu_busy_vs_shape.png",
) -> Path:
    """Grouped bars of GPU busy % per shape, baseline against optimized.

    Trace labels are expected to look like `<shape>_<variant>`, which is what
    `trace_small_baseline.json` yields; anything unparseable is grouped under
    its own name rather than dropped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    fig, ax = new_figure(7.5, 4.2)
    if not traces:
        ax.text(0.5, 0.5, "no traces yet — ask for the profiler exports",
                ha="center", va="center", transform=ax.transAxes, color=INK["muted"])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        finish(fig, ax, "GPU busy fraction by shape")
        fig.savefig(path)
        plt.close(fig)
        return path

    parsed: Dict[str, Dict[str, float]] = {}
    for label, frame in traces.items():
        shape, _, variant = str(label).rpartition("_")
        if not shape:
            shape, variant = str(label), "trace"
        parsed.setdefault(shape, {})[variant] = gpu_busy_fraction(frame)

    shapes = list(parsed)
    variants = sorted({v for values in parsed.values() for v in values})

    import numpy as np

    positions = np.arange(len(shapes))
    width = 0.8 / max(len(variants), 1)
    for index, variant in enumerate(variants):
        offset = (index - (len(variants) - 1) / 2) * (width + 0.02)
        ax.bar(
            positions + offset,
            [parsed[shape].get(variant, 0.0) for shape in shapes],
            width, color=color_for(index), label=variant,
        )

    ax.axhline(0.70, color=INK["secondary"], linewidth=1.2, zorder=1)
    ax.annotate("70%", xy=(0.002, 0.70), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK["secondary"], ha="left")

    ax.set_xticks(positions)
    ax.set_xticklabels(shapes)
    ax.set_ylabel("GPU busy fraction")
    # Headroom above the tallest bar so the legend never sits on the data.
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper left", ncols=len(variants))

    finish(fig, ax, "GPU busy fraction by shape",
           "fraction of the active window with a kernel running; "
           "below the 70% rule the GPU is starving between launches")
    fig.savefig(path)
    plt.close(fig)
    return path
