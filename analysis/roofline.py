"""Analytic FLOPs and bytes per config, and the roofline plot.

A speedup number says we got faster. It does not say whether we are *near the
machine's limit* or still leaving most of it on the floor. The roofline answers
that: plot achieved throughput against arithmetic intensity, draw the two roofs
(bandwidth-bound on the left, compute-bound on the right), and read off which
side of the ridge each configuration sits on.

The practical payoff is that the two sides call for different work. Left of the
ridge, more FLOPs are free and the fix is fusion — stop moving the same bytes
repeatedly. Right of it, only better math or better tensor-core utilisation
helps. The baseline sits far left at long sequences precisely because it writes
and re-reads a [B, H, S, S] score matrix, which is the whole argument for a
fused attention kernel.

All counts are analytic, not measured. They are stated here so the report can
show its working rather than quoting a black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from analysis.style import INK, MARKERS, SERIES, finish, new_figure

__all__ = [
    "MachineSpec", "RTX_4050_LAPTOP",
    "forward_flops", "forward_bytes", "explicit_attention_bytes",
    "arithmetic_intensity", "achieved_tflops", "ridge_point",
    "roofline_frame", "plot_roofline",
]


@dataclass(frozen=True)
class MachineSpec:
    """Peak numbers for one GPU. Person A confirms these from the spec sheet.

    Defaults describe the RTX 4050 Laptop (sm_89) the measurements come from.
    They are *placeholders until confirmed* — see docs/APPROVALS_NEEDED.md — and
    every ridge point in the report scales directly with them, so a wrong value
    here is a wrong conclusion, not a cosmetic error.
    """

    name: str = "RTX 4050 Laptop (sm_89)"
    peak_fp32_tflops: float = 12.0
    peak_bf16_tflops: float = 48.0
    bandwidth_gbs: float = 192.0


RTX_4050_LAPTOP = MachineSpec()


def _dims(config) -> tuple:
    """(B, S, d, H, f, L) from a TransformerConfig or a mapping/Series."""
    if hasattr(config, "batch_size"):
        return (config.batch_size, config.seq_len, config.d_model,
                config.num_heads, config.ffn_dim, config.num_layers)
    return (int(config["batch"]), int(config["seq_len"]), int(config["d_model"]),
            int(config["heads"]),
            int(config.get("ffn_dim") or 4 * int(config["d_model"])),
            int(config["layers"]))


def forward_flops(config) -> int:
    """Multiply-accumulates counted as 2 FLOPs, per layer:

        8 * B * S * d^2     Q, K, V and output projections (4 GEMMs)
        4 * B * S^2 * d     QK^T and PV (the two attention GEMMs)
        4 * B * S * d * f   the two FFN GEMMs

    LayerNorm, GELU, softmax and the residual adds are elementwise and
    negligible against these; they dominate *bytes*, not FLOPs, which is exactly
    why they matter to the left of the ridge and not to the right.
    """
    b, s, d, _h, f, layers = _dims(config)
    per_layer = 8 * b * s * d * d + 4 * b * s * s * d + 4 * b * s * d * f
    return int(layers * per_layer)


def _element_size(dtype: str) -> int:
    return 2 if str(dtype) in ("float16", "bfloat16", "half") else 4


def forward_bytes(config, dtype: str = "float32", include_attention: bool = True) -> int:
    """Bytes moved in one forward pass. An approximation, stated plainly.

    Counted:
      * every weight read once — `params * element_size`;
      * activation traffic, approximated as `layers * (14*B*S*d + 4*B*S*f) * e`.
        The 14 covers the residual stream being read and written across the two
        sublayers, the two LayerNorms, Q/K/V/context and the projections; the 4
        covers the FFN's hidden activation in and out.

    Not counted: cache reuse. Real traffic is lower wherever a tensor stays in
    L2 between adjacent kernels, so this *overestimates* bytes and therefore
    *underestimates* arithmetic intensity. Points sit slightly left of the truth
    — conservative in the direction that matters, since it makes us look more
    bandwidth-bound rather than less.

    `include_attention` adds the explicit score matrix. Leave it on for the
    baseline and off for a fused attention path: that single term is the
    difference between the two, and it is what moves them to opposite sides of
    the ridge.
    """
    b, s, d, _h, f, layers = _dims(config)
    element = _element_size(dtype)

    params = layers * (4 * d * d + 2 * d * f) + 2 * d
    weight_bytes = params * element
    activation_bytes = layers * (14 * b * s * d + 4 * b * s * f) * element

    total = weight_bytes + activation_bytes
    if include_attention:
        total += explicit_attention_bytes(config, dtype)
    return int(total)


def explicit_attention_bytes(config, dtype: str = "float32") -> int:
    """Traffic from materializing [B, H, S, S] scores, ~3 passes over it.

    Write the scores, read them for the fp32 softmax, write the probabilities,
    read them for PV. Rounded to three passes. This is the term that grows as
    S^2 and the reason the baseline goes bandwidth-bound at long sequences; a
    fused kernel never writes it to memory at all.
    """
    b, s, d, h, _f, layers = _dims(config)
    element = _element_size(dtype)
    return int(layers * 3 * b * h * s * s * element)


def arithmetic_intensity(config, dtype: str = "float32", include_attention: bool = True) -> float:
    """FLOPs per byte. Where a config sits on the roofline's x-axis."""
    return forward_flops(config) / forward_bytes(config, dtype, include_attention)


def achieved_tflops(config, median_ms: float) -> float:
    """Measured throughput: analytic FLOPs over measured wall time."""
    if not median_ms or median_ms <= 0:
        return float("nan")
    return forward_flops(config) / (median_ms / 1e3) / 1e12


def ridge_point(peak_tflops: float, bandwidth_gbs: float) -> float:
    """Arithmetic intensity where bandwidth stops binding and compute starts.

    Left of it a kernel cannot be fed fast enough; right of it the machine is
    genuinely running out of arithmetic. For the default fp32 numbers this is
    12.0e12 / 192e9 = 62.5 FLOP/byte.
    """
    return (peak_tflops * 1e12) / (bandwidth_gbs * 1e9)


def roofline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add intensity and achieved-throughput columns to passing rows.

    The baseline is charged for the explicit score matrix and every other
    strategy is not — that asymmetry is the point of the plot, not a fudge.
    """
    from analysis.load import usable

    subset = usable(frame).copy()
    if subset.empty:
        return subset

    def per_row(row: pd.Series) -> pd.Series:
        explicit = row["strategy_name"] == "baseline"
        dtype = str(row["dtype"])
        return pd.Series({
            "flops": forward_flops(row),
            "bytes": forward_bytes(row, dtype, include_attention=explicit),
            "intensity": arithmetic_intensity(row, dtype, include_attention=explicit),
            "achieved_tflops": achieved_tflops(row, row["optimized_median_ms"]),
            "baseline_achieved_tflops": achieved_tflops(row, row["baseline_median_ms"]),
        })

    return pd.concat([subset, subset.apply(per_row, axis=1)], axis=1)


def plot_roofline(
    frame: pd.DataFrame,
    out_dir: Path | str = "results/figures",
    machine: MachineSpec = RTX_4050_LAPTOP,
    filename: str = "roofline.png",
) -> Path:
    """Log-log roofline: two roofs, both ridge points, one point per measurement.

    Colour encodes dtype and *shape* encodes strategy. That split is deliberate:
    a scatter puts every colour pair on screen simultaneously, and only the
    first three categorical slots are guaranteed distinguishable under that
    condition — three dtypes fit exactly, and strategy identity moves to a
    channel that has no such limit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    data = roofline_frame(frame)
    fig, ax = new_figure(7.5, 5.0)

    if data.empty:
        ax.text(0.5, 0.5, "no passing rows yet", ha="center", va="center",
                transform=ax.transAxes, color=INK["muted"])
        finish(fig, ax, "Roofline", machine.name)
        fig.savefig(path)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path

    intensity = data["intensity"]
    x = _log_span(intensity.min(), intensity.max())

    # --- the roofs ---------------------------------------------------------
    for peak, label, style in (
        (machine.peak_bf16_tflops, f"bf16 tensor core peak {machine.peak_bf16_tflops:g} TFLOP/s", ":"),
        (machine.peak_fp32_tflops, f"fp32 peak {machine.peak_fp32_tflops:g} TFLOP/s", "-"),
    ):
        roof = [min(peak, machine.bandwidth_gbs * 1e9 * xi / 1e12) for xi in x]
        ax.plot(x, roof, style, color=INK["secondary"], linewidth=1.4,
                label=label, zorder=1)

    for peak, colour in ((machine.peak_fp32_tflops, INK["secondary"]),
                         (machine.peak_bf16_tflops, INK["muted"])):
        ridge = ridge_point(peak, machine.bandwidth_gbs)
        if x[0] <= ridge <= x[-1]:
            ax.plot([ridge], [peak], marker="|", color=colour, markersize=9, zorder=2)
            ax.annotate(
                f"ridge {ridge:.0f} FLOP/byte",
                xy=(ridge, peak), xytext=(4, -12), textcoords="offset points",
                fontsize=8, color=colour,
            )

    # --- the measurements --------------------------------------------------
    dtypes = sorted(data["dtype"].dropna().unique())
    strategies = sorted(data["strategy_name"].dropna().unique())
    dtype_colour: Dict[str, str] = {d: SERIES[i] for i, d in enumerate(dtypes[:3])}
    strategy_marker: Dict[str, str] = {
        s: MARKERS[i % len(MARKERS)] for i, s in enumerate(strategies)
    }

    for (dtype, strategy), group in data.groupby(["dtype", "strategy_name"], sort=False):
        ax.scatter(
            group["intensity"], group["achieved_tflops"],
            color=dtype_colour.get(str(dtype), INK["muted"]),
            marker=strategy_marker[str(strategy)],
            s=46, linewidths=0.8, edgecolors=INK["surface"], zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved throughput (TFLOP/s)")

    # Two legends: one per encoding channel, so neither is ambiguous.
    from matplotlib.lines import Line2D

    dtype_handles = [
        Line2D([], [], marker="o", linestyle="", color=colour, label=str(name))
        for name, colour in dtype_colour.items()
    ]
    strategy_handles = [
        Line2D([], [], marker=marker, linestyle="", color=INK["secondary"], label=str(name))
        for name, marker in strategy_marker.items()
    ]
    roof_handles, roof_labels = ax.get_legend_handles_labels()

    first = ax.legend(handles=roof_handles + dtype_handles, loc="upper left",
                      fontsize=8, title="roofs and dtype", title_fontsize=8)
    first.get_title().set_color(INK["secondary"])
    ax.add_artist(first)
    second = ax.legend(handles=strategy_handles, loc="lower right", fontsize=8,
                       title="strategy", title_fontsize=8)
    second.get_title().set_color(INK["secondary"])

    finish(fig, ax, "Roofline", f"{machine.name} · {machine.bandwidth_gbs:g} GB/s")
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _log_span(low: float, high: float, points: int = 64) -> list:
    import numpy as np

    low = max(low, 1e-3) / 3
    high = max(high, low * 10) * 3
    return list(np.logspace(np.log10(low), np.log10(high), points))
