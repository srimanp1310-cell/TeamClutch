"""Shared chart styling: one palette, one place, cited.

Every figure in `analysis/figures.py` and `analysis/roofline.py` draws its
colours from here, so the report's figures read as one set rather than seven
matplotlib defaults.

Palette provenance
------------------
These are the light-mode values of a pre-validated categorical palette, used in
its documented slot order. The order is the colourblind-safety mechanism, not a
cosmetic choice: it was selected so that *adjacent* slots clear a CVD separation
gate (worst adjacent CVD dE 9.1 on this surface, against a >=8 target, and worst
adjacent normal-vision dE 19.6 against a >=15 floor). Take the slots in order
and never cycle past the eighth.

Two consequences that shape the figures here:

  * **Line and bar charts** compare neighbouring series, so the adjacent-pair
    guarantee is the relevant one and up to eight series are safe.
  * **Scatter plots** put every pair on screen at once, and only the first three
    slots clear the gate under all-pairs. `roofline.py` therefore colours by
    dtype (three values) and uses marker *shape* for strategy, rather than
    spending eight colours on a scatter.

Figures are rendered light-only on purpose: they are PNGs embedded in a Markdown
report and README, not a themed web page, so there is no viewer preference to
respond to.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")  # no display on a CI box or over SSH
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "SERIES", "INK", "MARKERS", "STATUS",
    "color_for", "marker_for", "apply_style", "new_figure", "finish",
]

#: Categorical slots, in their validated order. Index 0 is slot 1.
SERIES: Sequence[str] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Chrome. Grid and axes are deliberately recessive; text never wears a series
#: colour, so identity is always carried by the mark beside it.
INK: Dict[str, str] = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

#: Reserved status colours. Never used for a series.
STATUS: Dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Shapes for composite encoding, where colour is already spent on another axis.
MARKERS: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X", "*")


def color_for(index: int) -> str:
    """Slot `index` (0-based). Raises past the eighth rather than cycling.

    Cycling would silently give two series the same colour; a ninth category
    belongs in an "other" bucket or a facet, which is a decision for the caller
    to make deliberately.
    """
    if not 0 <= index < len(SERIES):
        raise IndexError(
            f"categorical slot {index} is out of range: this palette has "
            f"{len(SERIES)} slots and must not be cycled. Fold the extra series "
            "into 'other' or use small multiples."
        )
    return SERIES[index]


def marker_for(index: int) -> str:
    return MARKERS[index % len(MARKERS)]


def apply_style() -> None:
    """Global rcParams. Thin marks, hairline grid, no dashed chrome."""
    plt.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",       # dashed gridlines read as data, never chrome
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
    })


def new_figure(width: float = 7.5, height: float = 4.5, **kwargs):
    apply_style()
    return plt.subplots(figsize=(width, height), **kwargs)


def finish(fig, axes, title: str, subtitle: str = "") -> None:
    """Title in primary ink, one-line subtitle in secondary. Never a series colour.

    The title is padded out of the subtitle's way explicitly. At the default pad
    the two land within a few pixels of each other and overlap — the kind of
    collision no palette check catches and only looking at the render does.
    """
    axes_list = axes if isinstance(axes, (list, tuple)) else [axes]
    top = axes_list[0]
    top.set_title(title, color=INK["primary"], loc="left",
                  pad=26 if subtitle else 10)
    if subtitle:
        top.text(
            0.0, 1.015, subtitle, transform=top.transAxes,
            fontsize=9, color=INK["secondary"], va="bottom", ha="left",
        )
    fig.tight_layout()
