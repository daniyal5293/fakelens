"""One visual system for every report figure.

* A model always keeps the same colour in every figure: the colour follows the entity, never its rank.
* Hues come from a CVD-validated categorical palette, used in a fixed order.
* Marks are thin, gridlines recede, text uses neutral ink (never the series colour), and every
  multi-series figure has a legend.
"""
from __future__ import annotations

import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Fixed entity -> colour mapping (main model first).
MODEL_COLORS = {
    "A2": PALETTE[0], "B": PALETTE[1], "A1": PALETTE[2], "A3": PALETTE[3], "A4": PALETTE[4],
    "A2_raw": PALETTE[6], "B_raw": PALETTE[7], "B_noaug": PALETTE[5],
}
MODEL_LABELS = {
    "A1": "A1 CLIP-L/14 linear", "A2": "A2 CLIP-L/14 multi-layer", "A3": "A3 CLIP-B/16 multi-layer",
    "A4": "A4 DINOv2-L/14 multi-layer", "B": "B ResNet-18", "A2_raw": "A2 (no debias)",
    "B_raw": "B (no debias)", "B_noaug": "B (no aug)",
}
CLASS_COLORS = {"real": PALETTE[0], "fake": PALETTE[1]}


def color(model: str) -> str:
    base = model.split("#")[0]
    return MODEL_COLORS.get(base, PALETTE[zlib.crc32(base.encode()) % len(PALETTE)])


def label(model: str) -> str:
    return MODEL_LABELS.get(model.split("#")[0], model)


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": TEXT,
        "font.size": 10, "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
