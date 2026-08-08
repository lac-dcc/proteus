import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LATTICE_ORDER = [
    "banded-16",
    "banded-32",
    "banded-64",
    "banded-128",
    "banded-192",
    "all-sparse",
]

LATTICE_COLORS = [
    "#86b6ef",
    "#5598e7",
    "#2a78d6",
    "#1c5cab",
    "#104281",
    "#0d366b",
]

FILL_PAD_BROADCAST_CONST_COLORS = [
    "#f3a282",
    "#ee7c4f",
    "#e85318",
    "#b54012",
    "#842f0d",
    "#6d270b",
]

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#898781"
TEXT_PRIMARY = "#0b0b0b"
SURFACE = "#fcfcfb"


def load_totals(csv_path: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    totals: dict[str, dict[str, int]] = defaultdict(dict)
    fill_pad_broadcast_const: dict[str, dict[str, int]] = defaultdict(dict)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_") or row["lattice"] not in LATTICE_ORDER:
                continue
            totals[row["lattice"]][row["model"]] = int(row["total"])
            fill_pad_broadcast_const[row["lattice"]][row["model"]] = int(row["fill_pad_broadcast_const"])
    return totals, fill_pad_broadcast_const


def plot(
    totals: dict[str, dict[str, int]],
    fill_pad_broadcast_const: dict[str, dict[str, int]],
    out_path: Path,
) -> None:
    models = sorted({m for per_model in totals.values() for m in per_model})
    x = range(len(models))
    total_x = [xi - 0.17 for xi in x]
    fill_pad_broadcast_const_x = [xi + 0.17 for xi in x]

    fig, ax = plt.subplots(figsize=(0.8 * len(models) + 3, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for lattice, color in reversed(list(zip(LATTICE_ORDER, LATTICE_COLORS))):
        values = [totals[lattice].get(m, 0) for m in models]
        ax.bar(total_x, values, width=0.3, color=color, label=lattice, zorder=2)

    for lattice, color in reversed(list(zip(LATTICE_ORDER, FILL_PAD_BROADCAST_CONST_COLORS))):
        values = [fill_pad_broadcast_const[lattice].get(m, 0) for m in models]
        ax.bar(fill_pad_broadcast_const_x, values, width=0.3, color=color, zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_MUTED)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    ax.set_ylabel("Sparse Fiber Count", color=TEXT_PRIMARY)
    ax.set_title(
        "Total inferred sparse fibers in the IR as seeded lattice grows",
        color=TEXT_PRIMARY,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles = list(reversed(handles)) + [Patch(color=FILL_PAD_BROADCAST_CONST_COLORS[2], label="Fill+Pad+Broadcast+Const")]
    labels = list(reversed(labels)) + ["Fill+Pad+Broadcast+Const"]
    ax.legend(
        handles, labels,
        title="Lattice Seed (Total)", frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)


totals, fill_pad_broadcast_const = load_totals(Path("csv_data/zero-counts.csv"))
plot(totals, fill_pad_broadcast_const, Path("assets/total_sparsity_growth.png"))
