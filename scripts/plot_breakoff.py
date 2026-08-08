import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

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

MATMUL_COLOR = "#eb6834"

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#898781"
TEXT_PRIMARY = "#0b0b0b"
SURFACE = "#fcfcfb"


def load_data(csv_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    breakoff: dict[str, dict[str, float]] = defaultdict(dict)
    matmul: dict[str, float] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_") or row["lattice"] not in LATTICE_ORDER:
                continue
            breakoff[row["lattice"]][row["model"]] = float(row["breakoff_pct"])
            if row["first_matmul_pct"] != "None":
                matmul[row["model"]] = float(row["first_matmul_pct"])
    return breakoff, matmul


def plot(breakoff: dict[str, dict[str, float]], matmul: dict[str, float], out_path: Path) -> None:
    models = sorted({m for per_model in breakoff.values() for m in per_model})
    y = range(len(models))

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(models) + 2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for lattice, color in reversed(list(zip(LATTICE_ORDER, LATTICE_COLORS))):
        values = [breakoff[lattice].get(m, 0) for m in models]
        ax.barh(list(y), values, height=0.6, color=color, label=lattice, zorder=2)

    matmul_y = [i for i, m in enumerate(models) if m in matmul]
    matmul_x = [matmul[m] for m in models if m in matmul]
    ax.scatter(
        matmul_x, matmul_y, marker="|", s=200, linewidths=1.5,
        color=MATMUL_COLOR, label="1st matmul", zorder=3,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(models, fontsize=9, color=TEXT_MUTED)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT_MUTED)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    ax.set_xlabel("Survived Sparsity Depth in IR Percentage", color=TEXT_PRIMARY)
    ax.set_title(
        "Depth of Survived Sparsity in the IR as seeded lattice grows",
        color=TEXT_PRIMARY,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles), reversed(labels),
        frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)


breakoff, matmul = load_data(Path("csv_data/zero-counts.csv"))
plot(breakoff, matmul, Path("assets/breakoff_growth.png"))
