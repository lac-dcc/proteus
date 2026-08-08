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

MODEL_COLORS = {
    "mbench_conv2d": "#2a78d6",
    "mbench_depthwise_conv2d": "#eb6834",
    "mbench_pooling_max": "#1baf7a",
    "mbench_pooling_sum": "#eda100",
}

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#898781"
TEXT_PRIMARY = "#0b0b0b"
SURFACE = "#fcfcfb"


X_LABELS = ["baseline"] + LATTICE_ORDER


def load_runtimes(csv_path: Path) -> dict[str, dict[str, float]]:
    runtimes: dict[str, dict[str, float]] = defaultdict(dict)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lattice"] not in LATTICE_ORDER:
                continue
            runtimes[row["model"]]["baseline"] = float(row["base_mean"])
            runtimes[row["model"]][row["lattice"]] = float(row["scf_mean"])
    return runtimes


def plot(runtimes: dict[str, dict[str, float]], out_path: Path) -> None:
    models = sorted(runtimes)
    x = range(len(X_LABELS))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for model in models:
        values = [runtimes[model][label] for label in X_LABELS]
        ax.plot(
            x, values, marker="o", markersize=5, linewidth=2,
            color=MODEL_COLORS[model], label=model, zorder=2,
        )

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(X_LABELS, rotation=45, ha="right", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT_MUTED)
    ax.grid(axis="y", which="both", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["left"].set_color(AXIS_COLOR)

    ax.set_ylabel("Operation Runtime (s)", color=TEXT_PRIMARY)
    ax.set_title(
        "Sparsity Rewritten Operation Runtime as seeded lattice grows",
        color=TEXT_PRIMARY,
    )
    ax.legend(frameon=True, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)


runtimes = load_runtimes(Path("csv_data/mbench-timings.csv"))
plot(runtimes, Path("assets/mbench_runtime_growth.png"))
