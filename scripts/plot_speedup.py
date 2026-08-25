from collections import defaultdict
from pathlib import Path
import csv

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


LATTICE_ORDER: list[str] = [
    "banded-16",
    "banded-32",
    "banded-64",
    "banded-128",
    "banded-192",
    "all-sparse",
]

MARKERS: list[str] = ["o", "s", "^", "D", "v", "P"]

GRID_STYLE = dict(
    linewidth=plt.rcParams["grid.linewidth"],
    linestyle=plt.rcParams["grid.linestyle"],
    alpha=plt.rcParams["grid.alpha"],
    color=plt.rcParams["grid.color"],
    zorder=1,
)


def load_data(timings_path: Path) -> defaultdict:
    speedups = defaultdict(dict)

    with timings_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_"):
                continue

            if row["lattice"] not in LATTICE_ORDER:
                continue

            speedups[row["model"]][row["lattice"]] = \
                float(row["scf_speedup"])

    return speedups


def plot(speedups: defaultdict, out_path: Path) -> None:
    models = sorted(speedups)
    x = [i for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, lattice in enumerate(LATTICE_ORDER):
        values = [speedups[model][lattice] for model in models]
        ax.scatter(x, values, marker=MARKERS[i % len(MARKERS)], label=lattice, s=100, edgecolor="black", linewidth=0.6, zorder=2)

    ax.axhline(2.68, **GRID_STYLE)
    ax.axhline(7.14, **GRID_STYLE)
    ax.axhline(16.48, **GRID_STYLE)
    ax.axhline(76.10, **GRID_STYLE)
    ax.axhline(1, linewidth=2, linestyle="--", zorder=2)
    ax.text(13.80, 0.80, "Baseline", ha="right", fontsize=16)

    ax.set_ylim(0.5, 101)
    ax.set_yscale("log")
    ax.set_yticks([2.68, 7.14, 16.48, 76.10], minor=True)
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda y, pos: f"{y:g}" if y in (2.68, 7.14, 16.48, 76.10) else "")
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.tick_params(axis="both", labelsize=14)
    ax.tick_params(axis="y", which="minor", labelsize=14)
    ax.set_ylabel("Speedup (×)", fontsize=16)
    ax.set_title("Execution Speedup versus Baseline Execution", fontsize=22)
    ax.grid()

    for spine in ["right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper center",
        frameon=False,
        ncols=4,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=16
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


speedups = load_data(Path("csv_data/timings.csv"))
plot(speedups, Path("assets/speedup.png"))
