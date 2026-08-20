from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt

LATTICE_ORDER: list[str] = [
    "baseline",
    "banded-16",
    "banded-32",
    "banded-64",
    "banded-128",
    "banded-192",
    "all-sparse",
]

LATTICE_COLORS: list[str] = {
    "mbench_conv2d": "darkgoldenrod",
    "mbench_depthwise_conv2d": "darkturquoise",
    "mbench_pooling_max": "cadetblue",
    "mbench_pooling_sum": "steelblue"
}


def load_data(path: Path) -> tuple[defaultdict]:
    timings = defaultdict(dict)

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lattice"] not in LATTICE_ORDER \
                    or not row["model"].startswith("mbench_"):
                continue

            timings[row["model"]]["baseline"] = float(row["base_mean"])
            timings[row["model"]][row["lattice"]] = float(row["scf_mean"])

        return timings


def plot(timings: defaultdict, out_path: Path) -> None:
    models = sorted(timings)
    x = [i for i in range(len(LATTICE_ORDER))]

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in models:
        values = [timings[model][label] for label in LATTICE_ORDER]
        ax.plot(x, values, marker="o", linewidth=2,
                color=LATTICE_COLORS[model], label=model, zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(LATTICE_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_ylabel("Operation Runtime (s)")
    ax.set_title(
        "Sparsity Rewritten Operation Runtime as seeded lattice grows"
    )

    ax.legend(
        loc="upper center",
        frameon=False,
        ncols=2,
        bbox_to_anchor=(0.5, -0.2)
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


timings = load_data(Path("csv_data/mbench-timings.csv"))
plot(timings, Path("assets/mbench_timings.png"))
