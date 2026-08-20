from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


LATTICE_ORDER: list[str] = [
    "banded-16",
    "banded-32",
    "banded-64",
    "banded-128",
    "banded-192",
    "all-sparse",
]

LATTICE_COLORS: dict[str, str] = {
    "banded-16": "cyan",
    "banded-32": "darkturquoise",
    "banded-64": "cadetblue",
    "banded-128": "steelblue",
    "banded-192": "royalblue",
    "all-sparse": "navy",
}

GEN_COLORS: dict[str, str] = {
    "banded-16": "papayawhip",
    "banded-32": "moccasin",
    "banded-64": "wheat",
    "banded-128": "tan",
    "banded-192": "orange",
    "all-sparse": "goldenrod",
}

INITIAL_LATTICES: dict[str, int] = {
    "banded-16": 32,
    "banded-32": 64,
    "banded-64": 128,
    "banded-128": 236,
    "banded-192": 384,
    "all-sparse": 448,
}


def load_data(path: Path) -> tuple[defaultdict, defaultdict]:
    totals = defaultdict(dict)
    gen_totals = defaultdict(dict)

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_") \
                    or row["lattice"] not in LATTICE_ORDER:
                continue

            totals[row["lattice"]][row["model"]] = int(row["total"]) \
                - INITIAL_LATTICES[row["lattice"]]
            gen_totals[row["lattice"]][row["model"]] = \
                int(row["fill_pad_broadcast_const"])

        return totals, gen_totals


def plot(total: defaultdict, gen_totals: defaultdict, out_path: Path) -> None:
    models = sorted({m for per_model in totals.values() for m in per_model})
    middle = len(models) // 2
    list_of_models = [models[:middle], models[middle:]]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    for i, models in enumerate(list_of_models):
        x = [i for i in range(len(models))]
        totals_x = [xi - 0.17 for xi in x]
        gen_totals_x = [xi + 0.17 for xi in x]

        for lattice in reversed(LATTICE_ORDER):

            values = [totals[lattice][m] for m in models]
            gen_values = [gen_totals[lattice][m] for m in models]

            axes[i].bar(
                totals_x,
                values,
                color=LATTICE_COLORS[lattice],
                label=lattice,
                width=0.3,
                zorder=2
            )

            axes[i].bar(
                gen_totals_x,
                gen_values,
                color=GEN_COLORS[lattice],
                width=0.3,
                zorder=2
            )

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        axes[i].grid(axis="y")

        for spine in ["top", "right", "left"]:
            axes[i].spines[spine].set_visible(False)

        axes[i].set_ylabel("Sparse Fiber Count")

    axes[0].set_title(
        "Total inferred sparse fibers in the IR as seed lattice grows"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    handles = list(reversed(handles)) \
        + [Patch(
            color=GEN_COLORS["banded-32"],
            label="Sparsity Generating Ops"
           )]

    labels = list(reversed(labels)) + ["Sparsity Generating Ops"]

    axes[1].legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        ncols=7,
        bbox_to_anchor=(0.5, -0.2)
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


totals, gen_totals = load_data(Path("csv_data/zero-counts.csv"))
plot(totals, gen_totals, Path("assets/zero-counts.png"))
