from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt

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


def load_data(path: Path) -> tuple[defaultdict, dict]:
    breakoffs = defaultdict(dict)
    matmuls = {}

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_") \
                    or row["lattice"] not in LATTICE_ORDER:
                continue

            breakoffs[row["lattice"]][row["model"]] = \
                float(row["breakoff_pct"])

            if row["first_matmul_pct"] != "None":
                matmuls[row["model"]] = \
                    float(row["first_matmul_pct"])

    return breakoffs, matmuls


def plot(breakoffs: defaultdict, matmuls: defaultdict, out_path: Path) -> None:
    models = sorted({m for per_model in breakoffs.values() for m in per_model})
    x = [i for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(12, 12))

    for lattice in reversed(LATTICE_ORDER):
        values = [breakoffs[lattice][m] for m in models]
        ax.barh(x, values, color=LATTICE_COLORS[lattice],
                label=lattice, height=0.8, zorder=2)

    matmul_y = [i for i, m in enumerate(models) if m in matmuls]
    matmul_x = [matmuls[m] for m in models if m in matmuls]

    ax.scatter(matmul_x, matmul_y, marker="|", s=200, linewidths=1.5,
               label="1st matmul", zorder=3)

    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x")

    for spine in ["top", "bottom", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Depth of Survived Sparsity in the IR as seeded lattice grows"
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        reversed(handles),
        reversed(labels),
        loc="upper center",
        frameon=False,
        ncols=4,
        bbox_to_anchor=(0.5, -0.1)
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


breakoffs, matmuls = load_data(Path("csv_data/zero-counts.csv"))
plot(breakoffs, matmuls, Path("assets/sparsity_survivability.png"))
