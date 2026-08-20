from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIELDS = [
    "total_mean",
    "rwscf_mean",
    "base_mean",
    "scf_mean",
]

LATTICE_ORDER: list[str] = [
    "banded-16",
    "banded-32",
    "banded-64",
    "banded-128",
    "banded-192",
    "all-sparse",
]

MODELS: list[str] = ["alexnet", "vgg16"]

MODEL_MARKERS: dict[str, str] = {
    "alexnet": "o",
    "vgg16": "*"
}

MODEL_LINES: dict[str, str] = {
    "alexnet": "-",
    "vgg16": "--"
}

LATTICE_COLORS: dict[str, str] = {
    "total_mean": "darkgoldenrod",
    "rwscf_mean": "darkturquoise",
    "base_mean": "cadetblue",
    "scf_mean": "steelblue"
}

LABELS: dict[str, str] = {
    "total_mean": "Analysis Runtime",
    "rwscf_mean": "Transformation Runtime",
    "base_mean": "Baseline IR",
    "scf_mean": "Transformed IR",
}

def load_data(path: Path) -> None:
    runtimes = defaultdict(lambda: defaultdict(dict))

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] not in MODELS:
                continue

            for field in FIELDS:
                runtimes[row["model"]][row["lattice"]][field] = float(row[field])

    return runtimes

def plot(timings: defaultdict, out_path: Path) -> None:
    x = [i for i in range(len(LATTICE_ORDER))]

    fig, ax = plt.subplots(figsize=(8, 6))


    for model in MODELS:
        for field in FIELDS:
            values = [timings[model][lattice][field] for lattice in LATTICE_ORDER]
            ax.plot(x, values, marker=MODEL_MARKERS[model], markersize=8, linestyle=MODEL_LINES[model], linewidth=2,
                    color=LATTICE_COLORS[field], label=LABELS[field], zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(LATTICE_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_ylabel("Runtime (s)")
    ax.set_title(
        "Runtimes across Passes and Execution of IRs as seed lattice grows."
    )

    field_handles = [Line2D([0], [0], color=LATTICE_COLORS[f], lw=2, label=LABELS[f]) for f in FIELDS]
    model_handles = [Line2D([0], [0], color="gray", marker=MODEL_MARKERS[m], linestyle=MODEL_LINES[m], label=m) for m in MODELS]
    
    legend1 = ax.legend(
        handles=field_handles,
        loc="upper center",
        frameon=False,
        ncols=2,
        bbox_to_anchor=(0.3, -0.2),
        title="Metric"
    )
    
    ax.add_artist(legend1)
    
    ax.legend(
        handles=model_handles,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.75, -0.2),
        title="Model"
    )


    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

plot(load_data(Path("csv_data/timings.csv")), Path("assets/runtimes.png"))
