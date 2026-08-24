from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt

EPSILON: float = 1e-4

IGNORE_MODELS: list[str] = ["resnet101", "googlenet", "mnasnet1_0", "resnet18", "resnet34", "resnet50"]

SEGMENT_COLORS: dict[str, str] = {
    "compilation": "darkgoldenrod",
    "analysis": "darkturquoise",
    "transform": "cadetblue",
    "execution": "steelblue",
}

FIELDS: list[str] = [
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


def load_data(timings_path: Path, compile_path: Path) -> tuple[defaultdict]:
    runtimes = defaultdict(lambda: defaultdict(dict))

    with timings_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_")  \
                    or row["model"] in IGNORE_MODELS:
                continue

            for field in FIELDS:
                runtimes[row["model"]][row["lattice"]][field] = \
                        float(row[field])

    with compile_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"].startswith("mbench_")  \
                    or row["model"] in IGNORE_MODELS:
                continue

            runtimes[row["model"]][row["lattice"]]["compile_mean"] = \
                float(row["compile_mean"])

            runtimes[row["model"]][row["lattice"]]["base_compile_mean"] = \
                float(row["base_compile_mean"])

    return runtimes


def plot(runtimes: defaultdict, out_path: Path) -> None:
    models = sorted(runtimes.keys())

    x_middle = [i for i in range(len(models))]
    x_left = [xi - 0.25 for xi in x_middle]
    x_right = [xi + 0.25 for xi in x_middle]

    fig, ax = plt.subplots(figsize=(9, 6))

    compile_time_left = [runtimes[m]["banded-64"]["base_compile_mean"] for m in models]
    execution_time_left = [runtimes[m]["banded-64"]["base_mean"] for m in models]

    analysis_time_middle = [runtimes[m]["banded-64"]["total_mean"] for m in models]
    transform_time_middle = [runtimes[m]["banded-64"]["rwscf_mean"] for m in models]
    compilation_time_middle = [runtimes[m]["banded-64"]["compile_mean"] for m in models]
    execution_time_middle = [runtimes[m]["banded-64"]["scf_mean"] for m in models]
    stacked_middle_1 = [a + t for a, t in zip(analysis_time_middle, transform_time_middle)]
    stacked_middle_2 = [s1 + c for s1, c in zip(stacked_middle_1, compilation_time_middle)]

    analysis_time_right = [runtimes[m]["banded-192"]["total_mean"] for m in models]
    transform_time_right = [runtimes[m]["banded-192"]["rwscf_mean"] for m in models]
    compilation_time_right = [runtimes[m]["banded-192"]["compile_mean"] for m in models]
    execution_time_right = [runtimes[m]["banded-192"]["scf_mean"] for m in models]
    stacked_right_1 = [a + t for a, t in zip(analysis_time_right, transform_time_right)]
    stacked_right_2 = [s1 + c for s1, c in zip(stacked_right_1, compilation_time_right)]

    ax.bar(
        x_left, compile_time_left, 0.2, color=SEGMENT_COLORS["compilation"],
        label=None,
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_left, execution_time_left, 0.2, color=SEGMENT_COLORS["execution"],
        bottom=compile_time_left,
        label=None,
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_middle, analysis_time_middle, 0.2, color=SEGMENT_COLORS["analysis"],
        bottom=EPSILON,
        label="Analysis Time",
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_middle, transform_time_middle, 0.2, color=SEGMENT_COLORS["transform"],
        bottom=[a + EPSILON for a in analysis_time_middle],
        label="Transform Time",
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_middle, compilation_time_middle, 0.2, color=SEGMENT_COLORS["compilation"],
        bottom=[s1 + EPSILON for s1 in stacked_middle_1],
        label="Lowering Time",
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_middle, execution_time_middle, 0.2, color=SEGMENT_COLORS["execution"],
        bottom=[s2 + EPSILON for s2 in stacked_middle_2],
        label="Execution Runtime",
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_right, analysis_time_right, 0.2, color=SEGMENT_COLORS["analysis"],
        bottom=EPSILON,
        label=None,
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_right, transform_time_right, 0.2, color=SEGMENT_COLORS["transform"],
        bottom=[a + EPSILON for a in analysis_time_right],
        label=None,
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_right, compilation_time_right, 0.2, color=SEGMENT_COLORS["compilation"],
        bottom=[s1 + EPSILON for s1 in stacked_right_1],
        label=None,
        zorder=2,
        edgecolor="black"
    )

    ax.bar(
        x_right, execution_time_right, 0.2, color=SEGMENT_COLORS["execution"],
        bottom=[s2 + EPSILON for s2 in stacked_right_2],
        label=None,
        zorder=2,
        edgecolor="black"
    )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_yscale("log")
    ax.set_ylim(bottom=EPSILON)
    ax.set_xticks(x_middle)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.tick_params(axis="x", pad=20)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.grid(axis="y")
    new_ticks = ax.secondary_xaxis("bottom")
    new_ticks.set_xticks(x_middle)
    new_ticks.set_xticklabels(["(0,64,192)"]
                              * len(x_middle), rotation=0, ha="center")

    ax.set_title(
        "Breakeven Point comparison between Execution and Compilation",
        fontsize=16
    )

    ax.legend(
        loc="upper center",
        frameon=False,
        ncols=4,
        bbox_to_anchor=(0.5, -0.3),
        fontsize=12
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


runtimes = load_data(Path("csv_data/timings.csv"), Path("csv_data/compile_timings.csv"))
plot(runtimes, Path("assets/compile-times.png"))
