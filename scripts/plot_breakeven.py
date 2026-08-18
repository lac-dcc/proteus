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

FIELDS = [
    "total_mean",
    "rwscf_mean",
    "base_mean",
    "scf_mean",
    "scf_speedup",
]

UNOBSERVED = ["n/a", "None", "", '']


def load_data(path: Path) -> tuple[defaultdict]:
    runtimes = defaultdict(lambda: defaultdict(dict))

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lattice"] not in LATTICE_ORDER \
                    or row["model"].startswith("mbench_"):
                continue

            for field in FIELDS:
                if row[field] in UNOBSERVED:
                    runtimes[row["model"]][row["lattice"]][field] = float(0.0)
                    continue

                runtimes[row["model"]][row["lattice"]][field] = \
                    float(row[field])
    return runtimes


def plot(runtimes: defaultdict, out_path: Path) -> None:
    models = sorted(runtimes)
    x = [i for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(12, 12))

    rewritten_x = [xi - 0.17 for xi in x]
    baseline_x = [xi + 0.17 for xi in x]
    baseline_values = [float(runtimes[m]["banded-192"]["base_mean"])
                       for m in models]

    for lattice in LATTICE_ORDER:
        rewrite_values = [float(runtimes[m][lattice]["total_mean"])
                          + float(runtimes[m][lattice]["rwscf_mean"])
                          + float(runtimes[m][lattice]["scf_mean"])
                          for m in models]

        ax.barh(rewritten_x, rewrite_values,
                label=f"Analysis + Transform + Runtime ({lattice})",
                color=LATTICE_COLORS[lattice], height=0.3, zorder=2)

    ax.barh(baseline_x, baseline_values, label="Baseline", color="papayawhip",
            height=0.3, zorder=2)

    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x")

    for spine in ["top", "bottom", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Breakeven Point comparison between Transformed IR and Baseline IR"
    )

    ax.legend(
        loc="upper center",
        frameon=False,
        ncols=2,
        bbox_to_anchor=(0.5, -0.1)
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def create_table(runtimes: defaultdict, out_path: Path) -> None:
    models = sorted(runtimes)
    lattice = "banded-192"

    with open(out_path, "w") as f:

        f.write(r"""% https://tinkertailorsoldiersponge.com/blog/2014/07/07/fancy-thesis-tables-in-latex
                    \newcommand{\ra}[1]{\renewcommand{\arraystretch}{#1}}
                    \begin{table}
                    \resizebox{\columnwidth}{!} {
                    \begin{tabular}{cccccc}\toprule
                    Model & Analysis & Transform & Baseline & Transformed & Speedup \\ \midrule""")

        for i, m in enumerate(models):
            if i % 2 == 0:
                f.write(rf"""\rowcolor{{black!20}} {m} & {runtimes[m][lattice]["total_mean"]:.4f} & {runtimes[m][lattice]["rwscf_mean"]:.4f} & {runtimes[m][lattice]["base_mean"]:.4f} & {runtimes[m][lattice]["scf_mean"]:.4f} & {runtimes[m][lattice]["scf_speedup"]:.4f} \\
                """)
            else:
                f.write(rf"""{m} & {runtimes[m][lattice]["total_mean"]:.4f} & {runtimes[m][lattice]["rwscf_mean"]:.4f}  & {runtimes[m][lattice]["base_mean"]:.4f}  & {runtimes[m][lattice]["scf_mean"]:.4f}  & {runtimes[m][lattice]["scf_speedup"]:.4f} \\
                """)

        f.write(r"""\bottomrule
                    \end{tabular}}
                    \caption{This table showcases runtimes of various models under a seed lattice element that is singularly banded across 192 slices on both spatial dimensions.}
                    \label{tab:runtimes}
                    \end{table}""")


runtimes = load_data(Path("csv_data/timings.csv"))
plot(runtimes, Path("assets/breakeven.png"))
create_table(runtimes, Path("assets/runtimes.tex"))
