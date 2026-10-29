"""Draw the paper's figures from results.json into <out dir>.

  diagnosis-time.pdf  per diagnosis step (collection, tree, diagnosis) and in total, the
                      wall-clock time to compute an instance's diagnosis
  repair-time.pdf     per model (one panel each), one pair of boxes per agent: the repair
                      time per run in the baseline and in FORECAST mode, without the
                      diagnosis; runs without a recorded runtime are left out

Boxes span the quartiles, with the median as a line and the mean as a yellow circle; whiskers
reach 1.5 times the interquartile range.

Usage: python3 plot.py <out dir> [--results results.json]
"""
import argparse, json, os, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import AGENTS, MODES, NAMES, overhead

#: Short agent names for the axis; the caption spells out ACR.
LABELS = {"mini-swe-agent": "mini-\nSWE-agent", "swe-agent": "SWE-\nagent", "acr": "ACR",
          "repair-agent": "Repair-\nAgent", "agentless": "Agentless"}
#: A colorblind-safe magenta and blue (FORECAST, the primary color); the baseline is drawn
#: hollow, so the modes also differ without color (grayscale print).
COLORS = {"baseline": "#D41159", "forecast": "#1A85FF"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
#: The mean: a yellow circle, outlined so it shows on hollow and filled boxes alike, and
#: larger than the faint outlier dots.
MEAN = dict(marker="o", markersize=4, markerfacecolor="#FFC20A", markeredgecolor=INK,
            markeredgewidth=0.6)
#: Diagnosis steps as the figure names them; collection includes instrumentation.
STEPS = (("Collection", lambda d: d["collect_seconds"]), ("Tree", lambda d: d["tree_seconds"]),
         ("Diagnosis", lambda d: d["diagnose_seconds"]), ("Total", overhead))


def box(ax, data, position, color, hollow=False, vertical=True, width=0.3):
    """One box in the figures' style."""
    ax.boxplot([data], positions=[position], widths=width, patch_artist=True,
               orientation="vertical" if vertical else "horizontal",
               showmeans=True, showfliers=True,
               boxprops=dict(facecolor="white" if hollow else color, edgecolor=color, linewidth=1),
               medianprops=dict(color=color if hollow else "white", linewidth=1.2),
               whiskerprops=dict(color=color, linewidth=1),
               capprops=dict(color=color, linewidth=1),
               meanprops=MEAN,
               flierprops=dict(marker="o", markersize=1.5, markerfacecolor=color,
                               markeredgecolor="none", alpha=0.5))


def style(ax, axis):
    """Recessive grid along `axis`, no top and right spines."""
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mean_handle():
    return plt.Line2D([], [], color="none", label="Mean", **MEAN)


def diagnosis_time(res, out):
    """Diagnosis time per step, one horizontal box each, in minutes."""
    ds = list(res["diagnoses"].values())
    fig, ax = plt.subplots(figsize=(3.1, 1.7))
    for k, (name, seconds) in enumerate(STEPS):
        box(ax, [seconds(d) / 60 for d in ds if seconds(d)], len(STEPS) - 1 - k, COLORS["forecast"],
            vertical=False, width=0.55)
    ax.set_xscale("log")
    ax.set_yticks(range(len(STEPS)))
    ax.set_yticklabels([name for name, _ in reversed(STEPS)])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlabel("Time per instance (min)")
    style(ax, "x")
    ax.legend(handles=[mean_handle()], loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/diagnosis-time.pdf")


def repair_time(res, out):
    """Repair time per run, per model, agent and mode."""
    models = list(res["repairs"])
    fig, axes = plt.subplots(1, len(models), figsize=(6.6, 2.4), sharey=True, squeeze=False)
    for col, m in enumerate(models):
        ax = axes[0][col]
        for k, agent in enumerate(AGENTS):
            for j, mode in enumerate(MODES):
                data = [r["seconds"] for r in res["repairs"][m][agent][mode].values() if r["seconds"]]
                box(ax, data, k + (j - 0.5) * 0.36, COLORS[mode], hollow=mode == "baseline")
        ax.set_yscale("log")
        ax.set_xticks(range(len(AGENTS)))
        ax.set_xticklabels([LABELS[g] for g in AGENTS])
        ax.tick_params(axis="x", length=0)
        style(ax, "y")
        ax.set_title(NAMES[m], fontsize=8, color=INK, loc="left")
        if col == 0:
            ax.set_ylabel("Repair time per run (s)")
    fig.legend(handles=[Patch(facecolor="white", edgecolor=COLORS["baseline"], label="Baseline"),
                        Patch(facecolor=COLORS["forecast"], edgecolor=COLORS["forecast"], label="FORECAST"),
                        mean_handle()],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{out}/repair-time.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "results", "results.json"))
    a = p.parse_args()
    with open(a.results) as f:
        res = json.load(f)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED, "pdf.fonttype": 42})
    os.makedirs(a.out, exist_ok=True)
    diagnosis_time(res, a.out)
    repair_time(res, a.out)


if __name__ == "__main__":
    main()
