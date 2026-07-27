"""Sweep figure for the qfz paper: accuracy vs epochs and vs training-set size.

Writes Paper/figures/sweeps.pdf (+ a PNG preview). Requires matplotlib + scipy.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Paper" / "figures"

MODELS = ["classical", "randconv", "rff", "quanv", "qpf"]
MODEL_LABELS = ["classical (trained)", "randconv (frozen)", "rff (frozen)",
                "quanv", "qpf"]
COLORS = {"classical": "#2a78d6", "randconv": "#eb6834", "rff": "#1baf7a",
          "quanv": "#eda100", "qpf": "#e87ba4"}
CLASSICAL = {"classical", "randconv", "rff"}
INK2, MUTED = "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def load(sweep):
    """-> {(dataset, model, value): [accs over seeds]}"""
    acc = {}
    for f in (REPO / "results" / "sweeps" / sweep).glob("*.json"):
        r = json.loads(f.read_text())
        if not isinstance(r, dict):  # skip aggregate summary.json
            continue
        c = r["config"]
        val = c["epochs"] if sweep == "epochs" else c["train_size"]
        acc.setdefault((c["dataset"], c["model"], val), []).append(
            r["metrics"]["test_accuracy"])
    return acc


ep = load("epochs")
bu = load("budget")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "pdf.fonttype": 42,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": MUTED,
})

fig = plt.figure(figsize=(7.3, 4.2))
gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.3)

PANELS = [
    (fig.add_subplot(gs[0, 0:2]), ep, "mnist", [3, 10, 30], "epochs", "MNIST"),
    (fig.add_subplot(gs[0, 2:4]), ep, "fashionmnist", [3, 10, 30], "epochs",
     "FashionMNIST"),
    (fig.add_subplot(gs[1, 0]), bu, "mnist", [500, 1000, 2000, 5000],
     "training images", "MNIST"),
    (fig.add_subplot(gs[1, 1]), bu, "fashionmnist", [500, 1000, 2000, 5000],
     "training images", "FashionMNIST"),
    (fig.add_subplot(gs[1, 2]), bu, "cifar10", [500, 1000, 2000, 5000],
     "training images", "CIFAR-10"),
    (fig.add_subplot(gs[1, 3]), bu, "svhn", [500, 1000, 2000, 5000],
     "training images", "SVHN"),
]

for ax, data, ds, values, xlabel, title in PANELS:
    for model, label in zip(MODELS, MODEL_LABELS):
        means = [np.mean(data[(ds, model, v)]) for v in values]
        stds = [np.std(data[(ds, model, v)], ddof=1) for v in values]
        ls = "--" if model in CLASSICAL else "-"
        ax.errorbar(values, means, yerr=stds, label=label,
                    color=COLORS[model], ls=ls, lw=1.3, marker="o", ms=2.6,
                    elinewidth=0.7, capsize=1.6, capthick=0.7)
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

for ax in (PANELS[0][0], PANELS[2][0]):
    ax.set_ylabel("test accuracy", fontsize=8)

handles, labels = PANELS[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
           fontsize=7.5, bbox_to_anchor=(0.5, 1.02), labelcolor=INK2)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "sweeps.pdf", bbox_inches="tight")
fig.savefig(OUT / "sweeps_preview.png", dpi=180, bbox_inches="tight")
print("written:", OUT / "sweeps.pdf")
