"""Publication chart for the qfz paper: main grid, mean +/- std, significance stars.

Writes Paper/figures/summary.pdf (+ a PNG preview). Requires matplotlib + scipy.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Paper" / "figures"

# --- data ---------------------------------------------------------------
runs = json.loads((REPO / "results/multiseed/summary.json").read_text())
accs = {}
for r in runs:
    c = r["config"]
    accs.setdefault((c["dataset"], c["model"]), []).append(r["metrics"]["test_accuracy"])

DATASETS = ["mnist", "fashionmnist", "cifar10", "svhn"]
DS_LABELS = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
MODELS = ["classical", "randconv", "rff", "quanv", "qpf", "pqc"]
MODEL_LABELS = ["classical (trained)", "randconv (frozen)", "rff (frozen)",
                "quanv", "qpf", "pqc"]
CLASSICAL = {"classical", "randconv", "rff"}

# categorical slots 1-6 (validated adjacent order, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
EDGE = {  # tone-on-tone darker step for hatch/edges on the baseline family
    "#2a78d6": "#1c5cab", "#eb6834": "#b34518", "#1baf7a": "#0e7d57",
    "#eda100": "#b37a00", "#e87ba4": "#c14e78", "#008300": "#005a00",
}


def holm(pvals):
    import numpy as np
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "pdf.fonttype": 42,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": MUTED,
})

fig, ax = plt.subplots(figsize=(7.3, 3.1))
n_ds, n_m = len(DATASETS), len(MODELS)
group_w = 0.82
bar_w = group_w / n_m * 0.94  # slim gap between adjacent bars

for j, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, COLORS)):
    xs, means, stds = [], [], []
    for i, ds in enumerate(DATASETS):
        vals = accs[(ds, model)]
        xs.append(i - group_w / 2 + (j + 0.5) * group_w / n_m)
        means.append(sum(vals) / len(vals))
        stds.append(stats.tstd(vals))
    hatch = "//" if model in CLASSICAL else None
    ax.bar(xs, means, width=bar_w, label=label, color=color,
           hatch=hatch, edgecolor=EDGE[color] if hatch else "none",
           linewidth=0.4 if hatch else 0.0, zorder=3)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=INK2,
                elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)
    # significance vs randconv (Holm-corrected within the model's family)
    if model not in CLASSICAL:
        raw = [stats.ttest_ind(accs[(ds, model)], accs[(ds, "randconv")],
                               equal_var=False).pvalue for ds in DATASETS]
        for i, p in enumerate(holm(raw)):
            s = star(p)
            if s:
                ax.text(xs[i], means[i] + stds[i] + 0.012, s, ha="center",
                        va="bottom", fontsize=7.5, color=INK2, zorder=5)

ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1.0)
ax.set_xticks(range(n_ds))
ax.set_xticklabels(DS_LABELS)
ax.tick_params(axis="x", length=0)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)

leg = ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=8,
                handlelength=1.4, handleheight=1.1, labelcolor=INK2)
fig.tight_layout(pad=0.4)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "summary.pdf", bbox_inches="tight")
fig.savefig(OUT / "summary_preview.png", dpi=180, bbox_inches="tight")
print("written:", OUT / "summary.pdf")
