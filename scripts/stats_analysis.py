"""Statistical analysis for the qfz benchmark records.

Reproduces every statistic reported in the paper, from the per-run JSON
records under results/:
  - Welch and paired t-tests with Holm-corrected p-values
  - Cohen's d effect sizes
  - 95% CIs for the trained-vs-frozen (quanv) ablation differences
  - exact figures quoted in the text (pqc deficit, std ratio)

Usage: python scripts/stats_analysis.py   (requires numpy + scipy)
Writes results/stats_report.md and prints it.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["mnist", "fashionmnist", "cifar10", "svhn"]


def load(dirname):
    accs = {}
    for p in (REPO / dirname).glob("*.json"):
        if p.name.startswith("summary"):
            continue
        r = json.loads(p.read_text())
        c = r["config"]
        key = (c["dataset"], c.get("variant") or c["model"])
        accs.setdefault(key, {})[c["seed"]] = r["metrics"]["test_accuracy"]
    return accs


def paired_arrays(accs, ds, a, b):
    sa, sb = accs[(ds, a)], accs[(ds, b)]
    seeds = sorted(set(sa) & set(sb))
    return np.array([sa[s] for s in seeds]), np.array([sb[s] for s in seeds])


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2)
                     / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled


def holm(pvals):
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def family(accs, model, base, paired=True):
    rows = []
    for ds in DATASETS:
        x, y = paired_arrays(accs, ds, model, base)
        w = stats.ttest_ind(x, y, equal_var=False)
        pr = stats.ttest_rel(x, y)
        rows.append({
            "dataset": ds, "diff": x.mean() - y.mean(),
            "welch_p": w.pvalue, "paired_p": pr.pvalue,
            "d": cohens_d(x, y), "n": len(x),
        })
    for key in ("welch_p", "paired_p"):
        adj = holm([r[key] for r in rows])
        for r, a in zip(rows, adj):
            r[key + "_holm"] = a
    return rows


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.0e}"


def main():
    main_accs = load("results/multiseed")
    abl_accs = load("results/ablations")

    lines = ["# Statistical reanalysis report", ""]

    for model, base in [("quanv", "randconv"), ("qpf", "randconv"),
                        ("rff", "randconv"), ("quanv", "rff"), ("qpf", "rff"),
                        ("quanv", "classical"), ("qpf", "classical"),
                        ("pqc", "classical"), ("pqc", "randconv")]:
        if not all((ds, model) in main_accs for ds in DATASETS):
            lines.append(f"## {model} vs {base}: (runs not yet available)\n")
            continue
        lines.append(f"## {model} vs {base}  (Holm family = 4 datasets)")
        lines.append("| dataset | diff | Welch p | Welch p (Holm) | paired p | paired p (Holm) | Cohen's d | n |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in family(main_accs, model, base):
            lines.append(
                f"| {r['dataset']} | {r['diff']:+.4f} | {fmt_p(r['welch_p'])} "
                f"| {fmt_p(r['welch_p_holm'])} | {fmt_p(r['paired_p'])} "
                f"| {fmt_p(r['paired_p_holm'])} | {r['d']:.2f} | {r['n']} |")
        lines.append("")

    # trained-vs-frozen quanv ablation: paired 95% CI per dataset
    lines.append("## quanv trained vs frozen (ablation, paired 95% CI of difference)")
    lines.append("| dataset | mean diff | 95% CI | paired p | n |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        x, y = paired_arrays(abl_accs, ds, "quanv-trained", "quanv-frozen")
        d = x - y
        se = d.std(ddof=1) / np.sqrt(len(d))
        tcrit = stats.t.ppf(0.975, len(d) - 1)
        p = stats.ttest_rel(x, y).pvalue
        lines.append(f"| {ds} | {d.mean():+.4f} | [{d.mean()-tcrit*se:+.4f}, "
                     f"{d.mean()+tcrit*se:+.4f}] | {fmt_p(p)} | {len(d)} |")
    lines.append("")

    # exact quoted figures
    x, y = paired_arrays(main_accs, "mnist", "pqc", "classical")
    lines.append("## Exact quoted figures")
    lines.append(f"- pqc vs classical (MNIST): diff = {x.mean()-y.mean():+.4f} "
                 f"({(x.mean()-y.mean())*100:+.1f} pts), "
                 f"std ratio = {x.std(ddof=1)/y.std(ddof=1):.1f}x, "
                 f"variance ratio = {x.var(ddof=1)/y.var(ddof=1):.1f}x")
    lines.append("")

    report = "\n".join(lines)
    (REPO / "results" / "stats_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
