"""Plotting helpers for feature maps and benchmark results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # work headless; callers can switch backends before import
import matplotlib.pyplot as plt
import torch


def plot_feature_maps(layer, image: torch.Tensor, save_path: str = None):
    """Visualize a quantum filter's output channels for one image.

    Args:
        layer: A quantum filter layer.
        image: Single image tensor ``[C, H, W]`` with values in [0, 1].
        save_path: If given, save the figure there; otherwise return it.

    Returns:
        The matplotlib figure.
    """
    with torch.no_grad():
        out = layer(image.unsqueeze(0))[0]  # [C', H', W']
    n_maps = out.shape[0]
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(2.2 * (n_maps + 1), 2.5))
    axes[0].imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")
    axes[0].set_title("input")
    for k in range(n_maps):
        axes[k + 1].imshow(out[k], cmap="viridis", vmin=-1, vmax=1)
        axes[k + 1].set_title(f"ch {k}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_benchmark_summary(summary_path: str, save_path: str = None):
    """Bar chart of test accuracy per (dataset, model) from a summary JSON.

    Runs sharing a (dataset, model) key — e.g. the same configuration
    trained with several seeds — are aggregated into a mean bar with a
    standard-deviation error bar.

    Args:
        summary_path: Path to ``summary.json`` written by
            ``qfz.benchmarks.run_all``.
        save_path: If given, save the figure there.

    Returns:
        The matplotlib figure.
    """
    from qfz.benchmarks.evaluate import aggregate_records

    runs = json.loads(Path(summary_path).read_text())
    rows = {(r["dataset"], r["model"]): r for r in aggregate_records(runs)}
    datasets = sorted({ds for ds, _ in rows})
    models = sorted({m for _, m in rows})

    fig, ax = plt.subplots(figsize=(2 + 2.2 * len(datasets), 4))
    width = 0.8 / len(models)
    for j, model in enumerate(models):
        means = [rows[(ds, model)]["test_accuracy_mean"] if (ds, model) in rows else 0.0
                 for ds in datasets]
        stds = [rows[(ds, model)]["test_accuracy_std"] if (ds, model) in rows else 0.0
                for ds in datasets]
        positions = [i + j * width for i in range(len(datasets))]
        ax.bar(positions, means, width=width, label=model,
               yerr=stds if any(stds) else None, capsize=3)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
