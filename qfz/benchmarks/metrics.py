"""Metric helpers: accuracy, parameter counts, timing."""

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters.

    Returns:
        Dict with ``trainable`` and ``total`` counts. Non-trainable
        quantum filter angles (stored as buffers) are not parameters and
        appear in neither count.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, device: str = "cpu") -> float:
    """Top-1 accuracy of ``model`` over ``loader``."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def measure_inference_time(model: nn.Module, loader, device: str = "cpu",
                           n_batches: int = 5) -> dict:
    """Average wall-clock inference time over the first ``n_batches``.

    Returns:
        Dict with ``ms_per_batch``, ``ms_per_image`` and the batch size used.
    """
    model.eval()
    times, batch_size = [], None
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(device)
        batch_size = images.shape[0]
        start = time.perf_counter()
        model(images)
        times.append(time.perf_counter() - start)
    mean_s = sum(times) / max(len(times), 1)
    return {
        "ms_per_batch": 1000 * mean_s,
        "ms_per_image": 1000 * mean_s / max(batch_size or 1, 1),
        "batch_size": batch_size,
    }
