"""Train a model on a registered dataset and record benchmark metrics.

Usage::

    python -m qfz.benchmarks.train --dataset mnist --model quanv \
        --epochs 3 --train-size 2000 --test-size 1000 --out results

Models:
    classical  Conv2d baseline, architecture-matched to the hybrids.
    quanv      Fixed random quanvolution filter + linear head.
    qpf        Fixed quantum preprocessing filter + linear head.
    pqc        Trainable parameterized quantum convolution + linear head.

Each run writes ``<out>/<dataset>_<model>.json`` containing the full
config and metrics (test accuracy, parameter counts, training time,
inference time), so experiments are self-describing and comparable.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from qfz.benchmarks.metrics import count_parameters, evaluate_accuracy, measure_inference_time
from qfz.datasets import get_dataloaders
from qfz.layers import PQCConv2D, QPF, Quanvolution2D
from qfz.models import ClassicalCNN, HybridCNN
from qfz.utils import set_seed

MODELS = ("classical", "quanv", "qpf", "pqc")


def build_model(name: str, info, seed: int = 42) -> nn.Module:
    """Build a benchmark model configured for a dataset.

    For multi-channel inputs the quantum filters run in per-channel mode
    (4 qubits regardless of channel count); all models produce
    ``4 * in_channels`` feature maps so the comparison is architecture-matched.

    Args:
        name: One of :data:`MODELS`.
        info: ``DatasetInfo`` describing the dataset.
        seed: Seed for quantum circuit structure / initial angles.

    Returns:
        A model mapping images to class logits.
    """
    channels, classes, size = info.in_channels, info.num_classes, info.img_size
    out_channels = 4 * channels
    per_channel = channels > 1

    if name == "classical":
        return ClassicalCNN(channels, classes, size, out_channels=out_channels)
    if name == "quanv":
        layer = Quanvolution2D(channels, out_channels, per_channel=per_channel, seed=seed)
    elif name == "qpf":
        layer = QPF(channels, entanglement="ring")
    elif name == "pqc":
        layer = PQCConv2D(channels, out_channels, per_channel=per_channel, seed=seed)
    else:
        raise ValueError(f"Unknown model '{name}'. Available: {MODELS}")
    return HybridCNN(layer, classes, channels, size)


def run_experiment(dataset: str, model_name: str, epochs: int = 3,
                   batch_size: int = 64, train_size: int = 2000,
                   test_size: int = 1000, lr: float = 1e-3, seed: int = 42,
                   root: str = None, device: str = "cpu",
                   out_dir: str = None, progress: bool = True) -> dict:
    """Train one (dataset, model) pair and return its benchmark record.

    Returns:
        Dict with ``config`` and ``metrics`` keys; also written to
        ``<out_dir>/<dataset>_<model>.json`` when ``out_dir`` is given.
    """
    set_seed(seed)
    train_loader, test_loader, info = get_dataloaders(
        dataset, root=root, batch_size=batch_size,
        train_size=train_size, test_size=test_size, seed=seed)
    model = build_model(model_name, info, seed=seed).to(device)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        iterator = tqdm(train_loader, desc=f"{dataset}/{model_name} epoch {epoch + 1}/{epochs}",
                        disable=not progress, leave=False)
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=f"{loss.item():.3f}")
    train_time = time.perf_counter() - train_start

    record = {
        "config": {
            "dataset": dataset, "model": model_name, "epochs": epochs,
            "batch_size": batch_size, "train_size": train_size,
            "test_size": test_size, "lr": lr, "seed": seed, "device": device,
        },
        "metrics": {
            "test_accuracy": evaluate_accuracy(model, test_loader, device),
            "parameters": count_parameters(model),
            "train_time_s": train_time,
            "inference": measure_inference_time(model, test_loader, device),
        },
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / f"{dataset}_{model_name}.json").write_text(
            json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="registered dataset name")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=2000,
                        help="training subset size (None-like -1 = full set)")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default=None,
                        help="data root (default: $QFZ_DATA_ROOT or ./data)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    record = run_experiment(
        args.dataset, args.model, epochs=args.epochs, batch_size=args.batch_size,
        train_size=None if args.train_size < 0 else args.train_size,
        test_size=None if args.test_size < 0 else args.test_size,
        lr=args.lr, seed=args.seed, root=args.root, device=args.device,
        out_dir=args.out)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
