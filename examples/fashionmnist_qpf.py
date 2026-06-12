"""FashionMNIST classification with the Quantum Preprocessing Filter.

The QPF is parameter-free: only the linear head is trained. Try the four
entanglement patterns (horizontal / vertical / diagonal / ring) and
compare.

Usage::

    python examples/fashionmnist_qpf.py [pattern]
"""

import sys

import torch

from qfz.benchmarks.metrics import count_parameters, evaluate_accuracy
from qfz.datasets import get_dataloaders
from qfz.layers import QPF
from qfz.models import HybridCNN
from qfz.utils import set_seed


def main(entanglement: str = "ring"):
    set_seed(42)
    train_loader, test_loader, info = get_dataloaders(
        "fashionmnist", batch_size=64, train_size=2000, test_size=1000)

    qpf = QPF(in_channels=1, entanglement=entanglement)
    model = HybridCNN(qpf, num_classes=info.num_classes,
                      in_channels=info.in_channels, img_size=info.img_size)
    print(f"entanglement pattern: {entanglement}")
    print("parameters:", count_parameters(model))

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        acc = evaluate_accuracy(model, test_loader)
        print(f"epoch {epoch + 1}: test accuracy {acc:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ring")
