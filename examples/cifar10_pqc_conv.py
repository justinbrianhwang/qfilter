"""CIFAR-10 classification with a trainable parameterized quantum convolution.

RGB images use per-channel mode: the same 4-qubit circuit filters each
color channel independently (12 output channels total), keeping the
simulation cheap. The circuit angles are trained end-to-end by
backpropagation through the simulator.

Usage::

    python examples/cifar10_pqc_conv.py
"""

import torch

from qfz.benchmarks.metrics import count_parameters, evaluate_accuracy
from qfz.datasets import get_dataloaders
from qfz.layers import PQCConv2D
from qfz.models import HybridCNN
from qfz.utils import set_seed


def main():
    set_seed(42)
    train_loader, test_loader, info = get_dataloaders(
        "cifar10", batch_size=64, train_size=2000, test_size=1000)

    pqc = PQCConv2D(in_channels=3, out_channels=12, per_channel=True,
                    circuit="hardware_efficient", n_layers=2)
    model = HybridCNN(pqc, num_classes=info.num_classes,
                      in_channels=info.in_channels, img_size=info.img_size)
    print("parameters:", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"epoch {epoch + 1} step {step:3d} loss {loss.item():.4f}")
        acc = evaluate_accuracy(model, test_loader)
        print(f"epoch {epoch + 1}: test accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
