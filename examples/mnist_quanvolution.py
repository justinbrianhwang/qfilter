"""MNIST classification with a fixed random quanvolution filter.

Runs a small CPU-friendly experiment: a non-trainable Quanvolution2D
front-end followed by a linear classifier, compared against nothing --
see qfz.benchmarks.run_all for the full comparison grid.

Usage::

    python examples/mnist_quanvolution.py
"""

import torch

from qfz.benchmarks.metrics import count_parameters, evaluate_accuracy
from qfz.datasets import get_dataloaders
from qfz.layers import Quanvolution2D
from qfz.models import HybridCNN
from qfz.utils import set_seed


def main():
    set_seed(42)
    train_loader, test_loader, info = get_dataloaders(
        "mnist", batch_size=64, train_size=2000, test_size=1000)

    quanv = Quanvolution2D(in_channels=1, out_channels=4, kernel_size=2,
                           stride=2, encoding="angle", circuit="random")
    model = HybridCNN(quanv, num_classes=info.num_classes,
                      in_channels=info.in_channels, img_size=info.img_size)
    print(model)
    print("parameters:", count_parameters(model))

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3)
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
