"""Hybrid quantum-classical model: quantum filter + linear head.

Wraps *any* quantum filter layer (Quanvolution2D, QPF, PQCConv2D, or a
custom :class:`~qfz.layers.base.QuantumFilter2d`) and attaches a linear
classification head sized automatically from the filter's output shape,
so it works with any input size, channel count, and number of classes.
"""

import torch
import torch.nn as nn


class HybridCNN(nn.Module):
    """Quantum filter followed by a linear classification head.

    Args:
        quantum_layer: Any module mapping ``[B, C, H, W]`` to
            ``[B, C', H', W']`` (typically a qfz quantum filter).
        num_classes: Number of target classes.
        in_channels: Number of input image channels.
        img_size: Input spatial size ``(H, W)``; used to size the head
            via a dummy forward pass.

    Example:
        >>> from qfz.layers import Quanvolution2D
        >>> layer = Quanvolution2D(in_channels=1, out_channels=4)
        >>> model = HybridCNN(layer, num_classes=10, in_channels=1, img_size=(28, 28))
        >>> model(torch.rand(2, 1, 28, 28)).shape
        torch.Size([2, 10])
    """

    def __init__(self, quantum_layer: nn.Module, num_classes: int,
                 in_channels: int, img_size: tuple):
        super().__init__()
        self.quantum = quantum_layer
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *img_size)
            flat_dim = self.quantum(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images ``[B, C, H, W]``."""
        return self.classifier(self.quantum(x))
