"""Classical CNN baseline, architecture-matched to the hybrid models.

To compare a quantum filter against classical convolution fairly, the
baseline mirrors the hybrid architecture exactly: one convolution with
the same geometry (kernel, stride, output channels) as the quantum
filter, a bounded activation (Tanh, matching the [-1, 1] range of
Pauli-Z expectation values), and the same linear classification head.
"""

import torch
import torch.nn as nn


class ClassicalCNN(nn.Module):
    """Convolution + Tanh + linear head, mirroring the hybrid models.

    Works with any input size, channel count, and number of classes.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of target classes.
        img_size: Input spatial size ``(H, W)``; used to size the head.
        out_channels: Feature maps produced by the conv layer. Defaults
            to ``4 * in_channels``, matching the quantum filters' default.
        kernel_size: Conv kernel size (default 2, like the quantum patch).
        stride: Conv stride. Defaults to ``kernel_size``.
    """

    def __init__(self, in_channels: int, num_classes: int, img_size: tuple,
                 out_channels: int = None, kernel_size: int = 2, stride: int = None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else 4 * in_channels
        stride = stride if stride is not None else kernel_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.Tanh(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *img_size)
            flat_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images ``[B, C, H, W]``."""
        return self.classifier(self.features(x))
