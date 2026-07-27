"""Spectrum-matched random Fourier feature filter: a classical control.

A fixed quantum filter with RY(pi*x) angle encoding realizes a truncated
Fourier series whose frequency spectrum is the set of integer vectors
``{-1, 0, 1}^n_qubits`` (Schuld et al., 2021, arXiv:2008.08605). Random
Fourier features drawn from that same spectrum are therefore the
dequantization-aware classical control for such filters (Landman et al.,
ICLR 2023, arXiv:2210.13200): each output channel is

    f_j(x) = cos(pi * <w_j, x> + b_j),   w_j ~ Uniform({-1,0,1}^d \\ {0}),
                                          b_j ~ Uniform[0, 2*pi).

:class:`RFFFilter2D` packages this feature map behind the exact same
:class:`~qfz.layers.base.QuantumFilter2d` geometry (kernel/stride/padding/
dilation, per-channel mode) as the quantum filters, so comparisons are
architecture-matched by construction. Like them, its outputs lie in
``[-1, 1]`` and it is a *fixed* (non-trainable) feature extractor.
"""

import math

import torch

from qfz.layers.base import QuantumFilter2d


class RFFFilter2D(QuantumFilter2d):
    """Frozen random Fourier feature filter over image patches.

    Interface mirrors :class:`~qfz.layers.quanvolution.Quanvolution2D`
    (minus the quantum-specific arguments); see
    :class:`~qfz.layers.base.QuantumFilter2d` for the geometry arguments.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output feature maps (frequency samples).
        kernel_size: Patch side length(s). Default 2.
        stride: Patch stride. Defaults to ``kernel_size``.
        padding: Zero-padding. Default 0.
        dilation: Patch element spacing. Default 1.
        per_channel: Apply the same filter to each channel independently.
        seed: Seed for the frequency and phase draws.
        max_patch_batch: Max patches evaluated at once.

    Example:
        >>> layer = RFFFilter2D(in_channels=1, out_channels=4)
        >>> layer(torch.rand(8, 1, 28, 28)).shape
        torch.Size([8, 4, 14, 14])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=2,
        stride=None,
        padding=0,
        dilation=1,
        per_channel: bool = False,
        seed: int = 42,
        max_patch_batch: int = 4096,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, per_channel, max_patch_batch)
        generator = torch.Generator().manual_seed(seed)
        freqs = torch.randint(-1, 2, (self.measured_qubits, self.n_qubits),
                              generator=generator)
        # The zero vector is a constant feature; resample any such rows.
        while (zero_rows := (freqs == 0).all(dim=1)).any():
            freqs[zero_rows] = torch.randint(
                -1, 2, (int(zero_rows.sum()), self.n_qubits), generator=generator)
        phases = 2 * math.pi * torch.rand(self.measured_qubits, generator=generator)
        self.register_buffer("freqs", freqs.float())
        self.register_buffer("phases", phases)

    def _run_circuit(self, patches: torch.Tensor) -> torch.Tensor:
        return torch.cos(math.pi * patches @ self.freqs.T + self.phases)
