"""Shared machinery for 2D quantum filter layers.

All quantum filter layers in qfz follow the same pipeline:

    [B, C, H, W] --unfold--> patches --quantum circuit--> expectation
    values --fold--> [B, out_channels, H_out, W_out]

:class:`QuantumFilter2d` implements the classical plumbing (patch
extraction with full ``nn.Conv2d``-style geometry parameters, chunking,
per-channel mode, folding) so that concrete layers only define the
quantum circuit via :meth:`QuantumFilter2d._run_circuit`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(value):
    """Normalize an int-or-tuple argument to a ``(h, w)`` tuple, as in PyTorch."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected an int or a 2-tuple, got {value!r}")
        return tuple(value)
    return (value, value)


class QuantumFilter2d(nn.Module):
    """Base class for quantum filters over 2D images.

    Geometry arguments mirror ``torch.nn.Conv2d`` and accept either an
    int or an ``(h, w)`` tuple.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output feature maps.
        kernel_size: Side length(s) of the input patch fed to the circuit.
        stride: Patch stride. Defaults to ``kernel_size`` (non-overlapping
            patches, the quanvolution convention) rather than 1.
        padding: Zero-padding added to both sides of the input.
        dilation: Spacing between patch elements.
        per_channel: If True, the *same* quantum filter is applied to each
            input channel independently (like a depthwise convolution) and
            the outputs are concatenated. This keeps the qubit count at
            ``kernel_h * kernel_w`` regardless of ``in_channels`` — strongly
            recommended for RGB images, where the stacked mode would need
            ``3 * kernel_h * kernel_w`` qubits. Requires ``out_channels``
            to be divisible by ``in_channels``.
        max_patch_batch: Maximum number of patches evaluated by the
            quantum simulator at once. Bounds memory usage; tune down if
            you run out of RAM with many qubits.

    Subclasses must set up their circuit and implement
    :meth:`_run_circuit`.
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
        max_patch_batch: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.per_channel = per_channel
        self.max_patch_batch = max_patch_batch

        patch_dim = self.kernel_size[0] * self.kernel_size[1]
        if per_channel:
            if out_channels % in_channels != 0:
                raise ValueError(
                    f"per_channel=True requires out_channels ({out_channels}) "
                    f"divisible by in_channels ({in_channels})."
                )
            self.n_qubits = patch_dim
            self.measured_qubits = out_channels // in_channels
        else:
            self.n_qubits = in_channels * patch_dim
            self.measured_qubits = out_channels

        if self.measured_qubits > self.n_qubits:
            raise ValueError(
                f"Requested {self.measured_qubits} measurements per circuit but "
                f"only {self.n_qubits} qubits are available "
                f"(= {'kernel_h * kernel_w' if per_channel else 'in_channels * kernel_h * kernel_w'})."
            )

    def _run_circuit(self, patches: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum circuit on a batch of patches.

        Args:
            patches: Tensor of shape ``[N, n_qubits]`` with values in [0, 1].

        Returns:
            Expectation values of shape ``[N, measured_qubits]``.
        """
        raise NotImplementedError

    def _output_size(self, height: int, width: int) -> tuple:
        """Spatial output size, following the standard convolution formula."""
        h_out = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return h_out, w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter.

        Args:
            x: Tensor of shape ``[B, in_channels, H, W]``, values in [0, 1].

        Returns:
            Tensor of shape ``[B, out_channels, H_out, W_out]``.
        """
        batch_size, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        h_out, w_out = self._output_size(height, width)

        if self.per_channel:
            # Treat each channel as an independent single-channel image.
            x = x.reshape(batch_size * channels, 1, height, width)

        # [B', n_qubits, L] -> [B' * L, n_qubits], L = h_out * w_out
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride,
                           padding=self.padding, dilation=self.dilation)
        patches = patches.transpose(1, 2).reshape(-1, self.n_qubits)
        patches = patches.clamp(0.0, 1.0)

        outputs = []
        for chunk in patches.split(self.max_patch_batch):
            out = self._run_circuit(chunk)
            if isinstance(out, (list, tuple)):
                out = torch.stack(list(out), dim=-1)
            if out.ndim == 1:  # single measured qubit
                out = out.unsqueeze(-1)
            outputs.append(out)
        out = torch.cat(outputs, dim=0)  # [B' * L, measured_qubits]

        n_patches = h_out * w_out
        if self.per_channel:
            # [B*C*L, q] -> [B, C, L, q] -> [B, C*q, h_out, w_out]
            out = out.reshape(batch_size, channels, n_patches, self.measured_qubits)
            out = out.permute(0, 1, 3, 2)
            out = out.reshape(batch_size, self.out_channels, h_out, w_out)
        else:
            out = out.reshape(batch_size, n_patches, self.out_channels)
            out = out.transpose(1, 2).reshape(batch_size, self.out_channels, h_out, w_out)
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"n_qubits={self.n_qubits}, per_channel={self.per_channel}"
        )
