"""Quanvolution2D: a quantum convolutional filter layer.

A quanvolutional layer (Henderson et al., 2020) slides a small window
over the input image, feeds each patch into a quantum circuit, and uses
qubit expectation values as the output feature map channels -- the
quantum analogue of a convolutional filter bank.
"""

import torch
import torch.nn as nn

import pennylane as qml

from qfz.circuits import get_circuit
from qfz.layers.base import QuantumFilter2d
from qfz.layers.encodings import get_encoding


class Quanvolution2D(QuantumFilter2d):
    """Quantum convolution layer for image tensors.

    Each patch is encoded onto qubits, passed through a quantum circuit,
    and the Pauli-Z expectation value of qubit ``k`` becomes output
    channel ``k`` at that spatial location.

    Input:  ``[B, in_channels, H, W]`` with values in ``[0, 1]``.
    Output: ``[B, out_channels, H_out, W_out]`` with values in ``[-1, 1]``,
    following the standard convolution output-size formula.

    Geometry arguments (``kernel_size``, ``stride``, ``padding``,
    ``dilation``) mirror ``torch.nn.Conv2d`` and accept an int or tuple.
    ``patch_size`` is accepted as an alias for ``kernel_size``.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output feature maps. Limited by the qubit
            count: ``in_channels * kernel_h * kernel_w`` (stacked mode) or
            ``kernel_h * kernel_w`` per channel (``per_channel=True``).
        kernel_size: Patch side length(s). Default 2.
        stride: Patch stride. Defaults to ``kernel_size`` (non-overlapping).
        padding: Zero-padding. Default 0.
        dilation: Patch element spacing. Default 1.
        encoding: Data encoding, ``"angle"`` or ``"basis"``.
        circuit: Circuit block name: ``"random"``, ``"hardware_efficient"``,
            or ``"iqp"``.
        backend: Quantum backend, currently only ``"pennylane"``.
        n_layers: Number of layers in the circuit block.
        trainable: If True, circuit angles are trainable parameters; if
            False (default), the layer is a fixed feature extractor.
        per_channel: Apply the same filter to each input channel
            independently (recommended for RGB; see
            :class:`~qfz.layers.base.QuantumFilter2d`).
        seed: Seed for circuit structure and initial angles.
        device_name: PennyLane device name (default ``"default.qubit"``).
        max_patch_batch: Max patches simulated at once (memory bound).
        patch_size: Alias for ``kernel_size`` (original quanvolution
            terminology); do not pass both.

    Example:
        >>> layer = Quanvolution2D(in_channels=1, out_channels=4)
        >>> x = torch.rand(8, 1, 28, 28)
        >>> layer(x).shape
        torch.Size([8, 4, 14, 14])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=None,
        stride=None,
        padding=0,
        dilation=1,
        encoding: str = "angle",
        circuit: str = "random",
        backend: str = "pennylane",
        n_layers: int = 1,
        trainable: bool = False,
        per_channel: bool = False,
        seed: int = 42,
        device_name: str = "default.qubit",
        max_patch_batch: int = 4096,
        patch_size=None,
    ):
        if kernel_size is not None and patch_size is not None:
            raise ValueError("Pass either kernel_size or patch_size, not both.")
        if kernel_size is None:
            kernel_size = patch_size if patch_size is not None else 2
        if backend != "pennylane":
            raise ValueError(f"Unsupported backend '{backend}'. Only 'pennylane' is supported.")

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, per_channel, max_patch_batch)
        self.trainable = trainable

        self.encode = get_encoding(encoding)
        self.circuit_block = get_circuit(circuit, n_qubits=self.n_qubits,
                                         n_layers=n_layers, seed=seed)

        weights = torch.tensor(self.circuit_block.init_weights(), dtype=torch.float32)
        if trainable:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer("weights", weights)

        dev = qml.device(device_name, wires=self.n_qubits)
        wires = list(range(self.n_qubits))
        n_measured = self.measured_qubits

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            self.encode(inputs, wires=wires)
            self.circuit_block.apply(weights, wires=wires)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_measured)]

        self.qnode = qnode

    def _run_circuit(self, patches: torch.Tensor) -> torch.Tensor:
        return self.qnode(patches, self.weights)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", trainable={self.trainable}"
