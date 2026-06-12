"""QPF: Quantum Preprocessing Filter.

A *fixed* (parameter-free) 4-qubit quantum filter applied to 2x2 image
patches: RY angle encoding followed by a chosen CNOT entanglement
pattern, then Pauli-Z measurement of all four qubits. Because it has no
trainable parameters, it acts as a deterministic feature transform that
can be applied once as preprocessing.

Qubit layout within a 2x2 patch::

    q0 q1
    q2 q3
"""

import torch

import pennylane as qml

from qfz.layers.base import QuantumFilter2d

# CNOT (control, target) pairs per entanglement pattern, indexing the
# 2x2 patch qubits as [[0, 1], [2, 3]].
ENTANGLEMENT_PATTERNS = {
    "horizontal": [(0, 1), (2, 3)],
    "vertical": [(0, 2), (1, 3)],
    "diagonal": [(0, 3), (1, 2)],
    "ring": [(0, 1), (1, 3), (3, 2), (2, 0)],
}


class QPF(QuantumFilter2d):
    """Fixed 4-qubit quantum preprocessing filter.

    Always applied per channel: each input channel is filtered
    independently by the same circuit and produces 4 output channels
    (one per qubit), so ``out_channels = 4 * in_channels``.

    Input:  ``[B, in_channels, H, W]`` with values in ``[0, 1]``.
    Output: ``[B, 4 * in_channels, H_out, W_out]`` with values in ``[-1, 1]``.

    Args:
        in_channels: Number of input image channels.
        entanglement: CNOT pattern, one of ``"horizontal"``,
            ``"vertical"``, ``"diagonal"``, ``"ring"``.
        stride: Patch stride. Defaults to 2 (non-overlapping patches);
            set ``stride=1`` for a denser, same-resolution-ish output.
        padding: Zero-padding added to the input. Default 0.
        device_name: PennyLane device name (default ``"default.qubit"``).
        max_patch_batch: Max patches simulated at once (memory bound).

    Example:
        >>> qpf = QPF(in_channels=1, entanglement="ring")
        >>> x = torch.rand(8, 1, 28, 28)
        >>> qpf(x).shape
        torch.Size([8, 4, 14, 14])
    """

    def __init__(
        self,
        in_channels: int = 1,
        entanglement: str = "ring",
        stride=2,
        padding=0,
        device_name: str = "default.qubit",
        max_patch_batch: int = 4096,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=4 * in_channels,
            kernel_size=2,
            stride=stride,
            padding=padding,
            per_channel=True,
            max_patch_batch=max_patch_batch,
        )
        if entanglement not in ENTANGLEMENT_PATTERNS:
            raise ValueError(
                f"Unknown entanglement '{entanglement}'. "
                f"Available: {sorted(ENTANGLEMENT_PATTERNS)}"
            )
        self.entanglement = entanglement
        pattern = ENTANGLEMENT_PATTERNS[entanglement]

        dev = qml.device(device_name, wires=4)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs):
            qml.AngleEmbedding(inputs * torch.pi, wires=range(4), rotation="Y")
            for control, target in pattern:
                qml.CNOT(wires=[control, target])
            return [qml.expval(qml.PauliZ(w)) for w in range(4)]

        self.qnode = qnode

    def _run_circuit(self, patches: torch.Tensor) -> torch.Tensor:
        return self.qnode(patches)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", entanglement='{self.entanglement}'"
