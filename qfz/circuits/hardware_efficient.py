"""Hardware-efficient ansatz: layered single-qubit rotations + ring CNOTs.

The standard variational circuit for near-term devices: each layer
applies parameterized RY and RZ rotations on every qubit followed by a
ring of CNOTs. All parameters are rotation angles, making this the
natural choice for *trainable* quantum filters.
"""

import numpy as np
import pennylane as qml


class HardwareEfficientCircuit:
    """Layered RY/RZ + ring-CNOT entangling block.

    Args:
        n_qubits: Number of qubits the block acts on.
        n_layers: Number of rotation+entangle layers.
        seed: Seed for the initial angles (the structure is fixed).
    """

    def __init__(self, n_qubits: int, n_layers: int = 1, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed

    @property
    def weights_shape(self) -> tuple:
        """Shape of the angle tensor expected by :meth:`apply`: (layers, qubits, 2)."""
        return (self.n_layers, self.n_qubits, 2)

    def init_weights(self) -> np.ndarray:
        """Sample initial angles uniformly from [0, 2*pi)."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0, 2 * np.pi, size=self.weights_shape)

    def apply(self, weights, wires):
        """Apply the block inside a qnode.

        Args:
            weights: Angle tensor of shape :attr:`weights_shape`.
            wires: Wires to act on (length ``n_qubits``).
        """
        wires = list(wires)
        for layer in range(self.n_layers):
            for i, wire in enumerate(wires):
                qml.RY(weights[layer, i, 0], wires=wire)
                qml.RZ(weights[layer, i, 1], wires=wire)
            if len(wires) > 1:
                for i in range(len(wires)):
                    qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
