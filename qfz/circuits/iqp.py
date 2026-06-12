"""IQP-style circuit block: Hadamards + diagonal (Z-basis) interactions.

Instantaneous Quantum Polynomial (IQP) circuits interleave Hadamard
layers with gates diagonal in the Z basis (single-qubit RZ and two-qubit
ZZ interactions). They are conjectured hard to simulate classically and
are a popular choice for quantum feature maps.
"""

import numpy as np
import pennylane as qml


class IQPCircuit:
    """Layered H + RZ + ring-IsingZZ block.

    Per layer: Hadamard on every qubit, then a parameterized RZ on every
    qubit, then parameterized ZZ interactions along a ring.

    Args:
        n_qubits: Number of qubits the block acts on.
        n_layers: Number of layers.
        seed: Seed for the initial angles (the structure is fixed).
    """

    def __init__(self, n_qubits: int, n_layers: int = 1, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        # A ring has n pairs for n > 2, one pair for n == 2, none for n == 1.
        self.n_pairs = 0 if n_qubits < 2 else (1 if n_qubits == 2 else n_qubits)

    @property
    def weights_shape(self) -> tuple:
        """Shape of the angle tensor expected by :meth:`apply`: (layers, qubits + pairs)."""
        return (self.n_layers, self.n_qubits + self.n_pairs)

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
        n = len(wires)
        for layer in range(self.n_layers):
            for wire in wires:
                qml.Hadamard(wires=wire)
            for i, wire in enumerate(wires):
                qml.RZ(weights[layer, i], wires=wire)
            for p in range(self.n_pairs):
                qml.IsingZZ(weights[layer, n + p], wires=[wires[p], wires[(p + 1) % n]])
