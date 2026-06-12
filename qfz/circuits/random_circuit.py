"""Seeded random variational circuit block.

This is the circuit used in the original quanvolution paper
(Henderson et al., "Quanvolutional Neural Networks", 2020): a randomly
structured sequence of single-qubit rotations and two-qubit gates whose
*structure* is fixed by a seed, while the rotation angles can either be
fixed random values (non-trainable filter) or trainable parameters.
"""

import numpy as np
import pennylane as qml


class RandomCircuit:
    """Random-structure entangling block built on ``qml.RandomLayers``.

    The gate structure (which wires, which rotations) is determined by
    ``seed`` and is identical on every call, which makes experiments
    reproducible. The angles are supplied externally as ``weights`` so
    the caller decides whether they are trainable.

    Args:
        n_qubits: Number of qubits the block acts on.
        n_layers: Number of random layers.
        n_rotations: Number of rotation gates per layer. Defaults to
            ``n_qubits``.
        seed: Seed fixing the random gate structure.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1, n_rotations: int = None, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_rotations = n_rotations if n_rotations is not None else n_qubits
        self.seed = seed

    @property
    def weights_shape(self) -> tuple:
        """Shape of the angle tensor expected by :meth:`apply`."""
        return (self.n_layers, self.n_rotations)

    def init_weights(self) -> np.ndarray:
        """Sample initial angles uniformly from [0, 2*pi).

        Returns:
            Array of shape :attr:`weights_shape`.
        """
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0, 2 * np.pi, size=self.weights_shape)

    def apply(self, weights, wires):
        """Apply the random block inside a qnode.

        Args:
            weights: Angle tensor of shape :attr:`weights_shape`.
            wires: Wires to act on (length ``n_qubits``).
        """
        qml.RandomLayers(weights, wires=wires, seed=self.seed)
