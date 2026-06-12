"""Classical-to-quantum data encoding strategies.

Each encoding function maps a batch of classical feature vectors onto
qubit states inside a PennyLane circuit. They are meant to be called
*inside* a qnode, after which an entangling/variational block is applied.

Conventions
-----------
- Inputs are assumed to be normalized to the range [0, 1]
  (e.g. image pixel intensities). Callers are responsible for scaling.
- ``wires`` is an iterable of qubit indices; the i-th feature is encoded
  on the i-th wire.
"""

import numpy as np
import pennylane as qml


def angle_encoding(inputs, wires, rotation: str = "Y"):
    """Encode features as single-qubit rotation angles.

    Each feature ``x`` in [0, 1] is mapped to a rotation ``R(pi * x)``
    on its own qubit, so the full intensity range covers half of the
    Bloch sphere (|0> at x=0, |1> at x=1 for rotation="Y").

    Args:
        inputs: Tensor of shape ``(..., n_features)`` with values in [0, 1].
            Leading batch dimensions are handled by PennyLane parameter
            broadcasting.
        wires: Qubit wires to encode onto. ``len(wires)`` must equal
            ``n_features``.
        rotation: Rotation axis, one of ``"X"``, ``"Y"``, ``"Z"``.
    """
    qml.AngleEmbedding(inputs * np.pi, wires=wires, rotation=rotation)


def basis_encoding(inputs, wires, threshold: float = 0.5):
    """Encode features as computational basis states via thresholding.

    Each feature is binarized (``x > threshold``) and the resulting bit
    decides whether the corresponding qubit is flipped from |0> to |1>.
    The flip is implemented as ``RX(pi * bit)``, which equals Pauli-X up
    to a global phase but, unlike ``qml.BasisEmbedding``, supports
    PennyLane parameter broadcasting over batched inputs.

    Note:
        Thresholding is non-differentiable, so this encoding is intended
        for *fixed* (non-trainable) preprocessing filters only.

    Args:
        inputs: Tensor of shape ``(..., n_features)`` with values in [0, 1].
        wires: Qubit wires to encode onto.
        threshold: Binarization threshold.
    """
    bits = (inputs > threshold).float()
    qml.AngleEmbedding(bits * np.pi, wires=wires, rotation="X")


ENCODINGS = {
    "angle": angle_encoding,
    "basis": basis_encoding,
}


def get_encoding(name: str):
    """Look up an encoding function by name.

    Args:
        name: One of ``"angle"`` or ``"basis"``.

    Returns:
        The encoding function.

    Raises:
        ValueError: If the encoding name is unknown.
    """
    try:
        return ENCODINGS[name]
    except KeyError:
        raise ValueError(
            f"Unknown encoding '{name}'. Available: {sorted(ENCODINGS)}"
        ) from None
