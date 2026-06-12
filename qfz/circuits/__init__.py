"""Variational / fixed quantum circuit blocks used by the filter layers.

A circuit block exposes ``weights_shape``, ``init_weights()`` and
``apply(weights, wires)`` so layers can treat all circuits uniformly.
Register custom blocks via :data:`CIRCUITS`.
"""

from qfz.circuits.hardware_efficient import HardwareEfficientCircuit
from qfz.circuits.iqp import IQPCircuit
from qfz.circuits.random_circuit import RandomCircuit

CIRCUITS = {
    "random": RandomCircuit,
    "hardware_efficient": HardwareEfficientCircuit,
    "iqp": IQPCircuit,
}


def get_circuit(name: str, **kwargs):
    """Instantiate a circuit block by name.

    Args:
        name: Circuit identifier, e.g. ``"random"``.
        **kwargs: Forwarded to the circuit constructor.

    Returns:
        A circuit block instance.

    Raises:
        ValueError: If the circuit name is unknown.
    """
    try:
        cls = CIRCUITS[name]
    except KeyError:
        raise ValueError(
            f"Unknown circuit '{name}'. Available: {sorted(CIRCUITS)}"
        ) from None
    return cls(**kwargs)


__all__ = ["RandomCircuit", "HardwareEfficientCircuit", "IQPCircuit", "CIRCUITS", "get_circuit"]
