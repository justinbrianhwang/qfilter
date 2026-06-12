"""Quantum Filter Zoo (qfz).

A research-oriented library of quantum filter layers that plug into
classical PyTorch models. Built on PennyLane.

This library is for research and educational experiments,
not a proven quantum advantage framework.
"""

__version__ = "0.2.0"

from qfz.layers import PQCConv2D, QPF, Quanvolution2D

__all__ = ["Quanvolution2D", "QPF", "PQCConv2D", "__version__"]
