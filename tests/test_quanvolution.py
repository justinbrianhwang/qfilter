"""Smoke tests for Quanvolution2D: shapes, value range, gradients, reproducibility."""

import pytest
import torch

from qfz.layers import Quanvolution2D


def test_output_shape_mnist():
    layer = Quanvolution2D(in_channels=1, out_channels=4, patch_size=2, stride=2)
    x = torch.rand(2, 1, 28, 28)
    out = layer(x)
    assert out.shape == (2, 4, 14, 14)


def test_output_range():
    layer = Quanvolution2D(in_channels=1, out_channels=4)
    x = torch.rand(2, 1, 8, 8)
    out = layer(x)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_out_channels_exceeds_qubits_raises():
    with pytest.raises(ValueError):
        Quanvolution2D(in_channels=1, out_channels=5, patch_size=2)


def test_non_trainable_by_default():
    layer = Quanvolution2D(in_channels=1, out_channels=4)
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 0


def test_trainable_gradients_flow():
    layer = Quanvolution2D(in_channels=1, out_channels=4, trainable=True)
    x = torch.rand(2, 1, 8, 8)
    layer(x).sum().backward()
    assert layer.weights.grad is not None
    assert torch.isfinite(layer.weights.grad).all()


def test_seed_reproducibility():
    x = torch.rand(2, 1, 8, 8)
    out1 = Quanvolution2D(in_channels=1, out_channels=4, seed=7)(x)
    out2 = Quanvolution2D(in_channels=1, out_channels=4, seed=7)(x)
    assert torch.allclose(out1, out2)


def test_basis_encoding_runs():
    layer = Quanvolution2D(in_channels=1, out_channels=4, encoding="basis")
    x = torch.rand(2, 1, 8, 8)
    assert layer(x).shape == (2, 4, 4, 4)
