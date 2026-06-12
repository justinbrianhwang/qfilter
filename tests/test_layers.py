"""Tests for QPF, PQCConv2D, per-channel mode, and Conv2d-style geometry."""

import pytest
import torch

from qfz.layers import PQCConv2D, QPF, Quanvolution2D
from qfz.layers.qpf import ENTANGLEMENT_PATTERNS


# ---------------------------------------------------------------- geometry

def test_conv2d_style_params_tuple():
    layer = Quanvolution2D(in_channels=1, out_channels=4,
                           kernel_size=(2, 2), stride=(1, 2), padding=(1, 0))
    x = torch.rand(2, 1, 8, 8)
    # H_out = (8 + 2 - 2)//1 + 1 = 9, W_out = (8 - 2)//2 + 1 = 4
    assert layer(x).shape == (2, 4, 9, 4)


def test_dilation():
    layer = Quanvolution2D(in_channels=1, out_channels=4, kernel_size=2,
                           stride=2, dilation=2)
    x = torch.rand(2, 1, 8, 8)
    # effective kernel = 3 -> H_out = (8 - 3)//2 + 1 = 3
    assert layer(x).shape == (2, 4, 3, 3)


def test_patch_size_alias():
    layer = Quanvolution2D(in_channels=1, out_channels=4, patch_size=2)
    assert layer.kernel_size == (2, 2)
    with pytest.raises(ValueError):
        Quanvolution2D(in_channels=1, out_channels=4, kernel_size=2, patch_size=2)


def test_stride_defaults_to_kernel_size():
    layer = Quanvolution2D(in_channels=1, out_channels=4, kernel_size=3)
    assert layer.stride == (3, 3)


# ------------------------------------------------------------- per-channel

def test_per_channel_rgb_shape():
    layer = Quanvolution2D(in_channels=3, out_channels=12, per_channel=True)
    assert layer.n_qubits == 4  # independent of in_channels
    x = torch.rand(2, 3, 8, 8)
    assert layer(x).shape == (2, 12, 4, 4)


def test_per_channel_requires_divisibility():
    with pytest.raises(ValueError):
        Quanvolution2D(in_channels=3, out_channels=10, per_channel=True)


def test_per_channel_matches_single_channel():
    """The same filter applied per channel must equal filtering each channel alone."""
    layer = Quanvolution2D(in_channels=3, out_channels=12, per_channel=True, seed=3)
    single = Quanvolution2D(in_channels=1, out_channels=4, seed=3)
    x = torch.rand(1, 3, 4, 4)
    out = layer(x)
    for c in range(3):
        expected = single(x[:, c:c + 1])
        assert torch.allclose(out[:, 4 * c:4 * (c + 1)], expected, atol=1e-6)


# --------------------------------------------------------------------- QPF

def test_qpf_shape_and_range():
    qpf = QPF(in_channels=1, entanglement="ring")
    x = torch.rand(2, 1, 28, 28)
    out = qpf(x)
    assert out.shape == (2, 4, 14, 14)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_qpf_all_patterns_differ():
    x = torch.rand(1, 1, 8, 8)
    outputs = {name: QPF(1, entanglement=name)(x) for name in ENTANGLEMENT_PATTERNS}
    names = list(outputs)
    assert any(not torch.allclose(outputs[a], outputs[b])
               for i, a in enumerate(names) for b in names[i + 1:])


def test_qpf_has_no_trainable_parameters():
    qpf = QPF(in_channels=3)
    assert sum(p.numel() for p in qpf.parameters()) == 0


def test_qpf_rgb():
    qpf = QPF(in_channels=3)
    assert qpf(torch.rand(2, 3, 8, 8)).shape == (2, 12, 4, 4)


def test_qpf_unknown_pattern_raises():
    with pytest.raises(ValueError):
        QPF(in_channels=1, entanglement="spiral")


# ---------------------------------------------------------------- PQCConv2D

def test_pqc_conv_trainable_by_default():
    layer = PQCConv2D(in_channels=1, out_channels=4)
    n_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert n_trainable == 2 * 4 * 2  # n_layers * n_qubits * 2 angles


def test_pqc_conv_gradients():
    layer = PQCConv2D(in_channels=1, out_channels=4)
    layer(torch.rand(2, 1, 8, 8)).sum().backward()
    assert torch.isfinite(layer.weights.grad).all()


def test_iqp_circuit_runs():
    layer = Quanvolution2D(in_channels=1, out_channels=4, circuit="iqp")
    assert layer(torch.rand(2, 1, 8, 8)).shape == (2, 4, 4, 4)


# ------------------------------------------------------------------ chunking

def test_patch_chunking_consistent():
    x = torch.rand(2, 1, 12, 12)
    full = Quanvolution2D(1, 4, seed=5, max_patch_batch=4096)(x)
    chunked = Quanvolution2D(1, 4, seed=5, max_patch_batch=7)(x)
    assert torch.allclose(full, chunked, atol=1e-6)
