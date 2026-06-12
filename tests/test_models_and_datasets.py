"""Tests for models and the dataset registry (no downloads required)."""

import pytest
import torch

from qfz.datasets import DATASETS, DatasetInfo, get_dataset, register_dataset
from qfz.layers import QPF, Quanvolution2D
from qfz.models import ClassicalCNN, HybridCNN


def test_classical_cnn_arbitrary_size():
    model = ClassicalCNN(in_channels=3, num_classes=7, img_size=(40, 56))
    assert model(torch.rand(2, 3, 40, 56)).shape == (2, 7)


def test_hybrid_cnn_with_quanvolution():
    layer = Quanvolution2D(in_channels=1, out_channels=4)
    model = HybridCNN(layer, num_classes=10, in_channels=1, img_size=(28, 28))
    assert model(torch.rand(2, 1, 28, 28)).shape == (2, 10)


def test_hybrid_cnn_with_qpf_rgb():
    layer = QPF(in_channels=3)
    model = HybridCNN(layer, num_classes=5, in_channels=3, img_size=(16, 16))
    assert model(torch.rand(2, 3, 16, 16)).shape == (2, 5)


def test_builtin_datasets_registered():
    assert {"mnist", "fashionmnist", "cifar10", "svhn"} <= set(DATASETS)


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("not_a_dataset")


def test_register_custom_dataset():
    @register_dataset("_toy")
    def _build_toy(root):
        images = torch.rand(20, 1, 8, 8)
        labels = torch.randint(0, 2, (20,))
        ds = torch.utils.data.TensorDataset(images, labels)
        return ds, ds, DatasetInfo("_toy", 1, 2, (8, 8))

    train, test, info = get_dataset("_toy", train_size=10)
    assert len(train) == 10 and info.num_classes == 2
    DATASETS.pop("_toy")
