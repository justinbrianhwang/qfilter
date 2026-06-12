"""Generic, registry-based dataset loading.

Any image classification dataset can be used with qfz, not just the
built-in ones. A dataset is registered as a *builder* function returning
``(train_dataset, test_dataset, DatasetInfo)``, where both datasets yield
``(image_tensor, label)`` pairs with pixel values in ``[0, 1]`` (required
by the quantum encodings).

Built-in datasets: ``mnist``, ``fashionmnist``, ``cifar10``, ``svhn``.

Registering a custom dataset::

    from qfz.datasets import register_dataset, DatasetInfo

    @register_dataset("mydata")
    def _build_mydata(root):
        train, test = ...  # any torch Dataset yielding (CxHxW in [0,1], int)
        return train, test, DatasetInfo("mydata", in_channels=3,
                                        num_classes=5, img_size=(64, 64))
"""

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms


@dataclass
class DatasetInfo:
    """Metadata a model needs to configure itself for a dataset.

    Attributes:
        name: Dataset identifier.
        in_channels: Number of image channels.
        num_classes: Number of target classes.
        img_size: Spatial size as ``(H, W)``.
    """

    name: str
    in_channels: int
    num_classes: int
    img_size: tuple


DATASETS = {}


def default_root() -> str:
    """Default data directory: ``$QFZ_DATA_ROOT`` if set, else ``./data``.

    Set the environment variable to keep all datasets in one place
    across projects instead of re-downloading per repository.
    """
    return os.environ.get("QFZ_DATA_ROOT", "data")

# ToTensor scales pixels to [0, 1], which the quantum encodings require.
# We deliberately skip mean/std normalization so classical and hybrid
# models see identical inputs.
_TO_TENSOR = transforms.ToTensor()


def register_dataset(name: str):
    """Decorator registering a dataset builder under ``name``.

    The builder receives the data root directory and must return
    ``(train_dataset, test_dataset, DatasetInfo)``.
    """

    def decorator(builder):
        DATASETS[name] = builder
        return builder

    return decorator


@register_dataset("mnist")
def _build_mnist(root):
    train = tv_datasets.MNIST(root, train=True, download=True, transform=_TO_TENSOR)
    test = tv_datasets.MNIST(root, train=False, download=True, transform=_TO_TENSOR)
    return train, test, DatasetInfo("mnist", 1, 10, (28, 28))


@register_dataset("fashionmnist")
def _build_fashionmnist(root):
    train = tv_datasets.FashionMNIST(root, train=True, download=True, transform=_TO_TENSOR)
    test = tv_datasets.FashionMNIST(root, train=False, download=True, transform=_TO_TENSOR)
    return train, test, DatasetInfo("fashionmnist", 1, 10, (28, 28))


@register_dataset("cifar10")
def _build_cifar10(root):
    train = tv_datasets.CIFAR10(root, train=True, download=True, transform=_TO_TENSOR)
    test = tv_datasets.CIFAR10(root, train=False, download=True, transform=_TO_TENSOR)
    return train, test, DatasetInfo("cifar10", 3, 10, (32, 32))


@register_dataset("svhn")
def _build_svhn(root):
    svhn_root = f"{root}/svhn"
    train = tv_datasets.SVHN(svhn_root, split="train", download=True, transform=_TO_TENSOR)
    test = tv_datasets.SVHN(svhn_root, split="test", download=True, transform=_TO_TENSOR)
    return train, test, DatasetInfo("svhn", 3, 10, (32, 32))


def get_dataset(name: str, root: str = None, train_size: int = None,
                test_size: int = None, seed: int = 42):
    """Build a registered dataset, optionally subsampled.

    Args:
        name: Registered dataset name (see :data:`DATASETS`).
        root: Directory where data is stored / downloaded. Defaults to
            :func:`default_root` (``$QFZ_DATA_ROOT`` or ``./data``).
        train_size: If given, randomly subsample the training set to
            this many examples (seeded, reproducible).
        test_size: Same for the test set.
        seed: Seed for the subsampling permutation.

    Returns:
        Tuple ``(train_dataset, test_dataset, DatasetInfo)``.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    try:
        builder = DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}. "
            "Use qfz.datasets.register_dataset to add your own."
        ) from None

    train, test, info = builder(root if root is not None else default_root())
    generator = torch.Generator().manual_seed(seed)
    if train_size is not None and train_size < len(train):
        idx = torch.randperm(len(train), generator=generator)[:train_size]
        train = Subset(train, idx.tolist())
    if test_size is not None and test_size < len(test):
        idx = torch.randperm(len(test), generator=generator)[:test_size]
        test = Subset(test, idx.tolist())
    return train, test, info


def get_dataloaders(name: str, root: str = None, batch_size: int = 64,
                    train_size: int = None, test_size: int = None,
                    num_workers: int = 0, seed: int = 42):
    """Build train/test dataloaders for a registered dataset.

    Args:
        name: Registered dataset name.
        root: Data root directory (defaults to :func:`default_root`).
        batch_size: Batch size for both loaders.
        train_size: Optional training subset size.
        test_size: Optional test subset size.
        num_workers: DataLoader workers (0 is safest on Windows).
        seed: Seed for subsampling and shuffling.

    Returns:
        Tuple ``(train_loader, test_loader, DatasetInfo)``.
    """
    train, test, info = get_dataset(name, root, train_size, test_size, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generator)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader, info
