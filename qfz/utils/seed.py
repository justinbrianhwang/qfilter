"""Global seeding for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch RNGs.

    Note that qfz circuit structures and initial angles are seeded
    separately via each layer's ``seed`` argument, so quantum filters
    are reproducible even without calling this. This function fixes the
    remaining sources of randomness (weight init, data shuffling, ...).

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
