"""Seed every RNG a run touches, so a rerun with the same config gives the same numbers."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's `random`, NumPy's global RNG and torch (CPU and all CUDA devices).

    Not covered here: DataLoader workers (`make_generator` below) and
    nondeterministic cuDNN kernels, so GPU runs match closely but not bit-for-bit.
    """
    random.seed(seed)
    # Legacy global RNG on purpose: third-party code (augmentations, the devkit) draws from it.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)  # seeds all CUDA devices too


def make_generator(seed: int) -> torch.Generator:
    """A generator for `DataLoader(generator=...)`: it fixes the shuffling order and the base seed
    from which PyTorch derives each worker's seed. PyTorch then seeds Python's `random`, torch and
    NumPy in every worker itself (checked in `torch/utils/data/_utils/worker.py`, torch 2.14)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
