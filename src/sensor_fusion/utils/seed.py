"""Seed every RNG a run touches, so a rerun with the same config gives the same numbers."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's `random`, NumPy's global RNG and torch (CPU and all CUDA devices).

    Not covered here: DataLoader workers (seeded via the DataLoader's generator, added
    with the training loop) and nondeterministic cuDNN kernels, so GPU runs match
    closely but not bit-for-bit.
    """
    random.seed(seed)
    # Legacy global RNG on purpose: third-party code (augmentations, the devkit) draws from it.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)  # seeds all CUDA devices too
