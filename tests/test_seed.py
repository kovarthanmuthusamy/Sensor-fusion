import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sensor_fusion.utils.seed import make_generator, seed_everything


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), torch.rand(1).item()  # noqa: NPY002


def test_same_seed_gives_same_draws() -> None:
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seeds_give_different_draws() -> None:
    seed_everything(1)
    first = _draw()
    seed_everything(2)
    assert _draw() != first


class _Draws(Dataset[Any]):
    """Each item reports what Python's and NumPy's RNGs produce inside a worker process."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[float, float]:
        return random.random(), float(np.random.rand())  # noqa: NPY002


def _epoch(seed: int) -> list[tuple[float, float]]:
    loader = DataLoader(_Draws(), batch_size=1, num_workers=2, generator=make_generator(seed))
    return [(float(a), float(b)) for a, b in loader]


def test_worker_draws_repeat_for_the_same_generator_seed_and_differ_otherwise() -> None:
    """Python's `random` and NumPy inside DataLoader workers follow the generator's seed."""
    first = _epoch(5)
    assert _epoch(5) == first
    assert _epoch(6) != first
    assert len({python for python, _ in first}) == len(first)  # workers do not repeat each other
