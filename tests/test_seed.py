import random

import numpy as np
import torch

from sensor_fusion.utils.seed import seed_everything


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
