"""Safeguards that stop a bad run early instead of after hours of GPU time (ADR 0010)."""

from collections.abc import Iterable, Mapping, Sequence

import torch

from sensor_fusion.models.losses import LossTerms


class TrainingDiverged(RuntimeError):
    """A loss term became NaN or infinite."""


def assert_scenes_disjoint(train_scenes: Iterable[str], val_scenes: Iterable[str]) -> None:
    """Fail if any scene is in both splits.

    Keyframes are 0.5 s apart, so a scene shared by train and val puts near-duplicate frames on
    both sides and the validation numbers measure memorisation (ML hygiene: split by scene).
    """
    shared = sorted(set(train_scenes) & set(val_scenes))
    if shared:
        raise ValueError(
            f"{len(shared)} scene(s) are in both the train and the val split, e.g. {shared[:3]}"
        )


def require_pinned_gpu(environ: Mapping[str, str], cuda_available: bool) -> None:
    """On a shared multi-GPU machine every run must choose its GPU on purpose.

    Without `CUDA_VISIBLE_DEVICES`, the run lands on GPU 0 whatever other people are doing there.
    """
    if cuda_available and "CUDA_VISIBLE_DEVICES" not in environ:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is not set: this is a shared machine, so pick a GPU after "
            "checking nvidia-smi (e.g. set CUDA_VISIBLE_DEVICES=0) or pass "
            "trainer.require_pinned_gpu=false"
        )


def assert_finite(terms: LossTerms, sample_tokens: Sequence[str]) -> None:
    """Raise `TrainingDiverged` naming the batch if any loss term is NaN or infinite."""
    values = terms._asdict()
    bad = [name for name, value in values.items() if not bool(torch.isfinite(value))]
    if bad:
        shown = {name: float(values[name].detach()) for name in values}
        raise TrainingDiverged(
            f"non-finite loss terms {bad}: {shown}; batch samples {list(sample_tokens)[:4]}..."
        )
