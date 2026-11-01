"""Optimiser and learning-rate schedule (ADR 0010).

Two learning rates: the pretrained backbone is nudged gently, the freshly initialised neck and
heads learn faster. Biases and normalisation parameters get no weight decay.
"""

import math

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

_BACKBONE_PREFIXES = ("conv1.", "stem.", "stages.")


def build_optimizer(
    model: nn.Module, *, lr_head: float, lr_backbone: float, weight_decay: float
) -> torch.optim.AdamW:
    """AdamW with four parameter groups: {backbone, head} x {decayed, not decayed}.

    The first convolution (including the added depth channel) belongs to the backbone group.
    """
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (backbone, decay): [] for backbone in (True, False) for decay in (True, False)
    }
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            buckets[(name.startswith(_BACKBONE_PREFIXES), parameter.ndim > 1)].append(parameter)
    groups = [
        {
            "params": parameters,
            "lr": lr_backbone if backbone else lr_head,
            "weight_decay": weight_decay if decay else 0.0,
        }
        for (backbone, decay), parameters in buckets.items()
        if parameters
    ]
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_fraction: float,
) -> LambdaLR:
    """Linear warm-up, then cosine decay down to `min_lr_fraction` of each group's base rate.

    Warm-up matters here: the heads start at a prior, AdamW's first steps are large relative to
    the weights, and the fp16 gradient scaler is still finding its scale.
    """
    if total_steps < 1 or warmup_steps < 0 or not 0 <= min_lr_fraction <= 1:
        raise ValueError("need total_steps >= 1, warmup_steps >= 0 and min_lr_fraction in [0, 1]")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_fraction + (1 - min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)
