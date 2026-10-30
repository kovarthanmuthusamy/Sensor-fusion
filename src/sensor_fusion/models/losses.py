"""Training loss for the detector: focal loss on the class grid + L1 on the regression grid.

The targets come from `data/targets.py`. Both terms are normalised by the number of objects in the
batch (floored at 1), so a batch with few objects is not under-weighted and an empty one gives a
finite loss. Everything is computed in float32 even under fp16 autocast: the log and power terms
overflow or lose precision in half precision (ADR 0009).
"""

from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from sensor_fusion.data.targets import REGRESSION_CHANNELS
from sensor_fusion.models.detector import Outputs

WEIGHT_KEYS = ("class", "size", "offset", "distance")
_SIZE = [REGRESSION_CHANNELS.index(name) for name in ("log_width", "log_height")]
_OFFSET = [REGRESSION_CHANNELS.index(name) for name in ("offset_x", "offset_y")]
_DISTANCE = [REGRESSION_CHANNELS.index("log_distance")]


class LossTerms(NamedTuple):
    total: Tensor
    class_: Tensor
    size: Tensor
    offset: Tensor
    distance: Tensor


def heatmap_focal_loss(logits: Tensor, target: Tensor, *, alpha: float, beta: float) -> Tensor:
    """Penalty-reduced focal loss (CenterNet / CornerNet) on the class grids.

    A cell is a positive where the target is exactly 1 (an object centre). Every other cell is a
    negative, but one just beside a centre has its penalty shrunk by `(1 - target)^beta`, because
    the network is not wrong to be a little confident there. `alpha` down-weights cells that are
    already predicted well, so the many easy empty cells do not dominate.
    """
    logits = logits.float()
    target = target.float()
    positive = target.eq(1).float()
    # log(sigmoid(x)) and log(1 - sigmoid(x)) written with logsigmoid so extreme logits stay finite.
    positive_loss = F.logsigmoid(logits) * torch.sigmoid(-logits).pow(alpha) * positive
    negative_loss = (
        F.logsigmoid(-logits)
        * torch.sigmoid(logits).pow(alpha)
        * (1 - target).pow(beta)
        * (1 - positive)
    )
    num_objects = positive.sum().clamp(min=1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / num_objects


def masked_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean absolute error per regression channel, over object-centre cells only.

    prediction, target: `(B, 5, h, w)`. mask: `(B, h, w)`, 1 at object centres. Returns `(5,)`.
    """
    error = (prediction.float() - target.float()).abs() * mask.float()[:, None]
    num_objects = mask.float().sum().clamp(min=1.0)
    return error.sum(dim=(0, 2, 3)) / num_objects


def detection_loss(
    outputs: Outputs,
    heatmap: Tensor,
    regression: Tensor,
    mask: Tensor,
    *,
    focal_alpha: float,
    focal_beta: float,
    weights: Mapping[str, float],
) -> LossTerms:
    """Weighted sum of the class loss and the three regression groups (size, offset, distance).

    Size and offset are each the mean of their two channels, so the weights are comparable.
    """
    if set(weights) != set(WEIGHT_KEYS):
        raise ValueError(f"weights must have exactly the keys {WEIGHT_KEYS}, got {sorted(weights)}")
    class_loss = heatmap_focal_loss(
        outputs.class_logits, heatmap, alpha=focal_alpha, beta=focal_beta
    )
    per_channel = masked_l1(outputs.regression, regression, mask)
    size = per_channel[_SIZE].mean()
    offset = per_channel[_OFFSET].mean()
    distance = per_channel[_DISTANCE].mean()
    total = (
        weights["class"] * class_loss
        + weights["size"] * size
        + weights["offset"] * offset
        + weights["distance"] * distance
    )
    return LossTerms(total, class_loss, size, offset, distance)
