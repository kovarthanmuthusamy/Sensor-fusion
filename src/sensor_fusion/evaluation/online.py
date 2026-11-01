"""A cheap mAP score that training computes after every epoch (ADR 0013).

The validation *loss* turned out to be the wrong yardstick for choosing a checkpoint: its class term
grows with overfitting (confident misses cost a lot) while detection quality does not fall. This
module scores the model with the metric that matters instead: AP at IoU 0.5 per class, mAP, and the
distance error, for all frames and for day and night separately. It uses a single IoU threshold, so
it is much cheaper than the final evaluation (`evaluate.py`), which keeps all ten and the bootstrap.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sensor_fusion.data.labels import CLASS_NAMES
from sensor_fusion.evaluation.inference import SceneInfo, collect_results
from sensor_fusion.evaluation.metrics import build_cell_data, cell_metrics, eligible_classes
from sensor_fusion.evaluation.report import CELLS

SCORE_CELLS = ("all", "day", "night")


def make_map_scorer(
    model: nn.Module,
    loader: DataLoader[Any],
    scene_info: SceneInfo,
    *,
    device: torch.device,
    amp: bool,
    stride: int,
    top_k: int,
    score_threshold: float,
    distance_score_threshold: float,
    min_objects: int,
) -> Callable[[], dict[str, float]]:
    """A function that scores the model's current weights on the loader's frames.

    Returns `{"<cell>/map50": ..., "<cell>/dist_rel_median": ..., "<cell>/ap50_<class>": ...}` for
    the cells all, day and night. A class gets an AP in a cell only with at least `min_objects`
    labelled objects there (ADR 0011); otherwise the value is nan.
    """

    def score() -> dict[str, float]:
        frames = collect_results(
            model,
            loader,
            scene_info,
            device=device,
            amp=amp,
            stride=stride,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        result: dict[str, float] = {}
        for cell in SCORE_CELLS:
            members = [f for f in frames if CELLS[cell](f)]
            if not members:
                continue
            data = build_cell_data(
                members,
                iou_thresholds=(0.5,),
                num_classes=len(CLASS_NAMES),
                distance_score_threshold=distance_score_threshold,
            )
            eligible = eligible_classes(data, min_objects)
            metrics = cell_metrics(data, np.ones(len(data.scene_names)), eligible)
            result[f"{cell}/map50"] = metrics.map50
            result[f"{cell}/dist_rel_median"] = metrics.dist_rel_median
            for name, ap50 in zip(CLASS_NAMES, metrics.ap50, strict=True):
                result[f"{cell}/ap50_{name}"] = ap50
        return result

    return score
