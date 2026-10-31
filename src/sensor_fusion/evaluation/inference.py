"""Run a trained model over a loader and turn its output grids into `FrameResult`s (ADR 0012)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sensor_fusion.data.targets import decode_outputs
from sensor_fusion.evaluation.matching import FrameResult
from sensor_fusion.models.detector import Outputs

SceneInfo = Callable[[str], tuple[str, str]]  # sample token -> (scene name, city)


def scene_info_from(nusc: Any) -> SceneInfo:
    """Look up the scene name and the city (Boston / Singapore) of a nuScenes sample token."""
    cache: dict[str, tuple[str, str]] = {}

    def info(token: str) -> tuple[str, str]:
        scene_token = nusc.get("sample", token)["scene_token"]
        if scene_token not in cache:
            scene = nusc.get("scene", scene_token)
            location = nusc.get("log", scene["log_token"])["location"]
            cache[scene_token] = (scene["name"], location.split("-")[0])
        return cache[scene_token]

    return info


def results_from_outputs(
    outputs: Outputs,
    batch: dict[str, Any],
    scene_info: SceneInfo,
    *,
    stride: int,
    top_k: int,
    score_threshold: float,
) -> list[FrameResult]:
    """Decode one batch of output grids into detections and pair them with the batch's labels.

    Detections and labels are both in network-input pixels, so no mapping back to the full-size
    photo is needed for a comparison between models.
    """
    results = []
    for i, token in enumerate(batch["sample_token"]):
        found = decode_outputs(
            torch.sigmoid(outputs.class_logits[i].float()),
            outputs.regression[i].float(),
            stride=stride,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        scene, city = scene_info(token)
        results.append(
            FrameResult(
                token=token,
                scene=scene,
                city=city,
                is_night=bool(batch["is_night"][i]),
                det_boxes=found.boxes.cpu().numpy().astype(np.float32),
                det_scores=found.scores.cpu().numpy().astype(np.float32),
                det_classes=found.classes.cpu().numpy().astype(np.int64),
                det_distances=found.distances.cpu().numpy().astype(np.float32),
                gt_boxes=batch["boxes"][i].numpy().astype(np.float32).reshape(-1, 4),
                gt_classes=batch["classes"][i].numpy().astype(np.int64),
                gt_distances=batch["distances"][i].numpy().astype(np.float32),
            )
        )
    return results


@torch.no_grad()
def collect_results(
    model: nn.Module,
    loader: DataLoader[Any],
    scene_info: SceneInfo,
    *,
    device: torch.device,
    amp: bool,
    stride: int,
    top_k: int,
    score_threshold: float,
) -> list[FrameResult]:
    """Detections for every frame of the loader, in loader order."""
    model.eval()
    results: list[FrameResult] = []
    for batch in loader:
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            outputs = model(batch["input"].to(device, non_blocking=True))
        results += results_from_outputs(
            outputs,
            batch,
            scene_info,
            stride=stride,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    return results
