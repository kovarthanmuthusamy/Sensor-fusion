"""Average precision and distance error for a group of frames ("cell"), with scene weights.

Everything that depends only on a frame is computed once (`build_cell_data`). A metric is then a
cheap function of per-scene weights: weight 1 everywhere gives the plain number, and random
multiplicities give one scene-level bootstrap sample (`bootstrap.py`). Resampling scenes with
multiplicity k is exactly the same as including that scene's detections k times, and the tests
check that identity.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sensor_fusion.evaluation.matching import FrameResult, match_frame

RECALL_POINTS = np.linspace(0.0, 1.0, 101)  # COCO's 101-point interpolation


@dataclass(frozen=True)
class ClassData:
    """All detections of one class in a cell, sorted by score (highest first)."""

    tp: np.ndarray  # (N, T) bool
    scene: np.ndarray  # (N,) int, index of the detection's scene within the cell
    n_gt_by_scene: np.ndarray  # (S,) float


@dataclass(frozen=True)
class DistanceData:
    """Matched detections (IoU >= thresholds[0], score >= the distance threshold), by error."""

    rel_error: np.ndarray  # (K,) |pred - gt| / gt, sorted ascending
    abs_error: np.ndarray  # (K,) metres, same order
    gt_distance: np.ndarray  # (K,) metres, same order
    scene: np.ndarray  # (K,) int


@dataclass(frozen=True)
class CellData:
    scene_names: tuple[str, ...]
    frames_by_scene: np.ndarray  # (S,) int
    classes: tuple[ClassData, ...]
    distance: DistanceData

    @property
    def n_gt(self) -> np.ndarray:
        """Labelled objects per class in the whole cell."""
        return np.array([c.n_gt_by_scene.sum() for c in self.classes])


@dataclass(frozen=True)
class Metrics:
    map50: float
    map: float  # mean of AP over IoU 0.50:0.05:0.95
    ap50: tuple[float, ...]  # per class, nan when the class is not eligible
    ap: tuple[float, ...]
    dist_rel_median: float
    dist_abs_median: float


def build_cell_data(
    frames: Sequence[FrameResult],
    *,
    iou_thresholds: tuple[float, ...],
    num_classes: int,
    distance_score_threshold: float,
) -> CellData:
    """Match every frame once and collect the per-scene pieces the metrics need."""
    scene_names = tuple(sorted({frame.scene for frame in frames}))
    index = {name: i for i, name in enumerate(scene_names)}
    n_scenes, n_thr = len(scene_names), len(iou_thresholds)
    frames_by_scene = np.zeros(n_scenes, dtype=np.int64)
    scores = [[] for _ in range(num_classes)]  # type: list[list[np.ndarray]]
    tps = [[] for _ in range(num_classes)]  # type: list[list[np.ndarray]]
    scenes = [[] for _ in range(num_classes)]  # type: list[list[np.ndarray]]
    n_gt = np.zeros((num_classes, n_scenes))
    rel, ab, gt_d, d_scene = [], [], [], []  # type: list[np.ndarray], list[np.ndarray], list, list

    for frame in frames:
        scene = index[frame.scene]
        frames_by_scene[scene] += 1
        for class_id, m in enumerate(match_frame(frame, iou_thresholds, num_classes)):
            n_gt[class_id, scene] += m.n_gt
            scores[class_id].append(m.scores)
            tps[class_id].append(m.tp)
            scenes[class_id].append(np.full(len(m.scores), scene, dtype=np.int64))
            keep = ~np.isnan(m.matched_gt_distances) & (m.scores >= distance_score_threshold)
            error = np.abs(m.pred_distances[keep] - m.matched_gt_distances[keep])
            rel.append(error / m.matched_gt_distances[keep])
            ab.append(error)
            gt_d.append(m.matched_gt_distances[keep])
            d_scene.append(np.full(int(keep.sum()), scene, dtype=np.int64))

    classes = []
    for class_id in range(num_classes):
        score = np.concatenate(scores[class_id]) if scores[class_id] else np.zeros(0)
        tp = np.concatenate(tps[class_id]) if tps[class_id] else np.zeros((0, n_thr), bool)
        owner = np.concatenate(scenes[class_id]) if scenes[class_id] else np.zeros(0, np.int64)
        order = np.argsort(-score, kind="stable")
        classes.append(ClassData(tp[order], owner[order], n_gt[class_id]))

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(0)

    rel_error, abs_error, gt_distance = cat(rel), cat(ab), cat(gt_d)
    d_idx = cat(d_scene).astype(np.int64)
    order = np.argsort(rel_error, kind="stable")
    distance = DistanceData(rel_error[order], abs_error[order], gt_distance[order], d_idx[order])
    return CellData(scene_names, frames_by_scene, tuple(classes), distance)


def average_precision(tp: np.ndarray, weights: np.ndarray, n_gt: float) -> np.ndarray:
    """COCO-style AP (101-point interpolation) per IoU threshold, with weighted detections.

    `tp` is `(N, T)` bool, rows already sorted by score. `weights` `(N,)` multiplies every
    detection (all ones for the plain number); `n_gt` is the weighted number of labelled objects.
    Returns `(T,)`; nan when there are no labelled objects.
    """
    n_thr = tp.shape[1]
    if n_gt <= 0:
        return np.full(n_thr, np.nan)
    if len(tp) == 0:
        return np.zeros(n_thr)
    w = weights[:, None]
    true_pos = np.cumsum(tp * w, axis=0)
    false_pos = np.cumsum((~tp) * w, axis=0)
    seen = true_pos + false_pos
    precision = np.divide(true_pos, seen, out=np.zeros_like(true_pos), where=seen > 0)
    recall = true_pos / n_gt
    ap = np.zeros(n_thr)
    for t in range(n_thr):
        envelope = np.maximum.accumulate(precision[::-1, t])[::-1]  # best precision from here on
        at = np.searchsorted(recall[:, t], RECALL_POINTS, side="left")
        found = at < len(envelope)
        values = np.zeros(len(RECALL_POINTS))
        values[found] = envelope[at[found]]
        ap[t] = values.mean()
    return ap


def weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    """Median of values (sorted ascending) where each carries a weight; nan if no weight.

    With all weights 1 this equals `numpy.median`: an even count averages the two middle values.
    """
    total = float(weights.sum())
    if len(sorted_values) == 0 or total <= 0:
        return float("nan")
    cumulative = np.cumsum(weights)
    half = total / 2
    at = int(np.searchsorted(cumulative, half, side="left"))
    at = min(at, len(sorted_values) - 1)
    if at + 1 < len(sorted_values) and np.isclose(cumulative[at], half):
        return float(0.5 * (sorted_values[at] + sorted_values[at + 1]))
    return float(sorted_values[at])


def eligible_classes(data: CellData, min_objects: int) -> list[int]:
    """Classes with enough labelled objects in the cell to report an AP (ADR 0011)."""
    return [c for c, n in enumerate(data.n_gt) if n >= min_objects]


def cell_metrics(data: CellData, scene_weights: np.ndarray, eligible: Sequence[int]) -> Metrics:
    """Metrics of a cell under per-scene weights (all ones = the plain numbers)."""
    n_classes = len(data.classes)
    ap50 = np.full(n_classes, np.nan)
    ap = np.full(n_classes, np.nan)
    for c in eligible:
        class_data = data.classes[c]
        per_threshold = average_precision(
            class_data.tp,
            scene_weights[class_data.scene],
            float((class_data.n_gt_by_scene * scene_weights).sum()),
        )
        ap50[c], ap[c] = per_threshold[0], per_threshold.mean()
    d = data.distance
    weights = scene_weights[d.scene]
    rel = weighted_median(d.rel_error, weights)
    # The absolute error is ordered by the relative error, so take it through the same positions.
    ab = _weighted_median_unsorted(d.abs_error, weights)
    return Metrics(
        map50=float(np.nanmean(ap50)) if len(eligible) else float("nan"),
        map=float(np.nanmean(ap)) if len(eligible) else float("nan"),
        ap50=tuple(float(v) for v in ap50),
        ap=tuple(float(v) for v in ap),
        dist_rel_median=rel,
        dist_abs_median=ab,
    )


def _weighted_median_unsorted(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    return weighted_median(values[order], weights[order])


def distance_by_range(data: CellData, edges: Sequence[float]) -> list[dict[str, float]]:
    """Median relative distance error and count of matched detections per range of true distance."""
    d = data.distance
    rows = []
    for low, high in itertools.pairwise(edges):
        inside = (d.gt_distance >= low) & (d.gt_distance < high)
        rel = np.sort(d.rel_error[inside])
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "matched": float(inside.sum()),
                "median_rel_error": weighted_median(rel, np.ones(len(rel))),
            }
        )
    return rows
