"""Matching detections to ground-truth boxes, one frame at a time (ADR 0012).

Matching follows the COCO rule: detections are visited from the highest score down; each takes the
not-yet-used ground-truth box of its own class with the highest IoU, if that IoU reaches the
threshold (then it is a true positive), otherwise it is a false positive. It is done per frame and
per class, which is why the results can be precomputed once and re-weighted by the bootstrap.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameResult:
    """Detections and labels of one frame, boxes `[x1, y1, x2, y2]` in network-input pixels."""

    token: str
    scene: str
    city: str
    is_night: bool
    det_boxes: np.ndarray  # (N, 4) float32
    det_scores: np.ndarray  # (N,) float32
    det_classes: np.ndarray  # (N,) int64
    det_distances: np.ndarray  # (N,) float32, metres
    gt_boxes: np.ndarray  # (M, 4) float32
    gt_classes: np.ndarray  # (M,) int64
    gt_distances: np.ndarray  # (M,) float32, metres


@dataclass(frozen=True)
class ClassMatches:
    """One class in one frame, detections sorted by score (highest first)."""

    scores: np.ndarray  # (n,)
    tp: np.ndarray  # (n, T) bool: true positive at each IoU threshold
    n_gt: int
    pred_distances: np.ndarray  # (n,)
    matched_gt_distances: np.ndarray  # (n,) distance of the matched box at thresholds[0], else nan


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over union of `(P, 4)` and `(Q, 4)` boxes: a `(P, Q)` matrix."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a, b = a.astype(np.float64), b.astype(np.float64)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def match_frame(
    frame: FrameResult, iou_thresholds: tuple[float, ...], num_classes: int
) -> list[ClassMatches]:
    """Match one frame's detections to its labels, per class.

    `iou_thresholds[0]` is the threshold whose matches are used for the distance error.
    """
    if not iou_thresholds:
        raise ValueError("need at least one IoU threshold")
    results = []
    for class_id in range(num_classes):
        det = np.flatnonzero(frame.det_classes == class_id)
        det = det[np.argsort(-frame.det_scores[det], kind="stable")]
        gt = np.flatnonzero(frame.gt_classes == class_id)
        ious = box_iou(frame.det_boxes[det], frame.gt_boxes[gt])
        tp = np.zeros((len(det), len(iou_thresholds)), dtype=bool)
        matched_distance = np.full(len(det), np.nan, dtype=np.float64)
        for t, threshold in enumerate(iou_thresholds):
            taken = np.zeros(len(gt), dtype=bool)
            for i in range(len(det)):
                if len(gt) == 0:
                    break
                available = np.where(taken, -1.0, ious[i])
                best = int(available.argmax())
                if available[best] >= threshold:
                    tp[i, t] = True
                    taken[best] = True
                    if t == 0:
                        matched_distance[i] = frame.gt_distances[gt[best]]
        results.append(
            ClassMatches(
                scores=frame.det_scores[det].astype(np.float64),
                tp=tp,
                n_gt=len(gt),
                pred_distances=frame.det_distances[det].astype(np.float64),
                matched_gt_distances=matched_distance,
            )
        )
    return results
