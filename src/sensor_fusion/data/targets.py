"""Target grids <-> boxes: the contract between the labels and the network's output (ADR 0006).

The network answers with grids at 1/stride of the input size:
    class grid       `(num_classes, h, w)`: likelihood that an object of that class is centred here.
    regression grid  `(5, h, w)`: read only at object centres, channels in `REGRESSION_CHANNELS`.
The encoder paints labels onto such grids; the decoder reads detections off them. They are exact
inverses of each other, so encode -> decode can be tested without a trained network.

Units: box sizes are learned as logs of the size in grid cells, the centre offset as a fraction of
a cell, the distance as the log of metres (relative error, ADR 0006). Decoded boxes are in input
pixels and decoded distances in metres.
"""

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

REGRESSION_CHANNELS = ("log_width", "log_height", "offset_x", "offset_y", "log_distance")

# Even a tiny object keeps a visible peak: the Gaussian spread never goes below this (grid cells).
_MIN_SIGMA = 0.4
# Network outputs are clamped before exp(): e^10 is far beyond any real size or distance, finite.
_LOG_CLAMP = 10.0


class Detections(NamedTuple):
    boxes: torch.Tensor  # (N, 4) float32 [x1, y1, x2, y2] in input pixels
    scores: torch.Tensor  # (N,) float32
    classes: torch.Tensor  # (N,) int64
    distances: torch.Tensor  # (N,) float32 metres


def _draw_peak(
    heat: np.ndarray, centre_x: int, centre_y: int, sigma_x: float, sigma_y: float
) -> None:
    """Paint an elliptical Gaussian (1 at the centre cell) into `heat`, keeping the maximum."""
    grid_h, grid_w = heat.shape
    radius_x, radius_y = max(1, math.ceil(3 * sigma_x)), max(1, math.ceil(3 * sigma_y))
    xs = np.arange(-radius_x, radius_x + 1)[None, :]
    ys = np.arange(-radius_y, radius_y + 1)[:, None]
    window = np.exp(-(xs**2) / (2 * sigma_x**2) - (ys**2) / (2 * sigma_y**2))
    left, right = min(centre_x, radius_x), min(grid_w - centre_x, radius_x + 1)
    top, bottom = min(centre_y, radius_y), min(grid_h - centre_y, radius_y + 1)
    region = heat[centre_y - top : centre_y + bottom, centre_x - left : centre_x + right]
    np.maximum(
        region,
        window[radius_y - top : radius_y + bottom, radius_x - left : radius_x + right],
        out=region,
    )


def encode_targets(
    boxes: np.ndarray,
    classes: np.ndarray,
    distances: np.ndarray,
    *,
    input_size: tuple[int, int],
    num_classes: int,
    stride: int,
    peak_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paint boxes, classes and distances onto the network's output grids.

    Args:
        boxes: `(K, 4)` float `[x1, y1, x2, y2]` in input pixels, with positive width and height.
        classes: `(K,)` ids in `[0, num_classes)`. distances: `(K,)` metres, all > 0.
        input_size: `(height, width)` of the network input; both must be multiples of `stride`.
        stride: input pixels per grid cell.
        peak_alpha: peak width per axis is `peak_alpha * size_in_cells / 6` (the TTFNet rule), so
            wide boxes get wide peaks and tall boxes get tall peaks.

    Returns:
        `heatmap` `(num_classes, h, w)` float32 in [0, 1]; `regression` `(5, h, w)` float32, valid
        only where `mask` is 1; `mask` `(h, w)` float32, 1 at each object's centre cell. If two
        objects share a centre cell, the nearer one wins.
    """
    height, width = input_size
    if height % stride or width % stride:
        raise ValueError(f"input size {input_size} must be a multiple of the stride {stride}")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (K, 4), got {boxes.shape}")
    if not len(boxes) == len(classes) == len(distances):
        raise ValueError("boxes, classes and distances must have the same length")
    if len(classes) and not ((classes >= 0) & (classes < num_classes)).all():
        raise ValueError(f"class ids must be in [0, {num_classes})")
    if (distances <= 0).any():
        raise ValueError("distances must be > 0 (they are learned as logs)")
    if ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).any():
        raise ValueError("boxes need positive width and height")

    grid_h, grid_w = height // stride, width // stride
    heatmap = np.zeros((num_classes, grid_h, grid_w), dtype=np.float32)
    regression = np.zeros((len(REGRESSION_CHANNELS), grid_h, grid_w), dtype=np.float32)
    mask = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Far objects first, so a nearer object written later wins a shared centre cell.
    for k in np.argsort(-distances, kind="stable"):
        x1, y1, x2, y2 = (float(v) for v in boxes[k])
        w_cells, h_cells = (x2 - x1) / stride, (y2 - y1) / stride
        centre_x, centre_y = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        cell_x, cell_y = min(int(centre_x), grid_w - 1), min(int(centre_y), grid_h - 1)
        _draw_peak(
            heatmap[classes[k]],
            cell_x,
            cell_y,
            max(peak_alpha * w_cells / 6, _MIN_SIGMA),
            max(peak_alpha * h_cells / 6, _MIN_SIGMA),
        )
        regression[:, cell_y, cell_x] = (
            math.log(w_cells),
            math.log(h_cells),
            centre_x - cell_x,
            centre_y - cell_y,
            math.log(float(distances[k])),
        )
        mask[cell_y, cell_x] = 1.0
    return heatmap, regression, mask


def decode_outputs(
    class_probs: torch.Tensor,
    regression: torch.Tensor,
    *,
    stride: int,
    top_k: int,
    score_threshold: float,
) -> Detections:
    """Read detections off one image's output grids: the inverse of `encode_targets`.

    Args:
        class_probs: `(num_classes, h, w)` probabilities in [0, 1] (apply the sigmoid first).
        regression: `(5, h, w)` raw regression outputs, channels as in `REGRESSION_CHANNELS`.
        stride: input pixels per grid cell.
        top_k: keep at most this many detections (the strongest).
        score_threshold: drop detections scoring below this.

    A detection is a cell that is the maximum of its 3x3 neighbourhood in its class grid, so no
    non-maximum suppression is needed. Boxes come out in input pixels, distances in metres.
    """
    if class_probs.ndim != 3 or regression.shape != (
        len(REGRESSION_CHANNELS),
        *class_probs.shape[1:],
    ):
        raise ValueError(
            f"class_probs must be (C, h, w) and regression (5, h, w), "
            f"got {tuple(class_probs.shape)} and {tuple(regression.shape)}"
        )
    _, grid_h, grid_w = class_probs.shape
    probs = class_probs.float()
    is_peak = F.max_pool2d(probs[None], kernel_size=3, stride=1, padding=1)[0] == probs
    peaks = torch.where(is_peak, probs, torch.zeros_like(probs))

    scores, flat = peaks.reshape(-1).topk(min(top_k, peaks.numel()))
    keep = scores >= score_threshold
    scores, flat = scores[keep], flat[keep]
    class_ids = flat // (grid_h * grid_w)
    cell = flat % (grid_h * grid_w)
    cell_y, cell_x = cell // grid_w, cell % grid_w

    values = regression.float()[:, cell_y, cell_x].clamp(-_LOG_CLAMP, _LOG_CLAMP)
    half_w = torch.exp(values[0]) * stride / 2
    half_h = torch.exp(values[1]) * stride / 2
    centre_x = (cell_x + values[2]) * stride
    centre_y = (cell_y + values[3]) * stride
    boxes = torch.stack(
        [centre_x - half_w, centre_y - half_h, centre_x + half_w, centre_y + half_h], dim=1
    )
    return Detections(boxes, scores, class_ids, torch.exp(values[4]))
