"""Input preparation: a raw dataset frame -> what the network consumes (ADR 0006).

Steps, in order: shrink photo and depth by an integer factor, crop to a multiple of 32, normalise,
choose the channels (3 = RGB, 4 = RGB + depth), and optionally mirror. The boxes follow every
geometric step. These are pure functions: no randomness here, the caller decides about mirroring.

Shapes: images are `(H, W, 3)` uint8, depth is `(H, W)` float32 metres (0 = no LiDAR return),
boxes are `(K, 4)` float32 `[x1, y1, x2, y2]` pixels, the network input is `(C, H, W)` float32.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

# The network halves the size five times, so its input size must be a multiple of 32.
SIZE_MULTIPLE = 32
# A box thinner than this (pixels, network input) after shrinking and cropping is dropped.
MIN_BOX_SIDE = 1.0


@dataclass(frozen=True)
class PreparedFrame:
    network_input: np.ndarray  # (C, H, W) float32
    boxes: np.ndarray  # (K, 4) float32 [x1, y1, x2, y2] in network-input pixels
    classes: np.ndarray  # (K,) int64
    distances: np.ndarray  # (K,) float32 metres (unchanged by shrinking, cropping or mirroring)
    crop_top: int  # rows removed at the top of the shrunk frame
    crop_left: int  # columns removed on the left of the shrunk frame
    flipped: bool


def shrink_depth(depth: np.ndarray, factor: int) -> np.ndarray:
    """Shrink a sparse depth image by an integer factor, keeping the nearest real depth.

    Each `factor` x `factor` block becomes its smallest positive depth, or 0 if the block has no
    LiDAR return. Never an average: averaging across a car's edge would invent depths.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    if factor < 1 or depth.shape[0] % factor or depth.shape[1] % factor:
        raise ValueError(f"factor {factor} must divide the depth size {depth.shape}")
    height, width = depth.shape
    blocks = depth.reshape(height // factor, factor, width // factor, factor)
    nearest = np.where(blocks > 0, blocks, np.inf).min(axis=(1, 3))
    nearest[np.isinf(nearest)] = 0.0
    return nearest.astype(np.float32)


def prepare_frame(
    image: np.ndarray,
    depth: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    distances: np.ndarray,
    *,
    downscale: int,
    depth_max_m: float,
    in_channels: int,
    rgb_mean: Sequence[float],
    rgb_std: Sequence[float],
    flip: bool,
) -> PreparedFrame:
    """Turn one dataset frame into a network input plus matching boxes.

    Args:
        image: `(H, W, 3)` uint8 RGB. depth: `(H, W)` float32 metres. boxes: `(K, 4)` pixels of
            the full-size photo. classes: `(K,)`. distances: `(K,)` metres.
        downscale: integer shrink factor (must divide H and W).
        depth_max_m: the depth channel is `depth / depth_max_m`, clipped to [0, 1].
        in_channels: 3 = RGB only (baseline), 4 = RGB + depth (fusion).
        rgb_mean, rgb_std: per-channel normalisation of the photo (ImageNet statistics).
        flip: mirror left-right (photo, depth and boxes together).
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be (H, W, 3) uint8, got {image.shape} {image.dtype}")
    if depth.shape != image.shape[:2]:
        raise ValueError(f"depth {depth.shape} must match the image {image.shape[:2]}")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (K, 4), got {boxes.shape}")
    if not len(boxes) == len(classes) == len(distances):
        raise ValueError("boxes, classes and distances must have the same length")
    if in_channels not in (3, 4):
        raise ValueError(f"in_channels must be 3 (RGB) or 4 (RGB + depth), got {in_channels}")
    if depth_max_m <= 0:
        raise ValueError(f"depth_max_m must be > 0, got {depth_max_m}")
    if len(rgb_mean) != 3 or len(rgb_std) != 3:
        raise ValueError("rgb_mean and rgb_std need 3 numbers each")

    # 1. Shrink. The photo is block-averaged; the depth keeps the nearest real value.
    photo = np.asarray(Image.fromarray(image).reduce(downscale), dtype=np.float32) / 255.0
    depth_small = shrink_depth(depth, downscale)

    # 2. Crop to a multiple of 32, equally on both sides.
    height, width = depth_small.shape
    crop_h = height // SIZE_MULTIPLE * SIZE_MULTIPLE
    crop_w = width // SIZE_MULTIPLE * SIZE_MULTIPLE
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    photo = photo[top : top + crop_h, left : left + crop_w]
    depth_small = depth_small[top : top + crop_h, left : left + crop_w]

    # 3. Normalise, 4. choose the channels.
    rgb = (photo - np.asarray(rgb_mean, dtype=np.float32)) / np.asarray(rgb_std, dtype=np.float32)
    channels = [rgb.transpose(2, 0, 1)]
    if in_channels == 4:
        channels.append(np.clip(depth_small / depth_max_m, 0.0, 1.0)[None])
    network_input = np.concatenate(channels, axis=0).astype(np.float32)

    # The boxes follow the same shrink and crop, then are clipped to what is left.
    new_boxes = boxes.astype(np.float32) / downscale
    new_boxes[:, [0, 2]] -= left
    new_boxes[:, [1, 3]] -= top
    new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, crop_w)
    new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, crop_h)
    keep = ((new_boxes[:, 2] - new_boxes[:, 0]) >= MIN_BOX_SIDE) & (
        (new_boxes[:, 3] - new_boxes[:, 1]) >= MIN_BOX_SIDE
    )
    new_boxes, new_classes = new_boxes[keep], classes[keep].astype(np.int64)
    new_distances = distances[keep].astype(np.float32)

    # 5. Mirror: the photo, depth and boxes flip together; distances do not change.
    if flip:
        network_input = np.ascontiguousarray(network_input[:, :, ::-1])
        new_boxes = new_boxes[:, [2, 1, 0, 3]].copy()
        new_boxes[:, [0, 2]] = crop_w - new_boxes[:, [0, 2]]

    return PreparedFrame(
        network_input=network_input,
        boxes=new_boxes,
        classes=new_classes,
        distances=new_distances,
        crop_top=top,
        crop_left=left,
        flipped=flip,
    )


def boxes_to_original(
    boxes: np.ndarray, *, downscale: int, crop_top: int, crop_left: int
) -> np.ndarray:
    """Map boxes from network-input pixels back to the full-size photo (undoes crop and shrink).

    For frames that were not mirrored: predictions at evaluation time never are.
    """
    original = boxes.astype(np.float32).copy()
    original[:, [0, 2]] += crop_left
    original[:, [1, 3]] += crop_top
    return original * downscale
