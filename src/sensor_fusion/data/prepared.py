"""The assembly line: raw dataset frame -> input + painted target grids, ready for batching.

Chains `NuScenesFrontDataset`, `transforms.prepare_frame` and `targets.encode_targets` (ADR 0006).
Mirroring is decided here, only in training, from torch's random generator, so the trainer's seed
(and DataLoader's per-worker seeding) controls it; evaluation is always deterministic.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.labels import CLASS_NAMES
from sensor_fusion.data.targets import encode_targets
from sensor_fusion.data.transforms import prepare_frame


class PreparedDataset(Dataset[dict[str, Any]]):
    """Item keys:
    input: `(C, H, W)` float32, C = in_channels (3 = RGB, 4 = RGB + depth).
    heatmap: `(num_classes, H/s, W/s)`, regression: `(5, H/s, W/s)`, mask: `(H/s, W/s)` float32.
    boxes `(K, 4)`, classes `(K,)`, distances `(K,)`: the labels in network-input pixels, kept for
    decoding checks and pictures. is_night: bool. sample_token: str.
    crop_top, crop_left: ints, to map boxes back to the full-size photo.
    """

    def __init__(
        self,
        base: NuScenesFrontDataset,
        *,
        in_channels: int,
        prep: Mapping[str, Any],
        train: bool,
    ) -> None:
        """Args:
        base: the raw dataset. in_channels: 3 or 4. train: mirror randomly when True.
        prep: the `prep` config group; its values are copied out, so no config object is pickled
            into DataLoader workers.
        """
        self.base = base
        self.in_channels = in_channels
        self.train = train
        self._downscale = int(prep["downscale"])
        self._depth_max_m = float(prep["depth_max_m"])
        self._flip_prob = float(prep["flip_prob"])
        self._rgb_mean = tuple(float(v) for v in prep["rgb_mean"])
        self._rgb_std = tuple(float(v) for v in prep["rgb_std"])
        self._stride = int(prep["output_stride"])
        self._peak_alpha = float(prep["peak_alpha"])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        flip = self.train and bool(torch.rand(()) < self._flip_prob)
        frame = prepare_frame(
            item["image"],
            item["depth"],
            item["boxes"],
            item["classes"],
            item["distances"],
            downscale=self._downscale,
            depth_max_m=self._depth_max_m,
            in_channels=self.in_channels,
            rgb_mean=self._rgb_mean,
            rgb_std=self._rgb_std,
            flip=flip,
        )
        heatmap, regression, mask = encode_targets(
            frame.boxes,
            frame.classes,
            frame.distances,
            input_size=(frame.network_input.shape[1], frame.network_input.shape[2]),
            num_classes=len(CLASS_NAMES),
            stride=self._stride,
            peak_alpha=self._peak_alpha,
        )
        return {
            "input": torch.from_numpy(frame.network_input),
            "heatmap": torch.from_numpy(heatmap),
            "regression": torch.from_numpy(regression),
            "mask": torch.from_numpy(mask),
            "boxes": torch.from_numpy(frame.boxes),
            "classes": torch.from_numpy(frame.classes),
            "distances": torch.from_numpy(frame.distances),
            "is_night": item["is_night"],
            "sample_token": item["sample_token"],
            "crop_top": frame.crop_top,
            "crop_left": frame.crop_left,
        }


def collate_prepared(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the fixed-size grids; keep the per-frame label lists (lengths differ) as lists."""
    out: dict[str, Any] = {
        key: torch.stack([sample[key] for sample in batch])
        for key in ("input", "heatmap", "regression", "mask")
    }
    out["is_night"] = torch.tensor([sample["is_night"] for sample in batch])
    for key in ("boxes", "classes", "distances", "sample_token", "crop_top", "crop_left"):
        out[key] = [sample[key] for sample in batch]
    return out
