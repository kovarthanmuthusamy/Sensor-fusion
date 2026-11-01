"""Serve an exported model: photo + LiDAR points + calibration in, boxes with distances out.

Everything except the network itself is the training code (projection, input preparation,
decoding), so the served model sees exactly the input it was trained on (ADR 0007).
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from sensor_fusion.data.projection import make_sparse_depth, project_to_image
from sensor_fusion.data.targets import decode_outputs
from sensor_fusion.data.transforms import boxes_to_original, prepare_frame
from sensor_fusion.serve.constants import INPUT_NAME

DEFAULT_TORCH_THREADS = 4


class InvalidInputError(ValueError):
    """The request is malformed (the API turns this into HTTP 422, never a crash)."""


@dataclass(frozen=True)
class Calibration:
    """How LiDAR points map into the photo.

    T_cam_from_lidar: `(4, 4)` homogeneous matrix, `p_cam = T @ p_lidar`, metres. For a moving
        vehicle it must already include the motion between the LiDAR scan and the photo.
    intrinsic: `(3, 3)` camera matrix in pixels of the full-size photo.
    """

    T_cam_from_lidar: np.ndarray
    intrinsic: np.ndarray


@dataclass(frozen=True)
class Detection:
    class_name: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the full-size photo
    distance_m: float  # along the camera's viewing direction


def _check_calibration(calibration: Calibration) -> None:
    t, k = calibration.T_cam_from_lidar, calibration.intrinsic
    if t.shape != (4, 4) or k.shape != (3, 3):
        raise InvalidInputError(
            f"calibration needs a 4x4 and a 3x3 matrix, got {t.shape}, {k.shape}"
        )
    if not (np.isfinite(t).all() and np.isfinite(k).all()):
        raise InvalidInputError("calibration contains NaN or infinity")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
        raise InvalidInputError("T_cam_from_lidar must be a homogeneous matrix (last row 0 0 0 1)")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise InvalidInputError("intrinsic focal lengths must be positive")


class OnnxDetector:
    """A model exported by `serve.export`, ready to answer detection requests."""

    def __init__(
        self,
        model_dir: Path,
        *,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        intra_op_threads: int | None = None,
        torch_threads: int | None = DEFAULT_TORCH_THREADS,
    ) -> None:
        """`torch_threads` sets PyTorch's thread count for the whole process (None leaves it).

        The preparation and decoding steps are many small tensor operations; on a many-core
        machine PyTorch's default (one thread per core) makes them 3x slower and far less
        predictable than a few threads (measured: 300 ms vs 102 ms on 36 cores).
        """
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        self.info: dict[str, Any] = json.loads((model_dir / "model.json").read_text("utf-8"))
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=list(providers)
        )
        self.in_channels: int = self.info["in_channels"]
        self.class_names: list[str] = self.info["class_names"]
        height, width = self.info["image_size"]
        self.image_size: tuple[int, int] = (int(height), int(width))
        self._prep: dict[str, Any] = self.info["prep"]
        self.needs_lidar = self.in_channels >= 4

    def prepare(
        self, image: np.ndarray, lidar_points: np.ndarray | None, calibration: Calibration | None
    ) -> tuple[np.ndarray, int, int]:
        """Validate the request and build the network input. Returns it with the crop offsets."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputError(
                f"image must be (H, W, 3) uint8, got {image.shape} {image.dtype}"
            )
        if image.shape[:2] != self.image_size:
            raise InvalidInputError(
                f"image must be {self.image_size[1]}x{self.image_size[0]} (width x height), "
                f"got {image.shape[1]}x{image.shape[0]}"
            )
        depth = np.zeros(self.image_size, dtype=np.float32)
        if self.needs_lidar:
            if lidar_points is None or calibration is None:
                raise InvalidInputError(
                    "this model uses LiDAR: send lidar points and calibration with the photo"
                )
            if lidar_points.ndim != 2 or lidar_points.shape[1] < 3 or len(lidar_points) == 0:
                raise InvalidInputError(f"lidar points must be (N, >=3), got {lidar_points.shape}")
            if not np.isfinite(lidar_points[:, :3]).all():
                raise InvalidInputError("lidar points contain NaN or infinity")
            _check_calibration(calibration)
            pixels, depths = project_to_image(
                lidar_points[:, :3],
                calibration.T_cam_from_lidar,
                calibration.intrinsic,
                self.image_size,
            )
            depth = make_sparse_depth(pixels, depths, self.image_size)
        frame = prepare_frame(
            image,
            depth,
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            downscale=int(self._prep["downscale"]),
            depth_max_m=float(self._prep["depth_max_m"]),
            in_channels=self.in_channels,
            rgb_mean=self._prep["rgb_mean"],
            rgb_std=self._prep["rgb_std"],
            flip=False,
            depth_thicken_px=int(self._prep.get("depth_thicken_px", 0)),
        )
        return frame.network_input, frame.crop_top, frame.crop_left

    def detect(
        self,
        image: np.ndarray,
        lidar_points: np.ndarray | None = None,
        calibration: Calibration | None = None,
        *,
        score_threshold: float | None = None,
    ) -> tuple[list[Detection], dict[str, float]]:
        """Detections for one frame and the time spent in each stage (milliseconds)."""
        decode = self.info["decode"]
        threshold = decode["score_threshold"] if score_threshold is None else score_threshold
        if not 0.0 <= threshold <= 1.0:
            raise InvalidInputError(f"score_threshold must be in [0, 1], got {threshold}")

        t0 = time.perf_counter()
        network_input, crop_top, crop_left = self.prepare(image, lidar_points, calibration)
        t1 = time.perf_counter()
        class_logits, regression = self.session.run(None, {INPUT_NAME: network_input[None]})
        t2 = time.perf_counter()
        found = decode_outputs(
            torch.sigmoid(torch.from_numpy(class_logits[0])),
            torch.from_numpy(regression[0]),
            stride=int(self._prep["output_stride"]),
            top_k=int(decode["top_k"]),
            score_threshold=float(threshold),
        )
        boxes = boxes_to_original(
            found.boxes.numpy(),
            downscale=int(self._prep["downscale"]),
            crop_top=crop_top,
            crop_left=crop_left,
        )
        detections = [
            Detection(
                class_name=self.class_names[int(c)],
                score=float(s),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                distance_m=float(d),
            )
            for b, s, c, d in zip(
                boxes,
                found.scores.numpy(),
                found.classes.numpy(),
                found.distances.numpy(),
                strict=True,
            )
        ]
        t3 = time.perf_counter()
        timings = {
            "prepare_ms": 1000 * (t1 - t0),
            "network_ms": 1000 * (t2 - t1),
            "decode_ms": 1000 * (t3 - t2),
            "total_ms": 1000 * (t3 - t0),
        }
        return detections, timings
