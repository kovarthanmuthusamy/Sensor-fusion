"""Build a service request (photo, raw LiDAR points, calibration) from a nuScenes frame.

Used by the benchmark, the demo and the tests, so they all send the service what a customer would.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image

from sensor_fusion.data.projection import build_lidar_to_camera
from sensor_fusion.serve.inference import Calibration


@dataclass(frozen=True)
class RequestData:
    sample_token: str
    image: np.ndarray  # (H, W, 3) uint8
    lidar_points: np.ndarray  # (N, 3) float32, LiDAR frame, metres
    calibration: Calibration


def request_from_nuscenes(
    nusc: Any, sample_token: str, *, camera: str = "CAM_FRONT", lidar: str = "LIDAR_TOP"
) -> RequestData:
    """The photo, the raw LiDAR scan and the calibration of one keyframe.

    The matrix includes the vehicle's motion between the LiDAR scan and the photo, as the dataset
    code does for training (the two sensors fire at slightly different moments).
    """
    sample = nusc.get("sample", sample_token)
    cam = nusc.get("sample_data", sample["data"][camera])
    lid = nusc.get("sample_data", sample["data"][lidar])
    cam_calib = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
    image = np.array(Image.open(os.path.join(nusc.dataroot, cam["filename"])).convert("RGB"))
    cloud = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lid["filename"]))
    transform = build_lidar_to_camera(
        nusc.get("calibrated_sensor", lid["calibrated_sensor_token"]),
        nusc.get("ego_pose", lid["ego_pose_token"]),
        nusc.get("ego_pose", cam["ego_pose_token"]),
        cam_calib,
    )
    return RequestData(
        sample_token=sample_token,
        image=image,
        lidar_points=cloud.points[:3].T.astype(np.float32),
        calibration=Calibration(
            transform, np.array(cam_calib["camera_intrinsic"], dtype=np.float64)
        ),
    )
