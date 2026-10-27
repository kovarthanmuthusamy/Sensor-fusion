"""Front-camera frames with an aligned sparse depth image and 2D labels (ADR 0004, ADR 0005).

The dataset returns raw arrays. Resizing, scaling and stacking the 4-channel input are a
separate, config-driven stage, so the RGB baseline and the fusion model differ only there.
"""

import os
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from torch.utils.data import Dataset

from sensor_fusion.data.labels import frame_labels
from sensor_fusion.data.projection import (
    build_lidar_to_camera,
    make_sparse_depth,
    project_to_image,
)
from sensor_fusion.data.splits import get_split_scenes, scene_is_night


class NuScenesFrontDataset(Dataset[dict[str, Any]]):
    """Keyframes of the scenes in a split, one front-camera frame per item.

    Item keys:
        image: `(H, W, 3)` uint8 RGB.
        depth: `(H, W)` float32, LiDAR depth in metres (camera z), `0` = no return.
        boxes: `(K, 4)` float32 `[x1, y1, x2, y2]` pixels.
        classes: `(K,)` int64, indices into `labels.CLASS_NAMES`.
        distances: `(K,)` float32, depth of each box centre in metres.
        is_night: bool, whether the frame's scene is a night scene.
        sample_token: str, the nuScenes sample token.
    """

    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        camera: str = "CAM_FRONT",
        lidar: str = "LIDAR_TOP",
    ) -> None:
        """Args:
        dataroot: folder holding `samples/` and the `<version>/` tables.
        version: e.g. `v1.0-mini`.
        split: a name known to `splits.get_split_scenes`. Every scene of the split must exist
            in `version`; mixing e.g. the trainval `train` split with the mini data is an error.
        camera, lidar: sensor channel names.
        """
        self.dataroot = dataroot
        self.version = version
        self.camera = camera
        self.lidar = lidar

        nusc = NuScenes(version, dataroot=dataroot, verbose=False)
        wanted = set(get_split_scenes(split))
        scenes = [scene for scene in nusc.scene if scene["name"] in wanted]
        missing = wanted - {scene["name"] for scene in scenes}
        if missing:
            raise ValueError(
                f"split {split!r} has {len(missing)} scene(s) that are not in {version}, "
                f"e.g. {sorted(missing)[:3]}: split and data version do not belong together"
            )

        # (sample token, is_night) in scene order, then time order: deterministic.
        self._entries: list[tuple[str, bool]] = []
        for scene in scenes:
            night = scene_is_night(scene)
            token = scene["first_sample_token"]
            while token:
                self._entries.append((token, night))
                token = nusc.get("sample", token)["next"]
        self._nusc: NuScenes | None = nusc

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers get a pickled copy. The devkit tables are large (hundreds of MB for
        # trainval), so leave them out and let each worker reload them on first use.
        state = self.__dict__.copy()
        state["_nusc"] = None
        return state

    @property
    def nusc(self) -> NuScenes:
        if self._nusc is None:
            self._nusc = NuScenes(self.version, dataroot=self.dataroot, verbose=False)
        return self._nusc

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_token, is_night = self._entries[index]
        nusc = self.nusc
        sample = nusc.get("sample", sample_token)
        cam = nusc.get("sample_data", sample["data"][self.camera])
        lidar = nusc.get("sample_data", sample["data"][self.lidar])
        cam_calib = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])

        image = np.array(Image.open(os.path.join(nusc.dataroot, cam["filename"])).convert("RGB"))
        image_size = (image.shape[0], image.shape[1])
        if image_size != (cam["height"], cam["width"]):
            raise ValueError(
                f"image is {image_size} but sample_data says {cam['height']}x{cam['width']}"
            )

        points = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar["filename"]))
        T_cam_from_lidar = build_lidar_to_camera(
            nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"]),
            nusc.get("ego_pose", lidar["ego_pose_token"]),
            nusc.get("ego_pose", cam["ego_pose_token"]),
            cam_calib,
        )
        pixels, depths = project_to_image(
            points.points[:3].T,
            T_cam_from_lidar,
            np.array(cam_calib["camera_intrinsic"]),
            image_size,
        )
        boxes, classes, distances = frame_labels(nusc, cam["token"])
        return {
            "image": image,
            "depth": make_sparse_depth(pixels, depths, image_size),
            "boxes": boxes,
            "classes": classes,
            "distances": distances,
            "is_night": is_night,
            "sample_token": sample_token,
        }
