"""LiDAR -> camera projection and the sparse depth image (ADR 0004).

The nuScenes frame chain for one LiDAR scan and one photo of the same keyframe:

    lidar --calibrated_sensor--> ego(t_lidar) --ego_pose--> global
          --ego_pose^-1--> ego(t_cam) --calibrated_sensor^-1--> camera

The two ego poses differ because the sensors fire at different moments (tens of
milliseconds apart) and the car moves in between.

Conventions (see .claude/rules/coordinate-frames.md):
- `T_<to>_from_<from>` is a 4x4 matrix acting on column vectors: `p_to = T @ p_from`.
- nuScenes quaternions are `[w, x, y, z]`.
- Camera frame: x right, y down, z forward, in metres.
- "Depth" is the camera z in metres, not the straight-line range.
- Image sizes are `(height, width)` in pixels. A pixel position is `(u, v)` = (column, row).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


def _quaternion_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """3x3 rotation matrix from a nuScenes quaternion `[w, x, y, z]` (it is normalised first)."""
    if len(quaternion) != 4:
        raise ValueError(f"quaternion must have 4 numbers [w, x, y, z], got {len(quaternion)}")
    values = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = values / np.linalg.norm(values)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rigid_transform(record: dict[str, Any]) -> np.ndarray:
    """4x4 transform from a nuScenes record with `translation` (m) and `rotation` (quaternion).

    For a `calibrated_sensor` this is `T_ego_from_sensor`; for an `ego_pose` it is
    `T_global_from_ego`.
    """
    transform = np.eye(4)
    transform[:3, :3] = _quaternion_to_rotation(record["rotation"])
    transform[:3, 3] = record["translation"]
    return transform


def _invert_rigid(transform: np.ndarray) -> np.ndarray:
    """Inverse of a rotation-plus-translation 4x4 matrix (a rotation is undone by its transpose)."""
    rotation = transform[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def build_lidar_to_camera(
    lidar_calib: dict[str, Any],
    lidar_pose: dict[str, Any],
    cam_pose: dict[str, Any],
    cam_calib: dict[str, Any],
) -> np.ndarray:
    """4x4 matrix `T_cam_from_lidar`: LiDAR-frame points -> camera-frame points (metres).

    Args:
        lidar_calib: `calibrated_sensor` record of the LiDAR (sensor -> ego).
        lidar_pose: `ego_pose` record at the LiDAR timestamp (ego -> global).
        cam_pose: `ego_pose` record at the camera timestamp (ego -> global).
        cam_calib: `calibrated_sensor` record of the camera (sensor -> ego).

    The LiDAR and camera `ego_pose` records are different on purpose; see the module docstring.
    """
    T_ego_from_lidar = _rigid_transform(lidar_calib)
    T_global_from_ego_lidar = _rigid_transform(lidar_pose)
    T_global_from_ego_cam = _rigid_transform(cam_pose)
    T_ego_from_cam = _rigid_transform(cam_calib)
    # lidar -> ego(t_lidar) -> global -> ego(t_cam) -> camera
    return (
        _invert_rigid(T_ego_from_cam)
        @ _invert_rigid(T_global_from_ego_cam)
        @ T_global_from_ego_lidar
        @ T_ego_from_lidar
    )


def project_to_image(
    points: np.ndarray,
    T_cam_from_lidar: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    min_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel position and depth of every LiDAR point that is visible in the photo.

    Args:
        points: `(N, 3)` x, y, z in the LiDAR frame, metres. (The devkit's `LidarPointCloud`
            stores `(4, N)`: take `.points[:3].T`.)
        T_cam_from_lidar: `(4, 4)` from `build_lidar_to_camera`.
        intrinsic: `(3, 3)` camera matrix K (pixels).
        image_size: `(height, width)` in pixels.
        min_depth: points with depth <= this (metres) are dropped before projecting. Must be > 0.

    Returns:
        `pixels` `(M, 2)` float32 as `(u, v)` with `0 <= u < width` and `0 <= v < height`, and
        `depths` `(M,)` float32 in metres (camera z). Order follows the input points.
    """
    height, width = image_size
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {points.shape} "
            "(the devkit's LidarPointCloud.points is (4, N): use .points[:3].T)"
        )
    if T_cam_from_lidar.shape != (4, 4):
        raise ValueError(f"T_cam_from_lidar must be (4, 4), got {T_cam_from_lidar.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must be (3, 3), got {intrinsic.shape}")
    if min_depth <= 0:
        raise ValueError(f"min_depth must be > 0 (we divide by depth), got {min_depth}")

    rotation, translation = T_cam_from_lidar[:3, :3], T_cam_from_lidar[:3, 3]
    points_cam = points.astype(np.float64) @ rotation.T + translation  # (N, 3), camera frame
    depths = points_cam[:, 2]

    # Filter before dividing by depth: points behind or next to the camera would blow up.
    in_front = depths > min_depth
    points_cam, depths = points_cam[in_front], depths[in_front]

    homogeneous = points_cam @ intrinsic.T  # (M, 3) = (u*z, v*z, z)
    pixels = (homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32)

    # Test the float32 values we return, so a pixel cannot round up to `width` afterwards.
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return pixels[inside], depths[inside].astype(np.float32)


def make_sparse_depth(
    pixels: np.ndarray, depths: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Sparse depth image: metres at the pixels a LiDAR point hit, `0` everywhere else.

    Args:
        pixels: `(M, 2)` as `(u, v)`, from `project_to_image` (so inside the image).
        depths: `(M,)` metres, all > 0 (zero is reserved for "no LiDAR return").
        image_size: `(height, width)` in pixels.

    Returns:
        `(height, width)` float32. If several points land on one pixel, the nearest wins.
    """
    height, width = image_size
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels must have shape (M, 2), got {pixels.shape}")
    if depths.shape != (pixels.shape[0],):
        raise ValueError(f"depths must have shape ({pixels.shape[0]},), got {depths.shape}")
    if (depths <= 0).any():
        raise ValueError("depths must be > 0: 0 means 'no LiDAR return' in the depth image")

    columns = np.floor(pixels[:, 0]).astype(np.intp)
    rows = np.floor(pixels[:, 1]).astype(np.intp)
    if ((columns < 0) | (columns >= width) | (rows < 0) | (rows >= height)).any():
        raise ValueError("pixels outside the image: filter them with project_to_image first")

    # `ufunc.at` is unbuffered, so repeated pixels are all taken into account. Plain assignment
    # (`image[rows, columns] = depths`) gives no guarantee about which duplicate is written last.
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(nearest, (rows, columns), depths.astype(np.float32))
    nearest[np.isinf(nearest)] = 0.0
    return nearest
