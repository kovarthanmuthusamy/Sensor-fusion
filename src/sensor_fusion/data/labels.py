"""3D boxes -> 2D boxes with a class and a distance, for one camera frame (ADR 0004, ADR 0005).

nuScenes only has 3D boxes. We project the 8 corners of each box into the photo and wrap the
smallest rectangle around them. Conventions as in `projection.py`: camera frame x right, y down,
z forward (metres); depth is camera z, not range; image sizes are `(height, width)`.
"""

import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

# Class ids are indices into this tuple. The detector wrapper adds the background class.
CLASS_NAMES = ("car", "large_vehicle", "pedestrian", "two_wheeler")
_DETECTION_TO_CLASS_NAME = {
    "car": "car",
    "truck": "large_vehicle",
    "bus": "large_vehicle",
    "trailer": "large_vehicle",
    "construction_vehicle": "large_vehicle",
    "pedestrian": "pedestrian",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
}

# nuScenes visibility levels: 1 = 0-40%, 2 = 40-60%, 3 = 60-80%, 4 = 80-100% of the object visible.
MIN_VISIBILITY = 2
# Keep a box only if at least this share of its projected rectangle lies inside the photo.
MIN_INSIDE_FRACTION = 0.5
# Corners closer than this to the camera plane (metres) count as behind it (as in the devkit).
MIN_CORNER_DEPTH = 0.1


def _to_four_classes(category_name: str) -> int | None:
    """Class id (index into `CLASS_NAMES`) of a nuScenes category, or None if we don't train it."""
    class_name = _DETECTION_TO_CLASS_NAME.get(category_to_detection_name(category_name))
    return None if class_name is None else CLASS_NAMES.index(class_name)


def box_to_2d(
    corners_cam: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    min_inside_fraction: float = MIN_INSIDE_FRACTION,
) -> tuple[np.ndarray, float] | None:
    """Flat box on the photo and the distance of one 3D box, or None if it is not usable.

    Args:
        corners_cam: `(8, 3)` the box corners in the camera frame, metres. (The devkit's
            `Box.corners()` is `(3, 8)`: pass `.T`.)
        intrinsic: `(3, 3)` camera matrix K (pixels).
        image_size: `(height, width)` in pixels.
        min_inside_fraction: share of the projected rectangle that must lie inside the photo.

    Returns:
        `(box, distance)`: `box` is `[x1, y1, x2, y2]` float32 pixels, clipped to the photo;
        `distance` is the depth (camera z, metres) of the box centre.
        None when a corner is behind or on the camera plane (such boxes are dropped), when the
        box lies outside the photo, or when too little of it is inside.
    """
    height, width = image_size
    if corners_cam.shape != (8, 3):
        raise ValueError(
            f"corners_cam must have shape (8, 3), got {corners_cam.shape} "
            "(the devkit's Box.corners() is (3, 8): pass .T)"
        )
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must be (3, 3), got {intrinsic.shape}")

    # Filter before dividing by depth: a corner behind the camera would project to nonsense.
    if (corners_cam[:, 2] <= MIN_CORNER_DEPTH).any():
        return None

    homogeneous = corners_cam @ intrinsic.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    full_area = (x2 - x1) * (y2 - y1)
    clipped_x1, clipped_y1 = max(x1, 0.0), max(y1, 0.0)
    clipped_x2, clipped_y2 = min(x2, float(width)), min(y2, float(height))
    if full_area <= 0 or clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None
    inside = (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1) / full_area
    if inside < min_inside_fraction:
        return None

    box = np.array([clipped_x1, clipped_y1, clipped_x2, clipped_y2], dtype=np.float32)
    return box, float(corners_cam[:, 2].mean())  # the mean of a box's 8 corners is its centre


def frame_labels(nusc: NuScenes, cam_token: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D boxes, class ids and distances of the usable objects in one camera frame.

    A 3D annotation is kept when its class is one of `CLASS_NAMES`, its nuScenes visibility
    level is at least `MIN_VISIBILITY`, and `box_to_2d` accepts it. Annotations are **not**
    filtered by LiDAR-point count, so an RGB-only model and a fusion model are scored on the
    same objects.

    Args:
        nusc: the loaded devkit.
        cam_token: `sample_data` token of a keyframe camera image.

    Returns:
        `boxes` `(K, 4)` float32 `[x1, y1, x2, y2]` pixels, `classes` `(K,)` int64,
        `distances` `(K,)` float32 metres (camera z of the box centre).
    """
    cam = nusc.get("sample_data", cam_token)
    image_size = (cam["height"], cam["width"])
    # BoxVisibility.NONE keeps every box; our own, documented rules are applied below.
    _, camera_boxes, intrinsic = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.NONE)

    boxes: list[np.ndarray] = []
    classes: list[int] = []
    distances: list[float] = []
    for camera_box in camera_boxes:
        class_id = _to_four_classes(camera_box.name)
        if class_id is None:
            continue
        annotation = nusc.get("sample_annotation", camera_box.token)
        if int(annotation["visibility_token"]) < MIN_VISIBILITY:
            continue
        flat = box_to_2d(camera_box.corners().T, np.asarray(intrinsic), image_size)
        if flat is None:
            continue
        boxes.append(flat[0])
        classes.append(class_id)
        distances.append(flat[1])

    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 4),
        np.array(classes, dtype=np.int64),
        np.array(distances, dtype=np.float32),
    )
