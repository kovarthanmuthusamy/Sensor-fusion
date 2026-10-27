from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from nuscenes.nuscenes import NuScenes

from sensor_fusion.data.labels import CLASS_NAMES, box_to_2d, frame_labels
from sensor_fusion.data.splits import scene_is_night

HEIGHT, WIDTH = 900, 1600
IMAGE_SIZE = (HEIGHT, WIDTH)
# Focal length 1000 px, principal point (800, 450): easy to compute by hand.
INTRINSIC = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]])
DATASET = Path(__file__).parents[1] / "dataset" / "v1.0-mini"
needs_mini = pytest.mark.skipif(
    not DATASET.exists(), reason="nuScenes-mini not found at dataset/v1.0-mini"
)


def _corners(center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    """`(8, 3)` corners of an axis-aligned box in the camera frame (extents along x, y, z)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return np.asarray(center) + signs * np.asarray(size) / 2


# --- box_to_2d: hand-computed cases ---------------------------------------------------------


def test_box_in_front_gives_the_smallest_rectangle_around_its_corners() -> None:
    # Box 2 m wide, 1 m high, 2 m deep, centred 10 m ahead: the near face (z = 9) is the widest.
    # u = 1000 * x / z + 800 -> 800 +- 1000 * 1 / 9 = 688.889 .. 911.111
    # v = 1000 * y / z + 450 -> 450 +- 1000 * 0.5 / 9 = 394.444 .. 505.556
    result = box_to_2d(_corners((0, 0, 10), (2, 1, 2)), INTRINSIC, IMAGE_SIZE)
    assert result is not None
    box, _ = result
    np.testing.assert_allclose(box, [688.889, 394.444, 911.111, 505.556], atol=1e-2)
    assert box.dtype == np.float32


def test_distance_is_the_depth_of_the_box_centre() -> None:
    result = box_to_2d(_corners((0, 0, 10), (2, 1, 2)), INTRINSIC, IMAGE_SIZE)
    assert result is not None
    assert result[1] == pytest.approx(10.0)  # not the nearest face (9.0) and not the range


def test_box_crossing_the_camera_plane_is_dropped() -> None:
    # A tiny box on the optical axis with z from 0.05 to 0.15: one face is within 0.1 m of the
    # camera plane. Without the rule it would project to u = 600..1000, inside the photo, so
    # only the corner-depth rule can reject it.
    assert box_to_2d(_corners((0, 0, 0.1), (0.02, 0.02, 0.1)), INTRINSIC, IMAGE_SIZE) is None


def test_the_same_tiny_box_further_ahead_is_kept() -> None:
    # Control for the test above: moved 10 m ahead, the identical box is a normal object.
    assert box_to_2d(_corners((0, 0, 10), (0.02, 0.02, 0.1)), INTRINSIC, IMAGE_SIZE) is not None


def test_box_entirely_behind_the_camera_is_dropped() -> None:
    assert box_to_2d(_corners((0, 0, -10), (2, 2, 2)), INTRINSIC, IMAGE_SIZE) is None


def test_box_outside_the_photo_is_dropped() -> None:
    assert box_to_2d(_corners((100, 0, 10), (2, 2, 2)), INTRINSIC, IMAGE_SIZE) is None


def test_zero_size_box_is_dropped() -> None:
    assert box_to_2d(_corners((0, 0, 10), (0, 0, 0)), INTRINSIC, IMAGE_SIZE) is None


def test_box_mostly_inside_is_kept_and_clipped_to_the_right_edge() -> None:
    # Box x from 6 to 8, z from 9 to 11: u from 1345.5 to 1688.9, so 74% is inside the photo.
    result = box_to_2d(_corners((7, 0, 10), (2, 1, 2)), INTRINSIC, IMAGE_SIZE)
    assert result is not None
    box, _ = result
    assert box[2] == WIDTH
    assert box[0] == pytest.approx(1345.45, abs=1e-2)


def test_box_mostly_outside_is_dropped_unless_the_threshold_is_lowered() -> None:
    # Box x from 7 to 9, z from 9 to 11: u from 1436.4 to 1800, so only 45% is inside the photo.
    corners = _corners((8, 0, 10), (2, 1, 2))
    assert box_to_2d(corners, INTRINSIC, IMAGE_SIZE) is None
    kept = box_to_2d(corners, INTRINSIC, IMAGE_SIZE, min_inside_fraction=0.3)
    assert kept is not None
    assert kept[0][2] == WIDTH


def test_box_cut_by_the_bottom_edge_is_clipped_in_v() -> None:
    # Box y from 3.5 to 5.5, z from 9 to 11: v from 768.2 to 1061.1, so 45% is inside the photo.
    kept = box_to_2d(_corners((0, 4.5, 10), (2, 2, 2)), INTRINSIC, IMAGE_SIZE, 0.3)
    assert kept is not None
    assert kept[0][3] == HEIGHT


def test_corners_in_the_devkit_layout_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(8, 3\)"):
        box_to_2d(_corners((0, 0, 10), (2, 2, 2)).T, INTRINSIC, IMAGE_SIZE)


def test_wrong_intrinsic_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="intrinsic"):
        box_to_2d(_corners((0, 0, 10), (2, 2, 2)), np.eye(4), IMAGE_SIZE)


# --- frame_labels on real data (needs the dataset) --------------------------------------------


@pytest.fixture(scope="module")
def nusc() -> NuScenes:
    return NuScenes("v1.0-mini", dataroot=str(DATASET), verbose=False)


@needs_mini
def test_labels_have_consistent_shapes_and_sensible_values(nusc: NuScenes) -> None:
    for sample in nusc.sample[::20]:
        boxes, classes, distances = frame_labels(nusc, sample["data"]["CAM_FRONT"])
        assert boxes.dtype == np.float32 and boxes.shape[1:] == (4,)
        assert classes.dtype == np.int64 and distances.dtype == np.float32
        assert len(boxes) == len(classes) == len(distances)
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert (boxes[:, [0, 2]] >= 0).all() and (boxes[:, [0, 2]] <= WIDTH).all()
        assert (boxes[:, [1, 3]] >= 0).all() and (boxes[:, [1, 3]] <= HEIGHT).all()
        assert ((classes >= 0) & (classes < len(CLASS_NAMES))).all()
        assert (distances > 0.1).all()


@needs_mini
def test_label_counts_over_all_mini_frames_match_the_measured_numbers(nusc: NuScenes) -> None:
    # Pinned from the measurement behind ADR 0005: 2,319 / 758 boxes pass the visibility rule
    # (day / night), and the 50%-inside rule removes 206 / 64 of them, leaving 2,113 / 694.
    # A change in any rule, or in the class mapping, shows up here first.
    counts: Counter[tuple[str, str]] = Counter()
    for sample in nusc.sample:
        tag = "night" if scene_is_night(nusc.get("scene", sample["scene_token"])) else "day"
        _, classes, _ = frame_labels(nusc, sample["data"]["CAM_FRONT"])
        for class_id in classes:
            counts[tag, CLASS_NAMES[class_id]] += 1
    assert counts == {
        ("day", "car"): 1057,
        ("day", "large_vehicle"): 315,
        ("day", "pedestrian"): 619,
        ("day", "two_wheeler"): 122,
        ("night", "car"): 450,
        ("night", "large_vehicle"): 51,
        ("night", "pedestrian"): 151,
        ("night", "two_wheeler"): 42,
    }
