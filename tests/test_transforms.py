from typing import Any

import numpy as np
import pytest

from sensor_fusion.data.transforms import (
    PreparedFrame,
    boxes_to_original,
    prepare_frame,
    shrink_depth,
)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HEIGHT, WIDTH = 100, 128  # shrunk by 2: 50 x 64, cropped to 32 x 64 (9 rows removed on top)
CROP_TOP, CROP_H, CROP_W = 9, 32, 64


def _frame(
    in_channels: int = 3, flip: bool = False, **overrides: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    data: dict[str, Any] = {
        "image": np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
        "depth": np.zeros((HEIGHT, WIDTH), dtype=np.float32),
        "boxes": np.array([[20.0, 40.0, 60.0, 80.0]], dtype=np.float32),
        "classes": np.array([2], dtype=np.int64),
        "distances": np.array([25.0], dtype=np.float32),
    }
    data.update(overrides)
    settings = {
        "downscale": 2,
        "depth_max_m": 100.0,
        "in_channels": in_channels,
        "rgb_mean": MEAN,
        "rgb_std": STD,
        "flip": flip,
    }
    return data, settings


def _prepare(in_channels: int = 3, flip: bool = False, **overrides: np.ndarray) -> PreparedFrame:
    data, settings = _frame(in_channels, flip, **overrides)
    return prepare_frame(**data, **settings)


# --- shrink_depth -----------------------------------------------------------------------


def test_shrinking_depth_keeps_the_nearest_real_value_of_each_block() -> None:
    depth = np.array([[0, 30, 0, 0], [10, 0, 0, 0]], dtype=np.float32)
    # Left block holds 30 and 10 (nearest 10); right block has no LiDAR return at all.
    np.testing.assert_array_equal(shrink_depth(depth, 2), [[10.0, 0.0]])


def test_shrinking_depth_never_averages_across_an_edge() -> None:
    depth = np.array([[10, 30], [0, 0]], dtype=np.float32)  # a near car next to a far wall
    assert shrink_depth(depth, 2)[0, 0] == 10.0  # an average would be 20.0: a depth nobody measured


def test_depth_must_be_a_two_dimensional_image() -> None:
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        shrink_depth(np.zeros((2, 4, 1), dtype=np.float32), 2)


def test_depth_factor_must_divide_the_size() -> None:
    with pytest.raises(ValueError, match="divide"):
        shrink_depth(np.zeros((5, 4), dtype=np.float32), 2)


# --- prepare_frame: shapes and channels -------------------------------------------------


def test_rgb_baseline_has_three_channels_and_fusion_four() -> None:
    assert _prepare(in_channels=3).network_input.shape == (3, CROP_H, CROP_W)
    assert _prepare(in_channels=4).network_input.shape == (4, CROP_H, CROP_W)


def test_baseline_and_fusion_inputs_differ_only_by_the_depth_channel() -> None:
    image = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb = _prepare(in_channels=3, image=image).network_input
    fusion = _prepare(in_channels=4, image=image).network_input
    np.testing.assert_array_equal(fusion[:3], rgb)


def test_depth_channel_is_metres_over_the_maximum_and_zero_means_no_return() -> None:
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth[40, 60] = 50.0  # full-size pixel (row 40, column 60)
    depth[60, 80] = 250.0  # beyond the maximum: clipped to 1
    channel = _prepare(in_channels=4, depth=depth).network_input[3]
    # shrunk by 2 -> (20, 30), cropped 9 rows off the top -> (11, 30)
    assert channel[11, 30] == pytest.approx(0.5)
    assert channel[21, 40] == 1.0
    assert np.count_nonzero(channel) == 2


def test_photo_is_normalised_with_the_given_statistics() -> None:
    white = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    network_input = _prepare(image=white).network_input
    expected = (1.0 - np.array(MEAN)) / np.array(STD)
    np.testing.assert_allclose(network_input[:, 0, 0], expected, rtol=1e-5)


# --- prepare_frame: the boxes follow the pixels -------------------------------------------


def test_boxes_follow_the_shrink_and_the_crop() -> None:
    frame = _prepare()
    # (20, 40, 60, 80) / 2 = (10, 20, 30, 40); the 9 cropped rows move y by -9.
    np.testing.assert_allclose(frame.boxes, [[10.0, 11.0, 30.0, 31.0]])
    assert (frame.crop_top, frame.crop_left) == (CROP_TOP, 0)
    np.testing.assert_array_equal(frame.classes, [2])
    np.testing.assert_allclose(frame.distances, [25.0])


def test_a_depth_pixel_and_a_box_corner_use_the_same_coordinates() -> None:
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth[40, 60] = 10.0  # full-size pixel (row 40, column 60) = the top-right corner of the box
    frame = _prepare(in_channels=4, depth=depth)
    row, column = np.argwhere(frame.network_input[3] > 0)[0]
    assert (column, row) == (30, 11)  # box corner (x2, y1) of the box above


def test_a_box_lost_in_the_cropped_rows_is_dropped_with_its_class_and_distance() -> None:
    boxes = np.array([[20.0, 40.0, 60.0, 80.0], [20.0, 0.0, 60.0, 10.0]], dtype=np.float32)
    frame = _prepare(
        boxes=boxes,
        classes=np.array([2, 1], dtype=np.int64),
        distances=np.array([25.0, 40.0], dtype=np.float32),
    )
    # The second box lies in rows 0-5 of the shrunk frame, which the crop removes.
    assert len(frame.boxes) == len(frame.classes) == len(frame.distances) == 1
    np.testing.assert_array_equal(frame.classes, [2])


def test_boxes_are_mapped_back_to_the_full_size_photo() -> None:
    frame = _prepare()
    restored = boxes_to_original(
        frame.boxes, downscale=2, crop_top=frame.crop_top, crop_left=frame.crop_left
    )
    np.testing.assert_allclose(restored, [[20.0, 40.0, 60.0, 80.0]])


# --- prepare_frame: mirroring -------------------------------------------------------------


def test_mirroring_flips_photo_depth_and_boxes_together() -> None:
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth[40, 60] = 10.0
    plain = _prepare(in_channels=4, depth=depth)
    mirrored = _prepare(in_channels=4, flip=True, depth=depth)
    np.testing.assert_array_equal(mirrored.network_input, plain.network_input[:, :, ::-1])
    # box x: (10, 30) in a 64-wide frame becomes (64 - 30, 64 - 10) = (34, 54); y is unchanged
    np.testing.assert_allclose(mirrored.boxes, [[34.0, 11.0, 54.0, 31.0]])
    assert mirrored.flipped and not plain.flipped


def test_mirroring_does_not_change_distances_or_classes() -> None:
    plain, mirrored = _prepare(), _prepare(flip=True)
    np.testing.assert_array_equal(plain.distances, mirrored.distances)
    np.testing.assert_array_equal(plain.classes, mirrored.classes)


# --- guards -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image": np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)}, "uint8"),
        ({"depth": np.zeros((10, 10), dtype=np.float32)}, "must match the image"),
        ({"boxes": np.zeros((1, 3), dtype=np.float32)}, r"\(K, 4\)"),
        ({"classes": np.zeros(2, dtype=np.int64)}, "same length"),
    ],
)
def test_malformed_frames_are_rejected(overrides: dict[str, np.ndarray], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare(**overrides)


def test_other_channel_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="in_channels"):
        _prepare(in_channels=5)


def test_non_positive_depth_maximum_is_rejected() -> None:
    data, settings = _frame()
    settings["depth_max_m"] = 0.0
    with pytest.raises(ValueError, match="depth_max_m"):
        prepare_frame(**data, **settings)


def test_normalisation_statistics_need_three_numbers() -> None:
    data, settings = _frame()
    settings["rgb_mean"] = (0.5, 0.5)
    with pytest.raises(ValueError, match="3 numbers"):
        prepare_frame(**data, **settings)
