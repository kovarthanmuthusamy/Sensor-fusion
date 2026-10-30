import math
from typing import Any

import numpy as np
import pytest
import torch

from sensor_fusion.data.targets import (
    REGRESSION_CHANNELS,
    Detections,
    decode_outputs,
    encode_targets,
)

STRIDE, ALPHA, NUM_CLASSES = 4, 0.54, 4
INPUT_SIZE = (64, 128)  # a 16 x 32 grid


def _encode(
    boxes: Any, classes: Any, distances: Any, input_size: tuple[int, int] = INPUT_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return encode_targets(
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64),
        np.asarray(distances, dtype=np.float32),
        input_size=input_size,
        num_classes=NUM_CLASSES,
        stride=STRIDE,
        peak_alpha=ALPHA,
    )


# --- encode_targets: hand-computed -----------------------------------------------------------


def test_one_box_paints_a_peak_and_regression_values_at_its_centre_cell() -> None:
    # Box 40 x 40 px at (42, 22)-(82, 62): centre (62, 42) px = (15.5, 10.5) cells -> cell (15, 10),
    # offset (0.5, 0.5); size 10 x 10 cells; distance 20 m.
    heatmap, regression, mask = _encode([[42, 22, 82, 62]], [1], [20.0])
    assert (
        heatmap.shape == (4, 16, 32) and regression.shape == (5, 16, 32) and mask.shape == (16, 32)
    )
    assert heatmap[1, 10, 15] == 1.0
    assert heatmap[[0, 2, 3]].max() == 0.0  # only the box's own class is painted
    expected = [math.log(10), math.log(10), 0.5, 0.5, math.log(20)]
    np.testing.assert_allclose(regression[:, 10, 15], expected, rtol=1e-5)
    assert mask.sum() == 1.0 and mask[10, 15] == 1.0
    assert tuple(REGRESSION_CHANNELS) == (
        "log_width",
        "log_height",
        "offset_x",
        "offset_y",
        "log_distance",
    )


def test_the_peak_is_the_unique_maximum_of_its_neighbourhood() -> None:
    heatmap, _, _ = _encode([[42, 22, 82, 62]], [1], [20.0])
    neighbours = heatmap[1, 9:12, 14:17].copy()
    neighbours[1, 1] = 0.0
    assert neighbours.max() < 1.0
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_a_wide_box_gets_a_peak_that_is_wider_than_it_is_tall() -> None:
    # 80 x 20 px = 20 x 5 cells: sigma_x = 0.54 * 20 / 6 = 1.8 cells, sigma_y stays at the minimum.
    heatmap, _, _ = _encode([[8, 24, 88, 44]], [0], [15.0])  # centre cell (12, 8)
    assert heatmap[0, 8, 12] == 1.0
    assert heatmap[0, 8, 13] == pytest.approx(math.exp(-1 / (2 * 1.8**2)), rel=1e-4)
    assert heatmap[0, 8, 12 + 3] > 0.2  # far along x it still carries signal
    assert heatmap[0, 8 + 3, 12] < 1e-6  # far along y it does not


def test_the_nearer_object_wins_a_shared_centre_cell() -> None:
    boxes = [[40, 20, 80, 60], [50, 30, 70, 50]]  # same centre (60, 40) px
    _, regression, mask = _encode(boxes, [0, 0], [50.0, 10.0])
    assert mask.sum() == 1.0
    assert regression[4, 10, 15] == pytest.approx(math.log(10.0), rel=1e-5)
    assert regression[0, 10, 15] == pytest.approx(math.log(5.0), rel=1e-5)  # 20 px = 5 cells


def test_an_object_in_the_last_cell_is_painted_without_leaving_the_grid() -> None:
    heatmap, _, mask = _encode([[124, 60, 128, 64]], [2], [30.0])  # centre cell (31, 15)
    assert heatmap[2, 15, 31] == 1.0 and mask[15, 31] == 1.0


def test_no_objects_gives_empty_grids() -> None:
    heatmap, regression, mask = _encode([], [], [])
    assert not heatmap.any() and not regression.any() and not mask.any()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([[0, 0, 8, 8]], [4], [10.0]), "class ids"),
        (([[0, 0, 8, 8]], [0], [0.0]), "distances must be > 0"),
        (([[8, 8, 8, 16]], [0], [10.0]), "positive width"),
        (([[0, 0, 8, 8]], [0, 1], [10.0]), "same length"),
    ],
)
def test_malformed_labels_are_rejected(args: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _encode(*args)


def test_boxes_must_have_four_numbers_each() -> None:
    with pytest.raises(ValueError, match=r"\(K, 4\)"):
        encode_targets(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=np.float32),
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            stride=STRIDE,
            peak_alpha=ALPHA,
        )


def test_input_size_must_be_a_multiple_of_the_stride() -> None:
    with pytest.raises(ValueError, match="multiple of the stride"):
        _encode([[0, 0, 8, 8]], [0], [10.0], input_size=(62, 128))


# --- decode_outputs ---------------------------------------------------------------------------


def _decode(
    heatmap: np.ndarray, regression: np.ndarray, top_k: int = 10, score_threshold: float = 0.5
) -> Detections:
    return decode_outputs(
        torch.from_numpy(heatmap),
        torch.from_numpy(regression),
        stride=STRIDE,
        top_k=top_k,
        score_threshold=score_threshold,
    )


def test_encoding_then_decoding_returns_the_same_objects() -> None:
    boxes = np.array(
        [[20, 20, 60, 60], [100, 30, 140, 90], [180, 50, 200, 110], [30, 80, 70, 110]],
        dtype=np.float32,
    )
    classes, distances = np.array([0, 1, 2, 3]), np.array([10.0, 25.5, 60.0, 12.0])
    heatmap, regression, _ = _encode(boxes, classes, distances, input_size=(128, 256))
    found = _decode(heatmap, regression)
    # Detections come out strongest first; put both sides in left-to-right order to compare.
    order = found.boxes[:, 0].argsort()
    expected = boxes[:, 0].argsort()
    np.testing.assert_allclose(found.boxes[order].numpy(), boxes[expected], atol=1e-3)
    np.testing.assert_array_equal(found.classes[order].numpy(), classes[expected])
    np.testing.assert_allclose(found.distances[order].numpy(), distances[expected], rtol=1e-4)
    np.testing.assert_allclose(found.scores.numpy(), 1.0)


def test_only_local_maxima_become_detections() -> None:
    heat = np.zeros((2, 8, 8), dtype=np.float32)
    ys, xs = np.mgrid[0:8, 0:8]
    heat[0] = 0.8 * np.exp(-((xs - 4) ** 2 + (ys - 4) ** 2) / 4)  # a smooth hill at (4, 4)
    found = _decode(heat, np.zeros((5, 8, 8), dtype=np.float32), score_threshold=0.3)
    assert len(found.scores) == 1 and found.scores[0] == pytest.approx(0.8)


def test_threshold_and_top_k_limit_the_detections() -> None:
    heat = np.zeros((2, 8, 8), dtype=np.float32)
    heat[0, 2, 2], heat[1, 5, 5], heat[0, 6, 1] = 0.9, 0.5, 0.2
    regression = np.zeros((5, 8, 8), dtype=np.float32)
    assert _decode(
        heat, regression, top_k=10, score_threshold=0.4
    ).scores.tolist() == pytest.approx([0.9, 0.5])
    assert len(_decode(heat, regression, top_k=1, score_threshold=0.4).scores) == 1
    assert len(_decode(heat, regression, top_k=10, score_threshold=0.95).scores) == 0


def test_absurd_network_outputs_still_give_finite_boxes() -> None:
    heat = np.zeros((1, 8, 8), dtype=np.float32)
    heat[0, 3, 3] = 0.9
    regression = np.full((5, 8, 8), 1e4, dtype=np.float32)
    found = _decode(heat, regression)
    assert torch.isfinite(found.boxes).all() and torch.isfinite(found.distances).all()


def test_decoder_rejects_mismatched_grids() -> None:
    with pytest.raises(ValueError, match=r"\(5, h, w\)"):
        decode_outputs(
            torch.zeros(4, 8, 8), torch.zeros(4, 8, 8), stride=STRIDE, top_k=5, score_threshold=0.1
        )
