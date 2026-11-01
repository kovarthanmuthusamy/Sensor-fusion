"""Matching, average precision, distance error and the bootstrap, on small hand-checkable data."""

import math

import numpy as np
import pytest

from sensor_fusion.evaluation.bootstrap import (
    bootstrap_samples,
    metrics_to_dict,
    percentile_interval,
)
from sensor_fusion.evaluation.matching import FrameResult, box_iou, match_frame
from sensor_fusion.evaluation.metrics import (
    average_precision,
    build_cell_data,
    cell_metrics,
    distance_by_range,
    eligible_classes,
    weighted_median,
)

THR = (0.5, 0.75)
CLASSES = ("car", "pedestrian")


def frame(
    scene: str = "s1",
    *,
    dets: list[tuple[list[float], float, int, float]] | None = None,
    gts: list[tuple[list[float], int, float]] | None = None,
    night: bool = False,
    city: str = "boston",
    token: str = "t",
) -> FrameResult:
    """dets: (box, score, class, distance); gts: (box, class, distance)."""
    dets, gts = dets or [], gts or []
    return FrameResult(
        token=token,
        scene=scene,
        city=city,
        is_night=night,
        det_boxes=np.array([d[0] for d in dets], dtype=np.float32).reshape(-1, 4),
        det_scores=np.array([d[1] for d in dets], dtype=np.float32),
        det_classes=np.array([d[2] for d in dets], dtype=np.int64),
        det_distances=np.array([d[3] for d in dets], dtype=np.float32),
        gt_boxes=np.array([g[0] for g in gts], dtype=np.float32).reshape(-1, 4),
        gt_classes=np.array([g[1] for g in gts], dtype=np.int64),
        gt_distances=np.array([g[2] for g in gts], dtype=np.float32),
    )


BOX = [0.0, 0.0, 10.0, 10.0]
SHIFTED = [2.0, 0.0, 12.0, 10.0]  # IoU with BOX = 80 / 120 = 2/3


# --- box_iou ---------------------------------------------------------------------------------


def test_iou_hand_computed() -> None:
    a = np.array([[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.float32)
    b = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [5, 5, 6, 6]], dtype=np.float32)
    iou = box_iou(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(1 / 3)  # intersection 2, union 6
    assert iou[0, 2] == 0.0
    assert iou.shape == (3, 3)


def test_iou_of_empty_sets_has_the_right_shape() -> None:
    assert box_iou(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert box_iou(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


# --- match_frame -----------------------------------------------------------------------------


def test_higher_score_takes_the_label_and_the_duplicate_is_a_false_positive() -> None:
    f = frame(dets=[(BOX, 0.6, 0, 10.0), (BOX, 0.9, 0, 11.0)], gts=[(BOX, 0, 10.0)])
    car = match_frame(f, THR, 2)[0]
    assert car.scores.tolist() == pytest.approx([0.9, 0.6])  # sorted by score
    assert car.tp[:, 0].tolist() == [True, False]
    assert car.n_gt == 1
    assert car.matched_gt_distances[0] == 10.0 and math.isnan(car.matched_gt_distances[1])


def test_iou_threshold_decides_true_positive() -> None:
    f = frame(dets=[(SHIFTED, 0.9, 0, 10.0)], gts=[(BOX, 0, 10.0)])
    tp = match_frame(f, (0.5, 0.75), 2)[0].tp[0]
    assert tp.tolist() == [True, False]  # IoU 2/3: above 0.5, below 0.75


def test_a_detection_never_matches_a_label_of_another_class() -> None:
    f = frame(dets=[(BOX, 0.9, 1, 10.0)], gts=[(BOX, 0, 10.0)])
    cars, pedestrians = match_frame(f, THR, 2)
    assert cars.n_gt == 1 and len(cars.scores) == 0
    assert pedestrians.n_gt == 0 and pedestrians.tp.sum() == 0


def test_no_labels_means_only_false_positives_and_no_detections_means_misses() -> None:
    no_labels = match_frame(frame(dets=[(BOX, 0.9, 0, 5.0)]), THR, 2)[0]
    assert no_labels.tp.sum() == 0 and no_labels.n_gt == 0
    no_dets = match_frame(frame(gts=[(BOX, 0, 5.0)]), THR, 2)[0]
    assert len(no_dets.scores) == 0 and no_dets.n_gt == 1


# --- average_precision -----------------------------------------------------------------------


def ap(tp: list[bool], n_gt: int, weights: list[float] | None = None) -> float:
    arr = np.array(tp, dtype=bool).reshape(-1, 1)
    w = np.ones(len(tp)) if weights is None else np.array(weights, dtype=float)
    return float(average_precision(arr, w, float(n_gt))[0])


def test_ap_hand_computed_cases() -> None:
    assert ap([True], 1) == pytest.approx(1.0)
    assert ap([False, True], 1) == pytest.approx(0.5)  # precision 1/2 at recall 1
    assert ap([True, True], 2) == pytest.approx(1.0)
    assert ap([True], 2) == pytest.approx(51 / 101)  # recall stops at 0.5: 51 of 101 points
    assert ap([False, False], 3) == 0.0
    assert ap([], 3) == 0.0
    assert math.isnan(ap([True], 0))


def test_ap_envelope_uses_the_best_precision_to_the_right() -> None:
    # TP FP TP: precision 1, 1/2, 2/3; recall 1/2, 1/2, 1. Envelope at recall <= 1/2 is 1.
    # recall points up to 0.5 (51 of them) score 1, the 50 above score 2/3.
    assert ap([True, False, True], 2) == pytest.approx((51 * 1 + 50 * (2 / 3)) / 101)


def test_zero_weight_detections_change_nothing() -> None:
    assert ap([False, True, False], 1, [0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_unit_weights_equal_plain_and_integer_weights_equal_repetition() -> None:
    tp = [True, False, True, False]
    assert ap(tp, 3, [1, 1, 1, 1]) == ap(tp, 3)
    # weight 2 on the first two detections = those two detections twice (as a second scene).
    # (Rows of a repeated scene sit at the same scores, so interleaving by score is the same.)
    assert ap([True, True, False, False], 2, [2, 0, 0, 0]) == ap([True, True], 2, [1, 1])


# --- agreement with the COCO reference implementation ----------------------------------------


def test_ap_matches_pycocotools_on_random_data() -> None:
    """Boxes here have positive size, like decoded ones (`centre +- exp(.)/2`). COCO silently
    ignores a box with negative area (bottom above top); ours would count it as a false positive.
    That cannot happen to a decoded box, so the test data must not produce one either."""
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(0)
    thresholds = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
    frames, images, gt_anns, dt_anns = [], [], [], []
    for image_id in range(1, 13):
        gts, dets = [], []
        for _ in range(int(rng.integers(1, 6))):
            x, y = rng.uniform(0, 80, 2)
            w, h = rng.uniform(15, 40, 2)
            cls = int(rng.integers(0, 2))
            gts.append(([x, y, x + w, y + h], cls, 10.0))
            if rng.random() < 0.8:  # a detection near the label, sometimes badly off
                jitter = np.clip(
                    rng.normal(0, 4, 4), -6, 6
                )  # keeps width and height positive (see below)
                box = [x + jitter[0], y + jitter[1], x + w + jitter[2], y + h + jitter[3]]
                dets.append((box, float(rng.uniform(0.05, 1)), cls, 10.0))
        for _ in range(int(rng.integers(0, 4))):  # clutter
            x, y = rng.uniform(0, 80, 2)
            dets.append(
                ([x, y, x + 20, y + 20], float(rng.uniform(0.05, 1)), int(rng.integers(0, 2)), 5.0)
            )
        frames.append(frame(f"s{image_id % 3}", dets=dets, gts=gts, token=str(image_id)))
        images.append({"id": image_id, "width": 200, "height": 200})
        for box, cls, _ in gts:
            area = (box[2] - box[0]) * (box[3] - box[1])
            gt_anns.append(
                {
                    "id": len(gt_anns) + 1,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "area": area,
                    "iscrowd": 0,
                }
            )
        for box, score, cls, _ in dets:
            dt_anns.append(
                {
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "score": score,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                }
            )

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": gt_anns,
        "categories": [{"id": 1}, {"id": 2}],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dt_anns)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = np.array(thresholds)
    evaluator.params.areaRng = [[0, 1e10]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.params.maxDets = [100]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # (T, R, K, A, M)

    data = build_cell_data(
        frames, iou_thresholds=thresholds, num_classes=2, distance_score_threshold=0.3
    )
    for k in range(2):
        mine = average_precision(
            data.classes[k].tp,
            np.ones(len(data.classes[k].tp)),
            float(data.classes[k].n_gt_by_scene.sum()),
        )
        for t in range(len(thresholds)):
            curve = precision[t, :, k, 0, 0]
            reference = float(np.mean(np.where(curve > -1, curve, 0)))
            assert mine[t] == pytest.approx(reference, abs=1e-9), (k, t)


# --- weighted median -------------------------------------------------------------------------


@pytest.mark.parametrize("values", [[1.0, 2.0, 7.0], [1.0, 2.0, 7.0, 9.0], [4.0]])
def test_weighted_median_with_unit_weights_is_the_median(values: list[float]) -> None:
    v = np.array(values)
    assert weighted_median(v, np.ones(len(v))) == pytest.approx(float(np.median(v)))


def test_weighted_median_follows_the_weights() -> None:
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([5.0, 1.0, 1.0])) == 1.0
    assert math.isnan(weighted_median(np.array([1.0]), np.array([0.0])))
    assert math.isnan(weighted_median(np.zeros(0), np.zeros(0)))


# --- cells -----------------------------------------------------------------------------------


def perfect_frames() -> list[FrameResult]:
    """Two scenes, a car and a pedestrian each, found exactly with the right distance."""
    return [
        frame(
            scene,
            dets=[(BOX, 0.9, 0, 10.0), ([20, 20, 30, 40], 0.8, 1, 25.0)],
            gts=[(BOX, 0, 10.0), ([20, 20, 30, 40], 1, 25.0)],
            token=scene,
        )
        for scene in ("a", "b")
    ]


def test_perfect_detections_score_one_with_no_distance_error() -> None:
    data = build_cell_data(
        perfect_frames(), iou_thresholds=THR, num_classes=2, distance_score_threshold=0.3
    )
    m = cell_metrics(data, np.ones(2), eligible=[0, 1])
    assert m.map50 == pytest.approx(1.0) and m.map == pytest.approx(1.0)
    assert m.dist_rel_median == 0.0 and m.dist_abs_median == 0.0
    assert data.n_gt.tolist() == [2.0, 2.0]
    assert data.scene_names == ("a", "b") and data.frames_by_scene.tolist() == [1, 1]


def test_ineligible_classes_are_left_out_of_the_mean_not_counted_as_zero() -> None:
    frames = perfect_frames()
    data = build_cell_data(frames, iou_thresholds=THR, num_classes=2, distance_score_threshold=0.3)
    assert eligible_classes(data, min_objects=2) == [0, 1]
    assert eligible_classes(data, min_objects=3) == []
    only_cars = cell_metrics(data, np.ones(2), eligible=[0])
    assert math.isnan(only_cars.ap50[1]) and only_cars.map50 == pytest.approx(1.0)
    none = cell_metrics(data, np.ones(2), eligible=[])
    assert math.isnan(none.map50)


def test_distance_error_uses_only_matched_confident_detections() -> None:
    frames = [
        frame("a", dets=[(BOX, 0.9, 0, 12.0)], gts=[(BOX, 0, 10.0)], token="1"),  # 20% error
        frame("a", dets=[(BOX, 0.2, 0, 50.0)], gts=[(BOX, 0, 10.0)], token="2"),  # not confident
        frame(
            "a", dets=[([50, 50, 60, 60], 0.9, 0, 99.0)], gts=[(BOX, 0, 10.0)], token="3"
        ),  # miss
    ]
    data = build_cell_data(frames, iou_thresholds=THR, num_classes=2, distance_score_threshold=0.3)
    m = cell_metrics(data, np.ones(1), eligible=[0])
    assert m.dist_rel_median == pytest.approx(0.2) and m.dist_abs_median == pytest.approx(2.0)


def test_distance_by_range_bins_by_true_distance() -> None:
    frames = [
        frame("a", dets=[(BOX, 0.9, 0, 11.0)], gts=[(BOX, 0, 10.0)], token="1"),
        frame("a", dets=[(BOX, 0.9, 0, 66.0)], gts=[(BOX, 0, 60.0)], token="2"),
        frame("a", dets=[(BOX, 0.9, 0, 90.0)], gts=[(BOX, 0, 100.0)], token="3"),
    ]
    data = build_cell_data(frames, iou_thresholds=THR, num_classes=2, distance_score_threshold=0.3)
    rows = distance_by_range(data, [0.0, 20.0, 40.0, math.inf])
    assert [int(r["matched"]) for r in rows] == [1, 0, 2]
    assert rows[0]["median_rel_error"] == pytest.approx(0.1)
    assert math.isnan(rows[1]["median_rel_error"])
    assert rows[2]["median_rel_error"] == pytest.approx(0.1)


def test_scene_weights_equal_repeating_the_scene() -> None:
    """The identity the bootstrap rests on: weight k on a scene = that scene included k times."""
    a = [frame("a", dets=[(BOX, 0.9, 0, 12.0)], gts=[(BOX, 0, 10.0)], token="1")]
    b = [
        frame("b", dets=[(BOX, 0.7, 0, 9.0)], gts=[(BOX, 0, 10.0)], token="2"),
        frame("b", dets=[([50, 50, 60, 60], 0.95, 0, 9.0)], gts=[(BOX, 0, 10.0)], token="3"),
    ]
    kwargs = {"iou_thresholds": THR, "num_classes": 2, "distance_score_threshold": 0.3}
    weighted = cell_metrics(build_cell_data(a + b, **kwargs), np.array([3.0, 1.0]), [0])  # type: ignore[arg-type]
    copies = [
        frame(f"a{i}", dets=[(BOX, 0.9, 0, 12.0)], gts=[(BOX, 0, 10.0)], token=f"c{i}")
        for i in range(3)
    ]
    repeated = cell_metrics(build_cell_data(copies + b, **kwargs), np.ones(4), [0])  # type: ignore[arg-type]
    assert weighted.ap50[0] == pytest.approx(repeated.ap50[0])
    assert weighted.map == pytest.approx(repeated.map)
    assert weighted.dist_rel_median == pytest.approx(repeated.dist_rel_median)


# --- bootstrap -------------------------------------------------------------------------------


def mixed_cell(n_scenes: int = 6, seed: int = 0) -> list[FrameResult]:
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_scenes):
        for k in range(3):
            hit = rng.random() < 0.7
            dets = [
                (
                    BOX if hit else [40, 40, 50, 50],
                    float(rng.uniform(0.3, 1)),
                    0,
                    float(10 + rng.normal(0, 1)),
                )
            ]
            frames.append(frame(f"s{s}", dets=dets, gts=[(BOX, 0, 10.0)], token=f"{s}-{k}"))
    return frames


def cell(frames: list[FrameResult]):  # type: ignore[no-untyped-def]
    return build_cell_data(frames, iou_thresholds=THR, num_classes=2, distance_score_threshold=0.3)


def test_bootstrap_is_reproducible_and_depends_on_the_seed() -> None:
    data = {"m": cell(mixed_cell())}
    one = bootstrap_samples(data, [0], CLASSES, n_boot=30, seed=1)["m"]["map50"]
    again = bootstrap_samples(data, [0], CLASSES, n_boot=30, seed=1)["m"]["map50"]
    other = bootstrap_samples(data, [0], CLASSES, n_boot=30, seed=2)["m"]["map50"]
    assert np.array_equal(one, again) and not np.array_equal(one, other)


def test_the_interval_brackets_the_point_estimate_and_widens_with_fewer_scenes() -> None:
    many, few = cell(mixed_cell(12)), cell(mixed_cell(3))
    point = cell_metrics(many, np.ones(12), [0]).map50
    low, high = percentile_interval(
        bootstrap_samples({"m": many}, [0], CLASSES, n_boot=200, seed=0)["m"]["map50"]
    )
    assert low <= point <= high
    l2, h2 = percentile_interval(
        bootstrap_samples({"m": few}, [0], CLASSES, n_boot=200, seed=0)["m"]["map50"]
    )
    assert (h2 - l2) > (high - low)


def test_two_identical_models_have_a_difference_of_exactly_zero() -> None:
    data = cell(mixed_cell())
    samples = bootstrap_samples({"a": data, "b": data}, [0], CLASSES, n_boot=20, seed=0)
    assert np.array_equal(samples["a"]["map50"], samples["b"]["map50"])


def test_models_on_different_scenes_cannot_be_bootstrapped_together() -> None:
    with pytest.raises(ValueError, match="same scenes"):
        bootstrap_samples(
            {"a": cell(mixed_cell(3)), "b": cell(mixed_cell(4))}, [0], CLASSES, n_boot=2, seed=0
        )


def test_interval_ignores_nan_and_handles_all_nan() -> None:
    assert percentile_interval(np.array([1.0, np.nan, 3.0]), 0.5) == pytest.approx((1.5, 2.5))
    low, high = percentile_interval(np.array([np.nan, np.nan]))
    assert math.isnan(low) and math.isnan(high)


def test_metrics_to_dict_names_every_class() -> None:
    m = cell_metrics(cell(mixed_cell(2)), np.ones(2), [0])
    flat = metrics_to_dict(m, CLASSES)
    assert {"map50", "map", "dist_rel_median", "ap50_car", "ap50_pedestrian", "ap_car"} <= set(flat)
    assert math.isnan(flat["ap50_pedestrian"])  # not eligible
