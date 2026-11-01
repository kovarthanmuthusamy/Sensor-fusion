"""The side-by-side demo: picking frames, drawing, comparing, and building the page."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("gradio")

from nuscenes.nuscenes import NuScenes
from test_serve import _prep, _random_model

from sensor_fusion.serve.demo import (
    DemoFrame,
    Item,
    as_items,
    build_demo,
    compare,
    draw,
    match_summary,
    pick_frames,
)
from sensor_fusion.serve.export import export_model
from sensor_fusion.serve.inference import Detection, OnnxDetector

ROOT = Path(__file__).parents[1]
MINI = ROOT / "dataset" / "v1.0-mini"
needs_mini = pytest.mark.skipif(not MINI.exists(), reason="nuScenes-mini not found")


@pytest.fixture(scope="module")
def detectors(tmp_path_factory: pytest.TempPathFactory) -> dict[str, OnnxDetector]:
    root = tmp_path_factory.mktemp("demo")
    out = {}
    for name, channels, head in (("rgb", 3, False), ("fusion", 5, True)):
        directory = export_model(
            _random_model(channels, head, seed=0),
            root / name,
            prep=_prep(),
            in_channels=channels,
            model_name=name,
            depth_head=head,
            epoch=0,
            checkpoint="random-weights",
        )
        out[name] = OnnxDetector(directory)
    return out


@pytest.fixture(scope="module")
def nusc() -> NuScenes:
    return NuScenes("v1.0-mini", dataroot=str(MINI), verbose=False)


def _item(
    name: str, box: tuple[float, float, float, float], distance: float, score: float | None
) -> Item:
    return (name, score, box, distance)


# --- matching a frame's detections to its labels -----------------------------------------------


def test_match_summary_counts_found_objects_and_their_distance_error() -> None:
    truth = [
        _item("car", (0, 0, 10, 10), 20.0, None),
        _item("pedestrian", (50, 50, 60, 70), 10.0, None),
    ]
    found = [
        _item("car", (1, 1, 10, 10), 22.0, 0.9),  # IoU 0.81, distance 10% off
        _item("pedestrian", (200, 200, 210, 220), 10.0, 0.8),  # nowhere near the label
    ]
    hits, error = match_summary(found, truth)
    assert hits == 1 and error == pytest.approx(0.10)


def test_a_label_is_matched_at_most_once_and_only_by_the_same_class() -> None:
    truth = [_item("car", (0, 0, 10, 10), 20.0, None)]
    twice = [_item("car", (0, 0, 10, 10), 20.0, 0.9), _item("car", (0, 0, 10, 10), 21.0, 0.8)]
    assert match_summary(twice, truth)[0] == 1
    assert match_summary([_item("pedestrian", (0, 0, 10, 10), 20.0, 0.9)], truth) == (0, None)


def test_the_conversion_from_detections_keeps_every_field() -> None:
    d = Detection("car", 0.7, (1.0, 2.0, 3.0, 4.0), 12.5)
    assert as_items([d]) == [("car", 0.7, (1.0, 2.0, 3.0, 4.0), 12.5)]


# --- drawing ------------------------------------------------------------------------------------


def test_drawing_marks_the_box_and_the_title_without_changing_the_input() -> None:
    image = np.full((100, 200, 3), 50, dtype=np.uint8)
    original = image.copy()
    drawn = np.asarray(
        draw(image, [_item("car", (20, 40, 80, 90), 18.0, 0.91)], (255, 0, 0), "model")
    )
    assert drawn.shape == image.shape and np.array_equal(image, original)
    assert tuple(drawn[65, 20]) == (255, 0, 0)  # on the left edge of the box
    assert not np.array_equal(drawn[:30], image[:30])  # the title bar


# --- with real frames ---------------------------------------------------------------------------


@needs_mini
def test_frames_cover_day_and_night(nusc: NuScenes) -> None:
    frames = pick_frames(nusc, split="mini_val", per_group=2)
    assert {f.is_night for f in frames} <= {False, True} and len(frames) >= 1
    assert all(f.label.split(":")[0] in ("day", "night") for f in frames)
    dev = pick_frames(nusc, split="mini_dev_val", per_group=3)
    assert {f.is_night for f in dev} == {False, True}  # the dev split holds both


@needs_mini
def test_compare_gives_the_labels_and_one_picture_per_model(
    nusc: NuScenes, detectors: dict[str, OnnxDetector]
) -> None:
    frame = pick_frames(nusc, split="mini_dev_val", per_group=1)[-1]  # a night frame
    images, summary = compare(nusc, frame, detectors, score_threshold=0.0)
    assert len(images) == 3  # labels, rgb, fusion
    assert all(image.size == (1600, 900) for image in images)
    assert frame.label in summary and "rgb:" in summary and "fusion:" in summary


@needs_mini
def test_the_page_builds(nusc: NuScenes, detectors: dict[str, OnnxDetector]) -> None:
    frames = pick_frames(nusc, split="mini_dev_val", per_group=1)
    page = build_demo(nusc, detectors, frames, results_markdown="# results")
    assert page is not None and isinstance(frames[0], DemoFrame)
