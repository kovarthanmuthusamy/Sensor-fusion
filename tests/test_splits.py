import json
from pathlib import Path

import pytest

from sensor_fusion.data.splits import get_split_scenes, scene_is_night

DATASET = Path(__file__).parents[1] / "dataset" / "v1.0-mini"
needs_mini = pytest.mark.skipif(
    not DATASET.exists(), reason="nuScenes-mini not found at dataset/v1.0-mini"
)


def _mini_scenes() -> dict[str, dict[str, str]]:
    records = json.loads((DATASET / "v1.0-mini" / "scene.json").read_text(encoding="utf-8"))
    return {record["name"]: record for record in records}


# --- the day/night tag -----------------------------------------------------------------


def test_night_is_detected_whatever_the_case() -> None:
    assert scene_is_night({"description": "Night, big street, bus stop"})
    assert scene_is_night({"description": "parked cars at NIGHT"})


def test_a_day_description_is_not_night() -> None:
    assert not scene_is_night({"description": "Parking lot, one parked truck, sunny"})


# --- splits ------------------------------------------------------------------------------


def test_official_train_and_val_share_no_scene() -> None:
    train, val = get_split_scenes("train"), get_split_scenes("val")
    assert (len(train), len(val)) == (700, 150)
    assert not set(train) & set(val)


def test_official_mini_splits_are_8_and_2_scenes() -> None:
    assert len(get_split_scenes("mini_train")) == 8
    assert len(get_split_scenes("mini_val")) == 2


def test_dev_split_repartitions_the_ten_mini_scenes() -> None:
    dev_train, dev_val = get_split_scenes("mini_dev_train"), get_split_scenes("mini_dev_val")
    mini = set(get_split_scenes("mini_train")) | set(get_split_scenes("mini_val"))
    assert not set(dev_train) & set(dev_val)
    assert set(dev_train) | set(dev_val) == mini
    assert (len(dev_train), len(dev_val)) == (8, 2)


def test_two_mini_train_scenes_are_official_validation_scenes() -> None:
    # Why mini and trainval splits must never be mixed: training on mini_train and validating
    # on trainval `val` would put these two scenes on both sides.
    overlap = set(get_split_scenes("mini_train")) & set(get_split_scenes("val"))
    assert overlap == {"scene-0553", "scene-0796"}


def test_unknown_split_is_rejected_and_the_valid_names_are_listed() -> None:
    with pytest.raises(ValueError, match="mini_dev_val"):
        get_split_scenes("validation")


def test_splits_do_not_share_state_between_calls() -> None:
    get_split_scenes("mini_train").clear()
    assert len(get_split_scenes("mini_train")) == 8


# --- against the real scene table (needs the dataset) ---------------------------------------


@needs_mini
def test_the_three_mini_night_scenes_are_tagged_night() -> None:
    night = {name for name, scene in _mini_scenes().items() if scene_is_night(scene)}
    assert night == {"scene-1077", "scene-1094", "scene-1100"}


@needs_mini
def test_official_mini_val_has_no_night_scene() -> None:
    scenes = _mini_scenes()
    assert not any(scene_is_night(scenes[name]) for name in get_split_scenes("mini_val"))


@needs_mini
def test_dev_val_holds_out_one_day_and_one_night_scene() -> None:
    scenes = _mini_scenes()
    tags = sorted(scene_is_night(scenes[name]) for name in get_split_scenes("mini_dev_val"))
    assert tags == [False, True]


@needs_mini
@pytest.mark.parametrize("split", ["mini_train", "mini_val", "mini_dev_train", "mini_dev_val"])
def test_every_mini_split_scene_exists_in_the_mini_table(split: str) -> None:
    assert set(get_split_scenes(split)) <= set(_mini_scenes())
