import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.labels import CLASS_NAMES

DATASET = Path(__file__).parents[1] / "dataset" / "v1.0-mini"
pytestmark = pytest.mark.skipif(
    not DATASET.exists(), reason="nuScenes-mini not found at dataset/v1.0-mini"
)


def _scene_table() -> dict[str, dict[str, Any]]:
    records = json.loads((DATASET / "v1.0-mini" / "scene.json").read_text(encoding="utf-8"))
    return {record["name"]: record for record in records}


@pytest.fixture(scope="module")
def dev_val() -> NuScenesFrontDataset:
    # scene-0103 (day) + scene-1100 (night): small, and it has both tags.
    return NuScenesFrontDataset(str(DATASET), "v1.0-mini", "mini_dev_val")


def test_length_is_the_number_of_keyframes_of_the_split_scenes(
    dev_val: NuScenesFrontDataset,
) -> None:
    scenes = _scene_table()
    assert len(dev_val) == scenes["scene-0103"]["nbr_samples"] + scenes["scene-1100"]["nbr_samples"]


def test_item_has_the_documented_keys_dtypes_and_shapes(dev_val: NuScenesFrontDataset) -> None:
    item = dev_val[0]
    assert set(item) == {
        "image",
        "depth",
        "boxes",
        "classes",
        "distances",
        "is_night",
        "sample_token",
    }
    assert item["image"].shape == (900, 1600, 3) and item["image"].dtype == np.uint8
    assert item["depth"].shape == (900, 1600) and item["depth"].dtype == np.float32
    assert item["boxes"].dtype == np.float32 and item["boxes"].shape[1:] == (4,)
    assert item["classes"].dtype == np.int64 and item["distances"].dtype == np.float32
    assert len(item["boxes"]) == len(item["classes"]) == len(item["distances"])
    assert isinstance(item["is_night"], bool) and isinstance(item["sample_token"], str)


def test_depth_image_is_sparse_and_non_negative(dev_val: NuScenesFrontDataset) -> None:
    depth = dev_val[0]["depth"]
    assert (depth >= 0).all()
    assert 0 < np.count_nonzero(depth) / depth.size < 0.01  # a few thousand pixels of 1.44 million


def test_night_flag_matches_the_scene(dev_val: NuScenesFrontDataset) -> None:
    night_frames = sum(dev_val[i]["is_night"] for i in range(len(dev_val)))
    assert night_frames == _scene_table()["scene-1100"]["nbr_samples"]


def test_items_are_deterministic(dev_val: NuScenesFrontDataset) -> None:
    first, second = dev_val[3], dev_val[3]
    assert first["sample_token"] == second["sample_token"]
    np.testing.assert_array_equal(first["depth"], second["depth"])
    np.testing.assert_array_equal(first["boxes"], second["boxes"])


def test_lidar_depth_inside_boxes_agrees_with_the_distance_labels(
    dev_val: NuScenesFrontDataset,
) -> None:
    # Cross-check of two independent pipeline stages (projection and labels): the LiDAR points
    # in the middle of a box should lie about at the box's distance. A LiDAR hits the near
    # surface, so for a car it reads about half a car length *below* the centre distance
    # (measured: median -1.8 m for cars, about 0 m for pedestrians; 91% within [-5, +2] m).
    # A frame or timestamp bug would break this agreement.
    differences = []
    for index in range(len(dev_val)):
        item = dev_val[index]
        for box, distance in zip(item["boxes"], item["distances"], strict=True):
            x1, y1, x2, y2 = box
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            window = item["depth"][
                int(cy - h / 4) : int(cy + h / 4) + 1, int(cx - w / 4) : int(cx + w / 4) + 1
            ]
            values = window[window > 0]
            if len(values) >= 5:
                differences.append(float(np.median(values)) - float(distance))
    assert len(differences) >= 100
    agree = np.mean([-5.0 <= d <= 2.0 for d in differences])
    assert agree >= 0.8, f"only {agree:.0%} of boxes agree with their LiDAR depth"


def test_class_ids_stay_inside_the_class_list(dev_val: NuScenesFrontDataset) -> None:
    for index in range(0, len(dev_val), 10):
        classes = dev_val[index]["classes"]
        assert ((classes >= 0) & (classes < len(CLASS_NAMES))).all()


def test_pickled_copy_is_small_and_still_works(dev_val: NuScenesFrontDataset) -> None:
    # DataLoader workers receive a pickled dataset. It must not drag the devkit tables along.
    payload = pickle.dumps(dev_val)
    assert len(payload) < 100_000
    restored = pickle.loads(payload)
    assert len(restored) == len(dev_val)
    assert restored[0]["sample_token"] == dev_val[0]["sample_token"]
    np.testing.assert_array_equal(restored[0]["depth"], dev_val[0]["depth"])


def test_split_from_another_data_version_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"not in v1\.0-mini"):
        NuScenesFrontDataset(str(DATASET), "v1.0-mini", "train")  # the 700 trainval scenes


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        NuScenesFrontDataset(str(DATASET), "v1.0-mini", "validation")


def test_image_size_that_disagrees_with_its_record_is_rejected(
    dev_val: NuScenesFrontDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Labels are computed from the record's size, the depth image from the real image: if they
    # ever differ, boxes and depth would silently live on different pixel grids.
    nusc = dev_val.nusc
    real_get = nusc.get

    def get_with_wrong_height(table: str, token: str) -> dict[str, Any]:
        record = real_get(table, token)
        if table == "sample_data" and record["sensor_modality"] == "camera":
            return {**record, "height": 450}
        return record

    monkeypatch.setattr(nusc, "get", get_with_wrong_height)
    with pytest.raises(ValueError, match="sample_data says"):
        dev_val[0]


def test_index_out_of_range_raises_index_error(dev_val: NuScenesFrontDataset) -> None:
    with pytest.raises(IndexError):
        dev_val[len(dev_val)]
