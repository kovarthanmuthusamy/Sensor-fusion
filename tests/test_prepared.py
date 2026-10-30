import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.labels import CLASS_NAMES
from sensor_fusion.data.prepared import PreparedDataset, collate_prepared
from sensor_fusion.data.targets import decode_outputs

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "dataset" / "v1.0-mini"
pytestmark = pytest.mark.skipif(
    not DATASET.exists(), reason="nuScenes-mini not found at dataset/v1.0-mini"
)
INPUT_H, INPUT_W, STRIDE = 448, 800, 4


@pytest.fixture(scope="module")
def prep() -> DictConfig:
    config = OmegaConf.load(ROOT / "configs" / "prep" / "default.yaml")
    assert isinstance(config, DictConfig)
    return config


@pytest.fixture(scope="module")
def base() -> NuScenesFrontDataset:
    return NuScenesFrontDataset(str(DATASET), "v1.0-mini", "mini_dev_val")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


# --- shapes and the fair-comparison guarantee --------------------------------------------------


@pytest.mark.parametrize("in_channels", [3, 4])
def test_item_shapes(base: NuScenesFrontDataset, prep: DictConfig, in_channels: int) -> None:
    item = PreparedDataset(base, in_channels=in_channels, prep=prep, train=False)[0]
    grid = (INPUT_H // STRIDE, INPUT_W // STRIDE)
    assert item["input"].shape == (in_channels, INPUT_H, INPUT_W)
    assert item["heatmap"].shape == (len(CLASS_NAMES), *grid)
    assert item["regression"].shape == (5, *grid)
    assert item["mask"].shape == grid
    assert len(item["boxes"]) == len(item["classes"]) == len(item["distances"])
    assert int(item["mask"].sum()) <= len(item["boxes"])  # shared centre cells can only reduce it


def test_baseline_and_fusion_get_the_same_photo_and_the_same_targets(
    base: NuScenesFrontDataset, prep: DictConfig
) -> None:
    rgb = PreparedDataset(base, in_channels=3, prep=prep, train=False)[5]
    fusion = PreparedDataset(base, in_channels=4, prep=prep, train=False)[5]
    assert torch.equal(fusion["input"][:3], rgb["input"])
    for key in ("heatmap", "regression", "mask", "boxes", "classes", "distances"):
        assert torch.equal(fusion[key], rgb[key]), key
    assert fusion["input"][3].max() > 0  # the extra channel really carries LiDAR depth


# --- mirroring ---------------------------------------------------------------------------------


def test_evaluation_never_mirrors_and_is_deterministic(
    base: NuScenesFrontDataset, prep: DictConfig
) -> None:
    always = OmegaConf.merge(prep, {"flip_prob": 1.0})
    dataset = PreparedDataset(base, in_channels=3, prep=always, train=False)
    assert torch.equal(dataset[2]["input"], dataset[2]["input"])
    plain = PreparedDataset(base, in_channels=3, prep=prep, train=False)[2]
    assert torch.equal(dataset[2]["input"], plain["input"])  # flip_prob is ignored when train=False


def test_training_mirrors_photo_and_boxes_together_when_the_probability_is_one(
    base: NuScenesFrontDataset, prep: DictConfig
) -> None:
    always = OmegaConf.merge(prep, {"flip_prob": 1.0})
    mirrored = PreparedDataset(base, in_channels=4, prep=always, train=True)[2]
    plain = PreparedDataset(base, in_channels=4, prep=prep, train=False)[2]
    assert torch.equal(mirrored["input"], plain["input"].flip(-1))
    np.testing.assert_allclose(
        mirrored["boxes"][:, [0, 2]].numpy(), INPUT_W - plain["boxes"][:, [2, 0]].numpy()
    )
    assert torch.equal(mirrored["distances"], plain["distances"])


def test_training_with_probability_zero_never_mirrors(
    base: NuScenesFrontDataset, prep: DictConfig
) -> None:
    never = OmegaConf.merge(prep, {"flip_prob": 0.0})
    train = PreparedDataset(base, in_channels=3, prep=never, train=True)[2]
    plain = PreparedDataset(base, in_channels=3, prep=prep, train=False)[2]
    assert torch.equal(train["input"], plain["input"])


# --- the labels survive painting and decoding ----------------------------------------------------


def test_every_object_that_does_not_share_a_cell_survives_painting_and_decoding(
    base: NuScenesFrontDataset, prep: DictConfig
) -> None:
    # Measured on 162 mini_train frames: 991 of 993 objects come back (99.8%), and the 2 lost ones
    # shared a centre cell with another object. Here: all 80 dev-validation frames, day and night.
    dataset = PreparedDataset(base, in_channels=3, prep=prep, train=False)
    total = recovered = extra = collisions = 0
    for index in range(len(dataset)):
        item = dataset[index]
        found = decode_outputs(
            item["heatmap"], item["regression"], stride=STRIDE, top_k=100, score_threshold=0.5
        )
        boxes, centres = item["boxes"].numpy(), set()
        for box in boxes:
            cell = (int((box[0] + box[2]) / (2 * STRIDE)), int((box[1] + box[3]) / (2 * STRIDE)))
            collisions += cell in centres
            centres.add(cell)
        used: set[int] = set()
        for box, class_id, distance in zip(
            boxes, item["classes"].numpy(), item["distances"].numpy(), strict=True
        ):
            for j in range(len(found.boxes)):
                if (
                    j not in used
                    and int(found.classes[j]) == class_id
                    and _iou(box, found.boxes[j].numpy()) > 0.99
                    and abs(float(found.distances[j]) / distance - 1) < 1e-3
                ):
                    used.add(j)
                    recovered += 1
                    break
        total += len(boxes)
        extra += len(found.boxes) - len(used)
    assert total > 300
    assert recovered >= total - collisions  # only colliding objects may be lost
    assert extra == 0  # and the decoder invents nothing


# --- batching ----------------------------------------------------------------------------------


def test_batches_stack_the_grids_and_keep_the_label_lists(
    base: NuScenesFrontDataset, prep: DictConfig
) -> None:
    dataset = PreparedDataset(base, in_channels=4, prep=prep, train=True)
    batch = next(
        iter(DataLoader(Subset(dataset, range(4)), batch_size=4, collate_fn=collate_prepared))
    )
    assert batch["input"].shape == (4, 4, INPUT_H, INPUT_W)
    assert batch["heatmap"].shape == (4, len(CLASS_NAMES), INPUT_H // STRIDE, INPUT_W // STRIDE)
    assert batch["mask"].shape == (4, INPUT_H // STRIDE, INPUT_W // STRIDE)
    assert batch["is_night"].dtype == torch.bool and batch["is_night"].shape == (4,)
    assert len(batch["boxes"]) == len(batch["sample_token"]) == 4  # lists, one entry per frame


def test_batches_come_through_loader_workers(base: NuScenesFrontDataset, prep: DictConfig) -> None:
    # Windows starts workers by pickling the dataset: fails if it can't be pickled or reloaded.
    dataset = PreparedDataset(base, in_channels=3, prep=prep, train=False)
    loader = DataLoader(
        Subset(dataset, range(4)), batch_size=2, num_workers=2, collate_fn=collate_prepared
    )
    batches = list(loader)
    assert [b["input"].shape[0] for b in batches] == [2, 2]


def test_prepared_dataset_pickles_small(base: NuScenesFrontDataset, prep: DictConfig) -> None:
    dataset = PreparedDataset(base, in_channels=3, prep=prep, train=True)
    assert len(pickle.dumps(dataset)) < 100_000
