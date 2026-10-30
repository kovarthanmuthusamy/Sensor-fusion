"""The training entry point's building blocks and the end-to-end wiring on nuScenes-mini."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from sensor_fusion.train import build_loaders, build_model, run_training

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs"
needs_mini = pytest.mark.skipif(
    not (ROOT / "dataset" / "v1.0-mini").exists(), reason="nuScenes-mini not found"
)


def _cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "model.pretrained=false",
                "trainer.num_workers=0",
                "trainer.batch_size=2",
                "trainer.wandb.mode=disabled",
                "trainer.require_pinned_gpu=false",
                *overrides,
            ],
        )


@pytest.mark.parametrize(("model", "channels"), [("rgb", 3), ("fusion", 4)])
def test_model_input_channels_follow_the_config(model: str, channels: int) -> None:
    net = build_model(_cfg(f"model={model}"))
    assert net.conv1.in_channels == channels


@needs_mini
@pytest.mark.parametrize(("model", "channels"), [("rgb", 3), ("fusion", 4)])
def test_loaders_give_batches_of_the_right_shape(model: str, channels: int) -> None:
    train, val = build_loaders(_cfg(f"model={model}"), augment=True)
    batch = next(iter(train))
    assert batch["input"].shape == (2, channels, 448, 800)
    assert batch["heatmap"].shape == (2, 4, 112, 200)
    assert batch["regression"].shape == (2, 5, 112, 200)
    assert val.dataset.train is False  # type: ignore[attr-defined]  # no mirroring in validation


@needs_mini
def test_overlapping_splits_are_refused_before_any_training() -> None:
    cfg = _cfg("data.val_split=mini_dev_train")
    with pytest.raises(ValueError, match="both the train and the val split"):
        build_loaders(cfg, augment=True)


@needs_mini
def test_the_dev_split_validates_on_one_day_and_one_night_scene() -> None:
    train, val = build_loaders(_cfg(), augment=False)
    val_scenes = val.dataset.base.scene_names  # type: ignore[attr-defined]
    train_scenes = train.dataset.base.scene_names  # type: ignore[attr-defined]
    assert sorted(val_scenes) == ["scene-0103", "scene-1100"]
    assert not set(val_scenes) & set(train_scenes)


@needs_mini
@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CPU run on 448x800 frames is too slow")
def test_a_whole_run_on_mini_writes_checkpoints(tmp_path: Path) -> None:
    cfg = _cfg("trainer.epochs=1", "trainer.batch_size=8", "trainer.log_every=5")
    run_training(cfg, tmp_path)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
