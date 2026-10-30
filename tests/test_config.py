"""Config composition tests: a broken YAML or override fails in CI, not 6 hours into a run."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=list(overrides))


def test_default_is_rgb_baseline_on_mini() -> None:
    cfg = _compose()
    assert cfg.model.in_channels == 3
    assert cfg.data.version == "v1.0-mini"


def test_fusion_adds_depth_channel() -> None:
    assert _compose("model=fusion").model.in_channels == 4


@pytest.mark.parametrize("model", ["rgb", "fusion"])
@pytest.mark.parametrize("data", ["mini", "trainval"])
def test_every_combination_resolves(model: str, data: str) -> None:
    cfg = _compose(f"model={model}", f"data={data}")
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def test_preparation_settings_are_shared_by_baseline_and_fusion() -> None:
    """Fair comparison: the same shrinking, normalisation and augmentation for both variants."""
    rgb = OmegaConf.to_container(_compose("model=rgb").prep)
    fusion = OmegaConf.to_container(_compose("model=fusion").prep)
    assert rgb == fusion


def test_output_stride_divides_the_input_size_multiple() -> None:
    # The cropped input is a multiple of 32; the grid needs it to be a multiple of the stride.
    assert 32 % _compose().prep.output_stride == 0


def test_baseline_and_fusion_differ_only_in_input() -> None:
    """The fair-comparison rule as a test: if someone tunes one variant only, this fails."""
    rgb = OmegaConf.to_container(_compose("model=rgb").model)
    fusion = OmegaConf.to_container(_compose("model=fusion").model)
    assert isinstance(rgb, dict) and isinstance(fusion, dict)
    differing = {k for k in rgb.keys() | fusion.keys() if rgb.get(k) != fusion.get(k)}
    assert differing == {"name", "in_channels"}


def test_loss_settings_are_shared_and_have_all_weight_groups() -> None:
    from sensor_fusion.models.losses import WEIGHT_KEYS

    rgb = OmegaConf.to_container(_compose("model=rgb").loss)
    fusion = OmegaConf.to_container(_compose("model=fusion").loss)
    assert rgb == fusion
    assert isinstance(rgb, dict) and set(rgb["weights"]) == set(WEIGHT_KEYS)
