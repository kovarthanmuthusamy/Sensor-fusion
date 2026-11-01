import math

import numpy as np
import pytest
import torch
from torch import nn

from sensor_fusion.data.targets import REGRESSION_CHANNELS, decode_outputs, encode_targets
from sensor_fusion.models.detector import Detector, _zero_conv_like, pool_depth
from sensor_fusion.models.losses import detection_loss

NUM_CLASSES, STRIDE = 4, 4
SIZE = (64, 128)  # a 16 x 32 output grid


def _model(in_channels: int = 3, **kwargs: float) -> Detector:
    return Detector(in_channels=in_channels, num_classes=NUM_CLASSES, pretrained=False, **kwargs)


# --- shapes and input checks -----------------------------------------------------------------


@pytest.mark.parametrize("in_channels", [3, 4, 5])
def test_output_grids_are_a_quarter_of_the_input(in_channels: int) -> None:
    out = _model(in_channels).eval()(torch.zeros(2, in_channels, *SIZE))
    assert out.class_logits.shape == (2, NUM_CLASSES, 16, 32)
    assert out.regression.shape == (2, len(REGRESSION_CHANNELS), 16, 32)


def test_real_input_size_gives_the_documented_grid() -> None:
    # 448 x 800 is the prepared photo size (ADR 0006): the grid must be 112 x 200.
    out = _model().eval()(torch.zeros(1, 3, 448, 800))
    assert out.class_logits.shape[-2:] == (112, 200)


def test_size_not_a_multiple_of_32_is_refused() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        _model()(torch.zeros(1, 3, 64, 100))


def test_unsupported_backbone_and_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError, match="resnet18"):
        Detector(in_channels=3, num_classes=4, backbone="resnet50", pretrained=False)
    with pytest.raises(ValueError, match="in_channels"):
        _model(2)
    with pytest.raises(ValueError, match="prior_prob"):
        _model(prior_prob=1.5)


# --- the depth channel starts at zero ---------------------------------------------------------


def test_depth_stem_has_the_photo_stems_geometry_and_zero_weights() -> None:
    conv = nn.Conv2d(3, 8, 7, stride=2, padding=3, bias=False)
    extra = _zero_conv_like(conv, 2)
    assert extra.weight.shape == (8, 2, 7, 7)
    assert extra.weight.abs().sum() == 0
    assert (extra.stride, extra.padding, extra.kernel_size) == (
        conv.stride,
        conv.padding,
        conv.kernel_size,
    )


def test_the_photo_stem_is_the_same_whatever_the_channel_count() -> None:
    assert _model(3).conv1_extra is None
    assert "conv1_extra.weight" not in _model(3).state_dict()  # RGB checkpoints keep their keys
    assert _model(4).conv1_extra.weight.shape[1] == 1  # type: ignore[union-attr]
    assert _model(5).conv1_extra.weight.shape[1] == 2  # type: ignore[union-attr]
    assert _model(5).conv1.weight.shape[1] == 3


@pytest.mark.parametrize("in_channels", [4, 5])
def test_fusion_starts_as_the_baseline_whatever_the_extra_channels_are(in_channels: int) -> None:
    torch.manual_seed(0)
    rgb, fusion = _model(3).eval(), _model(in_channels).eval()
    missing, unexpected = fusion.load_state_dict(rgb.state_dict(), strict=False)
    assert missing == ["conv1_extra.weight"] and not unexpected

    photo = torch.randn(1, 3, *SIZE)
    extra = torch.rand(1, in_channels - 3, *SIZE)
    expected = rgb(photo)
    got = fusion(torch.cat([photo, extra], dim=1))
    assert torch.allclose(got.class_logits, expected.class_logits, atol=1e-5)
    assert torch.allclose(got.regression, expected.regression, atol=1e-5)


def test_depth_weights_get_gradients_despite_starting_at_zero() -> None:
    """If zero init froze the depth stem, fusion could never beat the baseline."""
    torch.manual_seed(0)
    model = _model(5)
    out = model(torch.randn(1, 5, *SIZE))
    (out.class_logits.sum() + out.regression.sum()).backward()
    grad = model.conv1_extra.weight.grad  # type: ignore[union-attr]
    assert grad is not None and grad.abs().sum() > 0


def test_wrong_channel_count_is_refused() -> None:
    with pytest.raises(ValueError, match="expected 4 input channels, got 3"):
        _model(4)(torch.zeros(1, 3, *SIZE))


def test_checkpoints_from_before_the_depth_stem_still_load() -> None:
    """The first fusion runs stored depth as a 4th input of `conv1`; those files must still load."""
    torch.manual_seed(0)
    old_style = _model(4).eval()
    with torch.no_grad():
        old_style.conv1_extra.weight.normal_()  # type: ignore[union-attr]
    state = old_style.state_dict()
    legacy = dict(state)
    legacy["conv1.weight"] = torch.cat([state["conv1.weight"], state["conv1_extra.weight"]], dim=1)
    del legacy["conv1_extra.weight"]

    fresh = _model(4).eval()
    fresh.load_state_dict(legacy)  # strict: no missing or unexpected keys
    x = torch.randn(1, 4, *SIZE)
    assert torch.allclose(fresh(x).regression, old_style(x).regression, atol=1e-6)


# --- initial outputs ---------------------------------------------------------------------------


def test_initial_outputs_sit_at_the_prior_and_the_median_distance() -> None:
    torch.manual_seed(0)
    out = _model(prior_prob=0.1, distance_init_m=30.0).eval()(torch.randn(1, 3, *SIZE))
    assert torch.sigmoid(out.class_logits).mean().item() == pytest.approx(0.1, abs=0.02)
    log_distance = out.regression[:, REGRESSION_CHANNELS.index("log_distance")]
    assert log_distance.mean().item() == pytest.approx(math.log(30.0), abs=0.2)


# --- the whole chain: overfit one batch, then decode -------------------------------------------


def test_one_batch_can_be_memorised_and_decoded_back() -> None:
    """Overfit-one-batch (ML hygiene rule) through encoder -> network -> loss -> decoder."""
    torch.manual_seed(0)
    boxes = np.array([[16, 16, 56, 48], [72, 20, 112, 56]], dtype=np.float32)
    classes = np.array([0, 2], dtype=np.int64)
    distances = np.array([12.0, 45.0], dtype=np.float32)
    heat, reg, mask = encode_targets(
        boxes,
        classes,
        distances,
        input_size=SIZE,
        num_classes=NUM_CLASSES,
        stride=STRIDE,
        peak_alpha=0.54,
    )
    heatmap, regression, centres = (torch.from_numpy(a)[None] for a in (heat, reg, mask))
    photo = torch.randn(1, 3, *SIZE)

    model = _model()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    weights = {"class": 1.0, "size": 1.0, "offset": 1.0, "distance": 1.0}
    first = None
    for _ in range(150):
        out = model(photo)
        loss = detection_loss(
            out, heatmap, regression, centres, focal_alpha=2.0, focal_beta=4.0, weights=weights
        ).total
        first = loss.item() if first is None else first
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert loss.item() < 0.1 * first

    model.eval()
    with torch.no_grad():
        out = model(photo)
    found = decode_outputs(
        torch.sigmoid(out.class_logits[0]),
        out.regression[0],
        stride=STRIDE,
        top_k=10,
        score_threshold=0.3,
    )
    assert sorted(found.classes.tolist()) == [0, 2]
    order = found.classes.argsort()
    assert np.allclose(found.boxes[order].numpy(), boxes, atol=6.0)
    assert np.allclose(found.distances[order].numpy(), distances, rtol=0.15)


# --- the depth tower into the heads ------------------------------------------------------------


def _head_model(in_channels: int = 5) -> Detector:
    return Detector(
        in_channels=in_channels, num_classes=NUM_CLASSES, pretrained=False, depth_head=True
    )


def test_pooled_depth_statistics_are_hand_computable() -> None:
    depth = torch.zeros(1, 1, 4, 8)
    mask = torch.zeros(1, 1, 4, 8)
    depth[0, 0, 0, 0], depth[0, 0, 1, 1] = 0.2, 0.4  # two LiDAR values in the left 4x4 block
    mask[0, 0, 0, 0] = mask[0, 0, 1, 1] = 1.0
    pooled = pool_depth(depth, mask, 4)  # two blocks of 4 x 4
    assert pooled.shape == (1, 3, 1, 2)
    mean, nearest, density = pooled[0, :, 0, 0]
    assert mean.item() == pytest.approx(0.3) and nearest.item() == pytest.approx(0.2)
    assert density.item() == pytest.approx(2 / 16)
    assert pooled[0, :, 0, 1].tolist() == [0.0, 0.0, 0.0]  # an empty block is zeros, not a depth


def test_depth_head_keeps_the_output_shapes_for_both_input_recipes() -> None:
    for in_channels in (4, 5):
        out = _head_model(in_channels).eval()(torch.rand(1, in_channels, *SIZE))
        assert out.class_logits.shape == (1, NUM_CLASSES, 16, 32)
        assert out.regression.shape == (1, len(REGRESSION_CHANNELS), 16, 32)


def test_depth_head_needs_a_depth_channel() -> None:
    with pytest.raises(ValueError, match="depth_head needs a depth channel"):
        Detector(in_channels=3, num_classes=NUM_CLASSES, pretrained=False, depth_head=True)


def test_the_models_without_a_depth_head_are_unchanged() -> None:
    assert _model(5).depth_tower is None
    assert not any(k.startswith("depth_tower") for k in _model(3).state_dict())
    assert _model(3).class_head[0].weight.shape[1] == 64  # old checkpoints keep their head shape


def test_depth_head_starts_blind_to_depth_then_learns_to_use_it() -> None:
    """Zero start (like the depth stem): at first nothing depends on depth; the heads' new input
    weights get gradients at once, and the tower follows once those weights are non-zero."""
    torch.manual_seed(0)
    model = _head_model().eval()
    photo = torch.randn(1, 3, *SIZE)
    a = model(torch.cat([photo, torch.rand(1, 2, *SIZE)], dim=1))
    b = model(torch.cat([photo, torch.zeros(1, 2, *SIZE)], dim=1))
    assert torch.allclose(a.regression, b.regression, atol=1e-6)  # depth changes nothing yet

    model.train()
    x = torch.cat([photo, torch.rand(1, 2, *SIZE)], dim=1)
    out = model(x)
    (out.regression.sum() + out.class_logits.sum()).backward()
    first = model.regression_head[0].weight
    assert first.grad is not None and first.grad[:, 64:].abs().sum() > 0
    with torch.no_grad():  # as if training had made the new input weights non-zero
        for head in (model.class_head, model.regression_head):
            head[0].weight[:, 64:] += 0.05
    model.zero_grad()
    out = model(x)
    (out.regression.sum() + out.class_logits.sum()).backward()
    tower = next(model.depth_tower.parameters())  # type: ignore[union-attr]
    assert tower.grad is not None and tower.grad.abs().sum() > 0
