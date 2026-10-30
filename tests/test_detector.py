import math

import numpy as np
import pytest
import torch
from torch import nn

from sensor_fusion.data.targets import REGRESSION_CHANNELS, decode_outputs, encode_targets
from sensor_fusion.models.detector import Detector, _expand_first_conv
from sensor_fusion.models.losses import detection_loss

NUM_CLASSES, STRIDE = 4, 4
SIZE = (64, 128)  # a 16 x 32 output grid


def _model(in_channels: int = 3, **kwargs: float) -> Detector:
    return Detector(in_channels=in_channels, num_classes=NUM_CLASSES, pretrained=False, **kwargs)


# --- shapes and input checks -----------------------------------------------------------------


@pytest.mark.parametrize("in_channels", [3, 4])
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


def test_expanded_first_conv_keeps_rgb_filters_and_zeroes_the_rest() -> None:
    conv = nn.Conv2d(3, 8, 7, stride=2, padding=3, bias=False)
    wide = _expand_first_conv(conv, 4)
    assert wide.weight.shape == (8, 4, 7, 7)
    assert torch.equal(wide.weight[:, :3], conv.weight)
    assert wide.weight[:, 3:].abs().sum() == 0
    assert (wide.stride, wide.padding) == (conv.stride, conv.padding)
    with pytest.raises(ValueError, match="shrink"):
        _expand_first_conv(wide, 3)


def test_fusion_starts_as_the_baseline_whatever_the_depth_is() -> None:
    torch.manual_seed(0)
    rgb, fusion = _model(3).eval(), _model(4).eval()
    state = rgb.state_dict()
    state["conv1.weight"] = fusion.conv1.weight.detach().clone()
    state["conv1.weight"][:, :3] = rgb.conv1.weight.detach()
    fusion.load_state_dict(state)

    photo = torch.randn(1, 3, *SIZE)
    depth = torch.rand(1, 1, *SIZE)
    expected = rgb(photo)
    got = fusion(torch.cat([photo, depth], dim=1))
    assert torch.allclose(got.class_logits, expected.class_logits, atol=1e-5)
    assert torch.allclose(got.regression, expected.regression, atol=1e-5)


def test_depth_weights_get_gradients_despite_starting_at_zero() -> None:
    """If zero init froze the depth channel, fusion could never beat the baseline."""
    torch.manual_seed(0)
    model = _model(4)
    out = model(torch.randn(1, 4, *SIZE))
    (out.class_logits.sum() + out.regression.sum()).backward()
    grad = model.conv1.weight.grad
    assert grad is not None and grad[:, 3:].abs().sum() > 0


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
