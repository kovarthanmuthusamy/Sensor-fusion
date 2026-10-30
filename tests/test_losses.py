import math

import pytest
import torch

from sensor_fusion.data.targets import REGRESSION_CHANNELS
from sensor_fusion.models.detector import Outputs
from sensor_fusion.models.losses import (
    LossTerms,
    detection_loss,
    heatmap_focal_loss,
    masked_l1,
)

A, B = 2.0, 4.0  # focal exponent, penalty reduction
NUM_REG = len(REGRESSION_CHANNELS)


def _focal(logits: list[float], target: list[float]) -> float:
    shape = (1, 1, 1, len(target))
    return heatmap_focal_loss(
        torch.tensor(logits).reshape(shape), torch.tensor(target).reshape(shape), alpha=A, beta=B
    ).item()


# --- focal loss: hand-computed -----------------------------------------------------------------


def test_one_positive_at_probability_half() -> None:
    # -log(0.5) * (1 - 0.5)^2 = 0.693 * 0.25
    assert _focal([0.0], [1.0]) == pytest.approx(math.log(2) * 0.25)


def test_one_positive_and_one_far_negative() -> None:
    # Positive: 0.693 * 0.25. Negative (target 0, weight 1): -log(0.5) * 0.5^2 * 1 = the same.
    assert _focal([0.0, 0.0], [1.0, 0.0]) == pytest.approx(2 * math.log(2) * 0.25)


def test_a_negative_next_to_a_centre_is_penalised_less_than_a_far_one() -> None:
    near = _focal([0.0, 0.0], [1.0, 0.9])  # (1 - 0.9)^4 = 1e-4
    far = _focal([0.0, 0.0], [1.0, 0.0])
    assert near < far
    assert near == pytest.approx(math.log(2) * 0.25 * (1 + 1e-4), rel=1e-3)


def test_perfect_prediction_costs_almost_nothing_and_a_confident_miss_costs_a_lot() -> None:
    assert _focal([20.0, -20.0], [1.0, 0.0]) < 1e-6
    assert _focal([-20.0, 20.0], [1.0, 0.0]) > 10.0


def test_no_objects_is_finite_and_counts_only_confident_false_alarms() -> None:
    quiet = _focal([-8.0, -8.0], [0.0, 0.0])
    loud = _focal([8.0, 8.0], [0.0, 0.0])
    assert math.isfinite(quiet) and quiet < 1e-3
    assert loud > 1.0


def test_extreme_logits_stay_finite_and_half_precision_input_is_computed_in_float32() -> None:
    logits = torch.tensor([[[[1000.0, -1000.0]]]], dtype=torch.float16)
    loss = heatmap_focal_loss(logits, torch.tensor([[[[0.0, 1.0]]]]), alpha=A, beta=B)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_loss_is_normalised_by_the_number_of_objects() -> None:
    one = _focal([0.0], [1.0])
    two = heatmap_focal_loss(
        torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2), alpha=A, beta=B
    ).item()
    assert two == pytest.approx(one)  # two identical positives, divided by two


# --- masked L1 ---------------------------------------------------------------------------------


def test_masked_l1_reads_only_object_cells_and_averages_over_objects() -> None:
    pred = torch.zeros(1, NUM_REG, 2, 2)
    target = torch.zeros(1, NUM_REG, 2, 2)
    mask = torch.zeros(1, 2, 2)
    mask[0, 0, 0] = mask[0, 1, 1] = 1.0
    target[0, 4, 0, 0], target[0, 4, 1, 1] = 1.0, 3.0  # two objects, errors 1 and 3
    target[0, 4, 0, 1] = 99.0  # a non-object cell: must be ignored
    result = masked_l1(pred, target, mask)
    assert result.shape == (NUM_REG,)
    assert result[4].item() == pytest.approx(2.0)
    assert result[:4].abs().sum() == 0


def test_masked_l1_with_no_objects_is_zero_not_nan() -> None:
    result = masked_l1(
        torch.ones(1, NUM_REG, 2, 2), torch.zeros(1, NUM_REG, 2, 2), torch.zeros(1, 2, 2)
    )
    assert torch.equal(result, torch.zeros(NUM_REG))


# --- the combined loss ---------------------------------------------------------------------------


def _batch() -> tuple[Outputs, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    outputs = Outputs(torch.randn(2, 4, 3, 3), torch.randn(2, NUM_REG, 3, 3))
    heatmap = torch.zeros(2, 4, 3, 3)
    heatmap[0, 1, 1, 1] = 1.0
    mask = torch.zeros(2, 3, 3)
    mask[0, 1, 1] = 1.0
    return outputs, heatmap, torch.randn(2, NUM_REG, 3, 3), mask


def test_total_is_the_weighted_sum_of_the_terms() -> None:
    outputs, heatmap, regression, mask = _batch()
    weights = {"class": 2.0, "size": 0.5, "offset": 0.25, "distance": 3.0}
    terms = detection_loss(
        outputs, heatmap, regression, mask, focal_alpha=A, focal_beta=B, weights=weights
    )
    assert isinstance(terms, LossTerms)
    expected = 2.0 * terms.class_ + 0.5 * terms.size + 0.25 * terms.offset + 3.0 * terms.distance
    assert terms.total.item() == pytest.approx(expected.item())


def test_size_and_offset_terms_use_their_own_channels() -> None:
    outputs, heatmap, regression, mask = _batch()
    base = {"class": 0.0, "size": 0.0, "offset": 0.0, "distance": 0.0}
    for key, channel in (("size", 0), ("offset", 2), ("distance", 4)):
        perturbed = outputs.regression.clone()
        perturbed[0, channel, 1, 1] += 5.0  # the object-centre cell of image 0
        before = detection_loss(
            outputs, heatmap, regression, mask, focal_alpha=A, focal_beta=B, weights=base
        )
        after = detection_loss(
            Outputs(outputs.class_logits, perturbed),
            heatmap,
            regression,
            mask,
            focal_alpha=A,
            focal_beta=B,
            weights=base,
        )
        assert getattr(after, key) != pytest.approx(getattr(before, key))
        others = {"size", "offset", "distance"} - {key}
        for other in others:
            assert getattr(after, other).item() == pytest.approx(getattr(before, other).item())


def test_unknown_or_missing_weight_keys_are_refused() -> None:
    outputs, heatmap, regression, mask = _batch()
    with pytest.raises(ValueError, match="weights"):
        detection_loss(
            outputs,
            heatmap,
            regression,
            mask,
            focal_alpha=A,
            focal_beta=B,
            weights={"class": 1.0, "size": 1.0},
        )
