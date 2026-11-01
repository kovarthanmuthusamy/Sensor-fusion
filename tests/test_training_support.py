"""Safeguards, optimiser and schedule, checkpoints and tracking: the small parts of training."""

import itertools
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from sensor_fusion.models.detector import Detector
from sensor_fusion.models.losses import LossTerms
from sensor_fusion.training import tracking
from sensor_fusion.training.checkpoint import load_checkpoint, save_checkpoint
from sensor_fusion.training.checks import (
    TrainingDiverged,
    assert_finite,
    assert_scenes_disjoint,
    require_pinned_gpu,
)
from sensor_fusion.training.optim import build_optimizer, build_scheduler


def _model(in_channels: int = 4) -> Detector:
    return Detector(in_channels=in_channels, num_classes=4, pretrained=False)


# --- checks ----------------------------------------------------------------------------------


def test_disjoint_scenes_pass_and_shared_scenes_are_named() -> None:
    assert_scenes_disjoint(["scene-1", "scene-2"], ["scene-3"])
    with pytest.raises(ValueError, match=r"1 scene\(s\).*scene-2"):
        assert_scenes_disjoint(["scene-1", "scene-2"], ["scene-2", "scene-3"])


def test_gpu_must_be_pinned_only_when_cuda_exists() -> None:
    require_pinned_gpu({}, cuda_available=False)
    require_pinned_gpu({"CUDA_VISIBLE_DEVICES": "1"}, cuda_available=True)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        require_pinned_gpu({}, cuda_available=True)


def test_non_finite_loss_names_the_term_and_the_batch() -> None:
    good = LossTerms(*(torch.tensor(1.0) for _ in range(5)))
    assert_finite(good, ["a"])
    bad = good._replace(distance=torch.tensor(float("nan")), total=torch.tensor(float("inf")))
    with pytest.raises(TrainingDiverged, match=r"distance.*tok-1"):
        assert_finite(bad, ["tok-1", "tok-2"])


# --- optimiser and schedule --------------------------------------------------------------------


def test_every_parameter_is_in_exactly_one_group_and_the_rates_differ() -> None:
    model = _model()
    optimizer = build_optimizer(model, lr_head=5e-4, lr_backbone=1e-4, weight_decay=1e-4)
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())
    assert len(grouped) == len(set(grouped))
    assert {group["lr"] for group in optimizer.param_groups} == {5e-4, 1e-4}


def test_first_convolution_including_depth_channel_uses_the_backbone_rate() -> None:
    model = _model()
    optimizer = build_optimizer(model, lr_head=5e-4, lr_backbone=1e-4, weight_decay=1e-4)
    for group in optimizer.param_groups:
        if any(p is model.conv1.weight for p in group["params"]):
            assert group["lr"] == 1e-4 and group["weight_decay"] == 1e-4


def test_biases_and_norm_parameters_get_no_weight_decay() -> None:
    model = _model()
    optimizer = build_optimizer(model, lr_head=5e-4, lr_backbone=1e-4, weight_decay=1e-4)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            expected = 1e-4 if parameter.ndim > 1 else 0.0
            assert group["weight_decay"] == expected


def test_schedule_warms_up_then_decays_to_the_floor() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1.0}])
    scheduler = build_scheduler(optimizer, total_steps=100, warmup_steps=10, min_lr_fraction=0.1)
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(0.1)  # (0 + 1) / warm-up steps
    assert rates[9] == pytest.approx(1.0)  # end of warm-up
    assert all(a >= b for a, b in itertools.pairwise(rates[9:]))  # then only down
    assert rates[-1] == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 89 / 90)))
    assert rates[-1] < 0.11


def test_schedule_rejects_nonsense() -> None:
    optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))])
    with pytest.raises(ValueError, match="total_steps"):
        build_scheduler(optimizer, total_steps=0, warmup_steps=0, min_lr_fraction=0.1)


# --- checkpoints -------------------------------------------------------------------------------


def _training_objects() -> tuple[Detector, Any, Any, Any]:
    model = _model(3)
    optimizer = build_optimizer(model, lr_head=5e-4, lr_backbone=1e-4, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, total_steps=10, warmup_steps=2, min_lr_fraction=0.1)
    return model, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False)


def test_checkpoint_round_trip_restores_everything(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _training_objects()
    for _ in range(3):  # give the optimiser and schedule some state
        model(torch.randn(1, 3, 64, 128)).regression.sum().backward()
        optimizer.step()
        scheduler.step()
    path = tmp_path / "run" / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=4,
        step=3,
        best_val=0.5,
        config={"seed": 7},
    )
    assert not list(path.parent.glob("*.tmp"))  # the temporary file was renamed away

    other, other_opt, other_sched, other_scaler = _training_objects()
    state = load_checkpoint(
        path, model=other, optimizer=other_opt, scheduler=other_sched, scaler=other_scaler
    )
    assert (state.epoch, state.step, state.best_val, state.config) == (4, 3, 0.5, {"seed": 7})
    for a, b in zip(model.state_dict().values(), other.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert other_sched.last_epoch == scheduler.last_epoch
    assert other_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_model_only_load_works_without_training_state(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _training_objects()
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=0,
        step=0,
        best_val=1.0,
        config={},
    )
    fresh = _model(3)
    load_checkpoint(path, model=fresh)
    assert torch.equal(fresh.conv1.weight, model.conv1.weight)


# --- tracking ----------------------------------------------------------------------------------


def test_git_info_is_text_whatever_the_outcome(tmp_path: Path) -> None:
    info = tracking.git_info(cwd=tmp_path)  # a folder that is not a repository
    assert info == {"git_commit": "unknown", "git_dirty": "unknown"}
    here = tracking.git_info()
    assert set(here) == {"git_commit", "git_dirty"}


def test_online_mode_falls_back_to_offline_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(tracking.wandb, "login", lambda **_: False)
    monkeypatch.setattr(tracking.wandb, "init", lambda **kwargs: seen.update(kwargs) or "run")
    run = tracking.start_run(
        config={"seed": 1},
        mode="online",
        project="p",
        entity=None,
        name="n",
        tags=["t"],
        out_dir=tmp_path,
    )
    assert run == "run" and seen["mode"] == "offline"
    assert seen["config"]["seed"] == 1
    assert {"git_commit", "git_dirty", "env"} <= set(seen["config"])  # provenance is attached


def test_disabled_mode_never_logs_in_and_returns_a_usable_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(**_: Any) -> bool:
        raise AssertionError("must not log in when disabled")

    monkeypatch.setattr(tracking.wandb, "login", fail)
    run = tracking.start_run(
        config={}, mode="disabled", project="p", entity=None, name="n", tags=[], out_dir=tmp_path
    )
    run.log({"x": 1.0}, step=1)  # a no-op, but the call must work
    run.finish()
