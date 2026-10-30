"""The training loop on tiny synthetic data (CPU): fit, resume, budgets, validation, divergence."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sensor_fusion.data.prepared import collate_prepared
from sensor_fusion.data.targets import encode_targets
from sensor_fusion.models.detector import Detector, Outputs
from sensor_fusion.training.checks import TrainingDiverged
from sensor_fusion.training.loop import (
    LoopSettings,
    LossSettings,
    fit,
    make_scaler,
    overfit_one_batch,
    train_epoch,
    validate,
)
from sensor_fusion.training.optim import build_optimizer, build_scheduler

SIZE = (64, 128)  # a 16 x 32 output grid
LOSS = LossSettings(2.0, 4.0, {"class": 1.0, "size": 1.0, "offset": 1.0, "distance": 1.0})
CPU = torch.device("cpu")


class FakeFrames(Dataset[dict[str, Any]]):
    """Frames with two well-separated boxes each; every `night_every`-th frame is a night frame."""

    def __init__(self, count: int, night_every: int = 2, all_day: bool = False) -> None:
        self.count, self.night_every, self.all_day = count, night_every, all_day

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(index)
        boxes = np.array([[16, 16, 56, 48], [72, 20, 112, 56]], dtype=np.float32)
        classes = np.array([0, 2], dtype=np.int64)
        distances = np.array([12.0, 45.0], dtype=np.float32)
        heat, reg, mask = encode_targets(
            boxes, classes, distances, input_size=SIZE, num_classes=4, stride=4, peak_alpha=0.54
        )
        return {
            "input": torch.randn(3, *SIZE, generator=generator),
            "heatmap": torch.from_numpy(heat),
            "regression": torch.from_numpy(reg),
            "mask": torch.from_numpy(mask),
            "boxes": torch.from_numpy(boxes),
            "classes": torch.from_numpy(classes),
            "distances": torch.from_numpy(distances),
            "is_night": not self.all_day and index % self.night_every == 0,
            "sample_token": f"frame-{index}",
            "crop_top": 0,
            "crop_left": 0,
        }


class Recorder:
    def __init__(self) -> None:
        self.records: list[tuple[int | None, dict[str, Any]]] = []

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.records.append((step, data))

    def keys(self) -> set[str]:
        return {key for _, data in self.records for key in data}


def _loader(count: int = 4, batch_size: int = 2, night_every: int = 2) -> DataLoader[Any]:
    return DataLoader(
        FakeFrames(count, night_every), batch_size=batch_size, collate_fn=collate_prepared
    )


def _setup(total_steps: int = 8) -> tuple[nn.Module, Any, Any]:
    model = Detector(in_channels=3, num_classes=4, pretrained=False)
    optimizer = build_optimizer(model, lr_head=2e-3, lr_backbone=1e-3, weight_decay=1e-4)
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=1, min_lr_fraction=0.1
    )
    return model, optimizer, scheduler


def _settings(epochs: int = 2, **overrides: Any) -> LoopSettings:
    return LoopSettings(epochs=epochs, grad_clip=10.0, log_every=1, amp=True, **overrides)


def _fit(tmp_path: Path, run: Recorder, settings: LoopSettings, **kwargs: Any) -> Any:
    model, optimizer, scheduler = _setup()
    return fit(
        model,
        _loader(),
        _loader(),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=make_scaler(CPU, True),
        device=CPU,
        loss=LOSS,
        settings=settings,
        run=run,
        out_dir=tmp_path,
        config={"seed": 1},
        **kwargs,
    )


# --- fit ---------------------------------------------------------------------------------------


def test_fit_trains_validates_logs_and_saves(tmp_path: Path) -> None:
    run = Recorder()
    result = _fit(tmp_path, run, _settings(2))
    assert (result.epochs_run, result.step, result.stopped_early) == (2, 4, False)
    assert np.isfinite(result.best_val)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    assert {"train/total", "train/grad_norm", "train/lr_head", "val/all/total"} <= run.keys()
    assert {"val/day/total", "val/night/total", "val/night/objects"} <= run.keys()
    # The learning rates logged are the two different ones (head last, backbone first).
    first = next(data for _, data in run.records if "train/lr_head" in data)
    assert first["train/lr_head"] > first["train/lr_backbone"]


def test_resume_continues_the_step_count_and_skips_finished_epochs(tmp_path: Path) -> None:
    _fit(tmp_path, Recorder(), _settings(1))
    result = _fit(tmp_path, Recorder(), _settings(2), resume=tmp_path / "last.pt")
    assert (result.epochs_run, result.step) == (1, 4)  # one more epoch, steps 2 -> 4


def test_time_budget_stops_after_the_current_epoch_and_says_so(tmp_path: Path) -> None:
    result = _fit(tmp_path, Recorder(), _settings(3, max_minutes=1e-9))
    assert result.epochs_run == 1 and result.stopped_early


def test_cpu_run_skips_no_steps(tmp_path: Path) -> None:
    run = Recorder()
    _fit(tmp_path, run, _settings(1))
    epoch_logs = [data for _, data in run.records if "epoch/train_skipped_steps" in data]
    assert epoch_logs[0]["epoch/train_skipped_steps"] == 0.0


class _NanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> Outputs:
        nan = x.new_full((x.shape[0], 4, 16, 32), float("nan")) * self.weight
        return Outputs(nan, nan[:, :1].expand(-1, 5, -1, -1))


def test_divergence_stops_the_run_and_saves_the_offending_batch(tmp_path: Path) -> None:
    model = _NanModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(TrainingDiverged):
        train_epoch(
            model,
            _loader(),
            optimizer=optimizer,
            scheduler=build_scheduler(optimizer, total_steps=2, warmup_steps=0, min_lr_fraction=1),
            scaler=make_scaler(CPU, True),
            device=CPU,
            loss=LOSS,
            settings=_settings(),
            run=Recorder(),
            step=0,
            out_dir=tmp_path,
        )
    saved = json.loads((tmp_path / "bad_batch.json").read_text(encoding="utf-8"))
    assert saved["sample_tokens"] == ["frame-0", "frame-1"]


def test_an_empty_loader_is_an_error_not_a_silent_no_op(tmp_path: Path) -> None:
    model, optimizer, scheduler = _setup()
    with pytest.raises(ValueError, match="no batches"):
        train_epoch(
            model,
            _loader(count=0),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=make_scaler(CPU, True),
            device=CPU,
            loss=LOSS,
            settings=_settings(),
            run=Recorder(),
            step=0,
            out_dir=tmp_path,
        )


# --- validation --------------------------------------------------------------------------------


def test_validation_reports_day_night_and_all_and_they_add_up() -> None:
    model, _, _ = _setup()
    result = validate(model, _loader(count=6, batch_size=4), device=CPU, loss=LOSS, amp=True)
    assert set(result) == {"all", "day", "night"}
    assert result["day"]["frames"] == 3 and result["night"]["frames"] == 3
    assert result["all"]["frames"] == 6 and result["all"]["objects"] == 12
    # Per-object averages: the overall total is the object-weighted mix of day and night.
    mixed = (
        result["day"]["total"] * result["day"]["objects"]
        + result["night"]["total"] * result["night"]["objects"]
    ) / result["all"]["objects"]
    assert result["all"]["total"] == pytest.approx(mixed, rel=1e-4)


def test_a_group_without_frames_is_absent_not_zero() -> None:
    model, _, _ = _setup()
    day_only = DataLoader(FakeFrames(4, all_day=True), batch_size=2, collate_fn=collate_prepared)
    result = validate(model, day_only, device=CPU, loss=LOSS, amp=True)
    assert set(result) == {"all", "day"}


# --- the one-batch overfit check -----------------------------------------------------------------


def test_one_batch_can_be_memorised() -> None:
    torch.manual_seed(0)
    model = Detector(in_channels=3, num_classes=4, pretrained=False)
    optimizer = build_optimizer(model, lr_head=3e-3, lr_backbone=3e-3, weight_decay=0.0)
    batch = next(iter(_loader(count=2, batch_size=2)))
    history = overfit_one_batch(
        model,
        batch,
        optimizer=optimizer,
        scaler=make_scaler(CPU, True),
        device=CPU,
        loss=LOSS,
        steps=120,
        amp=True,
        grad_clip=10.0,
    )
    assert history[-1] < 0.1 * history[0]


# --- the half-precision path (needs a GPU) -------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_fp16_autocast_and_scaler_train_without_non_finite_values(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = Detector(in_channels=3, num_classes=4, pretrained=False).to(device)
    optimizer = build_optimizer(model, lr_head=1e-3, lr_backbone=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, total_steps=8, warmup_steps=1, min_lr_fraction=0.1)
    run = Recorder()
    step, stats = train_epoch(
        model,
        _loader(count=8, batch_size=2),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=make_scaler(device, True),
        device=device,
        loss=LOSS,
        settings=_settings(),
        run=run,
        step=0,
        out_dir=tmp_path,
    )
    assert step == 4 and np.isfinite(stats["total"])
    assert all(data["train/loss_scale"] > 0 for _, data in run.records if "train/total" in data)
