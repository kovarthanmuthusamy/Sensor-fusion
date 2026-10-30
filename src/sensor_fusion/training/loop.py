"""The training loop: one epoch of updates, validation split by day/night, and `fit` (ADR 0010).

Mixed precision follows the Volta rule: fp16 autocast with a gradient scaler (no bf16). Only the
network runs in half precision; the loss is computed in float32 (see `models/losses.py`).
"""

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sensor_fusion.models.detector import Outputs
from sensor_fusion.models.losses import LossTerms, detection_loss
from sensor_fusion.training.checkpoint import load_checkpoint, save_checkpoint
from sensor_fusion.training.checks import TrainingDiverged, assert_finite

log = logging.getLogger(__name__)

_TERMS = ("total", "class_", "size", "offset", "distance")


class RunLogger(Protocol):
    def log(self, data: dict[str, Any], step: int | None = None) -> Any: ...


@dataclass(frozen=True)
class LossSettings:
    focal_alpha: float
    focal_beta: float
    weights: Mapping[str, float]


@dataclass(frozen=True)
class LoopSettings:
    epochs: int
    grad_clip: float
    log_every: int
    amp: bool
    max_minutes: float | None = None


@dataclass(frozen=True)
class FitResult:
    epochs_run: int
    step: int
    best_val: float
    stopped_early: bool


def _autocast(device: torch.device, amp: bool) -> torch.autocast:
    return torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=_amp_on(device, amp)
    )


def _amp_on(device: torch.device, amp: bool) -> bool:
    return amp and device.type == "cuda"


def make_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=_amp_on(device, amp))


def _targets(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    inputs, heatmap, regression, mask = (
        batch[key].to(device, non_blocking=True)
        for key in ("input", "heatmap", "regression", "mask")
    )
    return inputs, heatmap, regression, mask


def _loss(
    outputs: Outputs, heatmap: Tensor, regression: Tensor, mask: Tensor, loss: LossSettings
) -> LossTerms:
    return detection_loss(
        outputs,
        heatmap,
        regression,
        mask,
        focal_alpha=loss.focal_alpha,
        focal_beta=loss.focal_beta,
        weights=loss.weights,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss: LossSettings,
    settings: LoopSettings,
    run: RunLogger,
    step: int,
    out_dir: Path,
) -> tuple[int, dict[str, float]]:
    """One pass over the training data. Returns the new step count and epoch statistics.

    Order inside a step (each line is there for a reason):
    backward on the scaled loss -> unscale -> clip -> step -> update the scale -> schedule.
    Gradients are unscaled before clipping, otherwise the clip threshold would apply to the
    scaled (65536x larger) gradients.
    """
    model.train()
    totals = dict.fromkeys(_TERMS, 0.0)
    batches = skipped = 0
    window_start, window_frames = time.perf_counter(), 0
    for batch in loader:
        inputs, heatmap, regression, mask = _targets(batch, device)
        with _autocast(device, settings.amp):
            outputs = model(inputs)
        terms = _loss(outputs, heatmap, regression, mask, loss)
        try:
            assert_finite(terms, batch["sample_token"])
        except TrainingDiverged:
            (out_dir / "bad_batch.json").write_text(
                json.dumps({"step": step, "sample_tokens": batch["sample_token"]}), encoding="utf-8"
            )
            raise

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(terms.total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # A smaller scale after `update` means the scaler found inf gradients and skipped the step.
        was_skipped = scaler.get_scale() < scale_before
        skipped += int(was_skipped)
        if not was_skipped:  # a skipped step took no update, so the schedule must not advance
            scheduler.step()

        step += 1
        batches += 1
        window_frames += inputs.shape[0]
        for name, value in zip(_TERMS, terms, strict=True):
            totals[name] += float(value.detach())
        if step % settings.log_every == 0:
            elapsed = time.perf_counter() - window_start
            run.log(
                {
                    **{
                        f"train/{name}": float(v.detach())
                        for name, v in zip(_TERMS, terms, strict=True)
                    },
                    "train/grad_norm": float(grad_norm),
                    "train/lr_head": optimizer.param_groups[-1]["lr"],
                    "train/lr_backbone": optimizer.param_groups[0]["lr"],
                    "train/loss_scale": scaler.get_scale(),
                    "train/frames_per_s": window_frames / max(elapsed, 1e-9),
                },
                step=step,
            )
            log.info(
                "step %d: loss %.4f, %.1f frames/s, lr %.2e",
                step,
                float(terms.total.detach()),
                window_frames / max(elapsed, 1e-9),
                optimizer.param_groups[-1]["lr"],
            )
            window_start, window_frames = time.perf_counter(), 0
    if batches == 0:
        raise ValueError("the training loader produced no batches (is the dataset empty?)")
    stats = {name: total / batches for name, total in totals.items()}
    stats["skipped_steps"] = float(skipped)
    return step, stats


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    loss: LossSettings,
    amp: bool,
) -> dict[str, dict[str, float]]:
    """Validation loss for all frames and separately for day and night frames.

    Each group's loss is the average over objects (a batch counts in proportion to the number of
    objects in it), not over batches: night frames are few and sparse, and a plain mean over
    batches would let a batch with two objects count as much as one with sixty.
    Returns `{"all"|"day"|"night": {term: value, "frames": n, "objects": n}}`; a group with no
    frames is absent.
    """
    model.eval()
    sums: dict[str, dict[str, float]] = {}
    for batch in loader:
        inputs, heatmap, regression, mask = _targets(batch, device)
        night = batch["is_night"].to(device)
        with _autocast(device, amp):
            outputs = model(inputs)
        for group, selected in (("all", torch.ones_like(night)), ("day", ~night), ("night", night)):
            if not bool(selected.any()):
                continue
            part = _loss(
                Outputs(outputs.class_logits[selected], outputs.regression[selected]),
                heatmap[selected],
                regression[selected],
                mask[selected],
                loss,
            )
            objects = float(mask[selected].sum().clamp(min=1.0))
            entry = sums.setdefault(
                group, dict.fromkeys(_TERMS, 0.0) | {"frames": 0.0, "objects": 0.0}
            )
            for name, value in zip(_TERMS, part, strict=True):
                entry[name] += float(value) * objects
            entry["frames"] += float(selected.sum())
            entry["objects"] += objects
    result = {}
    for group, entry in sums.items():
        result[group] = {name: entry[name] / entry["objects"] for name in _TERMS}
        result[group]["frames"] = entry["frames"]
        result[group]["objects"] = entry["objects"]
    return result


def overfit_one_batch(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss: LossSettings,
    steps: int,
    amp: bool,
    grad_clip: float,
) -> list[float]:
    """Train on one fixed batch and return the loss per step.

    The first sanity check for any new model or loss: if it cannot drive the loss near zero on
    one batch, there is a bug in the code and no amount of data or tuning will fix it.
    """
    model.train()
    inputs, heatmap, regression, mask = _targets(batch, device)
    history: list[float] = []
    for _ in range(steps):
        with _autocast(device, amp):
            outputs = model(inputs)
        terms: LossTerms = _loss(outputs, heatmap, regression, mask, loss)
        assert_finite(terms, batch["sample_token"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(terms.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        history.append(float(terms.total.detach()))
    return history


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss: LossSettings,
    settings: LoopSettings,
    run: RunLogger,
    out_dir: Path,
    config: dict[str, Any],
    resume: Path | None = None,
) -> FitResult:
    """Train for `settings.epochs`, validating after each epoch.

    Saves `last.pt` every epoch and `best.pt` whenever the overall validation loss improves
    (model selection on val is fine; the test split stays sealed). `max_minutes` is a safety
    stop: a run that hits it differs from its twin and must be reported as such.
    """
    start_epoch, step, best_val = 0, 0, float("inf")
    if resume is not None:
        state = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch, step, best_val = state.epoch + 1, state.step, state.best_val
        log.info("resumed from %s at epoch %d, step %d", resume, start_epoch, step)

    started = time.monotonic()
    epochs_run, stopped_early = 0, False
    for epoch in range(start_epoch, settings.epochs):
        step, stats = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            loss=loss,
            settings=settings,
            run=run,
            step=step,
            out_dir=out_dir,
        )
        val = validate(model, val_loader, device=device, loss=loss, amp=settings.amp)
        metrics: dict[str, Any] = {
            "epoch": epoch,
            **{f"epoch/train_{k}": v for k, v in stats.items()},
        }
        for group, entry in val.items():
            metrics.update({f"val/{group}/{key}": value for key, value in entry.items()})
        run.log(metrics, step=step)
        log.info(
            "epoch %d: train %.4f, val %.4f, skipped steps %d",
            epoch,
            stats["total"],
            val["all"]["total"],
            int(stats["skipped_steps"]),
        )

        best_val = min(best_val, val["all"]["total"])
        for name, keep in (("last.pt", True), ("best.pt", val["all"]["total"] <= best_val)):
            if keep:
                save_checkpoint(
                    out_dir / name,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    best_val=best_val,
                    config=config,
                )
        epochs_run += 1
        if (
            settings.max_minutes is not None
            and time.monotonic() - started > 60 * settings.max_minutes
        ):
            log.warning(
                "time budget of %.1f min reached: stopping after epoch %d",
                settings.max_minutes,
                epoch,
            )
            stopped_early = True
            break
    return FitResult(epochs_run, step, best_val, stopped_early)
