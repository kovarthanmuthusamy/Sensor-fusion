"""Save and restore everything a run needs to continue where it stopped (ADR 0010)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointState:
    epoch: int  # the last finished epoch (0-based)
    step: int  # optimiser steps taken so far
    best_val: float
    config: dict[str, Any]


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    step: int,
    best_val: float,
    config: dict[str, Any],
) -> None:
    """Write to a temporary file and rename, so a crash mid-save never leaves a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "config": config,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    """Restore the model and, if given, the optimiser, scheduler and gradient scaler."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    return CheckpointState(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        best_val=float(payload["best_val"]),
        config=dict(payload["config"]),
    )
