"""Experiment tracking with Weights & Biases (ADR 0010).

Every run records the seed and full resolved config (inside `config`), the git commit and whether
the working tree had uncommitted changes, and the environment, so a number on a chart can always
be traced back to the code that produced it.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

import wandb

from sensor_fusion.utils.env import collect_env_info

log = logging.getLogger(__name__)


def git_info(cwd: Path | None = None) -> dict[str, str]:
    """Commit hash and dirty flag of the working tree; "unknown" when git is unavailable."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=cwd, timeout=30
        ).stdout.strip()

    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": str(bool(run("status", "-s")))}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def start_run(
    *,
    config: dict[str, Any],
    mode: str,
    project: str,
    entity: str | None,
    name: str,
    tags: list[str],
    out_dir: Path,
) -> Any:
    """Start a W&B run. "online" falls back to "offline" (sync later with `wandb sync`) when no
    credentials are found, so a missing login never kills a long run."""
    if mode == "online":
        try:
            logged_in = bool(wandb.login(prompt=False, verify=False))
        except Exception:  # login failures come in many forms; all mean "not usable now"
            logged_in = False
        if not logged_in:
            log.warning("no W&B credentials found: running offline (use `wandb login`)")
            mode = "offline"
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        mode=mode,  # type: ignore[arg-type]
        dir=str(out_dir),
        config={**config, **git_info(), "env": collect_env_info()},
    )
