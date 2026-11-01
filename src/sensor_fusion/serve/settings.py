"""Service settings, read from environment variables (the twelve-factor way; ADR 0014).

Every variable starts with `SF_`. Defaults are safe for a laptop; a container sets them through its
environment, and `python -m sensor_fusion.serve.app` flags override them. A wrong value stops the
service at start-up with a message naming the variable, instead of failing later in a request.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

LOG_FORMATS = ("json", "text")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("/models")  # SF_MODEL_DIR: folder with model.onnx and model.json
    host: str = "127.0.0.1"  # SF_HOST (a container sets 0.0.0.0)
    port: int = 8000  # SF_PORT
    torch_threads: int = 4  # SF_TORCH_THREADS: for preparation and decoding (see ADR 0007)
    max_upload_mb: int = 64  # SF_MAX_UPLOAD_MB: whole request body; larger gets 413
    max_lidar_points: int = 300_000  # SF_MAX_LIDAR_POINTS: more gets 422 (a scan has ~35k)
    max_concurrent: int = 2  # SF_MAX_CONCURRENT: inferences running at the same time
    max_queue: int = 4  # SF_MAX_QUEUE: requests allowed to wait; beyond that: 503 + Retry-After
    log_level: str = "INFO"  # SF_LOG_LEVEL
    log_format: str = "json"  # SF_LOG_FORMAT: json (machines) or text (people)
    git_commit: str = "unknown"  # SF_GIT_COMMIT: set at image build time

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: Any) -> "Settings":
        """Settings from `SF_*` variables, then keyword overrides (e.g. command-line flags)."""
        env = os.environ if environ is None else environ

        def read(name: str, default: str) -> str:
            return env.get(f"SF_{name}", default)

        def integer(name: str, default: int, minimum: int) -> int:
            raw = read(name, str(default))
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"SF_{name} must be an integer, got {raw!r}") from None
            if value < minimum:
                raise ValueError(f"SF_{name} must be at least {minimum}, got {value}")
            return value

        defaults = cls()
        settings = cls(
            model_dir=Path(read("MODEL_DIR", str(defaults.model_dir))),
            host=read("HOST", defaults.host),
            port=integer("PORT", defaults.port, 1),
            torch_threads=integer("TORCH_THREADS", defaults.torch_threads, 1),
            max_upload_mb=integer("MAX_UPLOAD_MB", defaults.max_upload_mb, 1),
            max_lidar_points=integer("MAX_LIDAR_POINTS", defaults.max_lidar_points, 1),
            max_concurrent=integer("MAX_CONCURRENT", defaults.max_concurrent, 1),
            max_queue=integer("MAX_QUEUE", defaults.max_queue, 0),
            log_level=read("LOG_LEVEL", defaults.log_level).upper(),
            log_format=read("LOG_FORMAT", defaults.log_format).lower(),
            git_commit=read("GIT_COMMIT", defaults.git_commit),
        )
        settings = replace(settings, **{k: v for k, v in overrides.items() if v is not None})
        if settings.log_format not in LOG_FORMATS:
            raise ValueError(
                f"SF_LOG_FORMAT must be one of {LOG_FORMATS}, got {settings.log_format!r}"
            )
        if settings.log_level not in LOG_LEVELS:
            raise ValueError(
                f"SF_LOG_LEVEL must be one of {LOG_LEVELS}, got {settings.log_level!r}"
            )
        return settings
