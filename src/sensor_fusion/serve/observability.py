"""Structured logs, request IDs and Prometheus metrics for the service (ADR 0014).

One log line per request, as JSON, with a request ID that is also returned to the caller in
`X-Request-ID`, so a customer's report can be matched to the server's log. Request bodies (photos,
LiDAR points) are never logged. Metrics use one registry per app, so several apps (tests) do not
clash.
"""

import json
import logging
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Fields attached to log records with `extra=` that the JSON formatter copies into the line.
LOG_FIELDS = (
    "request_id",
    "method",
    "path",
    "status",
    "latency_ms",
    "detections",
    "model",
    "reason",
    "stage_ms",
)


def new_request_id(incoming: str | None) -> str:
    """Keep a caller's ID if it is short and plain, otherwise make one (never trust headers)."""
    if incoming and REQUEST_ID_PATTERN.match(incoming):
        return incoming
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy for a log collector, still readable by eye."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in LOG_FIELDS:
            if hasattr(record, field):
                line[field] = getattr(record, field)
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Send all logs (ours and uvicorn's) to stdout in one format. Safe to call twice."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_sf_handler", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler._sf_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


class Metrics:
    """The service's Prometheus metrics, in their own registry."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests by method, route and status",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration by method and route",
            ["method", "path"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )
        self.stage = Histogram(
            "inference_stage_seconds",
            "Time in each stage of a detection (prepare, network, decode)",
            ["stage"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
            registry=self.registry,
        )
        self.detections = Histogram(
            "detections_per_request",
            "Objects returned per detection request",
            buckets=(0, 1, 2, 5, 10, 20, 50, 100),
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "inference_in_flight", "Detections running right now", registry=self.registry
        )
        self.waiting = Gauge(
            "inference_waiting", "Detections waiting for a free slot", registry=self.registry
        )
        self.rejected = Counter(
            "rejected_requests_total",
            "Requests refused before inference, by reason",
            ["reason"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            "model_info",
            "The loaded model (value is always 1)",
            ["model", "epoch", "in_channels", "git_commit"],
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(seconds)


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self.start
