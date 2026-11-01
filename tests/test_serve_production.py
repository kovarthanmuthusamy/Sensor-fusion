"""Production behaviour of the service: settings, logs, request IDs, metrics, readiness, limits,
load shedding, a non-blocking event loop and safe error handling (ADR 0014).

Most tests use a stub detector (fast, and it can be slow, broken or busy on command); a few use a
real exported model.
"""

import asyncio
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("prometheus_client")

import httpx
import torch
from fastapi.testclient import TestClient

from sensor_fusion.models.detector import Detector
from sensor_fusion.serve.api import create_app
from sensor_fusion.serve.export import export_model
from sensor_fusion.serve.inference import Detection, OnnxDetector
from sensor_fusion.serve.observability import (
    JsonFormatter,
    configure_logging,
    new_request_id,
)
from sensor_fusion.serve.settings import Settings

ROOT = Path(__file__).parents[1]
CLASSES = ["car", "large_vehicle", "pedestrian", "two_wheeler"]


class StubDetector:
    """Looks like an `OnnxDetector` to the API. `delay` makes it slow, `error` makes it fail."""

    def __init__(self, delay: float = 0.0, error: str | None = None, warm_up_fails: bool = False):
        self.info: dict[str, Any] = {
            "model_name": "stub",
            "epoch": 3,
            "in_channels": 3,
            "class_names": CLASSES,
            "image_size": [900, 1600],
            "onnx_parity_max_abs_diff": {"class_logits": 0.0, "regression": 0.0},
        }
        self.image_size = (900, 1600)
        self.needs_lidar = False
        self.delay, self.error, self.warm_up_fails = delay, error, warm_up_fails
        self.calls = 0

    def warm_up(self) -> dict[str, float]:
        if self.warm_up_fails:
            raise RuntimeError("model is broken")
        return {"prepare_ms": 1.0, "network_ms": 2.0, "decode_ms": 0.5, "total_ms": 3.5}

    def detect(self, image: Any, lidar: Any, calibration: Any, *, score_threshold: Any = None):
        self.calls += 1
        time.sleep(self.delay)
        if self.error:
            raise RuntimeError(self.error)
        timings = {"prepare_ms": 1.0, "network_ms": 2.0, "decode_ms": 0.5, "total_ms": 3.5}
        return [Detection("car", 0.9, (1.0, 2.0, 30.0, 40.0), 12.0)], timings


def png() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((900, 1600, 3), 128, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


PHOTO = {"image": ("a.png", png(), "image/png")}


def app_for(detector: Any = None, warm_up: bool = False, **settings: Any):
    return create_app(detector or StubDetector(), Settings(**settings), warm_up=warm_up)


# --- settings ----------------------------------------------------------------------------------


def test_settings_have_safe_defaults_and_read_the_environment() -> None:
    assert Settings.from_env({}) == Settings()
    env = {
        "SF_PORT": "9000",
        "SF_MAX_UPLOAD_MB": "8",
        "SF_LOG_FORMAT": "TEXT",
        "SF_LOG_LEVEL": "debug",
    }
    settings = Settings.from_env(env)
    assert (settings.port, settings.max_upload_bytes) == (9000, 8 * 1024 * 1024)
    assert (settings.log_format, settings.log_level) == ("text", "DEBUG")  # case does not matter


def test_command_line_overrides_beat_the_environment_and_none_is_ignored() -> None:
    settings = Settings.from_env({"SF_PORT": "9000", "SF_HOST": "0.0.0.0"}, port=9100, host=None)
    assert (settings.port, settings.host) == (9100, "0.0.0.0")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("SF_PORT", "eight"),
        ("SF_PORT", "0"),
        ("SF_MAX_UPLOAD_MB", "0"),
        ("SF_MAX_QUEUE", "-1"),
        ("SF_LOG_FORMAT", "xml"),
        ("SF_LOG_LEVEL", "loud"),
    ],
)
def test_a_wrong_setting_stops_start_up_and_names_the_variable(variable: str, value: str) -> None:
    with pytest.raises(ValueError, match=variable):
        Settings.from_env({variable: value})


# --- logs --------------------------------------------------------------------------------------


def test_json_log_lines_carry_the_request_fields() -> None:
    record = logging.LogRecord("sensor_fusion.serve", logging.INFO, "f", 1, "request", (), None)
    record.request_id, record.status, record.latency_ms = "abc", 200, 12.5
    line = json.loads(JsonFormatter().format(record))
    assert {"ts", "level", "logger", "msg"} <= set(line)
    assert (line["request_id"], line["status"], line["latency_ms"]) == ("abc", 200, 12.5)
    assert line["ts"].endswith("+00:00")


def test_json_log_lines_include_the_traceback_of_an_exception() -> None:
    try:
        raise ValueError("bad thing")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, "f", 1, "oops", (), sys.exc_info())
    assert "ValueError: bad thing" in json.loads(JsonFormatter().format(record))["exception"]


def test_configuring_logging_twice_does_not_duplicate_output() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("INFO", "json")
        configure_logging("INFO", "text")
        ours = [h for h in root.handlers if getattr(h, "_sf_handler", False)]
        assert len(ours) == 1 and not isinstance(ours[0].formatter, JsonFormatter)
    finally:
        root.handlers = before


# --- request IDs -------------------------------------------------------------------------------


def test_request_ids_are_kept_when_plain_and_replaced_when_not() -> None:
    assert new_request_id("trace-123.abc_x") == "trace-123.abc_x"
    for bad in (None, "", "has space", "x" * 65, "semi;colon", "line\nbreak"):
        generated = new_request_id(bad)
        assert re.fullmatch(r"[0-9a-f]{32}", generated)


def test_every_response_carries_a_request_id_even_errors() -> None:
    client = TestClient(app_for())
    assert re.fullmatch(r"[0-9a-f]{32}", client.get("/health").headers["x-request-id"])
    assert (
        client.get("/health", headers={"X-Request-ID": "mine-1"}).headers["x-request-id"]
        == "mine-1"
    )
    missing = client.get("/does-not-exist")
    assert missing.status_code == 404 and "x-request-id" in missing.headers
    bad = client.post("/detect", files={"image": ("a.png", b"not an image", "image/png")})
    assert bad.status_code == 422 and "x-request-id" in bad.headers


def test_the_detect_response_repeats_the_request_id() -> None:
    client = TestClient(app_for())
    response = client.post("/detect", files=PHOTO, headers={"X-Request-ID": "abc-9"})
    assert response.status_code == 200
    assert response.json()["request_id"] == response.headers["x-request-id"] == "abc-9"


def test_each_request_is_logged_once_without_its_payload(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="sensor_fusion.serve")
    client = TestClient(app_for())
    client.get("/health")  # quiet: logged at DEBUG only
    client.post("/detect", files=PHOTO, headers={"X-Request-ID": "log-1"})
    lines = [r for r in caplog.records if r.getMessage() == "request"]
    assert len(lines) == 1
    line = lines[0]
    assert (line.path, line.status, line.request_id, line.detections) == (
        "/detect",
        200,
        "log-1",
        1,
    )
    assert isinstance(line.latency_ms, float)
    assert "PNG" not in json.dumps(JsonFormatter().format(line))  # no body content in the log


# --- metrics -----------------------------------------------------------------------------------


def test_metrics_count_requests_by_route_status_and_stage() -> None:
    client = TestClient(app_for())
    client.post("/detect", files=PHOTO)
    client.post("/detect", files={"image": ("a.png", b"junk", "image/png")})
    client.get("/health")
    client.get("/scanner/probe/1")
    client.get("/scanner/probe/2")
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/detect",status="200"} 1.0' in text
    assert 'http_requests_total{method="POST",path="/detect",status="422"} 1.0' in text
    assert (
        'http_requests_total{method="GET",path="unmatched",status="404"} 2.0' in text
    )  # one label
    assert 'http_request_duration_seconds_count{method="POST",path="/detect"} 2.0' in text
    assert 'inference_stage_seconds_count{stage="network"} 1.0' in text
    assert "detections_per_request_count 1.0" in text
    assert 'rejected_requests_total{reason="invalid"} 1.0' in text
    assert 'model_info{epoch="3",git_commit="unknown",in_channels="3",model="stub"} 1.0' in text


def test_two_apps_do_not_share_metrics() -> None:
    first, second = TestClient(app_for()), TestClient(app_for())
    first.get("/health")
    assert 'path="/health"' not in second.get("/metrics").text


# --- readiness ---------------------------------------------------------------------------------


def test_ready_only_after_the_start_up_warm_up_but_always_alive() -> None:
    app = app_for(warm_up=True)
    assert TestClient(app).get("/ready").status_code == 503  # no start-up has run yet
    with TestClient(app) as running:
        assert running.get("/ready").json() == {"status": "ready"}
        assert running.get("/health").status_code == 200


def test_a_model_that_fails_its_warm_up_stays_not_ready(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="sensor_fusion.serve")
    with TestClient(app_for(StubDetector(warm_up_fails=True), warm_up=True)) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200  # alive, so not restarted in a loop blindly
    assert any("warm-up failed" in r.getMessage() for r in caplog.records)


def test_without_a_warm_up_the_app_is_ready_at_once() -> None:
    assert TestClient(app_for(warm_up=False)).get("/ready").status_code == 200


# --- limits ------------------------------------------------------------------------------------


def test_a_body_over_the_limit_is_refused_with_413_before_it_is_parsed() -> None:
    client = TestClient(app_for(max_upload_mb=1))
    big = {"image": ("a.png", np.random.default_rng(0).bytes(2 * 1024 * 1024), "image/png")}
    response = client.post("/detect", files=big)
    assert response.status_code == 413 and "larger than 1 MB" in response.json()["detail"]
    assert "x-request-id" in response.headers
    assert 'rejected_requests_total{reason="too_large"} 1.0' in client.get("/metrics").text


def test_the_size_check_runs_before_the_body_is_parsed() -> None:
    """A huge body that is not even valid multipart must still get 413, not a parsing error:
    only the early check on Content-Length can say that without reading (and holding) the body.
    """
    client = TestClient(app_for(max_upload_mb=1))
    response = client.post(
        "/detect",
        content=b"x" * (2 * 1024 * 1024),
        headers={"content-type": "application/octet-stream"},
    )
    assert (
        response.status_code == 413
        and "request body is larger than 1 MB" in response.json()["detail"]
    )


def test_too_many_lidar_points_are_refused() -> None:
    client = TestClient(app_for(max_lidar_points=100))
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((500, 3), dtype=np.float32))
    response = client.post("/detect", files={**PHOTO, "lidar": ("l.npy", buffer.getvalue())})
    assert response.status_code == 422 and "limit is 100" in response.json()["detail"]


def test_a_tiny_file_declaring_a_huge_image_is_refused_before_decoding_pixels() -> None:
    """A 'decompression bomb': a few kB that would decode to a giant image."""
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((20000, 20000), dtype=np.uint8)).save(buffer, format="PNG")
    assert len(buffer.getvalue()) < 2_000_000
    response = TestClient(app_for()).post("/detect", files={"image": ("a.png", buffer.getvalue())})
    assert response.status_code == 422 and "image must be 1600x900" in response.json()["detail"]


# --- load shedding and a responsive event loop ------------------------------------------------


async def _fire(app: Any, count: int) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await asyncio.gather(*[client.post("/detect", files=PHOTO) for _ in range(count)])


def test_a_busy_service_sheds_load_with_503_and_retry_after() -> None:
    app = app_for(StubDetector(delay=0.4), max_concurrent=1, max_queue=1)  # room for two requests
    responses = asyncio.run(_fire(app, 5))
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 200, 503, 503, 503]
    assert all(r.headers["retry-after"] == "1" for r in responses if r.status_code == 503)
    assert (
        'rejected_requests_total{reason="overloaded"} 3.0' in TestClient(app).get("/metrics").text
    )


def test_health_answers_while_a_detection_is_running() -> None:
    """Inference runs in a worker thread: a slow model must not freeze liveness checks."""

    async def scenario() -> float:
        app = app_for(StubDetector(delay=0.8))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            slow = asyncio.create_task(client.post("/detect", files=PHOTO))
            await asyncio.sleep(0.15)
            started = time.perf_counter()
            health = await client.get("/health")
            elapsed = time.perf_counter() - started
            assert health.status_code == 200
            await slow
            return elapsed

    assert asyncio.run(scenario()) < 0.4


# --- failures never leak -----------------------------------------------------------------------


def test_an_unexpected_error_gives_a_generic_500_and_keeps_the_service_alive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="sensor_fusion.serve")
    client = TestClient(app_for(StubDetector(error="boom-secret-123")))
    response = client.post("/detect", files=PHOTO, headers={"X-Request-ID": "err-1"})
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "request_id": "err-1"}
    assert "boom-secret-123" not in response.text and "Traceback" not in response.text
    logged = [r for r in caplog.records if r.getMessage() == "unhandled error"]
    assert (
        logged and logged[0].exc_info and logged[0].request_id == "err-1"
    )  # details go to the log
    assert client.get("/health").status_code == 200
    assert (
        'http_requests_total{method="POST",path="/detect",status="500"} 1.0'
        in client.get("/metrics").text
    )


def test_version_reports_the_build_and_the_limits_in_force() -> None:
    client = TestClient(app_for(git_commit="abc1234", max_concurrent=3))
    body = client.get("/version").json()
    assert body["git_commit"] == "abc1234" and body["limits"]["max_concurrent"] == 3
    assert body["model"] == "stub" and isinstance(body["service_version"], str)


# --- with a real exported model ----------------------------------------------------------------


@pytest.fixture(scope="module")
def real_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    prep = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "prep" / "default.yaml"))
    assert isinstance(prep, dict)
    out = {}
    for name, channels, head in (("rgb", 3, False), ("fusion", 5, True)):
        torch.manual_seed(0)
        out[name] = export_model(
            Detector(in_channels=channels, num_classes=4, pretrained=False, depth_head=head),
            tmp_path_factory.mktemp(name),
            prep={str(k): v for k, v in prep.items()},
            in_channels=channels,
            model_name=name,
            depth_head=head,
            epoch=0,
            checkpoint="random-weights",
        )
    return out


@pytest.mark.parametrize("name", ["rgb", "fusion"])
def test_a_real_model_warms_up_and_the_service_becomes_ready(
    real_models: dict[str, Path], name: str
) -> None:
    detector = OnnxDetector(real_models[name])
    timings = detector.warm_up()
    assert set(timings) == {"prepare_ms", "network_ms", "decode_ms", "total_ms"}
    with TestClient(create_app(detector, Settings(), warm_up=True)) as client:
        assert client.get("/ready").status_code == 200
