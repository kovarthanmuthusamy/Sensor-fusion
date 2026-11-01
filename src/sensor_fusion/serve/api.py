"""A web service around an exported model (ADR 0007, hardened in ADR 0014).

    GET  /health    liveness: the process is up
    GET  /ready     readiness: the model is loaded and has run a warm-up detection
    GET  /version   which model and build is running, and the limits in force
    GET  /metrics   Prometheus metrics (requests, latency, stages, rejections, load)
    POST /detect    multipart form: `image` (JPEG/PNG), `lidar` (a .npy file, float32 (N, >=3), in
                    the LiDAR frame), `calibration` (JSON text with `T_cam_from_lidar` 4x4 and
                    `intrinsic` 3x3), optional `score_threshold`.

Bad input gets a clear 4xx message. An overloaded service answers 503 with `Retry-After` instead of
queueing without bound. An unexpected error returns a generic 500 with the request ID (the details
go to the log, never to the caller). Every response carries `X-Request-ID`.
"""

import io
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any

import anyio
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from sensor_fusion.serve.inference import Calibration, InvalidInputError, OnnxDetector
from sensor_fusion.serve.observability import Metrics, Timer, new_request_id
from sensor_fusion.serve.settings import Settings

log = logging.getLogger("sensor_fusion.serve")
KNOWN_PATHS = ("/health", "/ready", "/version", "/metrics", "/detect")
QUIET_PATHS = ("/health", "/ready", "/metrics")  # logged at DEBUG: orchestrators poll them


class DetectionOut(BaseModel):
    class_name: str
    score: float
    box: list[float]  # x1, y1, x2, y2, pixels of the full-size photo
    distance_m: float


class DetectResponse(BaseModel):
    model: str
    detections: list[DetectionOut]
    timings_ms: dict[str, float]
    request_id: str


def _bad(detail: str, status: int = 422) -> HTTPException:
    return HTTPException(status_code=status, detail=detail)


def _read_image(data: bytes, expected_hw: tuple[int, int]) -> np.ndarray:
    try:
        picture = Image.open(io.BytesIO(data))  # reads the header only
        if (picture.height, picture.width) != expected_hw:
            # Checked before decoding any pixels: a tiny file cannot make us allocate a huge image.
            raise _bad(
                f"image must be {expected_hw[1]}x{expected_hw[0]} (width x height), "
                f"got {picture.width}x{picture.height}"
            )
        return np.asarray(picture.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise _bad(f"image could not be decoded: {error}") from error


def _read_lidar(data: bytes, max_points: int) -> np.ndarray:
    try:
        points = np.load(io.BytesIO(data), allow_pickle=False)  # pickled files are refused
    except (ValueError, OSError, EOFError) as error:
        raise _bad(f"lidar must be a .npy file with a float array: {error}") from error
    if not np.issubdtype(points.dtype, np.floating):
        raise _bad(f"lidar must hold floats, got {points.dtype}")
    if points.ndim == 2 and points.shape[0] > max_points:
        raise _bad(f"lidar has {points.shape[0]:,} points, the limit is {max_points:,}")
    return points.astype(np.float32)


def _read_calibration(text: str) -> Calibration:
    try:
        raw = json.loads(text)
        return Calibration(
            T_cam_from_lidar=np.asarray(raw["T_cam_from_lidar"], dtype=np.float64),
            intrinsic=np.asarray(raw["intrinsic"], dtype=np.float64),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise _bad(
            "calibration must be JSON with numeric 'T_cam_from_lidar' (4x4) and 'intrinsic' (3x3): "
            f"{error!r}"
        ) from error


async def _upload(file: UploadFile | None, limit: int, name: str) -> bytes | None:
    if file is None:
        return None
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise _bad(f"{name} is larger than {limit // 2**20} MB", status=413)
    return data


def _package_version() -> str:
    try:
        return version("sensor-fusion")
    except PackageNotFoundError:
        return "unknown"


def create_app(
    detector: OnnxDetector, settings: Settings | None = None, *, warm_up: bool = True
) -> FastAPI:
    """Build the service around a loaded detector.

    With `warm_up` (the default) the app is ready only after a start-up detection has run, which
    happens when the server starts (the lifespan). Without it the app is ready at once.
    """
    settings = settings or Settings()
    metrics = Metrics()
    info = detector.info
    metrics.model_info.labels(
        info["model_name"], str(info["epoch"]), str(info["in_channels"]), settings.git_commit
    ).set(1)
    limiter = anyio.CapacityLimiter(settings.max_concurrent)
    capacity = settings.max_concurrent + settings.max_queue
    counters = {"active": 0}  # requests running or waiting; only touched on the event-loop thread

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if warm_up:
            try:
                timings = await run_in_threadpool(detector.warm_up)
                app.state.ready = True
                log.info(
                    "model warm and ready", extra={"model": info["model_name"], "stage_ms": timings}
                )
            except Exception:
                log.exception("warm-up failed: the service stays not ready")
        yield

    app = FastAPI(
        title="Night-time camera + LiDAR detector",
        version=_package_version(),
        lifespan=lifespan,
    )
    app.state.ready = not warm_up

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = new_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        timer = Timer()
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > settings.max_upload_bytes:
            metrics.rejected.labels("too_large").inc()
            response: Response = JSONResponse(
                {"detail": f"request body is larger than {settings.max_upload_mb} MB"},
                status_code=413,
            )
        else:
            try:
                response = await call_next(request)
            except Exception:
                log.exception(
                    "unhandled error",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                    },
                )
                response = JSONResponse(
                    {"error": "internal_error", "request_id": request_id}, status_code=500
                )
        route = request.scope.get("route")
        path = getattr(route, "path", None) or (
            request.url.path if request.url.path in KNOWN_PATHS else "unmatched"
        )
        seconds = timer.seconds()
        response.headers["X-Request-ID"] = request_id
        metrics.observe_request(request.method, path, response.status_code, seconds)
        extra: dict[str, Any] = {
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "latency_ms": round(1000 * seconds, 1),
        }
        if getattr(request.state, "detections", None) is not None:
            extra["detections"] = request.state.detections
        level = logging.DEBUG if path in QUIET_PATHS else logging.INFO
        log.log(level, "request", extra=extra)
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        if not app.state.ready:
            raise _bad("the model is not warm yet", status=503)
        return {"status": "ready"}

    @app.get("/version")
    async def version_info() -> dict[str, Any]:
        return {
            "model": info["model_name"],
            "epoch": info["epoch"],
            "in_channels": info["in_channels"],
            "needs_lidar": detector.needs_lidar,
            "classes": info["class_names"],
            "image_size_hw": info["image_size"],
            "onnx_parity_max_abs_diff": info["onnx_parity_max_abs_diff"],
            "service_version": _package_version(),
            "git_commit": settings.git_commit,
            "limits": {
                "max_upload_mb": settings.max_upload_mb,
                "max_lidar_points": settings.max_lidar_points,
                "max_concurrent": settings.max_concurrent,
                "max_queue": settings.max_queue,
            },
        }

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/detect",
        response_model=DetectResponse,
        responses={
            413: {"description": "the upload is larger than the limit"},
            422: {"description": "the input is malformed; the message says how"},
            503: {"description": "the service is busy; retry after the given delay"},
        },
    )
    async def detect(
        request: Request,
        image: Annotated[UploadFile, File()],
        lidar: Annotated[UploadFile | None, File()] = None,
        calibration: Annotated[str | None, Form()] = None,
        score_threshold: Annotated[float | None, Form()] = None,
    ) -> DetectResponse:
        if counters["active"] >= capacity:  # shed load instead of queueing without bound
            metrics.rejected.labels("overloaded").inc()
            raise HTTPException(
                status_code=503,
                detail="the service is busy, retry shortly",
                headers={"Retry-After": "1"},
            )
        counters["active"] += 1
        try:
            limit = settings.max_upload_bytes
            photo = _read_image(await _upload(image, limit, "image") or b"", detector.image_size)
            lidar_bytes = await _upload(lidar, limit, "lidar")
            points = _read_lidar(lidar_bytes, settings.max_lidar_points) if lidar_bytes else None
            calib = _read_calibration(calibration) if calibration else None

            def work() -> Any:
                metrics.waiting.dec()
                metrics.in_flight.inc()
                try:
                    return detector.detect(photo, points, calib, score_threshold=score_threshold)
                finally:
                    metrics.in_flight.dec()

            metrics.waiting.inc()
            try:
                found, timings = await anyio.to_thread.run_sync(work, limiter=limiter)
            except InvalidInputError as error:
                raise _bad(str(error)) from error
        except HTTPException as error:
            if error.status_code in (413, 422):
                metrics.rejected.labels(
                    "too_large" if error.status_code == 413 else "invalid"
                ).inc()
            raise
        finally:
            counters["active"] -= 1
        for stage in ("prepare", "network", "decode"):
            metrics.stage.labels(stage).observe(timings[f"{stage}_ms"] / 1000)
        metrics.detections.observe(len(found))
        request.state.detections = len(found)
        return DetectResponse(
            model=info["model_name"],
            detections=[
                DetectionOut(
                    class_name=d.class_name, score=d.score, box=list(d.box), distance_m=d.distance_m
                )
                for d in found
            ],
            timings_ms=timings,
            request_id=request.state.request_id,
        )

    return app
