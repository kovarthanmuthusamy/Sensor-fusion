"""A small web service around an exported model (ADR 0007).

    GET  /health   liveness
    GET  /version  which model is loaded and what input it needs
    POST /detect   multipart form: `image` (JPEG/PNG), `lidar` (a .npy file, float32 (N, >=3), in
                   the LiDAR frame), `calibration` (JSON text with `T_cam_from_lidar` 4x4 and
                   `intrinsic` 3x3), optional `score_threshold`.

Bad input gets a clear 4xx message, never a crash.
"""

import io
import json
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from sensor_fusion.serve.inference import Calibration, InvalidInputError, OnnxDetector

MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class DetectionOut(BaseModel):
    class_name: str
    score: float
    box: list[float]  # x1, y1, x2, y2, pixels of the full-size photo
    distance_m: float


class DetectResponse(BaseModel):
    model: str
    detections: list[DetectionOut]
    timings_ms: dict[str, float]


def _bad(detail: str, status: int = 422) -> HTTPException:
    return HTTPException(status_code=status, detail=detail)


def _read_image(data: bytes) -> np.ndarray:
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise _bad(f"image could not be decoded: {error}") from error


def _read_lidar(data: bytes) -> np.ndarray:
    try:
        points = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:  # pickled or corrupt files land here
        raise _bad(f"lidar must be a .npy file with a float array: {error}") from error
    if not np.issubdtype(points.dtype, np.floating):
        raise _bad(f"lidar must hold floats, got {points.dtype}")
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


async def _upload(file: UploadFile | None, name: str) -> bytes | None:
    if file is None:
        return None
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise _bad(f"{name} is larger than {MAX_UPLOAD_BYTES // 2**20} MB", status=413)
    return data


def create_app(detector: OnnxDetector) -> FastAPI:
    """Build the service around a loaded detector."""
    app = FastAPI(title="Night-time camera + LiDAR detector", version="1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, Any]:
        info = detector.info
        return {
            "model": info["model_name"],
            "epoch": info["epoch"],
            "in_channels": info["in_channels"],
            "needs_lidar": detector.needs_lidar,
            "classes": info["class_names"],
            "image_size_hw": info["image_size"],
            "onnx_parity_max_abs_diff": info["onnx_parity_max_abs_diff"],
        }

    @app.post("/detect", response_model=DetectResponse)
    async def detect(
        image: Annotated[UploadFile, File()],
        lidar: Annotated[UploadFile | None, File()] = None,
        calibration: Annotated[str | None, Form()] = None,
        score_threshold: Annotated[float | None, Form()] = None,
    ) -> DetectResponse:
        photo = _read_image(await _upload(image, "image") or b"")
        lidar_bytes = await _upload(lidar, "lidar")
        points = _read_lidar(lidar_bytes) if lidar_bytes else None
        calib = _read_calibration(calibration) if calibration else None
        try:
            found, timings = detector.detect(photo, points, calib, score_threshold=score_threshold)
        except InvalidInputError as error:
            raise _bad(str(error)) from error
        return DetectResponse(
            model=detector.info["model_name"],
            detections=[
                DetectionOut(
                    class_name=d.class_name, score=d.score, box=list(d.box), distance_m=d.distance_m
                )
                for d in found
            ],
            timings_ms=timings,
        )

    return app
