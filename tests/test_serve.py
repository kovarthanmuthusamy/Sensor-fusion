"""Export to ONNX, the detector service and the HTTP API, on random-weight models (ADR 0007)."""

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.prepared import PreparedDataset
from sensor_fusion.models.detector import Detector
from sensor_fusion.serve.api import create_app
from sensor_fusion.serve.export import check_parity, export_model
from sensor_fusion.serve.inference import (
    Calibration,
    InvalidInputError,
    OnnxDetector,
)
from sensor_fusion.serve.samples import request_from_nuscenes

ROOT = Path(__file__).parents[1]
MINI = ROOT / "dataset" / "v1.0-mini"
IMAGE_SIZE = (900, 1600)


def _prep() -> dict[str, Any]:
    container = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "prep" / "default.yaml"))
    assert isinstance(container, dict)
    return {str(k): v for k, v in container.items()}


def _random_model(in_channels: int, depth_head: bool, seed: int) -> Detector:
    torch.manual_seed(seed)
    model = Detector(
        in_channels=in_channels, num_classes=4, pretrained=False, depth_head=depth_head
    )
    with torch.no_grad():  # make depth matter: the zero start would hide any input difference
        if model.conv1_extra is not None:
            model.conv1_extra.weight.normal_(0, 0.05)
        for head in (model.class_head, model.regression_head):
            head[0].weight[:, 64:].normal_(0, 0.05)
    return model


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("serve")
    out = {}
    for name, channels, head in (("rgb", 3, False), ("fusion", 5, True)):
        out[name] = export_model(
            _random_model(channels, head, seed=0),
            root / name,
            prep=_prep(),
            in_channels=channels,
            model_name=name,
            depth_head=head,
            epoch=7,
            checkpoint="random-weights",
        )
    return out


@pytest.fixture(scope="module")
def rgb(exported: dict[str, Path]) -> OnnxDetector:
    return OnnxDetector(exported["rgb"])


@pytest.fixture(scope="module")
def fusion(exported: dict[str, Path]) -> OnnxDetector:
    return OnnxDetector(exported["fusion"])


def _request(seed: int = 0) -> tuple[np.ndarray, np.ndarray, Calibration]:
    """A grey photo, a LiDAR cloud in front of the car, and a camera looking along the car's x."""
    rng = np.random.default_rng(seed)
    image = np.full((*IMAGE_SIZE, 3), 120, dtype=np.uint8)
    points = np.column_stack(
        [rng.uniform(5, 60, 20000), rng.uniform(-15, 15, 20000), rng.uniform(-1.5, 2.5, 20000)]
    ).astype(np.float32)
    # camera x = -lidar y, camera y = -lidar z, camera z = lidar x (forward)
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    intrinsic = np.array([[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1]])
    return image, points, Calibration(transform, intrinsic)


# --- export ------------------------------------------------------------------------------------


def test_export_writes_the_model_and_its_input_recipe(exported: dict[str, Path]) -> None:
    info = json.loads((exported["fusion"] / "model.json").read_text(encoding="utf-8"))
    assert (info["in_channels"], info["depth_head"], info["input_size"]) == (5, True, [448, 800])
    assert info["image_size"] == [900, 1600] and info["epoch"] == 7
    assert info["prep"]["depth_thicken_px"] == 7
    assert max(info["onnx_parity_max_abs_diff"].values()) < 1e-3  # the exporter's own check
    assert (exported["fusion"] / "model.onnx").stat().st_size > 1_000_000


def test_parity_check_catches_a_model_that_is_not_the_exported_one(
    exported: dict[str, Path],
) -> None:
    other = _random_model(3, False, seed=1)
    with pytest.raises(ValueError, match="differs from PyTorch"):
        check_parity(other, exported["rgb"] / "model.onnx", in_channels=3, input_size=(448, 800))


# --- the detector --------------------------------------------------------------------------------


def test_detections_are_valid_and_cover_the_photo(rgb: OnnxDetector) -> None:
    image, _, _ = _request()
    found, timings = rgb.detect(image, score_threshold=0.0)
    assert 0 < len(found) <= 100
    for d in found:
        x1, y1, x2, y2 = d.box
        assert x1 < x2 and y1 < y2 and np.isfinite(d.box).all()
        assert d.class_name in rgb.class_names and 0.0 <= d.score <= 1.0 and d.distance_m > 0
    assert set(timings) == {"prepare_ms", "network_ms", "decode_ms", "total_ms"}
    assert all(v > 0 for v in timings.values())


def test_a_higher_threshold_keeps_only_confident_detections(rgb: OnnxDetector) -> None:
    image, _, _ = _request()
    everything, _ = rgb.detect(image, score_threshold=0.0)
    strict, _ = rgb.detect(image, score_threshold=0.5)
    assert len(strict) < len(everything)
    assert all(d.score >= 0.5 for d in strict)


def test_the_rgb_model_ignores_lidar_and_does_not_need_it(rgb: OnnxDetector) -> None:
    image, points, calibration = _request()
    without, _ = rgb.detect(image, score_threshold=0.0)
    with_lidar, _ = rgb.detect(image, points, calibration, score_threshold=0.0)
    assert without == with_lidar


def test_the_fusion_model_needs_lidar(fusion: OnnxDetector) -> None:
    image, points, _ = _request()
    with pytest.raises(InvalidInputError, match="uses LiDAR"):
        fusion.detect(image)
    with pytest.raises(InvalidInputError, match="uses LiDAR"):
        fusion.detect(image, points, None)


def test_the_fusion_model_answers_depend_on_the_lidar(fusion: OnnxDetector) -> None:
    image, points, calibration = _request()
    near, _ = fusion.detect(image, points, calibration, score_threshold=0.0)
    far, _ = fusion.detect(
        image,
        points * np.float32([1, 1, 1]) + np.float32([30, 0, 0]),
        calibration,
        score_threshold=0.0,
    )
    assert [d.distance_m for d in near] != [d.distance_m for d in far]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda im, pt, cal: (im[:, :-1], pt, cal), "image must be 1600x900"),
        (lambda im, pt, cal: (im.astype(np.float32), pt, cal), "uint8"),
        (lambda im, pt, cal: (im, pt[:, :2], cal), r"\(N, >=3\)"),
        (lambda im, pt, cal: (im, pt[:0], cal), r"\(N, >=3\)"),
        (lambda im, pt, cal: (im, np.where(pt > 50, np.nan, pt), cal), "NaN or infinity"),
        (lambda im, pt, cal: (im, pt, Calibration(cal.T_cam_from_lidar[:3], cal.intrinsic)), "4x4"),
        (
            lambda im, pt, cal: (im, pt, Calibration(cal.T_cam_from_lidar * 2, cal.intrinsic)),
            "homogeneous",
        ),
        (
            lambda im, pt, cal: (im, pt, Calibration(cal.T_cam_from_lidar, cal.intrinsic * 0)),
            "focal",
        ),
    ],
)
def test_malformed_requests_are_refused_with_a_clear_message(
    fusion: OnnxDetector, change: Any, message: str
) -> None:
    image, points, calibration = change(*_request())
    with pytest.raises(InvalidInputError, match=message):
        fusion.detect(image, points, calibration)


def test_score_threshold_must_be_a_probability(rgb: OnnxDetector) -> None:
    with pytest.raises(InvalidInputError, match="score_threshold"):
        rgb.detect(_request()[0], score_threshold=1.5)


# --- no drift between training and serving ------------------------------------------------------


@pytest.mark.skipif(not MINI.exists(), reason="nuScenes-mini not found")
def test_the_service_builds_the_same_input_as_the_training_pipeline(fusion: OnnxDetector) -> None:
    """Training-serving skew test: a real frame sent as a customer would (photo, raw LiDAR scan,
    calibration) must become exactly the tensor the dataset pipeline gives the network."""
    base = NuScenesFrontDataset(str(MINI), "v1.0-mini", "mini_dev_val")
    trained = PreparedDataset(base, in_channels=5, prep=_prep(), train=False)
    for index in (0, len(base) - 1):  # a day frame and a night frame
        token, _ = base._entries[index]
        request = request_from_nuscenes(base.nusc, token)
        served, _, _ = fusion.prepare(request.image, request.lidar_points, request.calibration)
        np.testing.assert_allclose(served, trained[index]["input"].numpy(), atol=1e-6)


# --- the HTTP API --------------------------------------------------------------------------------


def _png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _form(calibration: Calibration, **extra: str) -> dict[str, str]:
    payload = {
        "T_cam_from_lidar": calibration.T_cam_from_lidar.tolist(),
        "intrinsic": calibration.intrinsic.tolist(),
    }
    return {"calibration": json.dumps(payload), **extra}


def test_health_and_version(fusion: OnnxDetector) -> None:
    client = TestClient(create_app(fusion))
    assert client.get("/health").json() == {"status": "ok"}
    version = client.get("/version").json()
    assert version["model"] == "fusion" and version["needs_lidar"] is True
    assert version["classes"] == ["car", "large_vehicle", "pedestrian", "two_wheeler"]
    assert max(version["onnx_parity_max_abs_diff"].values()) < 1e-3


def test_detect_over_http(fusion: OnnxDetector) -> None:
    client = TestClient(create_app(fusion))
    image, points, calibration = _request()
    response = client.post(
        "/detect",
        files={"image": ("a.png", _png(image), "image/png"), "lidar": ("l.npy", _npy(points))},
        data=_form(calibration, score_threshold="0.0"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "fusion" and 0 < len(body["detections"]) <= 100
    first = body["detections"][0]
    assert set(first) == {"class_name", "score", "box", "distance_m"} and len(first["box"]) == 4
    assert body["timings_ms"]["total_ms"] > 0


def test_http_result_equals_the_direct_call(fusion: OnnxDetector) -> None:
    client = TestClient(create_app(fusion))
    image, points, calibration = _request()
    response = client.post(
        "/detect",
        files={"image": ("a.png", _png(image), "image/png"), "lidar": ("l.npy", _npy(points))},
        data=_form(calibration, score_threshold="0.2"),
    )
    direct, _ = fusion.detect(image, points, calibration, score_threshold=0.2)
    served = response.json()["detections"]
    assert [d["class_name"] for d in served] == [d.class_name for d in direct]
    np.testing.assert_allclose([d["distance_m"] for d in served], [d.distance_m for d in direct])


def test_the_rgb_model_works_over_http_without_lidar(rgb: OnnxDetector) -> None:
    client = TestClient(create_app(rgb))
    response = client.post(
        "/detect",
        files={"image": ("a.png", _png(_request()[0]), "image/png")},
        data={"score_threshold": "0.0"},
    )
    assert response.status_code == 200 and response.json()["detections"]


def _bad_requests() -> dict[str, tuple[bytes, bytes | None, dict[str, str], str]]:
    """name -> (image file, lidar file, form fields, text the error must contain)."""
    image, points, calibration = _request()
    png, form = _png(image), _form(calibration)
    return {
        "undecodable image": (b"not an image", _npy(points), form, "could not be decoded"),
        "wrong image size": (_png(image[:100, :100]), _npy(points), form, "image must be 1600x900"),
        "missing lidar": (png, None, form, "uses LiDAR"),
        "corrupt lidar file": (png, b"\x80\x04bad", form, "lidar must be a .npy"),
        "integer lidar": (png, _npy(points.astype(np.int32)), form, "must hold floats"),
        "lidar with two columns": (png, _npy(points[:, :2]), form, r"\(N, >=3\)"),
        "missing calibration": (png, _npy(points), {}, "uses LiDAR"),
        "calibration not json": (png, _npy(points), {"calibration": "{no"}, "must be JSON"),
        "calibration missing key": (
            png,
            _npy(points),
            {"calibration": json.dumps({"intrinsic": [[1, 0, 0]] * 3})},
            "must be JSON",
        ),
        "threshold out of range": (
            png,
            _npy(points),
            {**form, "score_threshold": "2.0"},
            "score_threshold",
        ),
    }


@pytest.mark.parametrize("name", list(_bad_requests()))
def test_bad_requests_get_a_422_with_a_reason_not_a_crash(fusion: OnnxDetector, name: str) -> None:
    image_file, lidar_file, form, message = _bad_requests()[name]
    files = {"image": ("a.png", image_file, "image/png")}
    if lidar_file is not None:
        files["lidar"] = ("l.npy", lidar_file, "application/octet-stream")
    response = TestClient(create_app(fusion)).post("/detect", files=files, data=form)
    assert response.status_code == 422, response.text
    assert re.search(message, json.dumps(response.json()["detail"]))


def test_the_service_does_not_import_the_training_stack() -> None:
    """The container only installs what serving needs, so importing the API must not need more."""
    code = (
        "import sys, sensor_fusion.serve.api;"
        "heavy = [m for m in ('hydra', 'wandb', 'nuscenes', 'torchvision', 'matplotlib')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


def test_the_service_sets_its_own_torch_thread_count(exported: dict[str, Path]) -> None:
    """Few threads are much faster for the many small preparation and decoding operations."""
    before = torch.get_num_threads()
    try:
        OnnxDetector(exported["rgb"], torch_threads=2)
        assert torch.get_num_threads() == 2
        OnnxDetector(exported["rgb"], torch_threads=None)  # None leaves the setting alone
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
