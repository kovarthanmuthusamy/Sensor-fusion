"""Write a random-weight exported model plus a sample request, for smoke-testing the service.

    python scripts/make_test_model.py ci

Creates `ci/model/` (model.onnx + model.json), `ci/image.png`, `ci/lidar.npy` and
`ci/calibration.json`. The weights are random: the point is to exercise the whole serving path
(container, API, projection, decoding), not to detect anything.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from sensor_fusion.models.detector import Detector
from sensor_fusion.serve.export import IMAGE_SIZE, export_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    prep = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "prep" / "default.yaml"))
    assert isinstance(prep, dict)
    torch.manual_seed(0)
    model = Detector(in_channels=5, num_classes=4, pretrained=False, depth_head=True)
    export_model(
        model,
        out / "model",
        prep={str(k): v for k, v in prep.items()},
        in_channels=5,
        model_name="random-weights-smoke-test",
        depth_head=True,
        epoch=0,
        checkpoint="random-weights",
    )

    rng = np.random.default_rng(0)
    Image.fromarray(np.full((*IMAGE_SIZE, 3), 120, dtype=np.uint8)).save(out / "image.png")
    points = np.column_stack(
        [rng.uniform(5, 60, 20000), rng.uniform(-15, 15, 20000), rng.uniform(-1.5, 2.5, 20000)]
    )
    np.save(out / "lidar.npy", points.astype(np.float32))
    calibration = {
        # camera x = -lidar y, camera y = -lidar z, camera z = lidar x (forward)
        "T_cam_from_lidar": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        "intrinsic": [[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1]],
    }
    (out / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    print(f"wrote {out}/model, image.png, lidar.npy, calibration.json")


if __name__ == "__main__":
    main()
