"""Write one real nuScenes frame as the files a customer would send to `/detect`.

    python scripts/make_sample_request.py outputs/sample

Creates `image.png`, `lidar.npy` and `calibration.json` in the folder; use them with `curl`
(see README) or with `scripts/load_test.py`. Needs the nuScenes data. The frame comes from the
validation split and is not committed anywhere (nuScenes images may not be redistributed).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image

from sensor_fusion.data.splits import get_split_scenes, scene_is_night
from sensor_fusion.serve.samples import request_from_nuscenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--root", default="dataset/v1.0-trainval")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--night", action="store_true", help="take a night frame (default: day)")
    args = parser.parse_args()

    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    wanted = set(get_split_scenes("val"))
    scene = next(s for s in nusc.scene if s["name"] in wanted and scene_is_night(s) == args.night)
    request = request_from_nuscenes(nusc, scene["first_sample_token"])
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(request.image).save(args.out / "image.png")
    np.save(args.out / "lidar.npy", request.lidar_points)
    calibration = {
        "T_cam_from_lidar": request.calibration.T_cam_from_lidar.tolist(),
        "intrinsic": request.calibration.intrinsic.tolist(),
    }
    (args.out / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    print(f"wrote a {'night' if args.night else 'day'} frame of {scene['name']} to {args.out}")


if __name__ == "__main__":
    main()
