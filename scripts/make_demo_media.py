"""Demo media for the README: a night drive, a night gallery and the LiDAR input picture.

    CUDA_VISIBLE_DEVICES="" python scripts/make_demo_media.py \\
        --model rgb=outputs/serve/rgb --model fusion=outputs/serve/fusion_dense_head

Writes to `docs/figures/`:
- `night_drive.gif`: consecutive keyframes of one night validation scene, camera only above
  camera + LiDAR, with `class distance` on each box.
- `night_gallery.jpg`: one frame from each of three night scenes, camera only | camera + LiDAR.
- `lidar_input.jpg`: one night photo, the projected LiDAR scan, and the dense depth the network
  actually receives.

Scenes are picked by how many labelled objects they hold, never by how the models did on them, so
the pictures are not cherry-picked. Only detections above `--threshold` are drawn. There is no
non-maximum suppression, so big close objects can carry two boxes. nuScenes images are
CC BY-NC-SA 4.0 (research use only).
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from nuscenes.nuscenes import NuScenes
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from sensor_fusion.data.projection import make_sparse_depth, project_to_image
from sensor_fusion.data.splits import get_split_scenes, scene_is_night
from sensor_fusion.data.transforms import thicken_depth
from sensor_fusion.serve.demo import MODEL_COLOURS, as_items, draw, ground_truth
from sensor_fusion.serve.inference import OnnxDetector
from sensor_fusion.serve.samples import RequestData, request_from_nuscenes

PREP_CONFIG = Path("configs/prep/default.yaml")  # where the training thickening size lives
NEAR_M, FAR_M = 3.0, 60.0  # depth range of the colour scale
GALLERY_SCENES = 3
GIF_FRAMES = 10  # 5 s of driving at the 2 Hz keyframe rate
GIF_FRAME_MS = 500  # real time: playback speed equals the keyframe rate
GIF_COLOURS = 64  # palette size: with no dithering this keeps the file under the 1 MB repo limit
DOT_RADIUS_PX = 3
HEADER_PX = 30
HEADER_FONT = ImageFont.load_default(size=22)


def keyframes(nusc: NuScenes, scene: dict, limit: int | None = None) -> list[str]:
    tokens, token = [], scene["first_sample_token"]
    while token and (limit is None or len(tokens) < limit):
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def busiest_night_scenes(nusc: NuScenes, count: int) -> list[dict]:
    """Night validation scenes with the most labelled objects per frame (not model results)."""
    wanted = set(get_split_scenes("val"))
    night = [s for s in nusc.scene if s["name"] in wanted and scene_is_night(s)]

    def density(scene: dict) -> float:
        tokens = keyframes(nusc, scene)
        return sum(len(ground_truth(nusc, t)) for t in tokens) / len(tokens)

    return sorted(night, key=density, reverse=True)[:count]


def detect_panels(
    request: RequestData, detectors: dict[str, OnnxDetector], threshold: float, width: int
) -> list[Image.Image]:
    """One drawn picture per model, scaled to `width` pixels."""
    panels = []
    for colour, (name, detector) in zip(MODEL_COLOURS, detectors.items(), strict=False):
        found, _ = detector.detect(
            request.image, request.lidar_points, request.calibration, score_threshold=threshold
        )
        panel = draw(request.image, as_items(found), colour, "")
        panel = panel.resize((width, round(panel.height * width / panel.width)))
        pen = ImageDraw.Draw(panel)  # a header at the final size: the full-size one shrinks to dust
        pen.rectangle((0, 0, width, HEADER_PX), fill=(0, 0, 0))
        noun = "detection" if len(found) == 1 else "detections"
        pen.text((8, 4), f"{name}: {len(found)} {noun}", fill=colour, font=HEADER_FONT)
        panels.append(panel)
    return panels


def stack(panels: list[Image.Image], horizontal: bool = False) -> Image.Image:
    size = (
        (sum(p.width for p in panels), panels[0].height)
        if horizontal
        else (panels[0].width, sum(p.height for p in panels))
    )
    sheet = Image.new("RGB", size)
    offset = 0
    for panel in panels:
        sheet.paste(panel, (offset, 0) if horizontal else (0, offset))
        offset += panel.width if horizontal else panel.height
    return sheet


def night_drive(
    nusc: NuScenes, scene: dict, detectors: dict[str, OnnxDetector], args: argparse.Namespace
) -> None:
    frames = []
    for token in keyframes(nusc, scene, GIF_FRAMES):
        request = request_from_nuscenes(nusc, token)
        sheet = stack(detect_panels(request, detectors, args.threshold, args.gif_width))
        frames.append(
            sheet.quantize(GIF_COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        )
    path = args.out / "night_drive.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
    )
    print(f"{path}: {scene['name']}, {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB")


def night_gallery(
    nusc: NuScenes,
    scenes: list[dict],
    detectors: dict[str, OnnxDetector],
    args: argparse.Namespace,
) -> None:
    rows = []
    for scene in scenes:
        tokens = keyframes(nusc, scene)
        request = request_from_nuscenes(nusc, tokens[len(tokens) // 2])  # the middle keyframe
        panels = detect_panels(request, detectors, args.threshold, args.gallery_width)
        rows.append(stack(panels, horizontal=True))
    path = args.out / "night_gallery.jpg"
    stack(rows).save(path, quality=85)
    print(f"{path}: {[s['name'] for s in scenes]}, {path.stat().st_size / 1e3:.0f} kB")


def depth_colour(depth_m: np.ndarray) -> np.ndarray:
    """Near = warm, far = cool, `(..., 3)` uint8."""
    t = np.clip((depth_m - NEAR_M) / (FAR_M - NEAR_M), 0.0, 1.0)
    rgba = np.asarray(matplotlib.colormaps["turbo_r"](t))  # a scalar gives a plain tuple
    return (rgba[..., :3] * 255).astype(np.uint8)


def lidar_input(nusc: NuScenes, scene: dict, thicken_px: int, args: argparse.Namespace) -> None:
    tokens = keyframes(nusc, scene)
    request = request_from_nuscenes(nusc, tokens[len(tokens) // 2])
    height, width = request.image.shape[:2]
    pixels, depths = project_to_image(
        request.lidar_points,
        request.calibration.T_cam_from_lidar,
        request.calibration.intrinsic,
        (height, width),
    )
    photo = Image.fromarray(request.image)
    sparse = photo.copy()
    pen = ImageDraw.Draw(sparse)
    for (u, v), z in zip(pixels, depths, strict=True):
        r = DOT_RADIUS_PX
        pen.ellipse((u - r, v - r, u + r, v + r), fill=tuple(int(c) for c in depth_colour(z)))

    dense = thicken_depth(make_sparse_depth(pixels, depths, (height, width)), thicken_px)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    canvas[dense > 0] = depth_colour(dense[dense > 0])
    titles = [
        "1. night photo",
        f"2. LiDAR scan projected into the photo ({len(depths):,} points)",
        f"3. network input: each point spread to {thicken_px}x{thicken_px} px (+ validity mask)",
    ]
    font = ImageFont.load_default(size=26)
    panels = []
    for title, image in zip(titles, [photo, sparse, Image.fromarray(canvas)], strict=True):
        image = image.copy()
        pen = ImageDraw.Draw(image)
        pen.rectangle((0, 0, width, 40), fill=(0, 0, 0))
        pen.text((10, 6), title, fill=(255, 255, 255), font=font)
        panels.append(
            image.resize((args.gallery_width, round(height * args.gallery_width / width)))
        )
    path = args.out / "lidar_input.jpg"
    stack(panels).save(path, quality=85)
    print(f"{path}: {scene['name']}, {path.stat().st_size / 1e3:.0f} kB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_DIR")
    parser.add_argument("--root", default="dataset/v1.0-trainval")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--gif-width", type=int, default=400)
    parser.add_argument("--gallery-width", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    detectors = {s.split("=", 1)[0]: OnnxDetector(Path(s.split("=", 1)[1])) for s in args.model}
    thicken_px = int(OmegaConf.load(PREP_CONFIG).depth_thicken_px)
    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    scenes = busiest_night_scenes(nusc, GALLERY_SCENES)
    args.out.mkdir(parents=True, exist_ok=True)
    night_drive(nusc, scenes[0], detectors, args)
    night_gallery(nusc, scenes, detectors, args)
    lidar_input(nusc, scenes[0], thicken_px, args)


if __name__ == "__main__":
    main()
