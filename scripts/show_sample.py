"""Draw one dataset frame for a visual check: LiDAR depth points, 2D boxes and distances.

    python scripts/show_sample.py --split mini_dev_val --index 10 50

Saves PNGs to outputs/debug/ (ignored by git). Use it after any change to projection or label
code: points should sit on the objects, and boxes should hug them.
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.labels import CLASS_NAMES

NEAR_M, FAR_M = 3.0, 60.0  # depth range of the colour scale: near = red, far = blue


def _depth_color(depth_m: float) -> tuple[int, int, int]:
    t = float(np.clip((depth_m - NEAR_M) / (FAR_M - NEAR_M), 0, 1))
    red, green, blue = colorsys.hsv_to_rgb(0.66 * t, 1.0, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def draw_sample(item: dict, out_path: Path) -> None:
    image = Image.fromarray(item["image"])
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)

    rows, columns = np.nonzero(item["depth"])
    for row, column in zip(rows, columns, strict=True):
        color = _depth_color(float(item["depth"][row, column]))
        draw.ellipse([column - 3, row - 3, column + 3, row + 3], fill=color)

    for box, class_id, distance in zip(
        item["boxes"], item["classes"], item["distances"], strict=True
    ):
        x1, y1, x2, y2 = (float(v) for v in box)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
        label = f"{CLASS_NAMES[class_id]} {distance:.1f} m"
        text_x = min(x1 + 2, image.width - draw.textlength(label, font=font) - 2)  # stay in frame
        draw.text(
            (text_x, max(y1 - 26, 0)),
            label,
            fill=(255, 255, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="dataset/v1.0-mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_dev_val")
    parser.add_argument("--index", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--out", default="outputs/debug")
    args = parser.parse_args()

    dataset = NuScenesFrontDataset(args.root, args.version, args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in args.index:
        item = dataset[index]
        out_path = out_dir / f"sample_{args.split}_{index}.png"
        draw_sample(item, out_path)
        print(
            f"{out_path}: {'night' if item['is_night'] else 'day'}, "
            f"{np.count_nonzero(item['depth'])} depth pixels, {len(item['boxes'])} boxes"
        )


if __name__ == "__main__":
    main()
