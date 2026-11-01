"""A local demo: the camera-only model and the camera + LiDAR model side by side (ADR 0007).

    python -m sensor_fusion.serve.demo \\
        --model rgb=outputs/serve/rgb --model fusion=outputs/serve/fusion_dense_head \\
        --results outputs/eval/retrain-best/results.md

Pick a day or a night frame from the validation split and compare each model's boxes and
distances with the labels. It runs from the dataset on this machine: nuScenes images are not
redistributable, so there is no hosted copy.
"""

import argparse
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sensor_fusion.data.labels import CLASS_NAMES, frame_labels
from sensor_fusion.data.splits import get_split_scenes, scene_is_night
from sensor_fusion.evaluation.matching import box_iou
from sensor_fusion.serve.inference import Detection, OnnxDetector
from sensor_fusion.serve.samples import request_from_nuscenes

Item = tuple[str, float | None, tuple[float, float, float, float], float]  # class, score, box, m
GROUND_TRUTH_COLOUR = (60, 220, 60)
MODEL_COLOURS = [(230, 90, 60), (70, 140, 255), (240, 200, 40)]
MATCH_IOU = 0.5


@dataclass(frozen=True)
class DemoFrame:
    label: str
    sample_token: str
    is_night: bool


def pick_frames(nusc: Any, split: str = "val", per_group: int = 6) -> list[DemoFrame]:
    """A few day and a few night frames, spread over different scenes (the middle frame of each)."""
    wanted = set(get_split_scenes(split))
    scenes = [s for s in nusc.scene if s["name"] in wanted]
    frames: list[DemoFrame] = []
    for night in (False, True):
        group = [s for s in scenes if scene_is_night(s) == night]
        step = max(1, len(group) // per_group)
        for scene in group[::step][:per_group]:
            token = scene["first_sample_token"]
            for _ in range(scene["nbr_samples"] // 2):
                token = nusc.get("sample", token)["next"] or token
            kind = "night" if night else "day"
            frames.append(DemoFrame(f"{kind}: {scene['name']}", token, night))
    return frames


def ground_truth(nusc: Any, sample_token: str) -> list[Item]:
    """The labelled objects of a frame with the training label rules (boxes in photo pixels)."""
    sample = nusc.get("sample", sample_token)
    boxes, classes, distances = frame_labels(nusc, sample["data"]["CAM_FRONT"])
    return [
        (CLASS_NAMES[int(c)], None, (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(d))
        for b, c, d in zip(boxes, classes, distances, strict=True)
    ]


def as_items(detections: Sequence[Detection]) -> list[Item]:
    return [(d.class_name, d.score, d.box, d.distance_m) for d in detections]


def draw(
    image: np.ndarray, items: Sequence[Item], colour: tuple[int, int, int], title: str
) -> Image.Image:
    """Boxes with `class distance (score)` labels, and a title bar."""
    canvas = Image.fromarray(image).copy()
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for name, score, (x1, y1, x2, y2), distance in items:
        pen.rectangle((x1, y1, x2, y2), outline=colour, width=3)
        text = f"{name} {distance:.0f} m" + (f" ({score:.2f})" if score is not None else "")
        left, top, right, bottom = pen.textbbox((x1, y1), text, font=font)
        pen.rectangle((left, top - 4, right + 4, bottom), fill=colour)
        pen.text((x1 + 2, top - 4), text, fill=(0, 0, 0), font=font)
    pen.rectangle((0, 0, canvas.width, 36), fill=(0, 0, 0))
    pen.text((8, 4), title, fill=colour, font=font)
    return canvas


def match_summary(detections: Sequence[Item], truth: Sequence[Item]) -> tuple[int, float | None]:
    """How many labelled objects were found (IoU >= 0.5, same class) and the median relative
    distance error over those. Quick and per frame; the real evaluation is `sensor_fusion.evaluate`.
    """
    found, errors, taken = 0, [], set()
    for name, _, box, distance in sorted(detections, key=lambda d: -(d[1] or 0.0)):
        candidates = [i for i, t in enumerate(truth) if t[0] == name and i not in taken]
        if not candidates:
            continue
        ious = box_iou(np.array([box]), np.array([truth[i][2] for i in candidates]))[0]
        best = int(ious.argmax())
        if ious[best] >= MATCH_IOU:
            taken.add(candidates[best])
            found += 1
            true_distance = truth[candidates[best]][3]
            errors.append(abs(distance - true_distance) / true_distance)
    return found, (statistics.median(errors) if errors else None)


def compare(
    nusc: Any,
    frame: DemoFrame,
    detectors: Mapping[str, OnnxDetector],
    score_threshold: float,
) -> tuple[list[Image.Image], str]:
    """One picture per model plus the labels, and a one-paragraph summary of the frame."""
    request = request_from_nuscenes(nusc, frame.sample_token)
    truth = ground_truth(nusc, frame.sample_token)
    images = [draw(request.image, truth, GROUND_TRUTH_COLOUR, f"labels ({len(truth)} objects)")]
    lines = [f"**{frame.label}**: {len(truth)} labelled objects."]
    for colour, (name, detector) in zip(MODEL_COLOURS, detectors.items(), strict=False):
        found, _ = detector.detect(
            request.image,
            request.lidar_points,
            request.calibration,
            score_threshold=score_threshold,
        )
        items = as_items(found)
        hits, error = match_summary(items, truth)
        images.append(draw(request.image, items, colour, f"{name} ({len(items)} detections)"))
        error_text = "n/a" if error is None else f"{100 * error:.1f}%"
        lines.append(f"- {name}: {hits} of {len(truth)} found, median distance error {error_text}")
    return images, "\n".join(lines)


def build_demo(
    nusc: Any,
    detectors: Mapping[str, OnnxDetector],
    frames: Sequence[DemoFrame],
    results_markdown: str = "",
) -> Any:
    """The Gradio page. Imported lazily so that the module works without Gradio installed."""
    import gradio as gr

    by_label = {f.label: f for f in frames}
    names = list(detectors)

    def run(label: str, threshold: float) -> tuple[Any, ...]:
        images, summary = compare(nusc, by_label[label], detectors, threshold)
        return (*images, summary)

    with gr.Blocks(title="Night-time camera + LiDAR detector") as page:
        gr.Markdown(
            "# Night-time camera + LiDAR detector\n"
            "Green: labels. Each model's boxes carry `class distance (score)`. "
            "Pick a night frame to see what LiDAR adds to the distance estimate."
        )
        with gr.Row():
            choice = gr.Dropdown(list(by_label), value=frames[0].label, label="Validation frame")
            threshold = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Score threshold")
        summary = gr.Markdown()
        with gr.Row():
            outputs = [gr.Image(label=t, type="pil") for t in ("labels", *names)]
        for trigger in (choice.change, threshold.change):
            trigger(run, [choice, threshold], [*outputs, summary])
        page.load(run, [choice, threshold], [*outputs, summary])
        if results_markdown:
            with gr.Accordion("Measured results on the whole validation split", open=False):
                gr.Markdown(results_markdown)
    return page


def main() -> None:
    from nuscenes.nuscenes import NuScenes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_DIR")
    parser.add_argument("--root", default="dataset/v1.0-trainval")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--results", type=Path, default=None, help="a results.md to show")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    detectors = {
        spec.split("=", 1)[0]: OnnxDetector(Path(spec.split("=", 1)[1])) for spec in args.model
    }
    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    results = args.results.read_text(encoding="utf-8") if args.results else ""
    build_demo(nusc, detectors, pick_frames(nusc), results).launch(server_port=args.port)


if __name__ == "__main__":
    main()
