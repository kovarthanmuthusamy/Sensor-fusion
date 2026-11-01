"""Pictures and numbers for night frames worth looking at: where LiDAR helps, where both models
miss, and where fusion is worse.

    python scripts/failure_cases.py --model rgb=outputs/serve/rgb \\
        --model fusion=outputs/serve/fusion_dense_head --night-step 3

Each picture stacks the labels, the first model and the second model (half size) and is written to
`docs/figures/`; `docs/failure_cases.md` lists the numbers. The ranking is by per-frame summaries
(`serve.demo.match_summary`); the real evaluation is `sensor_fusion.evaluate`.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from PIL import Image

from sensor_fusion.data.splits import get_split_scenes, scene_is_night
from sensor_fusion.serve.demo import (
    GROUND_TRUTH_COLOUR,
    MODEL_COLOURS,
    as_items,
    draw,
    ground_truth,
    match_summary,
)
from sensor_fusion.serve.inference import OnnxDetector
from sensor_fusion.serve.samples import request_from_nuscenes

MIN_OBJECTS = 4  # frames with fewer labelled objects say too little about a model


@dataclass
class FrameStats:
    scene: str
    token: str
    objects: int
    found: dict[str, int]
    error: dict[str, float | None]


def night_tokens(nusc: NuScenes, step: int) -> list[tuple[str, str]]:
    wanted = set(get_split_scenes("val"))
    out = []
    for scene in nusc.scene:
        if scene["name"] in wanted and scene_is_night(scene):
            token, index = scene["first_sample_token"], 0
            while token:
                if index % step == 0:
                    out.append((scene["name"], token))
                token = nusc.get("sample", token)["next"]
                index += 1
    return out


def collect(
    nusc: NuScenes,
    detectors: dict[str, OnnxDetector],
    frames: list[tuple[str, str]],
    threshold: float,
) -> list[FrameStats]:
    stats = []
    for scene, token in frames:
        request = request_from_nuscenes(nusc, token)
        truth = ground_truth(nusc, token)
        if len(truth) < MIN_OBJECTS:
            continue
        found: dict[str, int] = {}
        error: dict[str, float | None] = {}
        for name, detector in detectors.items():
            detections, _ = detector.detect(
                request.image, request.lidar_points, request.calibration, score_threshold=threshold
            )
            found[name], error[name] = match_summary(as_items(detections), truth)
        stats.append(FrameStats(scene, token, len(truth), found, error))
    return stats


def picture(
    nusc: NuScenes, detectors: dict[str, OnnxDetector], token: str, threshold: float, path: Path
) -> None:
    request = request_from_nuscenes(nusc, token)
    truth = ground_truth(nusc, token)
    panels = [draw(request.image, truth, GROUND_TRUTH_COLOUR, f"labels ({len(truth)} objects)")]
    for colour, (name, detector) in zip(MODEL_COLOURS, detectors.items(), strict=False):
        detections, _ = detector.detect(
            request.image, request.lidar_points, request.calibration, score_threshold=threshold
        )
        panels.append(draw(request.image, as_items(detections), colour, name))
    half = [p.resize((p.width // 2, p.height // 2)) for p in panels]
    sheet = Image.new("RGB", (half[0].width, sum(p.height for p in half)))
    top = 0
    for panel in half:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet.save(path, quality=85)  # JPEG: a few hundred kB instead of ~2 MB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_DIR")
    parser.add_argument("--root", default="dataset/v1.0-trainval")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--night-step", type=int, default=3, help="use every n-th night frame")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    detectors = {s.split("=", 1)[0]: OnnxDetector(Path(s.split("=", 1)[1])) for s in args.model}
    first, second = list(detectors)

    nusc = NuScenes(args.version, dataroot=args.root, verbose=False)
    stats = collect(nusc, detectors, night_tokens(nusc, args.night_step), args.threshold)
    both_measured = [s for s in stats if s.error[first] is not None and s.error[second] is not None]

    def gain(s: FrameStats) -> float:
        return (s.error[first] or 0.0) - (s.error[second] or 0.0)

    def recall(s: FrameStats) -> float:
        return max(s.found.values()) / s.objects

    groups = {
        f"{second}_fixes_distance": sorted(both_measured, key=gain, reverse=True),
        "both_miss_objects": sorted(stats, key=recall),
        f"{second}_worse_distance": sorted(both_measured, key=gain),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    lines = [
        "# Night frames worth looking at",
        "",
        f"{len(stats)} night validation frames with at least {MIN_OBJECTS} labelled objects "
        f"(every {args.night_step}rd/th frame), score threshold {args.threshold}. "
        f"Pictures: labels (top), {first} (middle), {second} (bottom). Errors are the median "
        "relative distance error over matched objects.",
        "",
    ]
    for group, ranked in groups.items():
        lines += [f"## {group.replace('_', ' ')}", ""]
        header = ["picture", "scene", "labelled", f"{first} found", f"{second} found"]
        header += [f"{first} err", f"{second} err"]
        lines += ["| " + " | ".join(header) + " |"]
        lines += ["|---|---|---|---|---|---|---|"]
        fresh = [s for s in ranked if s.token not in used][: args.per_group]
        used.update(s.token for s in fresh)  # each frame is shown once
        for rank, s in enumerate(fresh, start=1):
            name = f"{group}-{rank}.jpg"
            picture(nusc, detectors, s.token, args.threshold, args.out / name)
            err = {k: "n/a" if v is None else f"{100 * v:.1f}%" for k, v in s.error.items()}
            lines.append(
                f"| ![{name}](figures/{name}) | {s.scene} | {s.objects} | {s.found[first]} | "
                f"{s.found[second]} | {err[first]} | {err[second]} |"
            )
        lines.append("")
    Path("docs/failure_cases.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
