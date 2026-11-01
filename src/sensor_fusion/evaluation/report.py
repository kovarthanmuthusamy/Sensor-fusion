"""Turn the detections of one or two models into the day/night results tables (ADR 0011, 0012)."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sensor_fusion.evaluation.bootstrap import (
    HEADLINE_METRICS,
    bootstrap_samples,
    metrics_to_dict,
    percentile_interval,
)
from sensor_fusion.evaluation.matching import FrameResult
from sensor_fusion.evaluation.metrics import (
    build_cell_data,
    cell_metrics,
    distance_by_range,
    eligible_classes,
)

# The groups of frames results are reported for. Night exists only in Singapore (ADR 0008), so
# the Singapore pair is the control that holds the city fixed.
CELLS: dict[str, Callable[[FrameResult], bool]] = {
    "all": lambda f: True,
    "day": lambda f: not f.is_night,
    "night": lambda f: f.is_night,
    "boston_day": lambda f: f.city == "boston" and not f.is_night,
    "singapore_day": lambda f: f.city == "singapore" and not f.is_night,
    "singapore_night": lambda f: f.city == "singapore" and f.is_night,
}


# With fewer scenes than this, a bootstrap over scenes cannot say much (one scene gives an interval
# of zero width), so such cells are marked in the tables.
MIN_SCENES_FOR_INTERVAL = 10


@dataclass(frozen=True)
class EvalSettings:
    iou_thresholds: tuple[float, ...]  # the first one is also used for the distance error
    distance_score_threshold: float
    min_objects: int
    n_boot: int
    seed: int
    distance_edges: tuple[float, ...]
    confidence: float = 0.95


def _check_same_frames(frames: Mapping[str, Sequence[FrameResult]]) -> None:
    reference = [f.token for f in next(iter(frames.values()))]
    for model, results in frames.items():
        if [f.token for f in results] != reference:
            raise ValueError(
                f"model {model!r} was not evaluated on the same frames in the same order"
            )


def build_report(
    frames: Mapping[str, Sequence[FrameResult]],
    class_names: Sequence[str],
    settings: EvalSettings,
) -> dict[str, Any]:
    """Results for every cell: point estimates, bootstrap intervals, and (for exactly two models)
    the difference second minus first with its paired interval."""
    _check_same_frames(frames)
    models = list(frames)
    report: dict[str, Any] = {"settings": asdict(settings), "models": models, "cells": {}}
    for cell_name, belongs in CELLS.items():
        members = {m: [f for f in results if belongs(f)] for m, results in frames.items()}
        if not members[models[0]]:
            continue
        data = {
            m: build_cell_data(
                members[m],
                iou_thresholds=settings.iou_thresholds,
                num_classes=len(class_names),
                distance_score_threshold=settings.distance_score_threshold,
            )
            for m in models
        }
        first = data[models[0]]
        eligible = eligible_classes(first, settings.min_objects)
        samples = bootstrap_samples(
            data, eligible, class_names, n_boot=settings.n_boot, seed=settings.seed
        )
        entry: dict[str, Any] = {
            "scenes": len(first.scene_names),
            "frames": int(first.frames_by_scene.sum()),
            "objects": {name: int(n) for name, n in zip(class_names, first.n_gt, strict=True)},
            "eligible_classes": [class_names[c] for c in eligible],
            "models": {},
        }
        points: dict[str, dict[str, float]] = {}
        for m in models:
            points[m] = metrics_to_dict(
                cell_metrics(data[m], np.ones(len(first.scene_names)), eligible), class_names
            )
            entry["models"][m] = {
                "point": points[m],
                "ci": {
                    name: percentile_interval(values, settings.confidence)
                    for name, values in samples[m].items()
                },
                "by_range": distance_by_range(data[m], settings.distance_edges),
            }
        if len(models) == 2:
            a, b = models
            entry["difference"] = {
                name: {
                    "value": points[b][name] - points[a][name],
                    "ci": percentile_interval(
                        samples[b][name] - samples[a][name], settings.confidence
                    ),
                }
                for name in points[a]
            }
        report["cells"][cell_name] = entry
    return report


# --- markdown ------------------------------------------------------------------------------------


def _label(name: str, cell: Mapping[str, Any]) -> str:
    return f"{name} †" if cell["scenes"] < MIN_SCENES_FOR_INTERVAL else name


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100 * value:.1f}"


def _with_ci(value: float, ci: Sequence[float]) -> str:
    return "n/a" if math.isnan(value) else f"{_pct(value)} [{_pct(ci[0])}, {_pct(ci[1])}]"


def _metres(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


def to_markdown(report: Mapping[str, Any], class_names: Sequence[str]) -> str:
    """Readable tables. AP and relative error are shown in percent, brackets are 95% intervals
    from a bootstrap over scenes. Classes below the minimum object count show `-`."""
    models: list[str] = report["models"]
    cells: Mapping[str, Any] = report["cells"]
    s = report["settings"]
    out = [
        "# Evaluation results",
        "",
        f"IoU thresholds {s['iou_thresholds'][0]}..{s['iou_thresholds'][-1]}; a class gets an AP "
        f"in a cell only with at least {s['min_objects']} labelled objects (ADR 0011). Brackets: "
        f"{int(100 * s['confidence'])}% interval from {s['n_boot']} scene-level resamples.",
        "",
        f"† fewer than {MIN_SCENES_FOR_INTERVAL} scenes: the interval is not reliable.",
        "",
        "## Data behind each cell",
        "",
        "| cell | scenes | frames | " + " | ".join(class_names) + " |",
        "|---|---|---|" + "---|" * len(class_names),
    ]
    for name, cell in cells.items():
        counts = " | ".join(f"{cell['objects'][c]:,}" for c in class_names)
        out.append(f"| {_label(name, cell)} | {cell['scenes']} | {cell['frames']:,} | {counts} |")

    out += ["", "## Headline", ""]
    columns = list(models) + (["difference (2nd - 1st)"] if len(models) == 2 else [])
    for metric, title in (
        ("map50", "mAP@0.5 (%)"),
        ("map", "mAP@[0.5:0.95] (%)"),
        ("dist_rel_median", "median relative distance error (%)"),
    ):
        out += [
            f"### {title}",
            "",
            "| cell | " + " | ".join(columns) + " |",
            "|---|" + "---|" * len(columns),
        ]
        for name, cell in cells.items():
            row = [
                _with_ci(cell["models"][m]["point"][metric], cell["models"][m]["ci"][metric])
                for m in models
            ]
            if "difference" in cell:
                d = cell["difference"][metric]
                row.append(_with_ci(d["value"], d["ci"]))
            out.append(f"| {_label(name, cell)} | " + " | ".join(row) + " |")
        out.append("")

    out += ["## Per-class AP@0.5 (%)", "", "`-` = fewer than the minimum number of objects.", ""]
    out += [
        "| cell | model | " + " | ".join(class_names) + " |",
        "|---|---|" + "---|" * len(class_names),
    ]
    for name, cell in cells.items():
        for m in models:
            point, ci = cell["models"][m]["point"], cell["models"][m]["ci"]
            values = [
                _with_ci(point[f"ap50_{c}"], ci[f"ap50_{c}"])
                if c in cell["eligible_classes"]
                else "-"
                for c in class_names
            ]
            out.append(f"| {_label(name, cell)} | {m} | " + " | ".join(values) + " |")

    out += ["", "## Distance error by true distance (median relative error %, matched count)", ""]
    ranges = next(iter(cells.values()))["models"][models[0]]["by_range"]
    labels = [
        f"{r['low']:g}-{'inf' if math.isinf(r['high']) else format(r['high'], 'g')} m"
        for r in ranges
    ]
    out += ["| cell | model | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for name, cell in cells.items():
        for m in models:
            row = [
                f"{_pct(r['median_rel_error'])} (n={int(r['matched'])})"
                for r in cell["models"][m]["by_range"]
            ]
            out.append(f"| {_label(name, cell)} | {m} | " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


__all__ = ["CELLS", "HEADLINE_METRICS", "EvalSettings", "build_report", "to_markdown"]
