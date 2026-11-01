"""Decoding to results, the report and its tables, and the whole evaluation on nuScenes-mini."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.labels import CLASS_NAMES
from sensor_fusion.data.targets import encode_targets
from sensor_fusion.evaluate import evaluate_models, json_safe, write_results
from sensor_fusion.evaluation.inference import results_from_outputs
from sensor_fusion.evaluation.matching import FrameResult
from sensor_fusion.evaluation.metrics import build_cell_data, cell_metrics
from sensor_fusion.evaluation.report import CELLS, EvalSettings, build_report, to_markdown
from sensor_fusion.models.detector import Outputs
from sensor_fusion.train import build_model
from sensor_fusion.training.checkpoint import save_checkpoint
from sensor_fusion.training.optim import build_optimizer, build_scheduler

ROOT = Path(__file__).parents[1]
needs_mini = pytest.mark.skipif(
    not (ROOT / "dataset" / "v1.0-mini").exists(), reason="nuScenes-mini not found"
)
SIZE = (64, 128)
STRIDE = 4


# --- decoding: the targets, fed back as network output, must score perfectly ------------------


def _perfect_batch() -> tuple[Outputs, dict[str, Any]]:
    boxes = np.array([[16, 16, 56, 48], [72, 20, 112, 56]], dtype=np.float32)
    classes = np.array([0, 2], dtype=np.int64)
    distances = np.array([12.0, 45.0], dtype=np.float32)
    heat, reg, _ = encode_targets(
        boxes, classes, distances, input_size=SIZE, num_classes=4, stride=STRIDE, peak_alpha=0.54
    )
    probs = torch.from_numpy(heat).clamp(1e-6, 1 - 1e-6)[None]
    outputs = Outputs(torch.log(probs / (1 - probs)), torch.from_numpy(reg)[None])
    batch = {
        "sample_token": ["tok"],
        "is_night": torch.tensor([True]),
        "boxes": [torch.from_numpy(boxes)],
        "classes": [torch.from_numpy(classes)],
        "distances": [torch.from_numpy(distances)],
    }
    return outputs, batch


def test_targets_fed_back_as_output_are_found_exactly_and_score_perfectly() -> None:
    outputs, batch = _perfect_batch()
    (result,) = results_from_outputs(
        outputs,
        batch,
        lambda t: ("scene-x", "singapore"),
        stride=STRIDE,
        top_k=10,
        score_threshold=0.05,
    )
    assert (result.scene, result.city, result.is_night) == ("scene-x", "singapore", True)
    order = result.det_classes.argsort()
    assert result.det_classes[order].tolist() == [0, 2]
    assert np.allclose(result.det_boxes[order], batch["boxes"][0].numpy(), atol=1e-3)
    assert np.allclose(result.det_distances[order], [12.0, 45.0], rtol=1e-4)

    data = build_cell_data(
        [result], iou_thresholds=(0.5, 0.9), num_classes=4, distance_score_threshold=0.3
    )
    metrics = cell_metrics(data, np.ones(1), eligible=[0, 2])
    assert metrics.map50 == pytest.approx(1.0) and metrics.map == pytest.approx(1.0)
    assert metrics.dist_rel_median == pytest.approx(0.0, abs=1e-4)


# --- report ------------------------------------------------------------------------------------


def _frame(scene: str, token: str, *, city: str, night: bool, good: bool) -> FrameResult:
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    shifted = np.array([[40, 40, 50, 50]], dtype=np.float32)
    return FrameResult(
        token=token,
        scene=scene,
        city=city,
        is_night=night,
        det_boxes=box if good else shifted,
        det_scores=np.array([0.9], dtype=np.float32),
        det_classes=np.array([0]),
        det_distances=np.array([11.0], dtype=np.float32),
        gt_boxes=box,
        gt_classes=np.array([0]),
        gt_distances=np.array([10.0], dtype=np.float32),
    )


def _two_models() -> dict[str, list[FrameResult]]:
    layout = [  # (scene, city, night), three frames each
        ("b1", "boston", False),
        ("b2", "boston", False),
        ("s1", "singapore", False),
        ("s2", "singapore", False),
        ("n1", "singapore", True),
        ("n2", "singapore", True),
    ]
    frames: dict[str, list[FrameResult]] = {"baseline": [], "fusion": []}
    for scene, city, night in layout:
        for k in range(3):
            # fusion finds everything; the baseline misses all the night ones
            frames["baseline"].append(
                _frame(scene, f"{scene}-{k}", city=city, night=night, good=not night)
            )
            frames["fusion"].append(
                _frame(scene, f"{scene}-{k}", city=city, night=night, good=True)
            )
    return frames


SETTINGS = EvalSettings(
    iou_thresholds=(0.5, 0.75),
    distance_score_threshold=0.3,
    min_objects=5,
    n_boot=40,
    seed=0,
    distance_edges=(0.0, 20.0, math.inf),
)


def test_report_has_every_cell_with_counts_and_a_paired_difference() -> None:
    report = build_report(_two_models(), CLASS_NAMES, SETTINGS)
    assert list(report["cells"]) == list(CELLS)
    night = report["cells"]["singapore_night"]
    assert (night["scenes"], night["frames"], night["objects"]["car"]) == (2, 6, 6)
    assert night["eligible_classes"] == ["car"]  # 6 >= 5 labelled cars; other classes have none
    assert night["models"]["baseline"]["point"]["map50"] == 0.0
    assert night["models"]["fusion"]["point"]["map50"] == pytest.approx(1.0)
    gain = night["difference"]["map50"]
    assert gain["value"] == pytest.approx(1.0) and gain["ci"][0] == pytest.approx(1.0)
    day = report["cells"]["day"]["difference"]["map50"]
    assert day["value"] == pytest.approx(0.0) and day["ci"] == (0.0, 0.0)  # both models identical
    assert "boston_day" in report["cells"]


def test_the_minimum_object_rule_leaves_small_cells_without_an_ap() -> None:
    strict = EvalSettings(**{**SETTINGS.__dict__, "min_objects": 100})
    report = build_report(_two_models(), CLASS_NAMES, strict)
    cell = report["cells"]["night"]
    assert cell["eligible_classes"] == []
    assert math.isnan(cell["models"]["fusion"]["point"]["map50"])
    assert "-" in to_markdown(report, CLASS_NAMES)


def test_models_must_cover_the_same_frames() -> None:
    frames = _two_models()
    frames["fusion"] = frames["fusion"][:-1]
    with pytest.raises(ValueError, match="same frames"):
        build_report(frames, CLASS_NAMES, SETTINGS)


def test_markdown_has_the_tables_and_shows_intervals_in_percent() -> None:
    text = to_markdown(build_report(_two_models(), CLASS_NAMES, SETTINGS), CLASS_NAMES)
    for heading in ("## Data behind each cell", "### mAP@0.5 (%)", "## Per-class AP@0.5 (%)"):
        assert heading in text
    assert "singapore_night" in text and "100.0 [100.0, 100.0]" in text
    assert "0-20 m" in text and "20-inf m" in text


def test_json_output_has_no_nan_or_infinity() -> None:
    cleaned = json_safe({"a": float("nan"), "b": [float("inf"), 1.5], "c": (float("-inf"),)})
    assert cleaned == {"a": None, "b": ["inf", 1.5], "c": ["-inf"]}
    json.dumps(cleaned, allow_nan=False)  # would raise on a stray nan


def test_results_are_written_as_json_and_markdown(tmp_path: Path) -> None:
    report = build_report(_two_models(), CLASS_NAMES, SETTINGS)
    write_results(report, {"split": "val"}, tmp_path)
    loaded = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert loaded["provenance"]["split"] == "val" and "cells" in loaded["report"]
    assert (tmp_path / "results.md").read_text(encoding="utf-8").startswith("# Evaluation results")


# --- the whole thing on nuScenes-mini ----------------------------------------------------------


def _train_cfg(model: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=[f"model={model}", "model.pretrained=false"])


def _checkpoint(tmp_path: Path, model_name: str) -> str:
    cfg = _train_cfg(model_name)
    model = build_model(cfg)
    optimizer = build_optimizer(model, lr_head=1e-3, lr_backbone=1e-4, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, total_steps=2, warmup_steps=0, min_lr_fraction=0.1)
    path = tmp_path / f"{model_name}.pt"
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        epoch=3,
        step=2,
        best_val=1.0,
        config={str(k): v for k, v in container.items()},
    )
    return str(path)


@needs_mini
def test_evaluating_two_checkpoints_on_mini_produces_a_full_report(tmp_path: Path) -> None:
    baseline, fusion = _checkpoint(tmp_path, "rgb"), _checkpoint(tmp_path, "fusion")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="evaluate",
            overrides=[
                f"baseline={baseline}",
                f"fusion={fusion}",
                "data_root=dataset/v1.0-mini",
                "version=v1.0-mini",
                "split=mini_dev_val",
                "num_workers=0",
                "batch_size=4",
                "evaluation.n_boot=5",
                "evaluation.amp=false",
            ],
        )
    cfg.data_root = str(ROOT / cfg.data_root)
    report, provenance = evaluate_models(cfg, torch.device("cpu"))
    assert provenance["epochs"] == {"baseline": 3, "fusion": 3}
    assert {"all", "day", "night"} <= set(report["cells"])
    expected = len(NuScenesFrontDataset(cfg.data_root, "v1.0-mini", "mini_dev_val"))
    assert report["cells"]["all"]["frames"] == expected  # every frame of the split, once
    # night is Singapore in the dev split, and Boston has no night
    assert report["cells"]["singapore_night"]["frames"] == report["cells"]["night"]["frames"]
    assert "boston_night" not in report["cells"]
    assert "difference" in report["cells"]["all"]


def test_cells_with_few_scenes_are_marked_in_every_table() -> None:
    text = to_markdown(build_report(_two_models(), CLASS_NAMES, SETTINGS), CLASS_NAMES)
    assert "† fewer than 10 scenes" in text
    assert "| singapore_night † |" in text  # two scenes only
    assert text.count("| all † |") >= 4  # data, three headline tables, per-class, distance
