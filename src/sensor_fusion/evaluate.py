"""Evaluation entry point: baseline vs fusion on the validation split (ADR 0011, 0012).

    CUDA_VISIBLE_DEVICES=0 python -m sensor_fusion.evaluate \\
        baseline=outputs/runs/rgb-s42/best.pt fusion=outputs/runs/fusion-s42/best.pt

Each checkpoint carries the config it was trained with; the model and the input preparation are
rebuilt from it, so a model can never be evaluated with settings it was not trained with.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sensor_fusion.data.dataset import NuScenesFrontDataset
from sensor_fusion.data.labels import CLASS_NAMES
from sensor_fusion.data.prepared import PreparedDataset, collate_prepared
from sensor_fusion.evaluation.inference import collect_results, scene_info_from
from sensor_fusion.evaluation.matching import FrameResult
from sensor_fusion.evaluation.report import EvalSettings, build_report, to_markdown
from sensor_fusion.train import build_model
from sensor_fusion.training.tracking import git_info

log = logging.getLogger(__name__)


def json_safe(value: Any) -> Any:
    """JSON has no NaN or infinity: write them as null / strings so any reader can open the file."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return str(value) if math.isinf(value) else value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def load_for_evaluation(path: str, device: torch.device) -> tuple[torch.nn.Module, DictConfig, int]:
    """Rebuild a model from its checkpoint: weights, and the config it was trained with."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = OmegaConf.create(payload["config"])
    assert isinstance(config, DictConfig)
    config.model.pretrained = False  # the weights come from the checkpoint, not from ImageNet
    model = build_model(config)
    model.load_state_dict(payload["model"])
    return model.to(device), config, int(payload["epoch"])


def evaluate_models(cfg: DictConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run both models over the split and build the report. Returns (report, provenance)."""
    loaded = {name: load_for_evaluation(cfg[name], device) for name in ("baseline", "fusion")}
    preps = {name: OmegaConf.to_container(config.prep) for name, (_, config, _) in loaded.items()}
    if preps["baseline"] != preps["fusion"]:
        raise ValueError("the two checkpoints were trained with different input preparation")

    frames: dict[str, list[FrameResult]] = {}
    for name, (model, config, epoch) in loaded.items():
        base = NuScenesFrontDataset(
            cfg.data_root, cfg.version, cfg.split, config.data.camera, config.data.lidar
        )
        dataset = PreparedDataset(
            base, in_channels=config.model.in_channels, prep=config.prep, train=False
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=collate_prepared,
        )
        log.info("evaluating %s (epoch %d) on %s: %d frames", name, epoch, cfg.split, len(base))
        frames[name] = collect_results(
            model,
            loader,
            scene_info_from(base.nusc),
            device=device,
            amp=cfg.evaluation.amp,
            stride=int(config.prep.output_stride),
            top_k=cfg.evaluation.top_k,
            score_threshold=cfg.evaluation.score_threshold,
        )

    e = cfg.evaluation
    settings = EvalSettings(
        iou_thresholds=tuple(e.iou_thresholds),
        distance_score_threshold=e.distance_score_threshold,
        min_objects=e.min_objects,
        n_boot=e.n_boot,
        seed=e.seed,
        distance_edges=tuple(e.distance_edges),
        confidence=e.confidence,
    )
    report = build_report(frames, CLASS_NAMES, settings)
    provenance = {
        "checkpoints": {name: cfg[name] for name in loaded},
        "epochs": {name: epoch for name, (_, _, epoch) in loaded.items()},
        "split": cfg.split,
        **git_info(),
    }
    return report, provenance


def write_results(report: dict[str, Any], provenance: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json_safe({"provenance": provenance, "report": report})
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = to_markdown(report, CLASS_NAMES)
    (out_dir / "results.md").write_text(markdown, encoding="utf-8")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report, provenance = evaluate_models(cfg, device)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    write_results(report, provenance, out_dir)
    log.info("wrote %s and %s", out_dir / "results.md", out_dir / "results.json")


if __name__ == "__main__":
    main()
