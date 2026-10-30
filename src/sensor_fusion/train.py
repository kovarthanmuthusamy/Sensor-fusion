"""Training entry point (ADR 0010).

    CUDA_VISIBLE_DEVICES=0 python -m sensor_fusion.train model=fusion data=trainval
    CUDA_VISIBLE_DEVICES=0 python -m sensor_fusion.train trainer.overfit_one_batch=true

The RGB baseline and the fusion model run through exactly this code; only `model` differs.
"""

import logging
import os
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
from sensor_fusion.models.detector import Detector
from sensor_fusion.training.checks import assert_scenes_disjoint, require_pinned_gpu
from sensor_fusion.training.loop import (
    LoopSettings,
    LossSettings,
    fit,
    make_scaler,
    overfit_one_batch,
)
from sensor_fusion.training.optim import build_optimizer, build_scheduler
from sensor_fusion.training.tracking import start_run
from sensor_fusion.utils.env import collect_env_info
from sensor_fusion.utils.seed import make_generator, seed_everything

log = logging.getLogger(__name__)

OVERFIT_SUCCESS_RATIO = 0.1  # the loss must fall below this fraction of its first value


def build_model(cfg: DictConfig) -> Detector:
    return Detector(
        in_channels=cfg.model.in_channels,
        num_classes=len(CLASS_NAMES),
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        neck_channels=cfg.model.neck_channels,
        head_channels=cfg.model.head_channels,
        prior_prob=cfg.model.prior_prob,
        distance_init_m=cfg.model.distance_init_m,
    )


def build_loaders(cfg: DictConfig, *, augment: bool) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Train and val loaders. Fails if the two splits share a scene."""
    sources = [
        NuScenesFrontDataset(
            cfg.data.root, cfg.data.version, split, cfg.data.camera, cfg.data.lidar
        )
        for split in (cfg.data.train_split, cfg.data.val_split)
    ]
    assert_scenes_disjoint(sources[0].scene_names, sources[1].scene_names)
    train_set, val_set = (
        PreparedDataset(source, in_channels=cfg.model.in_channels, prep=cfg.prep, train=train)
        for source, train in zip(sources, (augment, False), strict=True)
    )
    workers = cfg.trainer.num_workers

    def loader(dataset: PreparedDataset, *, train: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=cfg.trainer.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_prepared,
            generator=make_generator(cfg.seed),
        )

    return loader(train_set, train=True), loader(val_set, train=False)


def run_training(cfg: DictConfig, out_dir: Path) -> None:
    """Everything `main` does once Hydra has composed the config and chosen the output folder."""
    seed_everything(cfg.seed)
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    config: dict[str, Any] = {str(key): value for key, value in container.items()}
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    for key, value in collect_env_info().items():
        log.info("%-22s%s", key, value)
    if cfg.trainer.require_pinned_gpu:
        require_pinned_gpu(os.environ, torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overfit = cfg.trainer.overfit_one_batch
    train_loader, val_loader = build_loaders(cfg, augment=not overfit)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(
        model,
        lr_head=cfg.trainer.lr_head,
        lr_backbone=cfg.trainer.lr_backbone,
        weight_decay=cfg.trainer.weight_decay,
    )
    scaler = make_scaler(device, cfg.trainer.amp)
    loss = LossSettings(cfg.loss.focal_alpha, cfg.loss.focal_beta, dict(cfg.loss.weights))

    if overfit:
        history = overfit_one_batch(
            model,
            next(iter(train_loader)),
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss=loss,
            steps=cfg.trainer.overfit_steps,
            amp=cfg.trainer.amp,
            grad_clip=cfg.trainer.grad_clip,
        )
        ratio = history[-1] / history[0]
        log.info(
            "overfit one batch: loss %.4f -> %.4f (ratio %.3f)", history[0], history[-1], ratio
        )
        if ratio > OVERFIT_SUCCESS_RATIO:
            raise SystemExit(f"FAILED: loss only fell to {ratio:.1%} of its start")
        log.info("PASSED: the model can memorise one batch")
        return

    total_steps = cfg.trainer.epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=cfg.trainer.warmup_steps,
        min_lr_fraction=cfg.trainer.min_lr_fraction,
    )
    run = start_run(
        config=config,
        mode=cfg.trainer.wandb.mode,
        project=cfg.trainer.wandb.project,
        entity=cfg.trainer.wandb.entity,
        name=f"{cfg.model.name}-{cfg.data.name}-s{cfg.seed}",
        tags=[cfg.model.name, cfg.data.name],
        out_dir=out_dir,
    )
    try:
        result = fit(
            model,
            train_loader,
            val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            loss=loss,
            settings=LoopSettings(
                epochs=cfg.trainer.epochs,
                grad_clip=cfg.trainer.grad_clip,
                log_every=cfg.trainer.log_every,
                amp=cfg.trainer.amp,
                max_minutes=cfg.trainer.max_minutes,
            ),
            run=run,
            out_dir=out_dir,
            config=config,
            resume=Path(cfg.trainer.resume) if cfg.trainer.resume else None,
        )
    finally:
        run.finish()
    log.info("finished: %s", result)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
