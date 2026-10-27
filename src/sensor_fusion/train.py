"""Training entry point.

Day 1: compose the config, seed, and log the environment. The training loop lands on Day 4.

    python -m sensor_fusion.train model=fusion data=mini
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from sensor_fusion.utils.env import collect_env_info
from sensor_fusion.utils.seed import seed_everything

log = logging.getLogger(__name__)


# config_path is relative to this file, so it needs the editable install (`pip install -e .`).
@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    for key, value in collect_env_info().items():
        log.info("%-22s%s", key, value)


if __name__ == "__main__":
    main()
