# 0003 — Hydra for experiment configuration

**Status:** Accepted · **Date:** 2026-10-18

## Context
Experiments vary the model (RGB vs fusion), the dataset (mini vs trainval) and hyperparameters. Every run's exact config must be recoverable afterwards, or results can't be reproduced or compared.

## Options
1. **Hydra**: YAML config groups, CLI overrides, and the resolved config saved automatically per run. Common in ML codebases, with a learning curve and some "magic".
2. **Dataclasses + OmegaConf**: typed and less magic, but we'd hand-write variant switching and config saving.
3. **argparse**: simplest. It doesn't scale and nothing saves the config.

## Decision
Hydra 1.3 with `version_base="1.3"` (it no longer changes the working directory per run). Configs live in `configs/` at the repo root, with the groups `model/` and `data/`.

## Consequences
- \+ `model=fusion data=trainval` swaps whole groups. The resolved config lands in `outputs/<date>/<time>/.hydra/`.
- \+ Multirun sweeps (`-m seed=1,2,3`) are available later for free.
- − A typo in a config key only fails at runtime, with no static typing. That's mitigated by `tests/test_config.py`, which composes every combination in CI. Structured configs could add typing later.
- − `config_path` is relative to `train.py`, so the package must be installed in editable mode.
