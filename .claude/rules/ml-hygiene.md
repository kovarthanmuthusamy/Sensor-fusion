---
paths:
  - "src/**/train*"
  - "src/**/eval*"
  - "src/**/models/**"
  - "src/**/datasets/**"
  - "configs/**"
  - "scripts/**"
---

# ML hygiene rules

- **Split by scene, never by frame.** Adjacent frames are nearly identical, so a frame-level split leaks data. Day and night splits come from `scene.description` containing "Night".
- **The test split is sealed.** Tune on val only. Evaluate on test only for the final reported numbers, and say so in the PR.
- **Seed everything** (python, numpy, torch, CUDA) from config. Log the seed, git commit hash and full resolved config with every run (W&B `config`).
- **Mixed precision: fp16 `torch.autocast` + `GradScaler`.** The GV100 (Volta) has **no bf16**. Watch for overflow and NaN in depth losses; clamp or normalise depth targets.
- **Pin a GPU:** runs set `CUDA_VISIBLE_DEVICES` explicitly. This is a shared lab machine, so check `nvidia-smi` before launching.
- **Fair comparison:** the RGB baseline and the RGB+depth fusion model share the architecture, schedule, augmentations and seed. Only the input channels differ. Any extra difference must be called out.
- **Overfit one batch first.** Every new model or loss must show it can drive the loss near zero on a single batch before a full run.
- **Metrics are always reported split by {day, night}** alongside the overall number.
- **No magic numbers in code.** Hyperparameters live in `configs/`.
