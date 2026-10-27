# Progress

Timebox: 2026-10-17 → ~2026-10-31. Tick items as PRs merge.

## Setup
- [x] Working agreement (CLAUDE.md) + Claude config (.claude/)
- [x] Portable Python 3.11.9 + project `.venv`
- [ ] User: create the empty GitHub repo + `wandb login`
- [ ] nuScenes account + trainval metadata + keyframe download (CAM_FRONT, LIDAR_TOP)

## Build
- [x] Day 1: repo skeleton: git, pyproject, Hydra configs, ruff/pre-commit, mypy, pytest+coverage, CI, README (PR #2)
- [x] Day 2: PR `feat/lidar-projection`: LiDAR→camera projection + sparse depth map, tested against the devkit ([ADR 0004](adr/0004-data-pipeline-choices.md))
- [ ] Day 3: PR `feat/nuscenes-dataset`: scene split + day/night tag, 3D→2D labels with distance, Dataset class, overlay picture
- [ ] Day 4–5: RGB-only baseline + training loop + W&B
- [ ] Day 6–7: RGB+depth fusion model + distance head
- [ ] Day 8–9: day/night evaluation, ablations, failure cases
- [ ] Day 10: README, demo GIF, cleanup

## Review
(weak topics from quizzes / mock interviews go here)
