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
- [x] Day 3: PR `feat/nuscenes-dataset`: scene split + day/night tag, 3D→2D labels with distance, Dataset class, overlay picture
- [ ] Day 4–5: RGB-only baseline + training loop + W&B
- [ ] Day 6–7: RGB+depth fusion model + distance head
- [ ] Day 8–9: day/night evaluation, ablations, failure cases
- [ ] Day 10: README, cleanup
- [ ] Day 11–13 (scope B, [ADR 0007](adr/0007-serving-and-demo.md), proposed): ONNX export + parity test, FastAPI `/detect`, latency table, CPU Docker image checked in CI, Gradio side-by-side demo, model card. Needs trained weights from Day 6–7.

## Review
Weak topics from quizzes (feeds `/mock-interview`):
- Scene-level splits and cross-split leakage: [log 000 Q1](learning/000-environment-setup.md), [log 003 Q14](learning/003-splits-labels-dataset.md). To do on Day 4: the training script checks that train and val scenes do not overlap.
- Protection rules versus team size (1 approval on a solo repo): [log 001 Q3](learning/001-repo-skeleton-and-git-workflow.md).
- Duplicates and unspecified order when writing into an array: [log 002 Q12](learning/002-lidar-projection.md).
- A systematic offset is not a frame bug (it grows with object size): [log 003 Q13](learning/003-splits-labels-dataset.md).
