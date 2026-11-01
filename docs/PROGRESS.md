# Progress

Timebox: 2026-10-17 → 2026-11-03 (extended on 2026-10-29). Tick items as PRs merge.

## Setup
- [x] Working agreement (CLAUDE.md) + Claude config (.claude/)
- [x] Portable Python 3.11.9 + project `.venv`
- [ ] User: create the empty GitHub repo + `wandb login`
- [x] nuScenes account + trainval metadata + keyframe download (CAM_FRONT, LIDAR_TOP): 34,149 files each, verified 2026-10-30

## Build
- [x] Day 1: repo skeleton: git, pyproject, Hydra configs, ruff/pre-commit, mypy, pytest+coverage, CI, README (PR #2)
- [x] Day 2: PR `feat/lidar-projection`: LiDAR→camera projection + sparse depth map, tested against the devkit ([ADR 0004](adr/0004-data-pipeline-choices.md))
- [x] Day 3: PR `feat/nuscenes-dataset`: scene split + day/night tag, 3D→2D labels with distance, Dataset class, overlay picture
- [x] Day 4–5: in three PRs ([ADR 0006](adr/0006-detector-input-and-targets.md)): A input preparation + target encoder + decoder (built, [log 004](learning/004-input-preparation-and-targets.md)), B network (with distance head) + loss (built, [log 005](learning/005-network-and-loss.md), [ADR 0009](adr/0009-network-and-loss.md)), C training loop + W&B + safeguards (built, [log 006](learning/006-training-loop.md), [ADR 0010](adr/0010-training-setup.md)); trainval is on disk, about 157 frames/s so about 3 min per epoch
<<<<<<< HEAD
- [ ] Day 6–7: **decide the epoch count first (ADR 0010)**, then train the RGB baseline and the RGB+depth fusion model on trainval (4th-channel initialisation); the feature-level (mid) fusion variant is dropped
- [ ] Day 8–9: day/night evaluation, ablations, failure cases
- [ ] Day 10: README, cleanup
- [ ] Day 11–13 (scope B, [ADR 0007](adr/0007-serving-and-demo.md), accepted): ONNX export + parity test, FastAPI `/detect`, latency table, CPU Docker image checked in CI, Gradio side-by-side demo, model card. Needs trained weights from Day 6–7.
=======
- [x] Day 6–7: first training runs (24 epochs; fusion matched the baseline, diagnosed in [log 006](learning/006-training-loop.md) and [log 008](learning/008-dense-depth-input.md)), then the retrain with dense depth + depth tower, 8 epochs ([ADR 0013](adr/0013-dense-depth-and-retrain-protocol.md), [log 010](learning/010-retrain-and-results.md))
- [x] Day 8–9: evaluation: code ([log 007](learning/007-evaluation.md), [ADR 0012](adr/0012-evaluation-protocol.md), [ADR 0011](adr/0011-evaluation-reporting-rules.md)), three-way comparison with ablation and a second checkpoint ([docs/results](results/README.md)), failure cases ([docs/failure_cases.md](failure_cases.md))
- [x] Day 10: README, model card ([docs/MODEL_CARD.md](MODEL_CARD.md)), cleanup
- [x] Day 11–13 (scope B, [ADR 0007](adr/0007-serving-and-demo.md), [log 009](learning/009-serving.md)): ONNX export with parity test, FastAPI `/detect`, latency table ([docs/benchmarks.md](benchmarks.md)), CPU Docker image (built in CI only), Gradio demo
- [ ] User: commit and push the branches, open the PRs, check that the `serve-container` CI job passes (the image has never been built on the lab machine), regenerate `requirements.lock.txt` if needed, log in to W&B and `wandb sync` the offline runs
- [ ] Optional if time: a second seed per model (to quantify seed noise at night), non-maximum suppression for duplicate boxes
>>>>>>> 7d776d1 (after training)

## Review
Weak topics from quizzes (feeds `/mock-interview`):
- Scene-level splits and cross-split leakage: [log 000 Q1](learning/000-environment-setup.md), [log 003 Q14](learning/003-splits-labels-dataset.md). To do on Day 4: the training script checks that train and val scenes do not overlap.
- Protection rules versus team size (1 approval on a solo repo): [log 001 Q3](learning/001-repo-skeleton-and-git-workflow.md).
- Duplicates and unspecified order when writing into an array: [log 002 Q12](learning/002-lidar-projection.md).
- A systematic offset is not a frame bug (it grows with object size): [log 003 Q13](learning/003-splits-labels-dataset.md).
