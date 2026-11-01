# Changelog

All notable changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) (for a research project: a minor version for a new capability or a changed result).

## [0.2.0] - 2026-11-01
### Added
- **Evaluation:** COCO-style mAP, distance error by range, six day/night/city cells, scene-level paired bootstrap, a reporting rule fixed beforehand (ADR 0011, 0012); `python -m sensor_fusion.evaluate`.
- **Dense LiDAR input:** thickened depth, validity mask, a separate depth stem, and a depth tower into the heads (`model=fusion_dense`, `fusion_dense_head`; ADR 0013).
- **Per-epoch validation mAP** and checkpoint selection by mAP; every epoch's checkpoint is kept.
- **Serving:** ONNX export with a parity check, FastAPI service (`/detect`), a CPU container, a Gradio demo, latency and load-test scripts (ADR 0007).
- **Production hardening of the service (ADR 0014):** settings from `SF_*` environment variables, structured JSON logs with request IDs, `/ready` and `/metrics`, a start-up warm-up, bounded concurrency with load shedding (503 + `Retry-After`), request size and LiDAR point limits, a generic 500 that never leaks details, inference off the event loop, a non-root container with a read-only filesystem, a compose file with limits.
- **Project files:** model card, runbook, security policy, contributing guide, Makefile, issue and PR templates, Dependabot, results and failure-case reports, README rewritten.

### Changed
- The final model is camera + LiDAR with the dense input and the depth tower. Median distance error falls from 8.1% to 4.3% (night: 10.9% to 3.6%); finding objects (mAP@0.5) is unchanged overall.
- Training length is 8 epochs (was 24); the best checkpoint is chosen by validation mAP instead of validation loss.
- The service sets its own PyTorch thread count (default 4): a call is about 3× faster and far more predictable on many-core machines.
- The 7×7 thickening is computed as two 1-D passes (100 ms to 29 ms), with identical results.

### Fixed
- The first fusion model was no better than the camera baseline because the LiDAR channel was 99% empty and "no data" looked like "0 m" (log 008).
- The validation loss rose for 24 epochs while detection held; it selected an under-trained checkpoint (log 006).

### Known limits
Night is one neighbourhood (15 validation scenes) and one seed per model; no non-maximum suppression (duplicate boxes); the Docker image is built and tested in CI only.

## [0.1.0] - 2026-10-30
### Added
- nuScenes data pipeline: LiDAR to camera projection, 3D to 2D labels with distance, scene-level splits, day/night tags, input preparation and target grids (ADR 0004 to 0006).
- Detector (ResNet-18, upsampling neck, class and regression heads) and loss (ADR 0009); training loop with safeguards, checkpoints and W&B tracking (ADR 0010).
- Repo skeleton: Hydra configs, ruff, mypy, pytest, pre-commit, CI (ADR 0002, 0003).
