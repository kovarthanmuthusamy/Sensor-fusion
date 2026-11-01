# 009 — Serving: ONNX export, the detection API, the container and the demo (Nov 1)

**PR:** `feat/serving` · **Date:** 2026-11-01 · **Tags:** mlops, design

## What was built
- **Export** (`serve/export.py`): the trained network to ONNX (dynamic batch, fixed image size) with a **parity check** that the PyTorch and ONNX outputs agree; a `model.json` next to it records the input recipe (channels, normalisation, thickening, decoding) and the measured parity.
- **Detector** (`serve/inference.py`): photo + raw LiDAR points + calibration in, boxes with class, score and distance out. Projection, input preparation and decoding are the training code; only the network runs in ONNX.
- **API** (`serve/api.py`): FastAPI with `/health`, `/version`, and `/detect` (multipart: image, `.npy` LiDAR, calibration JSON). Bad input gets a 4xx with a reason.
- **Container** (`Dockerfile`, CI job `serve-container`): a CPU image with only serving dependencies, built, started and sent a real request in CI.
- **Latency** (`scripts/benchmark_latency.py`) and **demo** (`serve/demo.py`, Gradio, local).
- 40 new tests for the exporter, the detector, the API, the training-serving skew check and the demo (337 in the whole suite, counted by pytest). Decisions: [ADR 0007](../adr/0007-serving-and-demo.md).

## Decisions made (by you)
- Request format (photo + raw LiDAR + calibration), network-only ONNX, Gradio, CPU image in CI → ADR 0007.

## Decisions I made inside the plan (veto any)
- Multipart upload with a `.npy` LiDAR file (JSON would be huge for ~34k points).
- The calibration is a single `T_cam_from_lidar` that must already include the vehicle's motion between the LiDAR scan and the photo, as in training.
- The image must be exactly 1600×900 (the size the model was trained on), refused otherwise.
- The demo is local only: nuScenes images cannot be redistributed in a hosted copy.

## Senior instinct
- **Make the exported file carry its own proof.** The exporter refuses to write a model that differs from PyTorch by more than 1e-3; the real fusion model differs by under 1e-5.
- **Test for training-serving skew with real data.** A real frame sent as a customer would (photo, raw LiDAR scan, calibration) must become *exactly* the tensor the training pipeline builds. One shared code path plus this test is what makes "it worked in training" mean something in production.
- **Keep the serving dependencies minimal.** The service first imported Hydra, W&B and the nuScenes devkit through a stray import; a test now fails if importing the API pulls in the training stack, so the image stays small.
- **Validate at the boundary.** Every malformed request (not an image, wrong size, pickled or integer LiDAR file, NaN points, a non-homogeneous matrix) gets a 422 with a reason; a pickled `.npy` is never loaded (`allow_pickle=False`).
- **Say what you could not run.** There is no Docker on the lab machine, so the image is built and tested in CI only; the README states it.
- **Install under constraints.** New packages were added with the existing lock file as a constraint, so nothing already pinned (and used by a training job running at the time) changed.
- **Latency numbers need context.** Median and p95, batch size, hardware named, "workstation, not edge", measured when no job was running.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| ONNX export failed in the depth tower: "not constant ... kernel sizes" | The pooling window was computed from tensor shapes, which the exporter treats as dynamic | The error names the node and scope; replace shape arithmetic by a constant when the value is fixed by the architecture (the output stride) |
| Wrong Pillow version in the Dockerfile | I wrote the pin from memory | Read versions from `pip list` of the tested environment, never from memory |
| Retrain queue: the last run failed immediately with `model=fusion_dense\r` | Python on Windows prints `\r\n`, and the shell's `read` kept the `\r` | Check the first lines of a failed run's log; strip `\r` when a shell script parses another program's output |
| Service imported Hydra, W&B and the devkit | A shared constant lived in a module that imported the training stack | Print `sys.modules` after importing the entry module, then move the constant |

## Quiz
**Q25:** The ONNX file contains only the network. Name two things that go wrong if the service re-implemented the input preparation or the box decoding instead of importing the training code, and how the skew test would catch them.
**Your answer:** _open_

**Q26:** The API accepts LiDAR as a `.npy` file with `allow_pickle=False`. What attack does that block, and why is a size limit on uploads also part of the same habit?
**Your answer:** _open_

Q7 to Q10, Q15 to Q24 are still open.

## Interview angle
"I exported the model to ONNX with a built-in parity check, wrapped it in a FastAPI service that takes a photo, raw LiDAR points and calibration, and kept one shared code path with training so a test can prove the served input is identical to the trained one. The container only carries serving dependencies, and CI builds it and sends it a real request."
