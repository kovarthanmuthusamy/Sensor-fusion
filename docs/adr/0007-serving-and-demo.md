# 0007 — Serving, export and demo (Day 11–13)

**Status:** Accepted · **Date:** 2026-10-29

## Context
Goal: show that the night-time fusion model can be handed to someone else and used, not only trained (CV target: ML and forward-deployed engineering roles). Chosen scope is "B": an inference service, an exported model, a latency table, a container checked in CI, and a side-by-side demo (RGB-only vs fusion, day vs night).
Constraints: locked-down Windows machine (Docker probably cannot run locally), shared GV100 GPUs, GitHub is the user's (Claude never pushes), no TensorRT/ROS 2 (CLAUDE.md out of scope). Code can be built and tested with random weights; the demo and the latency/accuracy story need the trained checkpoints from Day 6–7.

## Options and recommendations

### 1. What the client sends to `/detect`
1. **Photo + sparse depth image** (already projected): simplest, but the client must know our projection code.
2. **Photo + raw LiDAR points + calibration** *(recommended)*: what a real customer has; the server runs the same projection code as training. Costs a bigger request format and one more thing to validate.
3. Both accepted: most flexible, twice the tests.

### 2. What goes into the exported (ONNX) file
1. **Network only; decoding in our code** *(recommended)*: the peak-picking decoder is the exact inverse of the training targets and already tested; exporting it risks unsupported operations.
2. Network + decoder in one file: one artifact, but harder to export and to test.

### 3. Training-serving skew
**One preparation and decoding code path** *(recommended)*: serving imports the same shrink/crop/normalise and decode functions as training. A parity test compares PyTorch and ONNX outputs on the same frame.

### 4. Demo framework
1. **Gradio** *(recommended)*: image-in, image-out in ~50 lines; can be hosted on a user-owned Hugging Face Space.
2. Streamlit: similar, less suited to side-by-side image comparison.
3. Plain recorded clip only: cheapest, but not interactive.

### 5. Dependencies and container
Serving packages (onnx, onnxruntime, fastapi, uvicorn, gradio) go in an optional `serve` extra, so training and CI stay light. One CPU-only image (python-slim, pinned versions). Built and smoke-tested (start, `/health`, one `/detect`) in GitHub Actions on Ubuntu; if Docker is unavailable locally this is stated in the README, not hidden.

### 6. Latency reporting
Median and p95 over N warm runs, for PyTorch GPU, PyTorch CPU and ONNX CPU, with hardware named. Labelled "workstation numbers, not edge numbers".

## Decision
Decisions 1-4 accepted on 2026-10-29 as recommended: (1) photo + raw LiDAR points + calibration; (2) network only in the ONNX file, decoding in our code; (3) Gradio; (4) one CPU image built and smoke-tested in CI. Decisions 5-6 (serve extra, latency reporting) stand as written.

## Consequences
- \+ Shows the full path: data → model → export → API → container → demo, with an honest day/night model card.
- \+ Shared code and a parity test protect against silent training-serving mismatch.
- − About 3 extra days; real weights are needed for the demo, so this must follow Day 6–7 training.
- − CPU latency of a backbone at 800×448 may be slow; we report it instead of hiding it.

## Outcome (2026-11-01)
Built as decided, with these details and deviations:
- **Request format:** multipart upload: photo (JPEG/PNG), LiDAR as a `.npy` float array, calibration as JSON (`T_cam_from_lidar` 4×4 including vehicle motion, `intrinsic` 3×3). Only 1600×900 photos are accepted. Malformed input gets a 422 with a reason.
- **Export:** network only, ONNX, dynamic batch. It uses PyTorch's legacy TorchScript exporter (`dynamo=False`), which is deprecated from PyTorch 2.9; migrating is future work. The depth tower's pooling window had to become a constant for the export to work. The parity check is part of the exporter; the final model differs from PyTorch by at most 9.5e-06.
- **No drift:** serving reuses the training code for projection, preparation and decoding; a test sends a real nuScenes frame the way a customer would and requires the identical network input.
- **Container:** a lean CPU image with only serving dependencies (a test fails if the API imports the training stack). CI builds it, starts it and sends a request; it has not been built on the lab machine, which has no Docker.
- **Demo:** Gradio, local only (nuScenes images cannot be redistributed, so there is no hosted copy).
- **Latency** ([benchmarks](../benchmarks.md)): a whole call on CPU takes about 59 ms (camera only) and 112 ms (fusion), p95 67 and 129 ms. Two findings: PyTorch's default of one thread per core (36) made the many small operations 3× slower and the tail 5× worse (300 vs 102 ms), so the service sets its own thread count (4); and the 7×7 max-pool of the thickening was 100 ms of the preparation, now two 1-D passes (29 ms, identical output).
- **Not done:** TensorRT, ROS 2 and a GPU serving image (out of scope), a hosted demo.
