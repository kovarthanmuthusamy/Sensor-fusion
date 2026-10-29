# 0007 — Serving, export and demo (Day 11–13)

**Status:** Proposed · **Date:** 2026-10-29

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
Pending user review: the recommended choices above.

## Consequences
- \+ Shows the full path: data → model → export → API → container → demo, with an honest day/night model card.
- \+ Shared code and a parity test protect against silent training-serving mismatch.
- − About 3 extra days; real weights are needed for the demo, so this must follow Day 6–7 training.
- − CPU latency of a backbone at 800×448 may be slow; we report it instead of hiding it.
