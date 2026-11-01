# Runbook: the detection service

For whoever runs or is on call for the service. Design reasons are in [ADR 0014](adr/0014-production-hardening.md); the model's behaviour and limits are in the [model card](MODEL_CARD.md).

## What it is
A stateless HTTP service. A client sends one photo, the raw LiDAR points and the calibration; it answers with boxes, classes, scores and distances. It holds no data between requests, so any number of copies can run side by side. The model is a file (`model.onnx` + `model.json`) mounted into the container, not part of the image.

## Run it
```bash
# locally
make serve MODEL_DIR=outputs/serve/fusion_dense_head
# container (unprivileged, read-only, no capabilities)
make docker-build && make docker-run MODEL_DIR=outputs/serve/fusion_dense_head
# with limits and a health check
make compose-up MODEL_DIR=outputs/serve/fusion_dense_head
```
Create the model folder from a checkpoint with `make export FUSION=<path/to/best.pt> MODEL_DIR=<folder>`. The export refuses to write a file that differs from the PyTorch model by more than 1e-3.

## Endpoints
| route | meaning | healthy answer |
|---|---|---|
| `GET /health` | **liveness**: the process is up. Restart the container if this fails. | 200 |
| `GET /ready` | **readiness**: the model is loaded and has run a warm-up detection. Send traffic only when this is 200. | 200 (503 while starting or if the warm-up failed) |
| `GET /version` | model name, epoch, input channels, build commit, limits in force | 200 |
| `GET /metrics` | Prometheus metrics | 200 |
| `POST /detect` | the detection | 200 |

Every response has an `X-Request-ID` header (a caller's own ID is kept if it is short and plain). The same ID is in the log line for that request.

## Configuration
All settings are environment variables; a wrong value stops start-up with a message naming the variable.

| variable | default | when to change it |
|---|---|---|
| `SF_MODEL_DIR` | `/models` | where `model.onnx` and `model.json` are mounted |
| `SF_HOST` / `SF_PORT` | `127.0.0.1` / `8000` | the container sets the host to `0.0.0.0` |
| `SF_TORCH_THREADS` | `4` | **leave at 4 to 8.** One thread per core is 3× slower and much less predictable (log 009) |
| `SF_MAX_CONCURRENT` | `2` | detections running at once; raise only if CPU is idle at peak |
| `SF_MAX_QUEUE` | `4` | requests allowed to wait; beyond `max_concurrent + max_queue` the service answers 503 |
| `SF_MAX_UPLOAD_MB` | `64` | the largest request body |
| `SF_MAX_LIDAR_POINTS` | `300000` | a scan has about 35,000 points |
| `SF_LOG_FORMAT` / `SF_LOG_LEVEL` | `json` / `INFO` | `text` is easier to read by eye |
| `SF_GIT_COMMIT` | `unknown` | set by the image build (`--build-arg GIT_COMMIT=...`) |

## Logs
One JSON line per request on stdout (probes at `/health`, `/ready`, `/metrics` only at DEBUG), for example:
`{"ts": "...", "level": "INFO", "msg": "request", "request_id": "…", "method": "POST", "path": "/detect", "status": 200, "latency_ms": 112.4, "detections": 3}`.
Photos and LiDAR points are never logged. To follow one customer report, search the logs for its `request_id`.

## Metrics to watch
| metric | what it tells you |
|---|---|
| `http_requests_total{path,status}` | traffic and error rates by route (unknown paths share the label `unmatched`) |
| `http_request_duration_seconds` | latency; look at p95, not the mean |
| `inference_stage_seconds{stage}` | which stage is slow: `prepare`, `network` or `decode` |
| `rejected_requests_total{reason}` | `overloaded` (503), `too_large` (413), `invalid` (422) |
| `inference_in_flight`, `inference_waiting` | current load |
| `model_info{model,epoch,git_commit}` | what is running (always 1) |

Suggested alerts (starting points, tune them on real traffic): 5xx rate above 1% for 5 minutes; `/ready` failing for 2 minutes; p95 of `/detect` above 400 ms for 10 minutes; `rejected_requests_total{reason="overloaded"}` rising for 5 minutes (capacity).

## Capacity
Measured on the lab workstation (`docs/load_test.md`): see the table there. A single replica answers about as many requests per second as it has CPU for; `SF_MAX_CONCURRENT` bounds the work in flight, and more copies of the container add capacity linearly because the service is stateless. When busy, it sheds load (503 + `Retry-After: 1`) instead of building a queue that only turns overload into timeouts; clients should retry with a short back-off.

## Problems and what to do
| symptom | likely cause | what to do |
|---|---|---|
| container restarts, or never becomes ready | model folder missing or wrong, or a corrupt file | the log shows `model.json`/`model.onnx` not found or `warm-up failed`; check the mount and `SF_MODEL_DIR`; re-export the model (the exporter's parity check guards the file) |
| `/ready` is 503 but `/health` is 200 | the start-up warm-up failed | read the `warm-up failed` log line (it has the traceback); fix the model files, then restart |
| clients get 503 | more work than capacity | check `inference_waiting` and `rejected_requests_total{overloaded}`; add replicas, or raise `SF_MAX_CONCURRENT` if the CPU has room |
| clients get 422 "image must be 1600x900" | wrong photo size | the model was trained on 1600×900 nuScenes front-camera photos only; resize is not offered on purpose |
| clients get 422 about the calibration or lidar | malformed matrix or file | `T_cam_from_lidar` is 4×4 and must already include the vehicle's motion between scan and photo; `intrinsic` is 3×3; LiDAR is a float `.npy`, `(N, 3+)` |
| clients get 413 | body above `SF_MAX_UPLOAD_MB` | send smaller data, or raise the limit deliberately |
| latency high and erratic (p95 several times the median) | thread oversubscription or too few CPUs | `SF_TORCH_THREADS=4`; check the container's CPU limit; compare `inference_stage_seconds` per stage |
| 500 with `{"error": "internal_error", "request_id": …}` | a bug or an unexpected input | search the log for that `request_id`: the line has the full traceback (the caller never sees it); add a test that reproduces it |

## Updating the model or the service
1. Export the new checkpoint to a new folder (`make export`) and read `model.json` (epoch, parity).
2. Start the new version next to the old, wait for `/ready`, check `/version`, send a known request (`scripts/load_test.py` with one client works as a smoke test).
3. Move traffic, keep the old one for rollback. Rolling back is starting the previous image tag (images are tagged with the git commit) or the previous model folder.
4. Anything that changes behaviour belongs in an ADR and the changelog.

## Not covered here
Authentication, TLS and per-client rate limits are expected to come from the gateway in front of the service (see [SECURITY.md](../SECURITY.md)). There is no tracing, autoscaling or canary rollout yet.
