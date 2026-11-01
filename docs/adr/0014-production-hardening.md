# 0014 — Making the service operable: settings, logs, metrics, limits and a safer container

**Status:** Accepted · **Date:** 2026-11-01

## Context
The service of ADR 0007 answered correctly but was not something anyone could run for others: no readiness signal, no logs or metrics, a limit on upload size checked only after the whole body was read, inference blocking the server's event loop (so even a liveness probe stalled during a detection), an unbounded queue under load, tracebacks reaching callers on errors, and a container running as root. The user asked for "production style" and chose, within the Nov 3 deadline, **service hardening** and **repo and process polish**; a reproducible pipeline with model versioning and release automation were offered and declined.

## Options and decisions
1. **Configuration: environment variables** (`SF_*`, read once at start-up, validated, with defaults safe for a laptop) *(chosen)*. Alternatives: a config file or Hydra. Hydra is right for experiments but a deployed service is configured by its environment, and containers make variables the natural carrier. A wrong value stops start-up naming the variable.
2. **Logs: one JSON line per request on stdout**, with a request ID that is also returned in `X-Request-ID` *(chosen)*; shipping logs is the platform's job. Caller-supplied IDs are accepted only if short and plain. Bodies (photos, LiDAR) are never logged. Alternative: log files, rejected (a container's disk is not for logs).
3. **Metrics: `prometheus_client` and a `/metrics` endpoint** *(chosen)*: request count and latency by route and status, stage times, detections per request, in-flight and waiting load, rejections by reason, a model-info gauge. Route labels come from the route template, and unknown paths share one label, so a scanner cannot create unbounded series. Alternatives: hand-rolled text (reinvents the format), OpenTelemetry (more moving parts than one service needs).
4. **Liveness and readiness are different questions** *(chosen)*: `/health` = the process is up, `/ready` = the model is loaded and has run a warm-up detection at start-up. A broken model keeps the process alive but not ready, so it is not sent traffic and the log says why.
5. **Overload: bounded concurrency plus a bounded queue, then 503 with `Retry-After`** *(chosen)*. Inference is CPU-bound; a queue without a bound only turns overload into timeouts for everyone. Alternatives: unbounded queue (worse), autoscaling alone (needs a platform and still needs this on the way up).
6. **Inference runs in a worker thread** under a concurrency limiter *(chosen)*; the ONNX runtime releases the GIL, so a thread is enough and a process pool is not needed. Probes stay responsive while a detection runs (tested).
7. **Input defences** *(chosen)*: an early `Content-Length` check rejects oversized bodies with 413 before they are parsed; a cap on LiDAR points; the image header is checked before any pixel is decoded (a tiny file cannot allocate a huge image); `allow_pickle=False`.
8. **Errors: a generic 500 with the request ID**, the traceback only in the log *(chosen)*.
9. **Container:** non-root user (UID 10001), health check on `/ready`, image labelled and tagged with the git commit; the compose file adds a read-only filesystem, dropped capabilities, no new privileges and CPU and memory limits *(chosen)*.
10. **Declined or left out on purpose:** authentication, TLS and per-client rate limits (the gateway's job, stated in `SECURITY.md`), tracing, autoscaling, canary rollout, a model registry and reproducible pipeline (offered and declined), automatic image publishing and vulnerability scanning in CI.

## Decision
Items 1 to 9. Everything has a test; five bug injections (blocking the event loop, no load shedding, leaking exception text, no early size check, unbounded metric labels) are each caught by a test.

## Consequences
- \+ A reviewer or an on-call engineer can run it, see it, bound it, and diagnose a failure from a request ID.
- \+ Overload degrades predictably (fast 503s that clients can retry) instead of slowly for everyone.
- − The concurrency and queue limits (2 and 4) are starting values. They are set from the measured load test (`docs/load_test.md`) and should be revisited on the target hardware.
- − The parsing of a multipart body still happens before the capacity check for requests that declare a normal size; a gateway rate limit is the real protection against floods.
- − `prometheus_client` and `httpx` are new dependencies of the `serve` extra; the container image grows by a few MB.
- − The Docker image and compose file cannot be run on the lab machine (no Docker); CI builds and smoke-tests the image, including the non-root check.
