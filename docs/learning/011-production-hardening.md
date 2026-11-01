# 011 — Making the service operable (production hardening, Nov 1)

**PR:** `feat/production-service` · **Date:** 2026-11-01 · **Tags:** mlops, design

## What was built
- **Service** (`serve/`): settings from `SF_*` environment variables (`settings.py`), JSON logs with request IDs and Prometheus metrics (`observability.py`), and an API with `/ready`, `/metrics`, a start-up warm-up, bounded concurrency with load shedding (503 + `Retry-After`), early request-size and LiDAR-point limits, a header-first image check, inference in a worker thread, and a generic 500 (`api.py`, `app.py`, `inference.warm_up`).
- **Container:** non-root user, health check on `/ready`, image tagged and labelled with the git commit, a compose file with a read-only filesystem, dropped capabilities and CPU/memory limits. The CI job now also checks that the container is not root and that `/metrics` answers.
- **Project files:** `Makefile`, `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `docs/RUNBOOK.md`, a docs index, issue and PR templates, Dependabot, CODEOWNERS, `.editorconfig`; a load-test script and a sample-request script.
- 30 new tests, ADR 0014.

## Decisions made (by you)
- Scope: service hardening and repo/process polish, finished before Nov 3. A reproducible pipeline with model versioning, and release automation, were offered and declined → [ADR 0014](../adr/0014-production-hardening.md).

## Decisions I made inside the plan (veto any)
- Environment variables rather than a config file; stdout JSON logs rather than files; `prometheus_client` rather than OpenTelemetry.
- Starting limits of 2 concurrent detections and 4 waiting (set from the load test), and a 64 MB body limit.
- `/detect` is served even before the warm-up has finished; only `/ready` reflects the warm-up.

## Senior instinct
- **Liveness and readiness are separate questions.** "The process is up" and "the model is warm and can take traffic" need different probes; a broken model should stay out of rotation without being restarted in a loop.
- **Shed load, don't queue it.** For CPU-bound work an unbounded queue only turns overload into timeouts for everybody; a fast 503 with `Retry-After` lets clients and load balancers react.
- **Blocking code inside an async handler freezes everything.** Running inference on the event loop stalled even `/health`; a test now holds a slow detection open and requires `/health` to answer.
- **Never trust a header, and never log a body.** Caller request IDs are kept only if short and plain (log injection), photos and points are never logged.
- **Metric labels must be bounded.** Labelling by raw URL lets a scanner create unlimited time series; labels come from the route template and unknown paths share one.
- **Check the cheap thing first.** The size check on `Content-Length` rejects a huge body before it is parsed; checking only after reading means the damage (memory) is already done.
- **Test the test.** Five bug injections proved the tests; one slipped through (no early size check), because a later check produced an almost identical 413. The new test sends a body that is not even valid multipart, which only the early check can answer.
- **Say what you could not run.** There is no Docker here and no `make`; CI builds the image and dry-runs the Makefile, and the docs say so.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| The independent evaluation check hung, then crashed with "Safe importing of main module" | On Windows, data-loader workers re-import the script; mine had no `if __name__ == "__main__":` guard | Any script that starts worker processes needs the guard; read the first lines of the error, not the last |
| A mutation survived: removing the early size check changed nothing the test saw | Two checks gave nearly the same 413 message, so the test could not tell them apart | When a test passes with the code removed, ask what only that code can do, and test that (here: a non-multipart body) |
| `tail` not found in PowerShell | The review commands I gave were Linux/Git Bash | Give commands for the shell the person is using, or say which shell |

## Quiz
**Q29:** The service answers 503 when `max_concurrent + max_queue` requests are already in the system, instead of letting more wait. Why is that better for clients than a longer queue, and what must a well-behaved client do when it gets one?
**Your answer:** _open_

**Q30:** `/health` and `/ready` can disagree. Describe one situation where `/health` is 200 and `/ready` is 503, and say which of the two an orchestrator should use to restart a container, and which to decide whether to send it traffic.
**Your answer:** _open_

Q7 to Q10, Q15 to Q28 are still open.

## Interview angle
"I took a working model server and made it operable: environment-based settings, structured logs with request IDs, Prometheus metrics, readiness separate from liveness, bounded concurrency with load shedding, input limits that act before the body is parsed, a generic 500 that never leaks details, and a non-root container. Each behaviour has a test, and I checked the tests by injecting five bugs, which exposed one weak test that I then fixed."
