# Security

This is a research and portfolio project. The service is built to be operated safely, but it is **not a hardened public API**: read "What it assumes" before exposing it.

## What it assumes
- **A gateway in front of it.** The service has no authentication, no TLS and no per-client rate limit. Put it behind something that provides them (a reverse proxy, API gateway or service mesh) and do not expose it to the internet directly.
- **Trusted calibration.** It uses the calibration it is sent. A wrong matrix gives misleading distances; there is no way for the service to detect a plausible but wrong calibration.
- **Research use only.** The weights derive from nuScenes (CC BY-NC-SA 4.0). Do not use the output for any driving, driver-assistance or other safety decision.

## What it does to protect itself
- Input is validated before use: image type and size (the header is checked before any pixel is decoded, so a tiny file cannot make it allocate a huge image), LiDAR array type, shape, finiteness and point count, calibration shape and form.
- `.npy` files are loaded with `allow_pickle=False`: a pickled file would otherwise run arbitrary code when loaded.
- Request bodies above `SF_MAX_UPLOAD_MB` are refused with 413 before they are parsed; excess load is shed with 503 instead of queued without bound.
- Unexpected errors return a generic 500 with a request ID; the traceback goes to the log and never to the caller. Photos and LiDAR points are never logged.
- The container runs as an unprivileged user (UID 10001); the compose file also drops all capabilities, forbids new privileges and mounts the filesystem read-only. The model is mounted read-only.
- `X-Request-ID` values from callers are accepted only if short and plain; anything else is replaced, so a caller cannot inject text into the logs.

## Dependencies
Versions are pinned in `requirements.lock.txt`. Dependabot proposes updates weekly (pip, GitHub Actions and the Docker base image); each goes through CI like any other change. Run an image scan (for example Trivy) before deploying outside this repository: it is not yet part of CI.

## Reporting a vulnerability
Please use GitHub's private vulnerability reporting for this repository (Security tab, "Report a vulnerability") instead of a public issue. Include the version (`/version`), how to reproduce it and the impact you see. I will answer as soon as I can; this is a one-person project, so there is no guaranteed response time.
