# 0002 — Portable CPython 3.11 + venv + pip

**Status:** Accepted · **Date:** 2026-10-17

## Context
Locked-down lab workstation: no admin rights, the Python installer is blocked by policy (MSI error 1625), the system conda is broken (libmamba DLL access denied), and the shared system Python 3.13 must stay untouched. nuscenes-devkit requires `numpy<2`, which has no Python 3.13 wheels.

## Options
1. **uv**: fast, with a cross-platform lockfile. But the PyTorch CUDA index needs extra configuration, it ships its own TLS certificates (which can break behind university proxies), and it's young.
2. **Per-user Miniforge**: the conda ecosystem, but heavier, and it doesn't solve the PyTorch index question.
3. **System Python 3.13, no devkit**: parse the nuScenes JSON ourselves.
4. **Portable official CPython 3.11.9 + venv + pip**: the PSF NuGet package (signature verified), unzipped to `~/.local/python/3.11.9`.

## Decision
Option 4. Dependencies go in `pyproject.toml` as ranges, `requirements.lock.txt` pins exact versions from `pip freeze`, and PyTorch comes from the CUDA 12.6 wheel index (still ships Volta `sm_70` kernels).

## Consequences
- \+ Standard tooling that every reviewer knows. No admin needed, and the system Python is untouched.
- − pip has no cross-platform lockfile. `requirements.lock.txt` is specific to Windows + CUDA 12.6.
- − The portable Python never auto-updates, so security patches are manual.
- − The torch version is pinned in three places (lockfile, README, CI). Bump them together.
