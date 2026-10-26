# 000 — Environment setup on a locked-down lab machine

**PR:** — (bootstrap commit on `main`) · **Date:** 2026-10-17 · **Tags:** code-craft, design

## What was built
- Working agreement and Claude Code config: `CLAUDE.md` and `.claude/` (permissions, hooks, path-scoped rules, `/challenge`, `/mock-interview`).
- Portable Python 3.11.9 and the project `.venv` ([ADR 0002](../adr/0002-python-environment.md)).
- Decision records for the scope ([ADR 0001](../adr/0001-camera-lidar-fusion-scope.md)) and the environment.

## Decisions made (by you)
- Input-level camera+LiDAR fusion over late fusion or BEV → ADR 0001.
- Plain venv + pip over uv (uv's cons: the torch CUDA index config, TLS behind proxies, a young tool) → ADR 0002.
- A portable Python instead of waiting for IT, and 3.11 instead of 3.13 (the devkit needs `numpy<2`).
- You own GitHub: no `gh` and no pushes by agents.
- CLI-first, no MCP servers. Agents read the *installed* library source for API docs.

## Senior instinct
- **Verify an install by using it, not by seeing the files.** The unzip "succeeded" but left a Python without its standard library. Only running it exposed that.
- **Verify what you download**: a sha256 checksum or a code signature, every time you install a binary by hand.
- **Guardrails must fail open during bootstrap**, or people disable them.
- **Know your exit cost when picking a tool.** Standard formats (`pyproject.toml`, PEP 621) keep lock-in low.
- **Prefer a directory already on PATH over editing PATH.** `setx` silently truncates PATH at 1024 characters.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode character '→'` from the session hook | Windows console output defaults to cp1252, which can't encode `→` | Expect this from any Windows script that prints non-ASCII. Fix with `sys.stdout.reconfigure(encoding="utf-8")`, and run hooks by hand before trusting them. |
| Python installer exits with code 1625 | Group policy forbids installs | Look up the msiexec exit code (1625 means "forbidden by system policy"), then go portable or ask IT. |
| Portable Python fails with `ModuleNotFoundError: No module named 'encodings'` | Git Bash `unzip` extracted `tools/*` only one level deep, so the stdlib was missing | Compare file counts in the archive vs on disk. Extract with Python's `zipfile`. |
| Docs links 404 on GitHub / hook finds no TODOs on Linux, but all fine locally | Git tracks `Docs/`, we wrote `docs/`. Windows is case-insensitive, Linux and GitHub are not. | Run `git ls-files` (git's view) and compare with what the code references. Test links with exact case, and never use `Path.resolve()` for that: on Windows it silently fixes the case. |
| nuscenes-devkit can't be installed on Python 3.13 | It requires `numpy<2`, and numpy 1.x has no 3.13 wheels | Read a package's `Requires-Dist` metadata before choosing the Python version. |

## Quiz
**Q1 (warm-up):** nuScenes-mini has 10 scenes of ~40 frames each. If we split train/val by randomly shuffling *frames* instead of by *scene*, what goes wrong, and why would the validation metrics look *better* than reality?
**Your answer:** "It gives randomness in validation sets, instead of the bias when humans split manually."
**Feedback:** Incorrect. The issue is data leakage, not randomness. Keyframes are 0.5 s apart, so neighbouring frames of a scene are near-duplicates. A random frame split puts one in train and its twin in val, so val measures memorization and looks too good. Split by scene (the unit you deploy on), using nuScenes' official scene lists.

**Q2 (case bug):** How could you catch case-mismatch bugs on Windows?
**Your answer:** Enable per-folder case sensitivity in Windows.
**Feedback:** Valid, but it needs admin rights and some Windows tools break. Practical options: Linux CI, a link-case test (`tests/test_repo_links.py`), and pre-commit's `check-case-conflict`.

## Interview angle
"On a locked-down Windows workstation with no admin rights, I set up a reproducible Python/CUDA environment: a portable, signature-verified CPython, a venv, and CUDA wheels chosen to keep support for our Volta GPUs. The decisions are documented as ADRs."
