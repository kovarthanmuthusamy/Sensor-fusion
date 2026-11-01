# Contributing

Thanks for looking. This is a solo research project that is run like a team one: small reviewed changes, checks that must pass, and a written reason for every real decision.

## Set up
Python 3.11. Training used an NVIDIA GPU; everything else runs on CPU.
```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu   # or cu126 for a GPU
make install            # package + dev, serve and demo extras + git hooks
make help               # every task, one line each
```
Without `make` (for example on Windows), run the commands shown next to each task in the `Makefile`.

## The loop
1. **One task = one branch = one pull request**, branched from the *latest* `main`. After a squash merge the branch is finished: start the next task from `main` again. (Continuing on a squash-merged branch produces conflicts in every file the merge copied.)
2. Write the change **with its tests**. A bug fix starts with a test that fails for the bug. Behaviour that matters in deployment (limits, errors, readiness) has a test that fails if it is removed.
3. Run `make check` (lint, types, tests). CI runs the same on Ubuntu with CPU PyTorch.
4. Open the PR with the template: what changed, why, how it was tested, and **what was not verified**. Say so plainly when something could not be run.
5. Squash-merge. Update `CHANGELOG.md` for anything a user of the code or service would notice.

## Standards
- **Style:** `ruff` formats and lints (line length 100); `mypy` checks `src/`. The pre-commit hooks run on every commit; if a hook modifies files, stage them and commit again.
- **Seeds, configs and tracking:** hyperparameters live in `configs/` (Hydra), not in code; every run logs its seed, config, git commit and environment. Baseline and fusion models share everything except their input, and a test enforces it.
- **Splits are by scene, never by frame**, and the test split is not used for any choice.
- **Reporting:** results come with intervals and sample sizes; a reporting rule is fixed before the numbers exist (ADR 0011).
- **Decisions get an ADR** in `docs/adr/` (context, options, decision, consequences), written before or with the change. Errors and lessons go in the learning log (`docs/learning/`).
- **Commit messages:** an imperative summary line (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), then why if it is not obvious.
- **Large files** (data, checkpoints, outputs) are never committed (`.gitignore`).

## Where things live
| path | what |
|---|---|
| `src/sensor_fusion/data/` | projection, labels, splits, dataset, input preparation, targets |
| `src/sensor_fusion/models/`, `training/` | network, loss, optimiser, loop, checkpoints, tracking |
| `src/sensor_fusion/evaluation/`, `evaluate.py` | metrics, bootstrap, reports |
| `src/sensor_fusion/serve/` | export, detector, API, observability, settings, demo |
| `configs/` | Hydra config groups |
| `scripts/` | benchmarks, load test, figures, sample requests |
| `docs/` | model card, runbook, results, ADRs, learning log |

## Adding a model variant
Add `configs/model/<name>.yaml` (same keys as the others; a config test lists which keys may differ), build it in `models/detector.py` behind a config flag, add tests (shape, zero start, legacy loading if the structure changed), train and evaluate it with the same seed and schedule as the others, and record the decision in an ADR.

## Reporting problems
Use the issue templates. For a security problem, use private vulnerability reporting (see [SECURITY.md](SECURITY.md)).
