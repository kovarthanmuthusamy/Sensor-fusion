# 006 — Training loop, safeguards and tracking (Day 4–5, PR C)

**PR:** `feat/training-loop` · **Date:** 2026-10-30 · **Tags:** mlops, design

## What was built
- **Loop** (`training/loop.py`): one epoch of updates, validation split into all/day/night, `fit` with `last.pt`/`best.pt`, resume and an optional time cap, and the one-batch overfit check.
- **Safeguards** (`training/checks.py`): scenes of train and val must not overlap, the GPU must be pinned, and a non-finite loss stops the run and saves the offending batch.
- **Optimiser and schedule** (`training/optim.py`), **checkpoints** (`training/checkpoint.py`), **tracking** (`training/tracking.py`: W&B with commit hash, config and environment).
- **Entry point** (`train.py`, `configs/trainer/default.yaml`) and split names in the data configs. Decisions in [ADR 0010](../adr/0010-training-setup.md); the reporting rule in [ADR 0011](../adr/0011-evaluation-reporting-rules.md).
- New tests for all of it (217 in the suite, counted by pytest on a machine with a GPU; the GPU tests skip elsewhere).

## Decisions made (by you)
- Train on the natural day/night mix and evaluate for day and night separately; minimum-100-objects reporting rule (option A) → ADR 0011.

## Decisions I made inside the plan (veto any)
- AdamW with two learning rates (5e-4 heads, 1e-4 backbone), warm-up + cosine, grad clip 10, batch 16, 12 workers.
- W&B falls back to offline mode without credentials; `last.pt`/`best.pt` written atomically; no time cap by default.
- **Open for you:** the epoch count (default 12; I recommend 24). Both models must use the same value.

## Measurements
- One-batch overfit on real mini data, GPU, fp16: loss 11.42 → 0.57 (ratio 0.05).
- Real trainval, fusion model, batch 16, 12 workers: about 157 frames/s, so about 3 minutes per epoch. After about 1,700 steps (one epoch) the weighted loss was around 1.3.

## Senior instinct
- **Unscale before you clip.** With the gradient scaler, gradients are 65,536 times larger than the real ones until `unscale_`. Clipping first would apply the threshold to the wrong numbers.
- **A skipped step must not advance the schedule.** The scaler skips a step when it finds inf gradients (normal in the first steps while it finds its scale). PyTorch warned about stepping the schedule before the optimiser; the fix is to step the schedule only when the optimiser actually stepped.
- **Stop on the first bad number and keep the evidence.** A NaN loss stops the run and writes the sample tokens of that batch, so the next question ("which frame?") has an answer.
- **Make unsafe things fail at the start.** The scene-overlap check and the GPU-pinning check run before any GPU time is spent.
- **Verify library behaviour in the installed source.** I wrote a worker-seeding helper from memory of an old PyTorch problem, wrote a test that was supposed to show why it was needed, and the test failed: the installed `worker.py` already seeds Python and NumPy in each worker. I removed the helper and kept a test that pins the behaviour we rely on.
- **Average per object, not per batch.** Night validation batches hold few objects; a plain mean over batches would let a batch with two objects count as much as one with sixty.
- **Record provenance with every run.** Commit hash, a dirty flag and the full config go into W&B, so a chart point can always be traced to code.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| My new test "without `seed_worker` two runs differ" failed | Wrong premise from memory: PyTorch already seeds `random` and NumPy in workers | Read the installed library source (`torch/utils/data/_utils/worker.py`) before building a workaround |
| `UserWarning: lr_scheduler.step() before optimizer.step()` | The gradient scaler skipped the first optimiser step(s) because of inf gradients | Compare the loss scale before and after `scaler.update()`; step the schedule only if the scale did not drop |
| `UserWarning: Converting a tensor with requires_grad=True to a scalar` | Logging `float(loss)` on a tensor still attached to the graph | Use `.detach()` before converting for logs |
| ruff E501 (3 times) | Long lines in docstrings and a test | Trivial: shorten |

## Quiz
**Q19:** In `train_epoch` the gradients are unscaled (`scaler.unscale_`) before `clip_grad_norm_`. What would happen if the order were reversed, and what would you see in the logged gradient norm?
**Your answer:** _open_

**Q20:** Validation loss is averaged per object. Night frames are 10% of the validation frames but even fewer of its objects. Which two things go wrong if we averaged per batch instead, and how would that distort the night-versus-day comparison?
**Your answer:** _open_

Q7 to Q10, Q15 to Q18 are still open.

## Interview angle
"I wrote the training loop with the failure modes in mind: it refuses to start if train and validation share a scene or the GPU isn't pinned, stops on the first non-finite loss and records which frames caused it, and it logs the commit hash and config of every run. The baseline and the fusion model run through identical code, and I proved the loop works by overfitting one batch before any real training."
