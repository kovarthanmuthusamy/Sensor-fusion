# 0010 — Training setup and safeguards (Day 4–5, PR C)

**Status:** Accepted, except the epoch count (open, see Decision) · **Date:** 2026-10-30

## Context
PR A and PR B fixed the data, the network and the loss. PR C adds the loop that trains them. Constraints: a shared pair of Volta GPUs (fp16 only, no bf16), baseline and fusion must be trained under identical settings, and every number must be traceable to code and config. Measured on 2026-10-30 on `v1.0-trainval`, GPU 0, fusion model, batch 16, 12 loader workers, fp16: **about 157 frames/s**, so one epoch of the 28,130 training frames takes about 3 minutes. The one-batch overfit on real mini data (batch of 8) took the loss from 11.42 to 0.57 (ratio 0.05).

## Options and decisions
The choices below were made by me inside the plan; all values live in `configs/trainer/default.yaml` and can be changed.

1. **Optimiser:** AdamW *(chosen)*, with two learning rates: 5e-4 for the neck and heads, 1e-4 for the pretrained backbone (including the first convolution and its depth channel), weight decay 1e-4 except on biases and normalisation parameters. Alternative: one learning rate (simpler, but the pretrained layers are disturbed as hard as the fresh ones).
2. **Schedule:** linear warm-up (500 steps), then cosine decay to 1% *(chosen)*. Warm-up covers the heads starting at a prior and the fp16 gradient scaler finding its scale. The schedule does not advance on a step the scaler skipped.
3. **Precision:** fp16 autocast for the network, loss in float32, gradient scaler, gradients unscaled before clipping at norm 10 *(chosen)*.
4. **Batch size 16, 12 workers** *(chosen)*: fits next to other users on a 32 GB GPU; the loader keeps up (the 157 frames/s above).
5. **Safeguards** *(chosen)*: the run refuses to start if train and val share a scene; refuses without `CUDA_VISIBLE_DEVICES` when a GPU exists (switch `trainer.require_pinned_gpu`); stops on the first non-finite loss and writes the offending sample tokens to `bad_batch.json`; has an overfit-one-batch mode that fails the run if the loss does not fall to 10% of its start.
6. **Tracking:** W&B logs the full resolved config, the git commit and a dirty flag, and the environment. Without credentials it switches to offline mode with a warning, so a missing login does not kill a run.
7. **Checkpoints:** `last.pt` every epoch, `best.pt` when the overall validation loss improves, both with optimiser, scheduler, scaler and config, written atomically; `trainer.resume=<last.pt>` continues a run.
8. **Validation in this PR is the loss only**, reported for all, day and night frames, averaged per object (not per batch). mAP and distance error belong to the evaluation PR (ADR 0011).
9. **Fixed epochs, no time cap by default** *(chosen)*: a run stopped by `trainer.max_minutes` is not comparable with its twin, so the cap is off and a run that hits it is flagged in the log.
10. **Worker seeding:** a seeded generator fixes the shuffle order and the base seed of the workers. PyTorch seeds Python's `random` and NumPy inside each worker itself (read from the installed source). I first wrote my own worker seeding on a wrong assumption and removed it.

## Decision
The choices above, with **one open item for you: the epoch count.** The default is 12 (about 36 min per model at the measured speed). At this speed 24 epochs cost about 72 min per model and 36 epochs about 108 min. More epochs are affordable, but the value is shared by both models and must be chosen before either run starts. My recommendation is 24: long enough for the cosine schedule to settle, and both models fit in one afternoon on two GPUs.

## Consequences
- \+ Baseline and fusion are one code path with one config group; a test enforces identical trainer settings.
- \+ Failure modes that cost hours (leakage, unpinned GPU, silent NaN) are stopped at the start or on the first bad step.
- \+ Runs are traceable (commit, config, env in W&B) and resumable.
- − The learning rates and loss weights are untuned starting points, checked only by the overfit test and a few minutes of real training.
- − Selecting `best.pt` by the overall validation loss favours day (88% of frames). The evaluation PR should also look at the night loss before choosing which checkpoint to report, and must say how it chose.
- − Validation loss is not mAP: a lower loss does not always mean better detections.
