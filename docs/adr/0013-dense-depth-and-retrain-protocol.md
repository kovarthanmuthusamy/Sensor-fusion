# 0013 — Dense depth input, depth tower into the heads, and the retrain protocol

**Status:** Accepted · **Date:** 2026-10-31

## Context
The first 24-epoch runs left two problems (learning log 006, 008):
1. **Fusion was indistinguishable from the baseline**, including on distance, because the depth channel was too sparse and weak: 0.77% of pixels had a value, far objects got about 3 depth pixels, zero meant both "no data" and "0 m", and the depth filters had a low learning rate. ADR 0004 had promised "thickening at load time", which was never built.
2. **Choosing `best.pt` by validation loss picked epoch 1 of 24.** The class loss grows with overfitting (confident misses cost a lot) while detection quality does not fall.

## Decisions
1. **Thicken the LiDAR points** (7×7 window, nearest depth wins) after the crop; **add a validity mask** as a 5th input channel. Input becomes R, G, B, depth, mask. Still input-level fusion.
2. **Separate depth stem** (`conv1_extra`): the extra channels get their own first convolution, added to the photo's. Same maths as one convolution; separate parameters allow their own learning rate (`lr_depth`, 1e-3) and a zero start. (Adam is scale-invariant, so scaling a gradient would not do the same.) Old 4-channel checkpoints still load.
3. **Depth tower into the heads** (`fusion_dense_head`): local LiDAR statistics (mean depth, nearest depth, density in each 4×4 block) go through two small convolutions and are concatenated to the neck's output for both heads; the new head inputs start at zero. This goes beyond pure input-level fusion (ADR 0001); it is still one network, trained end to end.
4. **Per-epoch validation mAP** (AP@0.5 per class, mAP, distance error for all/day/night) and **every epoch's checkpoint kept**; `best.pt` is chosen by overall validation mAP@0.5 (`trainer.select_by: all/map50`). The final evaluation (ADR 0011, 0012) still uses all ten IoU thresholds and the bootstrap. Selecting on validation is allowed; the test split stays sealed.
5. **Retrain: 8 epochs** (the overfit set in after epoch 2 to 4), seed 42, identical settings for all models.

## Evidence (3-epoch experiments, same seed and schedule, validation distance loss at epoch 3)
Bars fixed before each run.

| model | distance loss | vs baseline | bar |
|---|---|---|---|
| RGB baseline | 0.1400 | | |
| dense input (decisions 1, 2) | 0.1215 | -13.2% | ≥20% below the baseline: **not met**, but consistent and lower at every epoch |
| dense input + depth tower (decision 3) | **0.1016** | **-27.4%** | ≥20% below the baseline: **met**; ≥5% below the dense model (≤ 0.1154): **met** |

## Run plan
The queue chose `fusion_dense_head` by the rule above and trains the RGB baseline and `fusion_dense_head` (8 epochs each, one per GPU), then `fusion_dense` as an ablation, so the final comparison is baseline vs dense input vs dense input plus tower.

## Consequences
- \+ The fusion model finally uses its sensor, and the effect is measurable early.
- \+ Checkpoints are chosen by the metric that matters, and all epochs are kept for inspection.
- − The model is no longer pure input-level fusion; the README must say so.
- − The experiments are 3 epochs and one seed; the claim rests on the full retrain and its intervals.
- − The per-epoch scoring costs about 2 to 3 minutes per epoch.
- − 12% of far objects still have no LiDAR point at all; no input change can fix that.

## Result of the retrain (2026-11-01)
Full tables in [`docs/results/`](../results/README.md); interpretation in [log 010](../learning/010-retrain-and-results.md). Fusion minus baseline, best epochs, validation split:
- Distance error roughly halves with the final model (all: -3.8 points [-4.2, -3.4]; night: -7.3 [-8.3, -5.4]); the dense input alone gives -1.2 and -4.0.
- mAP@[0.5:0.95] improves (all: +1.5 [0.4, 2.6]; night: +3.5 [1.8, 5.3]); mAP@0.5 overall is unchanged.
- Night mAP@0.5 gains: +7.2 [2.3, 10.7] (dense input only), +2.6 [-2.2, 8.0] (final model, best epoch), +7.1 [2.5, 11.2] (final model, last epoch): likely, but checkpoint-sensitive.
- The depth tower is credited with the distance gain, the dense input with the night detection gain.
