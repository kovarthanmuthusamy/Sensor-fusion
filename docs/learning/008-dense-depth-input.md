# 008 — Why fusion looked like the baseline, and the dense-depth input (debugging exercise)

**PR:** `feat/dense-depth-input` · **Date:** 2026-10-31 · **Tags:** av-robotics, mlops, design

## What was built
- **Thickening** (`data/transforms.py: thicken_depth`): each LiDAR point is spread over a 7×7 window, nearest depth wins.
- **Validity mask** as a 5th input channel (`in_channels: 5`, `configs/model/fusion_dense.yaml`), so "no measurement" is no longer the same number as "0 m".
- **Depth stem** (`models/detector.py`): the extra channels go through their own first convolution `conv1_extra`, added to the photo's. Same maths as one convolution, but separate parameters, so they can have their own learning rate (`lr_depth`, `optim.py`). Old 4-channel checkpoints still load.
- 25 new or changed tests. Experiment: 3-epoch runs of the RGB baseline and the dense model, identical schedule.

## The exercise
**Symptom:** the fusion model's training loss matched the baseline's to three decimals at every epoch, its validation distance loss was the same (0.137 vs 0.136), and the evaluation found no significant difference anywhere, although LiDAR measures distance directly.
**Diagnosis path:** memorisation was proposed (it explains equal *training* loss but not equal *validation* distance). The input was checked: only 0.77% of pixels had a depth value, 88% of output cells none, far objects about 3 depth pixels (18% of them none), values around 0.1 against RGB std 0.87, and the value 0 meant both "no data" and "0 m". ADR 0004 had promised "thickening applied at load time", and the code never did it.

## Experiment (pre-registered, 3 epochs, same seed and schedule)
Criterion fixed before the run: dense model's validation *distance* loss at epoch 3 at least 20% below the baseline's.

| validation distance loss | epoch 1 | epoch 2 | epoch 3 | night, epoch 3 |
|---|---|---|---|---|
| RGB baseline | 0.1760 | 0.1418 | 0.1400 | 0.1668 |
| dense fusion | 0.1376 | 0.1245 | 0.1215 | 0.1414 |
| reduction | 22% | 12% | **13%** | 15% |

**The 20% bar was not met at epoch 3 (13.2%).** But the effect is real and consistent: lower at every epoch, in day and night, and also lower in class and size loss, unlike the first fusion run, which never separated from the baseline. The gap narrows as the RGB model catches up on distance from size cues.

**Second experiment: depth statistics also fed into the heads** (a small depth tower; [ADR 0013](../adr/0013-dense-depth-and-retrain-protocol.md)). Bars fixed beforehand: ≥20% below the baseline, and ≥5% below the dense model.

| validation distance loss | epoch 1 | epoch 2 | epoch 3 | night, epoch 3 |
|---|---|---|---|---|
| RGB baseline | 0.1760 | 0.1418 | 0.1400 | 0.1668 |
| dense input | 0.1376 | 0.1245 | 0.1215 | 0.1414 |
| dense input + depth tower | 0.1222 | 0.1051 | **0.1016** | 0.1194 |

Both bars were met (27.4% below the baseline, 16% below the dense model), and unlike the dense input alone the advantage does not shrink with more epochs. The retrain uses this model. The lesson: after fixing what reaches the network, ask where the information has to be *used*; the distance head needed the local depth directly.

## Senior instinct
- **Measure the input before blaming the model.** Three numbers (density, scale, points per box) explained a result that looked like a training mystery.
- **A model that "does the same" with a new sensor is a signal about the input path, not about the sensor.**
- **Audit ADR promises against code.** A planned step ("thickening") was written down and never built.
- **Sparse data is not "mostly zeros": it needs a validity mask.** Zero as "no data" and zero as a value are different things.
- **Adam is scale-invariant**, so multiplying a gradient does not speed up learning; a slower layer needs its own parameter group and learning rate, hence the separate depth stem.
- **Pre-register the success criterion.** The experiment missed the bar we set, and we say so, instead of moving it.
- **Keep old checkpoints loadable** when a structure changes, or earlier results can no longer be reproduced.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| Fusion == baseline on loss, mAP and distance | Depth reached the network as a few dots (0.77% of pixels), with no validity mask, a small scale, and a low learning rate on its filters | Print density, scale and per-object point counts of every extra input channel before training |
| `shell: unexpected EOF` when patching files with a heredoc | Quotes inside a long heredoc broke the shell | Write the patch script with the file tool and run it |
| New test used class id 1 for "pedestrian" | Pedestrian is id 2 (id 1 is large_vehicle) | Look the class up by name in tests, not by number |

## Quiz
**Q23:** Why does multiplying the depth channel's gradient by 10 not make its filters learn faster with AdamW, and what does a separate learning rate change instead?
**Your answer:** _open_

**Q24:** The mask channel and the thickened depth are both zero in the same pixels. Why is the mask still useful to the network?
**Your answer:** _open_

Q7 to Q10, Q15 to Q22 are still open.

## Interview angle
"The fusion model matched the RGB baseline, which made no physical sense for a depth sensor. Instead of tuning, I measured the input: under 1% of pixels carried a LiDAR value, and distant objects got about three points. I added thickening, a validity mask and a separate depth stem with its own learning rate, tested the change with a pre-registered 3-epoch experiment (13% lower distance loss, short of my 20% bar), and reported that honestly."
