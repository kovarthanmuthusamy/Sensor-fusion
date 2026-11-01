# 010 — The retrain, the three-way comparison and what it says (Oct 31 – Nov 1)

**Date:** 2026-11-01 · **Tags:** av-robotics, mlops, design

## What was done
- Retrained with the fixes from log 008 ([ADR 0013](../adr/0013-dense-depth-and-retrain-protocol.md)): dense depth input, a separate depth stem, 8 epochs, validation mAP after every epoch, checkpoints by mAP. The queue picked the fusion model by a rule written down before the experiment finished (`docs/results/head-experiment-decision.json`).
- Three models, same seed, same schedule, same data: the **RGB baseline**, **dense input only** (ablation), and **dense input + depth tower into the heads** (the final model).
- Evaluated all three on the 6,019 validation frames with the scene-level paired bootstrap, and the final pair twice (best epoch and last epoch) to see whether the checkpoint choice matters. Full tables in [`docs/results/`](../results/README.md).

## The result (best epochs, fusion minus baseline, 95% interval)
| | dense input only | dense input + depth tower |
|---|---|---|
| median distance error, all (baseline 8.1%) | 6.8% (**-1.2** [-1.6, -0.9]) | **4.3%** (**-3.8** [-4.2, -3.4]) |
| median distance error, night (baseline 10.9%) | 6.9% (**-4.0** [-5.2, -2.1]) | **3.6%** (**-7.3** [-8.3, -5.4]) |
| mAP@0.5, common classes, all (baseline 51.2) | +0.8 [-1.9, 3.4] | -0.2 [-2.4, 1.9] |
| mAP@0.5, common classes, night (baseline 41.4) | **+7.2** [2.3, 10.7] | +2.6 [-2.2, 8.0] (final epoch: **+7.1** [2.5, 11.2]) |
| mAP@[0.5:0.95], common classes, all (baseline 21.9) | **+1.2** [0.1, 2.3] | **+1.5** [0.4, 2.6] |
| mAP@[0.5:0.95], common classes, night (baseline 16.4) | **+4.9** [2.7, 7.0] | **+3.5** [1.8, 5.3] |

**Reading it:** (1) the dense input (thickened depth + mask + depth stem) is what helps *detection* at night; (2) the depth tower into the heads is what halves the *distance* error; (3) overall detection at IoU 0.5 is unchanged; boxes get tighter (mAP@[0.5:0.95]); (4) the baseline's distance error gets worse at night (10.9% vs 8.0%) while the fusion models' does not.

## Senior instinct
- **Test the claim at the unit that decides it.** The distance gain is the most robust result (it holds at every range, in day and night, for both checkpoints); the night detection gain moves from +2.6 to +7.1 with the checkpoint, so it is reported as likely but not nailed down.
- **Run the ablation, not just the winner.** Without it the story would have been "the depth tower helps everything"; with it the tower is credited for distance, and the dense input for night detection.
- **Re-evaluate at more than one checkpoint** when the interesting cell is tiny (15 night scenes): the baseline's own night numbers moved from 10.9% to 7.6% distance error between checkpoints.
- **Automate the decision, then read it.** The queue applied the pre-registered rule by itself and wrote the numbers to a file; the human reviews the file, not a memory of the rule.
- **Fix the first-order cost, then measure again.** The profile showed one function (7×7 max-pool) was 100 ms of 146 ms of preparation; splitting it into two 1-D passes gave identical output (tested against the definition) in 29 ms.
- **Thread counts are a deployment parameter.** On a 36-core machine PyTorch's default made small operations 3× slower and the tail 5× worse than 4 threads (300 → 102 ms); the service now sets its own.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| Queue's ablation run died after 6 s | `model=fusion_dense\r`: Python on Windows prints `\r\n`; the shell kept the `\r` | Read the first lines of the failed run's log; strip `\r` when a shell script parses another program's output |
| Validation loss rose for 24 epochs while mAP held | Overfitting shows up in the confident-miss term of the loss first | Compare checkpoints on the metric that matters before trusting the loss (log 006) |
| Night detection gain differed between checkpoints (+2.6 vs +7.1) | 15 night scenes in one neighbourhood: a few scenes decide the number | Evaluate more than one checkpoint; report the range, not the better one |

## Quiz
**Q27:** The depth tower halves the distance error but does not change mAP@0.5 overall, while the dense input improves night mAP@0.5 but helps distance less. Why would giving the *heads* local depth help distance more than detection, and giving the *first layers* a dense depth map help detection more?
**Your answer:** _open_

**Q28:** For the night detection gain, one checkpoint gave +2.6 [-2.2, 8.0] and another +7.1 [2.5, 11.2]. What would you do, with the time and compute you have left, to find out which statement is closer to the truth?
**Your answer:** _open_

Q7 to Q10, Q15 to Q26 are still open.

## Interview angle
"I traced why my LiDAR model matched the camera-only baseline to a sparse, unmasked depth input, fixed it with thickening, a validity mask and a separate depth stem, and then added a depth path into the heads. I ran the ablation to credit each part, evaluated with scene-level bootstrap intervals at two checkpoints, and report that the distance error halves (8.1% to 4.3%, and 10.9% to 3.6% at night) while the night detection gain is real but sensitive to the checkpoint."
