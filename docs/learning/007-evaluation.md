# 007 — Evaluation: mAP, distance error and the scene bootstrap (Day 8–9)

**PR:** `feat/evaluation` · **Date:** 2026-10-30 · **Tags:** mlops, av-robotics, design

## What was built
- **Matching and AP** (`evaluation/matching.py`, `metrics.py`): COCO-style matching and 101-point AP with scene weights, distance error, and the minimum-object rule from ADR 0011.
- **Bootstrap** (`evaluation/bootstrap.py`): scene-level, paired between the two models.
- **Inference and report** (`evaluation/inference.py`, `report.py`): decoding to per-frame results, six cells, markdown and JSON tables.
- **Entry point** (`evaluate.py`, `configs/evaluate.yaml`, `configs/evaluation/default.yaml`): `python -m sensor_fusion.evaluate baseline=<ckpt> fusion=<ckpt>`.
- Protocol in [ADR 0012](../adr/0012-evaluation-protocol.md). 34 new tests; the AP agrees with pycocotools at all ten IoU thresholds.

## Decisions made (by you)
- Reporting rule (option A) and training on the natural mix → [ADR 0011](../adr/0011-evaluation-reporting-rules.md).

## Decisions I made inside the plan (veto any)
- AP@0.5 as the headline plus AP@[0.5:0.95]; median relative distance error on matched detections with score ≥ 0.3; the distance ranges 0/20/40/60 m; 1000 bootstrap resamples; † for cells with fewer than 10 scenes.

## Senior instinct
- **Test your metric against the reference implementation.** AP is easy to get subtly wrong. Agreement with pycocotools on random data, at ten thresholds, turns "I think this is right" into evidence.
- **Know what the reference does that you don't.** My AP first disagreed with COCO by about 0.003. Matching was identical in every frame, and my AP function gave the same answer as mine when fed COCO's own match list, so the difference sat in accumulation: one random test box was inverted (bottom above top). COCO gives that box a negative area and silently ignores it; I counted it as a false positive. Decoded boxes can never be inverted, so production was unaffected, and I fixed the test data.
- **A bootstrap sample is a weight vector.** Resampling scenes with replacement is the same as giving each scene a count; computing per-frame results once and re-weighting is far cheaper than recomputing, and a test pins that weight *k* equals including the scene *k* times.
- **Resample the unit that is independent.** Frames in a scene are near-duplicates, so frames are not independent; scenes are.
- **Pair the comparison.** Scoring both models on the same resamples makes the interval for the difference much tighter than comparing two separate intervals, because the scene-to-scene variation cancels.
- **Decide the reporting rule before seeing results**, and decide eligibility once from the full data, not per resample (otherwise the set of classes changes between resamples).
- **Rebuild from the checkpoint's own config.** Evaluation reads the model and preparation settings from the checkpoint, so a mismatch is impossible by construction.
- **A confidence interval of zero width is a warning, not a result.** One scene gives that, so small cells are marked †.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| My AP differed from COCO's by 0.003 (0.2950 vs 0.2979) | Test data contained a box with bottom above top, which COCO ignores (negative area) and I counted as a false positive | Compare in stages: first the per-frame matches (identical), then feed the reference's own match list to your accumulation (identical), then find the remaining difference in what each side counts |
| `GlobalHydra is already initialized` in a test | A helper opened a second Hydra context inside the first | Build whatever needs Hydra before entering the outer context |
| Test expected 81 mini frames, there were 80 | I assumed the scene sizes instead of asking the dataset | Take expected counts from the data source |

## Quiz
**Q21:** My AP and COCO's differed on one frame by a detection with a negative-area box. Why would a decoded detection from this model never have a negative area, and which line of the decoder guarantees it?
**Your answer:** _open_

**Q22:** For the fusion-minus-baseline difference we use the same resampled scenes for both models. Why does that give a tighter interval than subtracting the two separate intervals, and when would it not help?
**Your answer:** _open_

Q7 to Q10, Q15 to Q20 are still open.

## Interview angle
"I wrote the detection metrics myself so I could put scene weights into them, then verified the AP against the COCO reference at ten IoU thresholds. Uncertainty comes from a scene-level paired bootstrap, because frames within a scene are near-duplicates, and the rule for which classes get reported was fixed before any results existed."
