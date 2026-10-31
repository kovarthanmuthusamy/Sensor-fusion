# 0012 — Evaluation protocol: metrics, matching and the bootstrap

**Status:** Accepted · **Date:** 2026-10-30

## Context
ADR 0001 asks for mAP and distance error, split by day and night. ADR 0008 and 0011 fix which cells to report and when a class may be scored. This ADR fixes how the numbers are computed. The decisions below were made by me inside the plan; the values live in `configs/evaluation/default.yaml`.

## Decisions
1. **AP:** COCO-style average precision with 101-point interpolation. Headline is AP at IoU 0.5 (`map50`); `map` averages over IoU 0.50, 0.55, …, 0.95. Own implementation (the bootstrap needs scene weights, which pycocotools lacks), checked against pycocotools at all ten IoU thresholds in a test.
2. **Matching:** COCO's rule, per frame and per class: detections from the highest score down, each takes the unused same-class label with the highest IoU if that reaches the threshold. Detections are decoded with `top_k=100` and a low score threshold of 0.05, because AP needs the whole precision–recall curve.
3. **Space:** boxes are compared in network-input pixels (800×448), where the labels already are, so both models are scored in the same space. No mapping back to the full-size photo.
4. **Distance error:** only for detections matched at IoU ≥ 0.5 **and** confident (score ≥ 0.3). Reported as the median relative error `|pred − true| / true`, the median absolute error in metres, and the median relative error by true-distance range (0–20, 20–40, 40–60, 60+ m) with the number of matched objects in each range. Median, not mean, because a few wild far-range errors would dominate a mean.
5. **Cells:** all, day, night, Boston day, Singapore day, Singapore night (ADR 0008). Cells with fewer than 10 scenes are marked †, because a bootstrap over so few scenes is not reliable.
6. **Bootstrap:** resample **scenes** with replacement (1000 times), never frames. Both models are scored on the same resamples, so the interval of the difference (fusion − baseline) is a paired interval. Which classes are scored (ADR 0011's ≥ 100 objects) is decided once from the full cell, not per resample.
7. **Implementation trick:** every per-frame quantity is computed once; a resample is just a weight (the multiplicity) per scene. Weight *k* on a scene is exactly the same as including it *k* times, which a test verifies.
8. **Provenance:** each checkpoint carries the config it was trained with. Evaluation rebuilds the model and the input preparation from it (and refuses two checkpoints with different preparation), so a model cannot be evaluated with settings it was not trained with.
9. **Split:** `val` only. The test split stays sealed.

## Consequences
- \+ Numbers are checkable against the COCO reference, and uncertainty is stated for every cell.
- \+ The paired interval answers the real question ("does fusion beat the baseline, and by how much?") more sharply than two separate intervals.
- − Distance error is measured only on matched, confident detections. A model that detects fewer far objects can look better on distance, so the matched counts per range are always shown next to it.
- − AP at 800×448 input-pixel space is not comparable with published nuScenes numbers (4 merged classes, front camera, 2D boxes).
- − The cells' scenes overlap (all contains day and night), so cells are not independent of each other.
