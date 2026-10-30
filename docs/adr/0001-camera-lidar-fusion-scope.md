# 0001 — Camera + LiDAR fusion for 2D detection and object distance

**Status:** Accepted · **Date:** 2026-10-17

## Context
Goal: show, as a portfolio project, that LiDAR helps camera perception at night. Constraints: 1–2 weeks, a shared Windows workstation (2× Quadro GV100), and no custom CUDA ops (spconv/mmdet3d are unreliable on Windows).

## Options
1. **Late fusion**: a pretrained 2D detector, then the median LiDAR depth inside each box. About 3 days and no training, but there's little ML in it.
2. **Input-level (early) fusion**: project LiDAR into the camera as a sparse depth image, stack it as a 4th channel, and train one CNN for boxes + distance. Pure PyTorch.
3. **BEV 3D detection** (BEVFusion style): the industry state of the art, but it needs sparse-conv CUDA ops. High risk on Windows in 2 weeks.

## Decision
Option 2, with an **RGB-only twin as the baseline**: the same architecture, schedule and seed, with only `in_channels` 3 vs 4. Evaluate mAP and distance error split by {day, night}.

Terminology: stacking depth at the input is *early* fusion. *Mid* (feature-level) fusion uses a separate depth branch and fuses feature maps. That was a candidate ablation for Days 6–7. **Update 2026-10-27:** dropped, to protect the timebox. The project stays input-level (early) fusion only.

## Consequences
- \+ A controlled experiment: only the input differs, so a night-time gap is attributable to LiDAR.
- \+ Pure PyTorch, so it runs on Windows.
- − Output is 2D boxes + distance, not full 3D boxes.
- − Input-level fusion is sensitive to calibration and timestamp errors and to LiDAR sparsity at range.
- − A pretrained backbone expects 3 channels. How to initialise the 4th is a later decision.
