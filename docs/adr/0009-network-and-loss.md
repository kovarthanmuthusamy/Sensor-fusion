# 0009 — Network and loss (Day 4–5, PR B)

**Status:** Accepted · **Date:** 2026-10-29

## Context
ADR 0006 fixed the output contract: a class grid and a 5-channel regression grid at 1/4 of the input size (112×200 cells for the 448×800 input). The network must be the same for the RGB baseline and the RGB+depth fusion model, differing only in input channels. Constraints: shared GV100 GPUs, a bounded training budget (deadline Nov 3), ImageNet weights are reachable (checked 2026-10-29).

## Options and decisions

### 1. Backbone
**ResNet-18** *(chosen)*: 47 MB, fastest on shared GPUs, enough for a 2-day budget. Alternatives: ResNet-34 (about twice the compute for a small gain), ResNet-50 (slower still; the extra capacity is hard to use in a short run). Only `resnet18` is built; the config key exists so a change is a visible decision.

### 2. Neck
**Top-down upsampling with skip connections** *(chosen)*: coarse features are upsampled and added to finer stages, ending at stride 4, so far-away small objects keep their detail. Alternatives: CenterNet-style deconvolution without skips (weaker on small objects), torchvision FPN (heavier, built around anchor-based detectors).

### 3. Depth channel initialisation
**Zero weights for the added channel** *(chosen)*: the fusion model starts as exactly the baseline and learns how much depth to use. Tests show the output is identical to the baseline at the start for any depth, and that the zero weights still receive gradients. Alternatives: mean of the RGB filters (depth starts with arbitrary influence), random (weakest start).

### 4. Loss
**Penalty-reduced focal loss on the class grid + L1 on the regression grid at object centres** *(chosen)*. The regression targets are already in log units (size, distance), so L1 is simple and the distance error is relative. Alternative: add a GIoU box loss (better overlap in principle, a third thing to tune).

### 5. Smaller details that follow (reversible, set in config)
- Initial class probability 0.1 (`prior_prob`) and initial distance 30 m (`distance_init_m`, the label median), set through the last layer's bias. Last-layer weights start near zero so the bias decides the first outputs.
- Loss weights start at 1 for the class term and 0.1 for each regression group (size, offset, distance). Because the targets are in log units (small numbers), these may be too low; the loss curves decide, not this ADR.
- The loss is computed in float32 even under fp16 autocast.
- Loss terms are divided by the number of objects in the batch (at least 1), not by the number of cells.
- Hyperparameters live in `configs/model/*.yaml` (identical except `name`, `in_channels`) and `configs/loss/default.yaml`.

## Decision
The choices marked *(chosen)*.

## Consequences
- \+ One class, two configs: baseline and fusion cannot drift apart (a config test enforces it).
- \+ The whole chain (encode → network → loss → decode) is tested by memorising one batch on CPU.
- \+ Export-friendly: the network returns plain grids; decoding stays outside (ADR 0007).
- − ResNet-18 caps accuracy; the README must not oversell absolute mAP.
- − Zero-initialised depth may learn slowly if the LiDAR signal is sparse; watch the early curves of the fusion run.
- − The starting loss weights are a guess. Revisit after the first real run, and change them for both variants together.
