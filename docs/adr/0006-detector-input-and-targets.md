# 0006 — Detector design, input preparation and targets (Day 4–5)

**Status:** Accepted · **Date:** 2026-10-27

## Context
ADR 0001 asks for one CNN that outputs 2D boxes and a distance per object, with an RGB-only twin as the baseline. Constraints measured on 2026-10-27: both GPUs are shared (about 8.5 GB and 73% used by others), one loader process delivers 14 frames/s (71 ms per frame), only `v1.0-mini` is on disk, and the timebox ends about Oct 31.

## Options and decisions

### 1. Detector design
1. **One centre-based network** *(chosen)*: each object is a peak on a grid, and box size, centre fine-offset and distance are read from further grids at that spot. Only the ImageNet backbone is pretrained; heads, targets, loss and decoder are ours.
2. Two-stage Faster R-CNN with a distance branch: mature and detection-pretrained, but heavier on shared GPUs and the distance branch means modifying library internals.
3. Off-the-shelf detector plus a separate distance model: least code, but two models, not "one CNN" (ADR 0001 would change).

### 2. When the distance output joins
**From the start** *(chosen)*: baseline and fusion are the same network from the first run; Day 6–7 only changes the input channels. The alternative would force retraining the baseline for a fair comparison.

### 3. Input resolution
**Half, 800×448** *(chosen)*. The photo is shrunk by 2 (1600×900 → 800×450), then one row is dropped at the top and bottom so the height is a multiple of 32 (448). Alternatives: full 1600×896 (about 4× slower on shared GPUs) or quarter 640×352 (far objects become a few pixels).

### 4. How distance error is learned
**Relative error via log-distance** *(chosen)*: a 10% error costs the same at 10 m and 80 m. Distances range from 8 m to 155 m (median 30 m). The reported error on Day 8 stays in metres and percent. Alternative: absolute metres, where far objects dominate the loss.

### 5. Smaller details that follow (reversible, set in config)
- Box sizes are also learned in log units, for the same reason as the distance.
- The network output grid is 1/4 of the input size (stride 4): 112×200 cells.
- Depth is shrunk by taking the **nearest valid depth** in each block, never an average (an average across a car's edge invents depths). Depth is then divided by 100 m; `0` still means "no LiDAR return".
- Augmentation: horizontal mirror only, probability 0.5, training only, applied to photo, depth and boxes together. Identical for baseline and fusion.
- A peak's width on the target grid follows the box per axis: σ = 0.54 × size in cells / 6 (the TTFNet rule, checked in the paper; 0.54 is the paper's own value, "not carefully selected"). Wide boxes get wide peaks, tall boxes tall peaks, and a small floor keeps tiny objects visible. I first remembered CenterNet's radius formula, but working out its numbers showed it did not behave as I recalled, so I used a rule I could verify.
- If two objects share a centre cell, the nearer one wins.
- The channel recipe comes from `model.in_channels` (3 = RGB, 4 = RGB + depth). It is the only difference between baseline and fusion.

## Decision
The choices marked *(chosen)*, built as three PRs: A (input preparation, target encoder, decoder), B (network and loss), C (trainer with W&B and safeguards).

## Consequences
- \+ Distance is one more output grid, so "one CNN → boxes + distance" stays literally true.
- \+ Targets and decoder are inverse of each other, so a round trip (encode, then decode) tests both without a trained network.
- \+ Light and fast enough for shared GPUs, with 8–16 loader workers to cover the 71 ms per frame.
- − More of our own code to review than with an off-the-shelf detector.
- − Two objects whose centres fall in the same grid cell can't both be represented (the nearer wins). Measured on mini: 2 of 993 objects (0.2%) were lost this way; the round-trip test pins that every other object comes back.
- − At half resolution an object 80 m away is about 15 px tall. The Day 8 evaluation should report accuracy by distance.
