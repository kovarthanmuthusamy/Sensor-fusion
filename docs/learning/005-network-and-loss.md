# 005 — Network and loss (Day 4–5, PR B)

**PR:** `feat/network-and-loss` (stacked on `feat/input-and-targets`) · **Date:** 2026-10-29 · **Tags:** design, mlops, av-robotics

## What was built
- **Detector** (`models/detector.py`): ResNet-18 → top-down upsampling with skip connections → a class head and a regression head, both at 1/4 of the input size. Baseline and fusion are the same class; `in_channels` is the only difference.
- **Loss** (`models/losses.py`): penalty-reduced focal loss on the class grid plus L1 on the five regression channels at object centres, normalised by the number of objects.
- **Config**: `configs/model/{rgb,fusion}.yaml` extended, new `configs/loss/default.yaml`. Decisions in [ADR 0009](../adr/0009-network-and-loss.md).
- 25 more tests than PR A (175 in the suite, counted by pytest); both new modules at 100% line and branch coverage; mypy clean.

## Decisions made (by you)
- ResNet-18, skip-connected upsampling, zero-initialised depth channel, focal + L1 → [ADR 0009](../adr/0009-network-and-loss.md).

## Decisions I made inside the plan (veto any)
- Heads start at a **prior**: class probability 0.1 and distance 30 m, via the last layer's bias.
- Loss weights 1 / 0.1 / 0.1 / 0.1 (class / size / offset / distance), as promised. They are only a first guess.
- Loss computed in float32; normalised by objects, not cells.
- Added `torchvision.*` to mypy's untyped-import list (torchvision ships no type markers).

## Senior instinct
- **Initialise the last layer's bias, not just the weights.** With 4 × 22,400 cells and a handful of objects, a class output starting at probability 0.5 makes the first gradients all about suppressing empty cells. Starting at 0.1 (CenterNet's trick) avoids that. A test pins the initial output.
- **Zero init is not zero gradient.** The depth filters start at 0, but their gradient depends on the input, not on their own value, so they still learn. This is the opposite of zero-initialising a whole layer's weights *and* its successor. A test checks the gradient is non-zero.
- **Make "fusion starts as the baseline" a test.** If it holds, any later gap between the models is learned, not an initialisation accident.
- **Overfit one batch through the whole chain.** One image, two objects: the loss drops by 10×, and decoding returns the right classes, boxes (within 6 px) and distances (within 15%). That proves the encoder, network, loss and decoder agree, before any GPU time is spent.
- **Compute losses in float32.** `log`, `sigmoid` and powers are the places fp16 overflows or loses precision on the GV100. Writing the log terms with `logsigmoid` keeps extreme logits finite.
- **Inject bugs to test your tests.** 10 realistic bugs (flipped bias sign, non-zero depth init, dropped skip connection, wrong distance channel, gradients cut off, no penalty reduction, unmasked L1, missing normalisation, swapped weight) were all caught.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| 8 mypy errors in `detector.py` | Indexing `nn.Sequential[-1]` gives `Tensor \| Module`, `Conv2d.bias` is `Tensor \| None`, and torchvision has no type markers | Small typed helpers (`_last_conv`, `_last_bias`) with an `assert isinstance`; the assertion also documents the assumption |
| My bug-injection script crashed on its first run (wrong Python path) and **left a flipped bias sign in `detector.py`** | The script patched the file, then failed before restoring it | I noticed the leftover `.bak` file, restored the file, and made the script restore in a `finally` block and use an absolute path. Always make a mutate-and-restore tool restore even on failure, and check `git diff` afterwards |

## Quiz
**Q17:** The class head's bias starts at −log(9) (probability 0.1). What would the first training steps look like if it started at 0 (probability 0.5), and why does the focal loss alone not fix that?
**Your answer:** _open_

**Q18:** The depth channel's filters start at zero, yet they get gradients. Why does a zero weight still get a non-zero gradient here, and in which situation would zero initialisation truly freeze learning?
**Your answer:** _open_

Q7 to Q10 and Q15, Q16 are still open.

## Interview angle
"I built one detector class for both the camera-only baseline and the camera+LiDAR model; they differ only in input channels, and the extra depth channel starts at zero so the fusion model begins as the baseline. I checked that with tests, and that the whole encode–network–loss–decode chain can memorise one batch, before spending GPU time."
