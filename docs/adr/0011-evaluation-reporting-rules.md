# 0011 — How day/night results are trained, scored and reported

**Status:** Accepted · **Date:** 2026-10-30

## Context
ADR 0008 found that night exists only in Singapore. `docs/object_counts.md` (`scripts/count_objects.py`, 2026-10-29) shows what the validation split can support:

| val cell | frames | objects | car | large_vehicle | pedestrian | two_wheeler |
|---|---|---|---|---|---|---|
| Boston day | 3,090 | 17,074 | 10,321 | 3,730 | 2,762 | 261 |
| Singapore day | 2,327 | 9,073 | 3,751 | 1,226 | 3,412 | 684 |
| Singapore night | 602 | 1,302 | 1,110 | **46** | **32** | 114 |

An AP computed from 32 pedestrians or 46 large vehicles can move by many points from a single missed detection, and a mean over the four classes would give these noisy classes the same weight as the car AP.

## Options
A. **A minimum-count rule** *(chosen)*. B. All four classes with bootstrap intervals only. C. Cars only at night.

## Decision
1. **Training:** both models train on the natural mix of the official `train` split (day and night together, 12% night frames). No oversampling or reweighting. Any rebalancing would be a separate experiment applied to both models.
2. **Evaluation:** on the official `val` split, reported for overall, day and night, plus the Singapore-day vs Singapore-night control (ADR 0008).
3. **Reporting rule, fixed before any result exists:** a class gets an AP in a cell only if that cell has at least **100 validation objects of the class**. Below that the cell shows "not enough data (n = …)" and the class is left out of the cell's mean. With the current counts, night reports car (1,110) and two-wheeler (114); large vehicle (46) and pedestrian (32) are shown as counts only.
4. **Uncertainty:** every reported number gets a 95% interval from a bootstrap that resamples **scenes**, not frames.
5. **Always shown:** scenes, frames and objects behind each cell.
6. **Checkpoint choice:** decided and stated before test-time numbers are looked at; model selection uses validation only; the test split stays sealed.

## Consequences
- \+ The headline night number cannot be driven by a handful of pedestrians.
- \+ The rule is fixed in advance, so it cannot be tuned to make the result look better.
- − Night results cover cars and two-wheelers only, and one neighbourhood (Holland Village). The README says so.
- − The threshold of 100 is a judgment call: two-wheelers at night (114) pass it only barely.
