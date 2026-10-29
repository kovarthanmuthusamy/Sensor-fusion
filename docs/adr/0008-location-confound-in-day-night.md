# 0008 — The night gap is confounded with location

**Status:** Accepted · **Date:** 2026-10-29

## Context
The headline result (ADR 0001) is the day-versus-night gap of the fusion model against the RGB baseline. Night is tagged by the word "Night" in the scene description. Counts from `v1.0-trainval` metadata (850 scenes) and `v1.0-mini` (10 scenes), 2026-10-29:

| Split | Day scenes | Night scenes | Night locations |
|---|---|---|---|
| trainval train | 616 (Boston 390, Singapore 226) | 84 | Holland Village 51, Queenstown 33 |
| trainval val | 135 (Boston 77, Singapore 58) | 15 (602 frames) | Holland Village 15 only |
| mini | 7 | 3 | Holland Village 3 only |

Facts that follow:
- **All 99 night scenes in trainval are in Singapore.** Boston has none, so "night" is never observed in a city that drives on the right. City, driving side, road layout and vehicle mix move together with the light.
- Singapore has 284 day scenes, so a comparison with the city held fixed exists.
- Val night is one neighbourhood (Holland Village), although train also has Queenstown night. Val night is only 602 frames.
- Mini cannot separate night from location at all (Holland Village has no day scene there).

## Options
1. **Report only {day, night}**: simplest, but a reader cannot tell a light effect from a city effect.
2. **Report {day, night} and a Singapore-only control** *(chosen)*: the headline stays as planned. Next to it, the same metrics on Singapore day versus Singapore night. Every number carries its scene and frame count (and object count per class).
3. **Cross-city test (train Boston, test Singapore or the reverse)**: rejected, because there is no Boston night to train or test on, so it would measure city shift and nothing about night.
4. **Rebalance or re-split to remove the confound**: rejected. The official split is kept so results compare with published numbers, and re-splitting is not possible anyway when one city has no night.

## Decision
Option 2.
- Keep the official scene-level split.
- The Day 8–9 evaluation reports mAP and distance error for {day, night} (headline) and for Singapore {day, night} (control), with sample sizes in every table cell.
- Night numbers are called indicative where val has few frames, and the README says val night is one neighbourhood.
- Before Day 8, count objects per class in each cell; if a class has too few night objects, report it as "not enough data" instead of a number.

## Consequences
- \+ The central claim is honest: we can say how much of the gap survives within Singapore.
- \+ Costs only a second slice in the evaluation, no extra training.
- − The conclusion is limited to Singapore night. We cannot claim the same gap for Boston night, because no such data exists in nuScenes.
- − Small val-night sample gives wide uncertainty; the README must not present a single night mAP as precise.
- − Mini results cannot support any day/night claim and are only for development.
