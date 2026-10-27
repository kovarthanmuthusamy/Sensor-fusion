# 0005 — Label rules and dataset output (Day 3)

**Status:** Accepted · **Date:** 2026-10-27

## Context
nuScenes gives 3D boxes only; ADR 0004 fixed that the 2D boxes are derived from them (4 merged classes, visibility ≥ 40%, no filtering by LiDAR points). Measured on nuScenes-mini, `CAM_FRONT`, 4 classes (day / night):

| What happens to a 3D box | day | night |
|---|---|---|
| All 4-class boxes around the car | 11,192 | 3,471 |
| Entirely behind the camera | 5,014 | 1,652 |
| Straddles the camera plane (some corners behind) | 352 | 65 |
| In front, outside the photo | 2,581 | 815 |
| In front, at least partly in the photo | 3,245 | 939 |
| …and visibility ≥ 40% | 2,319 | 758 |

Also: 16% of the kept boxes are cut by the photo edge; no kept box is shorter than 10 px (5th percentile 31 px); 17% (day) and 21% (night) of kept boxes contain zero LiDAR points; distance (camera z) has median 30 m, 95th percentile 82 m, maximum 155 m. Two of the eight official `mini_train` scenes (`scene-0553`, `scene-0796`) are official **validation** scenes.

## Options and decisions

### 1. Boxes that cross the camera plane
1. **Drop them** *(chosen)*: the same rule as the devkit's own `box_in_image`. Costs about 10% (day) and 6% (night) of the in-photo candidates, mostly very close objects.
2. Clip the 3D box at the camera plane: geometrically correct, about 30 more lines, harder to test.
3. Project only the corners in front: easy, but the 2D boxes come out too small.

### 2. Boxes cut by the photo edge
1. **Keep if at least 50% of the projected box is inside the photo** *(chosen)*, with the box clipped to the photo. Drops about 9% of kept boxes. The 50% is a judgment, held in one constant (`MIN_INSIDE_FRACTION`).
2. Keep any visible part: slivers are hard to learn and to judge a distance for.
3. Keep only fully inside: loses 16% of boxes, mostly roadside objects.

### 3. Distance label
**Depth (camera z) of the box centre** *(chosen, as in ADR 0004)*. Alternatives: nearest face (about 2 m closer for a car, depends on rotation), or storing both. The example car: centre 14.9 m, nearest face 12.8 m.

### 4. What the dataset returns
1. **Raw arrays, transforms later** *(chosen)*: `image` `(H, W, 3)` uint8; `depth` `(H, W)` float32 metres, `0` = no return; `boxes` `(K, 4)` float32 `[x1, y1, x2, y2]` pixels (torchvision convention); `classes` `(K,)` int64, 0-based indices into `CLASS_NAMES` (the detector wrapper adds the background class); `distances` `(K,)` float32 metres; `is_night` bool; `sample_token` str. Resizing, scaling and stacking the 4-channel input are a separate Day 4 stage driven by config.
2. Ready-made tensors: fewer files now, but data loading gets mixed with experiment choices.

### 5. Splits
Official `train`, `val`, `mini_train`, `mini_val`, plus the dev-only `mini_dev_train` / `mini_dev_val` from ADR 0004. `mini_dev_val` = `scene-0103` (day) + `scene-1100` (night). Never mix mini splits with trainval splits.

## Decision
The choices marked *(chosen)*. No minimum box size (none is needed). Labels are **not** filtered by LiDAR-point count; the share of boxes without LiDAR points is reported separately on Day 8.

## Consequences
- \+ One pure function (`box_to_2d`) holds all the geometry rules, with hand-computed tests.
- \+ Raw arrays make the dataset testable without torch transforms, and the RGB-vs-fusion difference lives only in the Day 4 config.
- − Very close objects (those crossing the camera plane) never appear in training or evaluation.
- − A 2D box from projected corners is slightly larger than the visible object (ADR 0004).
- − 17–21% of labelled objects have no LiDAR points: the fusion model cannot get depth help there, which caps its gain.
- − Training on `mini_train` and validating on trainval `val` would leak: two mini training scenes are validation scenes there.
