# 0004 — Data pipeline choices (Day 2–3)

**Status:** Accepted · **Date:** 2026-10-27

## Context
The data layer must give, for every `CAM_FRONT` frame: the RGB image, a sparse depth image from `LIDAR_TOP`, 2D boxes with a class and a distance each, and a day/night tag. Facts found in nuScenes-mini (10 scenes, 404 keyframes, 283 day and 121 night, images 1600×900):
- nuScenes ships **only 3D boxes**, so 2D boxes must be derived.
- The devkit's `map_pointcloud_to_image` goes sensor → ego (at lidar time) → global → ego (at camera time) → camera. The two ego poses differ because the sensors fire at different moments.
- The official `mini_val` scenes (`scene-0103`, `scene-0916`) contain **no night scene**. All three night scenes (`1077`, `1094`, `1100`) are in `mini_train`.
- Around the car (all six cameras), night has 3,641 boxes. Only cars (1,662) and pedestrians (1,464) are plentiful there.

## Options and decisions

### 1. Projection: how LiDAR points reach the image
1. **Own pure function, devkit as a test oracle** *(chosen)*: points + calibrations in, pixels + depths out, with the transform chain explicit. A test compares it with `map_pointcloud_to_image`. Costs about 40 lines more.
2. **Wrap the devkit function**: less code, but it mixes file loading with the maths, and it teaches little about frames.

### 2. Classes
1. **4 merged classes** *(chosen)*: `car`; `large_vehicle` (truck, bus, trailer, construction vehicle); `pedestrian`; `two_wheeler` (bicycle, motorcycle). Built from the devkit's `category_to_detection_name` plus a 10→4 map. Dropped: `barrier`, `traffic_cone`, and categories the devkit does not map (stroller, wheelchair, personal mobility, animals, debris).
2. **All 10 detection classes**: comparable to the benchmark, but several classes have too few night examples on mini.
3. **2 classes (car, pedestrian)**: cleanest signal, weaker as a portfolio piece.

### 3. Splits on nuScenes-mini
1. **Official lists + a dev-only split** *(chosen)*: the default is `mini_train`/`mini_val`. A clearly labelled split `mini_dev` holds out one whole night scene (proposal: `scene-1100`) and one whole day scene (proposal: `scene-0103`). It exists only to smoke-test the day/night evaluation code and is never reported as a result.
2. **Official lists only**: simplest, but the evaluation code would first meet night data on Day 8.

Splitting is always **by scene**, never by frame (neighbouring keyframes are near-duplicates).

### 4. Defaults (config-driven, cheap to change)
- **Distance label** = depth (`z` in the camera frame) of the 3D box centre, the same quantity as the depth channel. Not the straight-line range.
- **Depth image** stored raw: metres, `0` = no LiDAR return, one pixel per point, nearest point wins. Scaling and thickening are applied at load time.
- **Ground-truth filter**: nuScenes visibility level ≥ 40% (tokens 2–4) and part of the box inside the image. **Never filter by LiDAR-point count**: that would give the fusion model an easier test set than the RGB-only baseline.

## Decision
The choices marked *(chosen)* above, plus the defaults in section 4. The code is split into one stage per file under `src/sensor_fusion/data/`: `projection`, `splits`, `labels`, `dataset`. Two PRs: projection first, then split + labels + dataset.

## Consequences
- \+ The projection is a pure function that is easy to unit-test. The dataset and the picture script reuse it.
- \+ The day/night code path runs on real data (one night scene) long before trainval arrives.
- − 4 classes means our mAP is **not comparable** with the nuScenes leaderboard. The README must say so.
- − A 2D box from projected 3D corners is the smallest rectangle around them, so it is slightly larger than the visible object, especially for rotated cars.
- − The `mini_dev` night validation has about 40 frames: it checks the code and says nothing about model quality.
- − Visibility ≥ 40% drops heavily hidden objects, from training and evaluation alike (same for both models).
