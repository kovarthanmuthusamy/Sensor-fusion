# 003 — Scene splits, 3D→2D labels and the dataset

**PR:** `feat/nuscenes-dataset` · **Date:** 2026-10-27 · **Tags:** av-robotics, mlops, design

## What was built
- `src/sensor_fusion/data/splits.py`: official and dev-only scene splits, and the day/night tag.
- `src/sensor_fusion/data/labels.py`: `box_to_2d` (all the geometry rules) and `frame_labels` (3D annotations → 2D boxes, classes, distances).
- `src/sensor_fusion/data/dataset.py`: `NuScenesFrontDataset`, which returns raw arrays (photo, sparse depth, boxes, classes, distances, night flag, token).
- `scripts/show_sample.py`: draws depth points, boxes and distances on a frame for the visual check.
- Tests: 98 in the whole suite (43 new), 100% line and branch coverage on all four data modules; `pillow` is now a declared dependency.
- Decisions are in [ADR 0005](../adr/0005-label-rules-and-dataset-output.md).

## Decisions made (by you)
- Boxes crossing the camera plane are dropped; boxes are kept when at least 50% is inside the photo; the distance label is the box centre's depth; the dataset returns raw arrays and transforms come later → ADR 0005. Each came with measured numbers on nuScenes-mini.

## Decisions I made inside the plan (veto any)
- The label rules (`MIN_VISIBILITY`, `MIN_INSIDE_FRACTION`, `MIN_CORNER_DEPTH`) are named constants in code, not config. They define the benchmark's ground truth, not an experiment hyperparameter, so changing them would change what "correct" means for both models. `.claude/rules/ml-hygiene.md` says hyperparameters live in `configs/`, so this is a judgement call.
- The night tag is a case-insensitive substring match on "night".
- Class ids are 0-based; the detector wrapper adds the background class.
- The dataset refuses a split whose scenes are not in the chosen data version.
- The devkit tables are left out of the pickled dataset, so DataLoader workers reload them.
- `frame_labels` takes `(nusc, cam_token)`: the sample is implied by the camera token.

## Senior instinct
- **Measure before you decide.** Counting what happens to every 3D box (behind, straddling, outside, cut, hidden) turned each rule into a decision with a number, and showed that no minimum-size rule is needed.
- **A test can pass for the wrong reason.** The mutation check showed the straddler test still passed with the rule removed, because the box was rejected by another filter. The fix is a box that only that rule can reject, plus a control that proves the same box is kept elsewhere.
- **Predict, then compare.** The measurement predicted 2,113 / 694 labelled boxes (day / night) before the code existed; the code gave exactly that. Two independent routes to the same number are worth pinning in a test.
- **Check that stages agree with each other.** The LiDAR depth inside a box should sit near that box's distance label: 91% of boxes agree within −5 to +2 m. For cars the median is 1.8 m *below* the label, about half a car length, because a LiDAR hits the near surface and the label is the centre.
- **A guard no test triggers is a guess.** Coverage flagged the image-size check; a monkeypatched test now proves it fires.
- **Pickling matters for DataLoader.** Workers get a copy of the dataset; the devkit tables are huge for trainval, so they are dropped from the copy.
- **Never mix split lists across data versions.** Two of the eight `mini_train` scenes are official validation scenes; a test records that fact.
- **Conventions differ at library boundaries.** The devkit passes image sizes as `(width, height)`, we use `(height, width)`; the frame rules ask for documented shapes at every boundary.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| Straddler test passed with the rule deleted | The test box was rejected by the "outside the photo" filter instead | Mutation check: remove one rule at a time and see which tests fail |
| `archspec check` warned about helper `_MINI_DEV_VAL` | The checker lists functions as helpers, and this one is a constant | Read the warning; constants don't belong in `helpers` |
| ruff RUF043 / RUF059 in tests | Unescaped regex in `match=`; an unused unpacked variable | `ruff check`; trivial |
| A label ran off the right edge of the picture | Text was placed at the box's left edge | Look at the picture; clamp the text position to the image width |

## Quiz
**Q13:** In the agreement test, cars read about 1.8 m *below* their distance label. Is that a bug? What does it mean for what the model has to learn?
**Your answer:** _open_

**Q14:** A teammate trains on `mini_train` and validates on the trainval `val` split. What goes wrong, and where in the code would you add a check?
**Your answer:** _open_

Earlier questions Q7 to Q10 are also still open.

## Interview angle
"I built the nuScenes data layer as small, tested stages: scene-level splits with a day/night tag, 2D labels derived from 3D boxes under rules I chose after measuring what happens to every box, and a dataset that returns aligned image, sparse LiDAR depth and labels. I cross-checked the stages against each other and verified them with injected bugs and a picture."
