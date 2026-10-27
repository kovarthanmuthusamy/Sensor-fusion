# 002 — LiDAR → camera projection and the sparse depth image

**PR:** `feat/lidar-projection` · **Date:** 2026-10-27 · **Tags:** av-robotics, code-craft

## What was built
- `src/sensor_fusion/data/projection.py`: `build_lidar_to_camera` (the 4-hop chain as one 4×4 matrix `T_cam_from_lidar`), `project_to_image` (pinhole step, with the filters), `make_sparse_depth` (depth image, nearest point wins).
- `tests/test_projection.py`: 25 tests. Hand-computed points (a point 10 m ahead hits the principal point), error cases, a float32-rounding regression, and a comparison with the devkit's `map_pointcloud_to_image` on the first frame of all 10 mini scenes. That last test is skipped when `dataset/` is absent, so **CI cannot run it**; it only protects local runs.
- The three nodes in `architecture.yaml` are now `built`.
- Visual check (required by `.claude/rules/coordinate-frames.md`): LiDAR-on-photo overlays for a day frame (`scene-0061`) and a night frame (`scene-1077`), saved under `outputs/debug/` (not committed).

## Decisions made (by you)
- Own pure projection function, with the devkit as a test oracle → [ADR 0004](../adr/0004-data-pipeline-choices.md), section 1.

## Decisions I made inside that plan (veto any of them)
- Quaternions are converted explicitly in numpy, so `src/` has no new `pyquaternion` dependency. `pyquaternion` is used only as an oracle in a test.
- Image sizes are `(height, width)`, matching array shapes; a pixel is `(u, v)` = (column, row).
- `min_depth` defaults to 1.0 m, the same as the devkit.
- Pixel → depth-image cell uses `floor`, and the pixel must lie in `[0, size)`.

## Senior instinct
- **Test the strength of your tests, not only that they pass.** I injected six realistic bugs (collapsed ego poses, `[x, y, z, w]` quaternions, un-inverted camera calibration, range instead of depth, no behind-camera filter, plain assignment instead of nearest-wins). All six were caught. The real-data comparison catches the collapsed-ego-pose bug because its tolerance (0.05 px) is tighter than the effect (about 0.8 px for the traced point).
- **Test the values you return, not the values you computed.** Filtering in float64 and returning float32 lets a pixel at 1599.999999999 round up to 1600.0, which is outside a 1600-column image. A boundary regression test pins this.
- **Don't rely on unspecified order.** NumPy gives no guarantee which duplicate wins in `image[rows, cols] = values`. `np.minimum.at` is unbuffered and states what it does.
- **Hand-computed tests and an oracle test do different jobs.** Hand-computed ones run everywhere and say which hop is wrong. The oracle covers real calibrations but needs the dataset.
- **Look at it.** The overlay shows whether points sit on the objects: the day minivan, and at night a dark car and bushes the camera barely shows, but the LiDAR does.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| ruff E501 on a docstring line | Line longer than 100 characters | `ruff check`; the format hook can't fix long strings |
| Coverage 87% on the new module | Five shape guards that no test triggered | `pytest --cov-report=term-missing`; add a test per guard |
| Spec text said motion "shifts points sideways" | I described the effect without measuring it | Trace one real point with numbers (here: under 1 px for a far point, depth changes with the car's travel) |

## Quiz
**Q11:** The devkit comparison already passes on real data. Why did I also write hand-computed tests such as the point 10 m ahead?
**Your answer:** _open_

**Q12:** What would break if `make_sparse_depth` used `image[rows, cols] = depths` and two points hit the same pixel?
**Your answer:** _open_

Earlier questions Q7 to Q10 are also still open; answer them in chat and I'll record them in the logs.

## Interview angle
"I wrote the LiDAR-to-camera projection as a pure function, with each hop of the nuScenes frame chain explicit. I tested it against the devkit on real data, against hand-computed points, and with injected bugs to prove the tests would notice. I also checked by eye with an overlay on day and night frames."
