---
paths:
  - "src/**/data/**"
  - "src/**/geometry/**"
  - "src/**/calib*"
  - "tests/**/test_*data*"
  - "tests/**/test_*geometry*"
---

# Coordinate frames & calibration rules

Most fusion bugs are frame bugs. Every agent touching this code follows these rules:

- **Name transforms by their frames:** `T_cam_from_lidar` maps points *from* lidar *into* cam, so `p_cam = T_cam_from_lidar @ p_lidar`. Never use bare names like `T`, `R` or `extrinsic`.
- **nuScenes chain:** lidar → ego(t_lidar) → global → ego(t_cam) → camera. The two ego poses differ because the sensors fire at different timestamps. Don't collapse them into one ego frame without a comment saying why.
- **nuScenes quaternions are `[w, x, y, z]`.** Use `pyquaternion` or convert explicitly. Never pass them straight to scipy, which expects `[x, y, z, w]`.
- **Units in names or docstrings:** metres, pixels, radians. Depth means the camera z-axis (forward), **not** Euclidean range. Say which one.
- **Assert shapes at function boundaries:** points `(N, 3)` or `(3, N)` must be documented and asserted. A silent transpose is a classic bug.
- **Filter before projecting:** drop points with `z_cam <= ~0.1 m` before dividing by z, then drop pixels outside the image bounds.
- **4×4 homogeneous matrices** for rigid transforms. Build them with a helper and never hand-assemble them inline.
- **Every new geometry function gets a test** with a known hand-computed point (e.g. a point 10 m straight ahead projects to the principal point).
- **Visual check:** any change to projection code must be verified with a LiDAR-on-image overlay saved to `outputs/debug/`. Mention it in the PR.
