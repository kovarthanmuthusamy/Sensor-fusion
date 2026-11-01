# Model card: night-time camera + LiDAR detector with distance

A small research model: it finds four kinds of road users in the front-camera image and says how far each one is. It exists to answer one question honestly: *does adding LiDAR depth help a camera detector, especially at night?* It is not a safety system.

## Model details
- **Architecture:** one CNN. ResNet-18 backbone (ImageNet-pretrained), a top-down neck that merges coarse and fine features, and two heads on a grid at 1/4 of the input size. The class head gives a heat map of object centres; the regression head reads box size, centre offset and the log of the distance at each centre.
- **Three variants, identical except for the input:**
  | name | input | extra |
  |---|---|---|
  | `rgb` | photo | the baseline |
  | `fusion_dense` | photo + thickened LiDAR depth + validity mask (5 channels) | the depth has its own first convolution |
  | `fusion_dense_head` (**final**) | as above | a small depth tower gives local LiDAR statistics to both heads |
- **Input:** a 1600×900 front-camera photo; for the fusion models also the raw LiDAR points, `T_cam_from_lidar` and the camera intrinsics. Internally the photo is shrunk to 800×450 and cropped to 800×448.
- **Output:** per object: class (`car`, `large_vehicle`, `pedestrian`, `two_wheeler`), score, a box in photo pixels, and the distance in metres along the camera's viewing direction (not the straight-line range).
- **Training data:** nuScenes v1.0-trainval, official `train` split: 700 scenes, 28,130 front-camera frames (88% day, 12% night), with the top LiDAR scan of the same moment. Boxes are 2D boxes projected from nuScenes' 3D boxes (visibility ≥ 40%).
- **Training:** 8 epochs, AdamW, fp16, seed 42, horizontal-flip augmentation, focal loss for the heat map and L1 on log-sizes and log-distance. The checkpoint is the epoch with the best validation mAP@0.5 (baseline: after its 4th epoch, final model: after its 6th).
- **Software:** PyTorch; served through ONNX Runtime (the exported final model matches PyTorch to better than 1e-5).
- **Licence:** the code is MIT. The weights are derived from nuScenes (CC BY-NC-SA 4.0), so they are for **non-commercial research use only**.

## Intended use
Studying LiDAR–camera fusion, as a worked example of an evaluation that reports uncertainty, and as a template for serving a fusion model. **Out of scope:** any driving, driver-assistance or safety decision; any other camera, sensor set, or country than the data it was trained on.

## Evaluation
Validation split of nuScenes (150 scenes, 6,019 frames; 135 day scenes, 15 night scenes). Brackets are 95% intervals from a bootstrap over scenes (frames of a scene are near-duplicates, so scenes are the independent unit). The test split was never used. mAP is over the classes with at least 100 labelled objects in *every* cell (car and two-wheeler), so day and night are comparable. Full tables: [`docs/results/`](results/README.md).

| | baseline (`rgb`) | `fusion_dense` | `fusion_dense_head` (final) |
|---|---|---|---|
| median distance error, all frames | 8.1% | 6.8% | **4.3%** [4.1, 4.5] |
| median distance error, **night** | 10.9% | 6.9% | **3.6%** [3.1, 4.5] |
| median distance error, 60 m and beyond | 9.1% | 8.3% | **5.0%** |
| mAP@0.5, all frames | 51.2 | 51.9 | 51.0 |
| mAP@0.5, night (Singapore) | 41.4 | **48.7** | 44.0 |
| mAP@[0.5:0.95], all frames | 21.9 | 23.0 | 23.4 |
| mAP@[0.5:0.95], night | 16.4 | 21.3 | 19.9 |

**What the intervals allow us to say** (fusion minus baseline, for the final model unless noted):
- Distance error roughly halves: -3.8 points overall [-4.2, -3.4] and -7.3 points at night [-8.3, -5.4]. This holds at every distance range and with a second checkpoint (-3.2 and -3.9).
- Box precision improves: mAP@[0.5:0.95] +1.5 [0.4, 2.6] overall and +3.5 [1.8, 5.3] at night.
- Finding objects at IoU 0.5 does **not** change overall (-0.2 [-2.4, 1.9]). At night it improves for the dense-input ablation (+7.2 [2.3, 10.7]), but for the final model it is +2.6 [-2.2, 8.0] at the best epoch and +7.1 [2.5, 11.2] at the last, so a night detection gain is likely but sensitive to the checkpoint.
- The baseline's distance error is worse at night than by day (10.9% vs 8.0%); the fusion models' is not.

## Where it fails
Pictures and numbers: [`docs/failure_cases.md`](failure_cases.md).
- **Small, distant objects in the dark.** Two-wheelers at night: AP@0.5 about 12 (baseline) and 15–24 (fusion), from only 114 labelled objects. Pedestrians and large vehicles at night are not scored at all (32 and 46 objects).
- **Duplicate boxes.** The decoder keeps every local peak of the heat map and has no non-maximum suppression, so one large nearby object can get two or three overlapping boxes.
- **Objects with no LiDAR point.** In a sample of 71 training frames, 12% of the labelled objects 40 m or farther had no LiDAR return even after thickening; there the fusion model can only use the photo.
- **Fusion is occasionally worse.** In a few night frames the fusion distance error is slightly higher than the baseline's (for example 3.9% vs 2.5%).

## Limits and caveats
- **Night means one neighbourhood.** All 99 night scenes in nuScenes trainval are in Singapore, and the 15 validation night scenes are in Holland Village, so day vs night is confounded with place. The Singapore day vs night comparison controls the city but not the neighbourhood.
- **One seed per model.** Checkpoint-to-checkpoint changes at night (for example the baseline's night distance error: 10.9% vs 7.6%) show how noisy 15 scenes are.
- **The best epoch was chosen on the validation split** that is also reported, which flatters both models slightly; the final-epoch evaluation shows the effect is small for the main findings.
- **Front camera only; 4 merged classes; 2D boxes.** Not comparable with the nuScenes leaderboard.
- **Serving needs a good calibration.** The service trusts `T_cam_from_lidar`, which must already include the vehicle's motion between the LiDAR scan and the photo. A wrong matrix makes the LiDAR channels misleading; the model was not trained to cope with miscalibration.
- **Only 1600×900 photos** from the same camera model are accepted.

## Serving and speed
`python -m sensor_fusion.serve.app` serves the exported model (photo + LiDAR + calibration in, detections out); a CPU container image is built and smoke-tested in CI. Measured on a shared workstation (36 cores, batch 1; not an embedded device), see [`docs/benchmarks.md`](benchmarks.md):

| | network only (ONNX, CPU) | whole service call on CPU |
|---|---|---|
| `rgb` | about 14 ms | about 59 ms |
| `fusion_dense_head` | about 19 ms | about 112 ms |

Preparing the input (projection, thickening, normalising) is the largest part of a call, not the network.

## How this was made
All code, decisions, mistakes and lessons are in this repository: design decisions in [`docs/adr/`](adr/), a log with the errors hit and what was learned in [`docs/learning/`](learning/). Two decisions in particular shaped the result: a failed first attempt (the LiDAR channel was too sparse to be used, [log 008](learning/008-dense-depth-input.md)) and the choice of checkpoints by mAP instead of validation loss ([log 006](learning/006-training-loop.md)).
