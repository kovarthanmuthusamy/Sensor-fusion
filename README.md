# Night-time Camera + LiDAR Fusion

Cameras struggle at night. Does adding LiDAR depth close the day–night gap? This project trains the **same** 2D detector with a per-object distance head twice on [nuScenes](https://www.nuscenes.org/): once on RGB only, and once on RGB plus projected LiDAR depth. It then compares the two on day vs night scenes.

> **Status:** Day 1 of ~10. Repo skeleton is in place. See [progress](docs/PROGRESS.md).

## Approach
```
Camera (CAM_FRONT) ─────────────────────────┐
                                            ├─► [R,G,B,D] ─► CNN ─► 2D boxes + distance
LiDAR (LIDAR_TOP) ─► project into camera ───┘   (sparse depth image)

Baseline: the same network on [R,G,B].   Metrics: mAP and distance error, split by day / night.
```
Design decisions are recorded in [`docs/adr/`](docs/adr/).

## Setup
Requires Python 3.11 and an NVIDIA GPU. Developed on Quadro GV100s (Volta), which is why PyTorch comes from the CUDA 12.6 index.
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
pre-commit install
python -m sensor_fusion.utils.env   # is your GPU supported by the installed torch build?
```
Exact versions for the dev machine (Windows, CUDA 12.6) are in [`requirements.lock.txt`](requirements.lock.txt).

## Data
nuScenes requires free registration at nuscenes.org, and is not included in this repo. By default the code expects
`dataset/v1.0-mini/` (development) and `dataset/v1.0-trainval/` (keyframes, training). Stored elsewhere? Pass `data.root=<path>`.

## Usage
```bash
python -m sensor_fusion.train                       # RGB baseline on mini
python -m sensor_fusion.train model=fusion          # RGB + LiDAR depth
python -m sensor_fusion.train data=trainval seed=7  # any config value can be overridden
```
Every run saves its fully resolved config to `outputs/<date>/<time>/.hydra/`.

## Development
```bash
pre-commit run --all-files   # lint + format (also runs on every commit)
mypy                         # type check
pytest --cov                 # tests; GPU tests skip themselves without CUDA
```

## How this repo is built
Code is written with AI coding agents (Claude Code) under a review-every-diff workflow: see [`CLAUDE.md`](CLAUDE.md). Design decisions, bugs and lessons are logged in [`docs/learning/`](docs/learning/).

## License
Code: [MIT](LICENSE). nuScenes is licensed separately (CC BY-NC-SA 4.0, non-commercial).
