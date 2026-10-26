# Sensor Fusion — Night-time Camera + LiDAR

Portfolio project (target: AV / robotics employers in Germany). Timebox: 1–2 weeks.

## Scope (decided 2026-10-17)
- **Mid-fusion 2D detection + per-object distance**: input = RGB + projected sparse LiDAR depth (4 ch) → one CNN → 2D boxes + distance.
- **Baseline**: identical network, RGB only (3 ch). Only the input changes.
- **Eval**: mAP and distance error, each split by {day, night}. The headline result is the night gap.
- **Data**: nuScenes. `dataset/v1.0-mini` for development; trainval keyframes only (CAM_FRONT + LIDAR_TOP, no sweeps) for real runs. Night split = scene description contains "Night".
- Out of scope unless time is left: BEV/3D detection, ROS 2, TensorRT, synthetic data.

## Environment constraints
- Windows Server, shared lab workstation. 2× Quadro GV100 32GB (Volta): **fp16 AMP only, no bf16**. Always pin a GPU with `CUDA_VISIBLE_DEVICES`.
- Locked-down machine: system conda (`C:\ProgramData\anaconda3`) is **broken** (libmamba DLL access denied), and there's no admin, node or winget. Install everything **per-user**.
- Pure PyTorch / torchvision. **No custom CUDA ops** (mmdet3d, spconv) because they're unreliable on Windows.
- `dataset/` is never committed or modified.

## Tooling (CLI-first, no MCP servers)
- **Plain `venv` + pip on a portable Python 3.11.9** at `C:\Users\muthusamy\.local\python\3.11.9` (the official PSF NuGet zip, because IT policy blocks installers with error 1625; it's not on PATH and `py` doesn't know it). Recreate the venv with `C:/Users/muthusamy/.local/python/3.11.9/python.exe -m venv .venv`. Shell state doesn't persist between agent calls, so always call the venv binaries explicitly: `.venv/Scripts/python -m pytest`, `.venv/Scripts/python -m sensor_fusion...`. Never install into the system Python 3.13 (nuscenes-devkit needs `numpy<2`, which has no 3.13 wheels).
- Dependencies: `pyproject.toml` declares ranges (installed with `pip install -e .[dev]`), and `requirements.lock.txt` (from `pip freeze`) pins exact versions. PyTorch comes from the **cu126** index (it still supports Volta sm_70). Verify that `torch.cuda.get_arch_list()` contains `sm_70`.
- **GitHub belongs to the user.** Claude never pushes, adds remotes or uses `gh` (these are denied in settings). Claude works with *local* git only (branch, add, commit). At the end of each task, Claude gives the user the GitHub steps: the exact push command, a PR title and description ready to paste, what to check in CI, and the merge/cleanup commands.
- **wandb** for experiment tracking (cloud). Query runs with small `wandb.Api()` scripts in `scripts/`, not by hand.
- **Library docs:** read the *installed* version first (`uv run python -c "help(...)"`, `inspect.getsource`, site-packages), then web search. Never guess APIs from memory.
- Progress lives in `docs/PROGRESS.md` (the SessionStart hook prints the open TODOs). Path-scoped rules are in `.claude/rules/`. Skills: `/challenge`, `/mock-interview`.

## Working agreement: how Claude works with the user
The user is an ML/robotics MSc student: strong theory, building industry instinct. **The point of the project is the user's learning. Speed comes from agents doing the typing.**

1. **Roles.** Agents write the code. The user reviews every diff and makes the design decisions. Claude never makes a meaningful design decision silently. Present 2–3 options with trade-offs and a recommendation, then let the user choose.
2. **Task flow.** One task = one local feature branch = one PR that the user opens. Claude writes the PR description (what / why / how tested). Keep PRs small and reviewable. CI (GitHub Actions) runs lint and tests.
3. **Learning log.** Every task ends with a note in `docs/learning/NNN-<slug>.md` that follows `docs/learning/_TEMPLATE.md`. Chat replies stay short and the depth goes in the log.
4. **Quiz after each task.** Ask 1–2 questions about the diff or concept just built (non-blocking). Prefer "why" and "what would break if" questions over recall. Record the user's answer and the correction in that task's log.
5. **Errors, by severity.**
   - Trivial (typo, import, path): fix silently, one line in the log at most.
   - Classic / instructive (coordinate frames, calibration, timestamp sync, data leakage, train/val split by scene, NaN/AMP overflow, OOM, dataloader bottleneck, eval bugs): **stop and turn it into a challenge**. Show the symptom, give a hint, and let the user diagnose first. Then fix it and write it up as a "Senior instinct" entry.
6. **Decisions.** Non-trivial choices get a short ADR in `docs/adr/NNNN-<slug>.md` (context / options / decision / consequences).
7. **Name the practice.** When using an industry practice (Hydra config, seeding, scene-level split, pre-commit, etc.), name it and give one line on why seniors do it.
