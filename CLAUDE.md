# Sensor Fusion — Night-time Camera + LiDAR

Portfolio project (target: AV / robotics employers in Germany). Timebox: 1–2 weeks.

## Scope (decided 2026-10-17)
- **Camera + LiDAR fusion for 2D detection + per-object distance** (ADR 0001): input = RGB + projected sparse LiDAR depth (4 ch, i.e. input-level/early fusion) → one CNN → 2D boxes + distance. A feature-level (mid) fusion variant is a Day 6–7 decision.
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
- **No Claude attribution** (decided 2026-10-27). Commit messages and PR descriptions carry no `Co-Authored-By: Claude` trailer and no "Generated with Claude Code" footer. This overrides the default attribution Claude Code adds.
- **wandb** for experiment tracking (cloud). Query runs with small `wandb.Api()` scripts in `scripts/`, not by hand.
- **Library docs:** read the *installed* version first (`.venv/Scripts/python -c "help(...)"`, `inspect.getsource`, site-packages), then web search. Never guess APIs from memory.
- **Config:** Hydra (`configs/`, ADR 0003). Quality gates: pre-commit (ruff), mypy on `src/`, pytest with coverage. CI = GitHub Actions on Ubuntu with CPU torch.

## Commands
```
.venv/Scripts/python -m pytest --cov                    # tests (+coverage)
.venv/Scripts/python -m mypy                            # type check
.venv/Scripts/pre-commit run --all-files                # lint + format
.venv/Scripts/python -m sensor_fusion.utils.env         # torch/GPU arch check
.venv/Scripts/python -m sensor_fusion.train model=fusion data=mini
.venv/Scripts/pip freeze --exclude-editable             # regenerate requirements.lock.txt (keep its header)
```
- Progress lives in `docs/PROGRESS.md` (the SessionStart hook prints the open TODOs). Path-scoped rules are in `.claude/rules/`. Skills: `/challenge`, `/mock-interview`.

## Working agreement: how Claude works with the user
The user is an ML/robotics MSc student: strong theory, building industry instinct. **The point of the project is the user's learning. Speed comes from agents doing the typing.**

1. **Roles.** Agents write the code. The user reviews every diff and makes the design decisions. Claude never makes a meaningful design decision silently. Present 2–3 options with trade-offs and a recommendation, then let the user choose.
2. **Task flow.** One task = one local feature branch = one PR that the user opens. Claude writes the PR description (what / why / how tested). Keep PRs small and reviewable. CI (GitHub Actions) runs lint and tests.
3. **Learning log.** Every task ends with a note in `docs/learning/NNN-<slug>.md` that follows `docs/learning/_TEMPLATE.md`. Chat replies stay short and the depth goes in the log.
4. **Quiz after each task.** Ask 1–2 questions about the diff or concept just built (non-blocking). Prefer "why" and "what would break if" questions over recall. Record the user's answer and the correction in that task's log. **A squash-merged branch is finished:** never push more commits to it. The user may merge at any moment, so put quiz answers in the first commit of the next task's branch, unless the user says the PR is still open (a leftover commit moves with `git cherry-pick` onto a fresh branch from the updated `main`).
5. **Errors, by severity.**
   - Trivial (typo, import, path): fix silently, one line in the log at most.
   - Classic / instructive (coordinate frames, calibration, timestamp sync, data leakage, train/val split by scene, NaN/AMP overflow, OOM, dataloader bottleneck, eval bugs): **stop and turn it into a challenge**. Show the symptom, give a hint, and let the user diagnose first. Then fix it and write it up as a "Senior instinct" entry.
6. **Decisions.** Non-trivial choices get a short ADR in `docs/adr/NNNN-<slug>.md` (context / options / decision / consequences).
7. **Name the practice.** When using an industry practice (Hydra config, seeding, scene-level split, pre-commit, etc.), name it and give one line on why seniors do it.

<!-- archspec:start (managed by `archview init`; rerun it to update this section) -->
## Architecture spec (architecture.yaml)

`architecture.yaml` is the contract for this project's structure. The user reviews it as a
diagram (`/archspec:view`) before code is written, and may edit it there.

### Workflow
1. **Plan.** Before implementing a feature, add or update its spec entries with
   `status: planned`. Then stop and tell the user it is ready to review in the viewer.
   Do not write code until the user says to go ahead.
2. **Build.** Re-read architecture.yaml first (the user may have edited it) and build
   against it as it is now.
3. **Keep it in step.** When you create or change code that has a node, update that node
   in the same step: `status: built`, `signature`, and `purpose`/`explain` if the logic
   changed. A built node whose spec changed but whose code is not updated yet is `changed`.
4. **Check.** Run the spec check (the `archspec:check` skill, or `python -m archspec check`
   where the package is pip-installed) and fix every error it reports.
5. **Patch, don't rewrite.** Edit only the affected entries; keep comments and unknown fields.

### What gets a node
- A function, class or method gets a node only if it is an entry point, is used from
  another script, or is on the main data path. Everything else goes in its parent's
  `helpers` list. Aim for at most ~6 nodes per script.
- One script = one pipeline stage. Write modular, one-stage-per-script code.
- A method gets its own node (id `Class.method`) only if data flows into or out of it;
  other methods go in the class's `methods` list.
- No variables, no implementation detail.

### Arrows
- `edges` are data flow, not calls. `from`/`to` use `script_id.node_id`.
- `data`: a short plain label of what travels ("raw customer table").
- `shape`: optional, for arrays ("X (N, 32) float32").

### Writing for the reader
The user is strong in math and statistics but not in programming.
- `purpose`: one plain line saying what the step achieves.
- `explain`: 2-4 sentences: what comes in, what is done to it, what goes out, and why the
  next step needs it. Describe the logic in data terms ("one row per customer",
  "rescaled so features are comparable"), never code terms (DataFrame, tensor, dict,
  kwargs, instance). No formulas.

### Format
```yaml
scripts:
  - id: preprocess                 # one script = one stage
    file: src/preprocess.py
    purpose: Turn raw exports into a clean feature table
    nodes:
      - id: build_features
        kind: function             # function | class | method | stage
        status: planned            # planned | built | changed
        purpose: Turn customer records into numbers the model can learn from
        explain: >-
          Takes the customer table and separates the answer column from the rest.
          Text categories become yes/no columns and numbers are rescaled so all
          features are comparable. Out comes one row of numbers per customer.
        signature: "build_features(df: pd.DataFrame) -> pd.DataFrame"  # required once built
        helpers: [_one_hot]
edges:
  - from: preprocess.build_features
    to: train.Trainer.fit
    data: features X and labels y
    shape: "X (N, 32) float32, y (N,)"
```
<!-- archspec:end -->
