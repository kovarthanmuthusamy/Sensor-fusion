---
name: mock-interview
description: Run a mock technical interview about this project in the style of a German AV/robotics company (perception / sensor-fusion engineer role). User-triggered at milestones.
disable-model-invocation: true
---

# /mock-interview

## Prepare (silently)
Read `docs/learning/*.md`, `docs/adr/*.md`, `docs/PROGRESS.md` and the README. Only ask about things the project has actually built or decided so far.

## Format (~8 questions, one at a time, wait for each answer)
1. **Project pitch (1):** "Walk me through your project in 2 minutes."
2. **Design decisions (2):** "Why input-level (early) fusion and not late fusion?" or "Why this split?" Probe the trade-offs.
3. **Domain depth (2):** calibration and frames, sensor sync, LiDAR sparsity at range, night-time camera failure modes, metric choice.
4. **Debugging story (1):** "Tell me about a bug you hit." Use the "Errors hit" entries and expect the STAR format.
5. **ML engineering (1):** reproducibility, experiment tracking, fair baselines, fp16 on Volta.
6. **Stretch (1):** "How would you deploy this in a car?" (latency, TensorRT, ROS 2, failure handling).

If an answer is shallow, ask one follow-up "why?" before moving on, as real interviewers do.

## Debrief
- A score per area (1–5) and the 2–3 weakest answers, each with a model answer.
- Append a dated summary to `docs/learning/interviews.md`.
- Add any weak topic as a `- [ ]` item under "Review" in `docs/PROGRESS.md`.

User argument, if given (e.g. `/mock-interview data`), focuses the session on that area.
