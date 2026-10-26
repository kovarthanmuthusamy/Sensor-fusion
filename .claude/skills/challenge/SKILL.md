---
name: challenge
description: Turn an instructive bug into a guided debugging exercise for the user (symptom → hypothesis → graded hints → reveal → fix → log). Use when a classic ML/AV bug appears (coordinate frames, calibration, timestamp sync, data leakage, NaN/fp16 overflow, OOM, dataloader bottleneck, eval bugs), or when the user runs /challenge to practise.
---

# /challenge: debugging exercise

The user is building senior debugging instinct. **Do not fix the bug first.** Teach diagnosis.

## If invoked with no live bug
Pick a realistic bug from the current codebase area (e.g. a transposed rotation, a frame-level split or a missing `GradScaler`). Describe it as a symptom only. Don't modify the code.

## Steps
1. **Symptom only.** Show what's observed: error text, a wrong plot, a metric that's too good, NaN at step N. Include the minimal repro command. Don't name the cause.
2. **Ask:** "What's your hypothesis, and what's the *first* thing you'd check?" Then wait for the answer.
3. **Graded hints**, one per reply and only if needed:
   - Hint 1: which *area* to look at (file, stage, tensor).
   - Hint 2: which *check* would expose it (print shapes, plot overlay, compare the two timestamps).
   - Hint 3: reveal the root cause.
   Praise a correct diagnostic *process* even when the guess is wrong.
4. **Fix**, as a minimal diff the user reviews. Add a regression test that would have caught it.
5. **Log** a row in the current task's `docs/learning/NNN-*.md` "Errors hit" table (symptom / root cause / how to diagnose next time) and one "Senior instinct" line.

## Tone
Like a patient senior engineer pairing with a junior. Short messages, one question at a time.
