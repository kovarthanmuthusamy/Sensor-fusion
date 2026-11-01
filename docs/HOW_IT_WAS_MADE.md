# How this project was made with an AI coding agent

This project was built by one person (an ML and robotics MSc student) working with **Claude Code**, an AI coding agent that runs in the terminal. This page says what that meant in practice: who decided, who typed, what kept the agent honest, and where it went wrong. The aim was a project I could defend line by line in an interview, not a project that merely exists.

## The split of work
| | me | the agent |
|---|---|---|
| **Design decisions** (scope, fusion at the input, what to cut, how to evaluate) | decides, from 2–3 options with trade-offs and a recommendation | proposes options, never decides silently |
| **Code** (data pipeline, network, training, evaluation, service) | reviews every diff | writes it and the tests |
| **Git** (branches, commits, pushes, PRs) | does all of it | never runs git; says when a commit is needed and writes the message and PR text |
| **Results** | reads the numbers, decides what is claimed | runs the experiments and reports intervals, not single numbers |
| **Learning** | answers quiz questions, diagnoses bugs first | writes a log of every task and quizzes me afterwards |

Speed came from the agent doing the typing. Understanding had to come from me, so the setup was built to force it.

## How the agent was set up
- **A written working agreement** ([`CLAUDE.md`](../CLAUDE.md)): the scope, the roles above, the rules for git, and the environment limits of a locked-down shared lab machine (no admin rights, Volta GPUs with fp16 only, no custom CUDA code). It is loaded at the start of every session, so the agent does not need re-briefing.
- **Rules scoped to topics** ([`.claude/rules/`](../.claude/rules/)): *coordinate frames* (name every transform by its frames, `T_cam_from_lidar`; filter before projecting; a visual overlay check for any projection change) and *ML hygiene* (split by scene, never by frame; keep the test split sealed; same schedule and seed for the baseline and the fusion model; overfit one batch first).
- **Hooks** ([`.claude/hooks/`](../.claude/hooks/)): the code formatter runs after every file write, and each session starts by printing the open to-dos from the progress file.
- **Two custom skills**: `/challenge` turns a real bug into a debugging exercise (symptom first, graded hints, then the fix), and `/mock-interview` interviews me on what the project has actually built.
- **Persistent memory**: preferences that outlive a session, such as "the agent never touches git".
- **Paper trail by default**: every non-trivial choice has an ADR ([`docs/adr/`](adr/), 14 of them), and every task ends with a learning log ([`docs/learning/`](learning/), 12) of what was built, which errors were hit and a quiz.

## What went right
- **A negative result was taken seriously.** The first LiDAR model was no better than the camera-only model. Instead of tuning until it looked better, the agent was asked to measure the input: 99% of the LiDAR depth image was empty. The fix (thicken the points, add a validity mask, give depth its own entry path into the network) was chosen with an experiment whose pass/fail rule was written down first ([log 008](learning/008-dense-depth-input.md), [ADR 0013](adr/0013-dense-depth-and-retrain-protocol.md)).
- **The evaluation was checked by something other than the code that wrote it.** The mAP code is my own, so it was compared with `pycocotools` on the real detections: all four classes match to three decimals. The distance error was recomputed by independent code and matched (8.09%).
- **Tests were tested.** For the service, five bugs were injected on purpose (blocking the event loop, no load shedding, leaking exception text, no early size check, unbounded metric labels). Four were caught at once. One slipped through, because two checks produced almost the same error, so a new test now exercises the case only the early check can answer.
- **Performance was measured, not assumed.** Profiling found that PyTorch's default of 36 threads made the service three times slower on the lab machine; the service now sets 4.
- **Statistics matched the data.** Frames of one scene are near-duplicates, so every interval is a bootstrap over scenes, and the results say plainly which differences are established and which are only likely.

## What went wrong, and what it changed
| What happened | Lesson now written into the process |
|---|---|
| An API key appeared in the agent's tool output after a tooling warning. | Rotate it at once; never paste secrets into a chat; keep credentials out of anything the agent reads. |
| After a merge on the remote, eight production files silently went back to old versions; CI caught it as a missing dependency. | Read the diff stat of every merge, and do not merge `main` into a branch whose history was squash-merged. |
| A long-running training job was killed twice by a shared-machine memory guard. | Checkpoint every epoch and make resuming a tested path, not an afterthought. |
| A test generator produced a box COCO ignores, so my AP and the reference disagreed in the 3rd decimal. | The generator was wrong, not the metric; test data must be able to occur in real output. |

## Why work this way
- It makes the agent a **collaborator that can be audited**: decisions are in ADRs, behaviour is in tests, mistakes are in a log.
- It keeps the **human responsible for the claims**. The agent is fast and fluent, which makes an unchecked agent dangerous: it will write plausible code for a wrong idea. The independent checks above exist because of that.
- It is the way a small team would use such a tool: written conventions, automated gates (formatter, ruff, mypy, 372 tests, CI), human review of every change, and no silent decisions.

## Numbers
14 decision records · 12 learning logs · 372 tests at 95% coverage · CI on every pull request (lint, types, tests, container build and smoke test) · every experiment configured through Hydra and logged to Weights & Biases with its seed and config.
