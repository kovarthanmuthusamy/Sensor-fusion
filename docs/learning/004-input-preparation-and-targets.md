# 004 — Input preparation, target grids and the decoder (Day 4–5, PR A)

**PR:** `feat/input-and-targets` · **Date:** 2026-10-27 · **Tags:** mlops, av-robotics, design

## What was built
A system view of PR A, the first of three Day 4–5 PRs ([ADR 0006](../adr/0006-detector-input-and-targets.md)):
- **Input preparation** (`data/transforms.py`): shrinks the photo and depth, crops to a multiple of 32, normalises, picks 3 or 4 channels, and optionally mirrors. The boxes follow every geometric step.
- **Target encoder and decoder** (`data/targets.py`): paint labels onto the network's output grids, and read detections back off them. They are exact inverses.
- **Assembly line** (`data/prepared.py`): chains dataset, preparation and encoder into one training sample, and batches it. Settings are in the shared config group `configs/prep/`.
- 52 new tests (150 in the suite); all new modules at 100% line and branch coverage.

## Decisions made (by you)
- One centre-based network with the distance output from the start; half resolution (800×448); relative (log) distance error → ADR 0006.
- Dropping the optional mid-fusion variant, to protect the timebox (ADR 0001 updated).
- No real GPU job until trainval is on disk.

## Decisions I made inside the plan (veto any)
- **Peak width rule:** σ = 0.54 × size / 6 per axis (TTFNet), with α in config. I first used a remembered formula, found it behaved differently from my memory, and replaced it with one verified in the paper.
- Box sizes are learned in log units; the output grid is 1/4 of the input; the shrink factor must be an integer, which makes the nearest-depth rule exact.
- Preparation settings (including the mirror probability) are one shared config group, so baseline and fusion cannot drift apart.
- If two objects share a centre cell, the nearer one wins.

## Senior instinct
- **Test inverse pairs against each other.** Encode then decode must give the labels back, with no trained network. It found no bug here, but it will be the first thing to fail when someone changes one half.
- **Measure the weakness you accepted.** The design loses objects that share a centre cell; the real-data round trip puts a number on it (2 of 993, 0.2%), and the test pins "every other object survives".
- **Verify a remembered algorithm before building on it.** My recollection of a peak-radius formula would have painted one-cell peaks for typical objects. Working through its numbers exposed it, and the paper confirmed the replacement.
- **Don't average across edges.** Averaging depth across a car's edge invents depths no LiDAR measured; take the nearest real value.
- **Make the fair comparison structural.** One shared preparation config and a test that the baseline and fusion inputs differ only by the depth channel.
- **Inject bugs to test your tests.** 17 realistic bugs (averaged depth, unshifted boxes, a mirror that skips the boxes, a swapped sign, a decoder that forgets the stride) were all caught, each by a focused test.
- **Check that it works through loader workers.** On Windows, workers receive a pickled dataset; a test runs real workers.

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| Test imports vanished (`NameError: Any`) | The format hook runs `ruff --fix`, which deletes an import that is unused at that moment; I added the import before the code that used it | Add imports after the code that uses them, or re-run after the edit |
| Round-trip test failed | Test bug: I sorted only one side before comparing | Compare like with like: sort both sides the same way |
| Peaks for typical boxes came out a single cell wide | A peak-radius formula recalled from memory did not behave as I recalled | Work through numbers for a typical object before trusting a remembered formula |
| ruff E501 (4 times), `type: ignore` noise | Long lines; I kept adding ignores to untyped test helpers | Type the helpers; mypy only checks `src/` |
| W&B login not found from my environment | The login may have been made in a different shell or profile | `wandb.login(prompt=False)` tells whether credentials are visible; re-run `wandb login` in this session |

## Quiz
**Q15:** The depth image is shrunk by taking the nearest valid depth in each block, not the average. What would the network see at the edge of a car if we averaged, and why does that matter?
**Your answer:** _open_

**Q16:** The decoder needs no non-maximum suppression. Why not, and in what case can two real objects still end up as one detection?
**Your answer:** _open_

Q7 to Q10 are still open.

## Interview angle
"I designed the detector's input and target pipeline as small, exactly invertible stages: preparation, painting labels onto output grids, and decoding them back. Because they're inverses I could prove them on real data without a trained network (99.8% of objects round-trip, and the rest are the ones sharing a grid cell), and I verified my tests with injected bugs."
