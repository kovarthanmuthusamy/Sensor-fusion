# Evaluation reports

Full tables written by `python -m sensor_fusion.evaluate` on the 6,019 validation frames (150 scenes). Every number has a 95% interval from a scene-level paired bootstrap (1000 resamples). A class gets an AP in a cell only with at least 100 labelled objects (ADR 0011), and the headline mAP is over the classes scored in every cell (car, two-wheeler). "difference" is the second model minus the first.

| file | what it compares | checkpoints |
|---|---|---|
| [baseline-vs-fusion_dense_head-best.md](baseline-vs-fusion_dense_head-best.md) | RGB baseline vs dense input + depth tower (**the final model**) | best epoch by validation mAP@0.5 (baseline: epoch 3, fusion: epoch 5) |
| [baseline-vs-fusion_dense_head-final-epoch.md](baseline-vs-fusion_dense_head-final-epoch.md) | the same pair, sensitivity check | last epoch (8) of both |
| [baseline-vs-fusion_dense-best.md](baseline-vs-fusion_dense-best.md) | RGB baseline vs dense input **without** the tower (ablation) | best epoch by validation mAP@0.5 |
| [first-run-24-epochs-overfit.md](first-run-24-epochs-overfit.md) | the first, failed attempt: sparse depth input, 24 epochs | last epoch; kept to show the problem that was diagnosed (learning log 006 and 008) |
| [head-experiment-decision.json](head-experiment-decision.json) | the pre-registered rule that chose the depth tower | 3-epoch experiment |

Validation split only: the test split was never used. The best epoch of each model was chosen on this same validation split, which makes its numbers slightly optimistic; the final-epoch file shows how much that matters.

The file `first-run-24-epochs-overfit.md` was written before the common-class mAP existed, so it only has the per-cell mAP (not comparable across cells).
