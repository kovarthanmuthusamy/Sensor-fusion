# Documentation index

## Start here
| document | for |
|---|---|
| [`../README.md`](../README.md) | what the project is, the result, and how to try it |
| [`MODEL_CARD.md`](MODEL_CARD.md) | what the model does, its measured behaviour and its limits |
| [`RUNBOOK.md`](RUNBOOK.md) | running and operating the service: settings, endpoints, metrics, problems and fixes |
| [`../SECURITY.md`](../SECURITY.md) | what the service assumes and protects against; how to report a problem |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | how to change the project: workflow, checks, standards |
| [`../CHANGELOG.md`](../CHANGELOG.md) | what changed in each version |

## Evidence
| document | contents |
|---|---|
| [`results/`](results/README.md) | the full evaluation reports (baseline vs fusion, the ablation, a second checkpoint) |
| [`benchmarks.md`](benchmarks.md) | latency of each way of running the model |
| [`load_test.md`](load_test.md) | throughput and latency of the service under concurrent load |
| [`failure_cases.md`](failure_cases.md) | night frames where LiDAR helps, where both models miss, where fusion is worse |
| [`object_counts.md`](object_counts.md) | labelled objects per split, city, time of day and class |

## Decisions (ADRs)
| | decision |
|---|---|
| [0001](adr/0001-camera-lidar-fusion-scope.md) | camera + LiDAR fusion for 2D detection and distance |
| [0002](adr/0002-python-environment.md), [0003](adr/0003-hydra-config.md) | Python environment, Hydra configuration |
| [0004](adr/0004-data-pipeline-choices.md), [0005](adr/0005-label-rules-and-dataset-output.md) | data pipeline, label rules |
| [0006](adr/0006-detector-input-and-targets.md), [0009](adr/0009-network-and-loss.md) | detector input and targets, network and loss |
| [0007](adr/0007-serving-and-demo.md), [0014](adr/0014-production-hardening.md) | serving and demo, making the service operable |
| [0008](adr/0008-location-confound-in-day-night.md), [0011](adr/0011-evaluation-reporting-rules.md), [0012](adr/0012-evaluation-protocol.md) | night is confounded with location, reporting rules, evaluation protocol |
| [0010](adr/0010-training-setup.md), [0013](adr/0013-dense-depth-and-retrain-protocol.md) | training setup, dense depth input and the retrain |

## Learning log
What each task built, which errors were hit, and what each taught: [`learning/`](learning/) (000 environment, 001 repo and workflow, 002 projection, 003 splits and labels, 004 input and targets, 005 network and loss, 006 training loop, 007 evaluation, 008 dense depth, 009 serving, 010 retrain and results, 011 production hardening).

[`PROGRESS.md`](PROGRESS.md) tracks the plan.
