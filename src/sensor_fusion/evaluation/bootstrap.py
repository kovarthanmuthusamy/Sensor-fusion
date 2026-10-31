"""Scene-level bootstrap for the metrics of one or more models on the same frames (ADR 0011).

Frames of one scene are near-duplicates, so the unit that is resampled is the scene, never the
frame (resampling frames would pretend there is far more independent data than there is). Every
model is scored on the *same* resampled scenes, so the interval of a difference between two
models (fusion minus baseline) is a paired interval and much tighter than comparing two
independent intervals.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from sensor_fusion.evaluation.metrics import CellData, Metrics, cell_metrics

HEADLINE_METRICS = ("map50", "map", "dist_rel_median", "dist_abs_median")


def metrics_to_dict(metrics: Metrics, class_names: Sequence[str]) -> dict[str, float]:
    """Flat `{name: value}` view: the headline numbers plus `ap50_<class>` / `ap_<class>`."""
    flat = {name: float(getattr(metrics, name)) for name in HEADLINE_METRICS}
    for class_name, ap50, ap in zip(class_names, metrics.ap50, metrics.ap, strict=True):
        flat[f"ap50_{class_name}"] = ap50
        flat[f"ap_{class_name}"] = ap
    return flat


def bootstrap_samples(
    cells: Mapping[str, CellData],
    eligible: Sequence[int],
    class_names: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """`samples[model][metric]` is an array with one value per bootstrap resample.

    All models in `cells` must hold the same scenes (checked): they share the resampling.
    """
    models = list(cells)
    scene_names = cells[models[0]].scene_names
    if any(cells[m].scene_names != scene_names for m in models):
        raise ValueError("models must be evaluated on the same scenes to be bootstrapped together")
    n_scenes = len(scene_names)
    rng = np.random.default_rng(seed)
    runs: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(0, n_scenes, n_scenes), minlength=n_scenes).astype(float)
        for model in models:
            runs[model].append(
                metrics_to_dict(cell_metrics(cells[model], weights, eligible), class_names)
            )
    return {
        model: {name: np.array([run[name] for run in rows]) for name in rows[0]} if rows else {}
        for model, rows in runs.items()
    }


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Central interval of the bootstrap samples (nan-aware); nan, nan if there are none."""
    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(valid, [tail, 100 - tail])
    return float(low), float(high)
