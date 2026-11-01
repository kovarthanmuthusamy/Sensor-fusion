"""Draw the README chart from the evaluation report: distance error by true distance, day vs night.

    python scripts/make_readme_figures.py

Reads `docs/results/baseline-vs-fusion_dense_head-best.json` and writes
`docs/figures/distance_error_by_range.png`. Colours are the first two slots of the reference
palette (blue, orange); two series in a fixed order, one axis, a legend, light grid, no dual axis.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path("docs/results/baseline-vs-fusion_dense_head-best.json")
OUT = Path("docs/figures/distance_error_by_range.png")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {"baseline": ("Camera only", "#2a78d6"), "fusion": ("Camera + LiDAR", "#eb6834")}


def main() -> None:
    cells = json.loads(REPORT.read_text(encoding="utf-8"))["report"]["cells"]
    ranges = cells["day"]["models"]["baseline"]["by_range"]
    labels = [
        f"{r['low']:g}-{r['high']:g} m" if not math.isinf(float(r["high"])) else f"{r['low']:g}+ m"
        for r in ranges
    ]

    night_counts = [r["matched"] for m in SERIES for r in cells["night"]["models"][m]["by_range"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, facecolor=SURFACE)
    width, gap = 0.36, 0.02
    for ax, cell, title in zip(axes, ("day", "night"), ("Day", "Night"), strict=True):
        ax.set_facecolor(SURFACE)
        for offset, (model, (name, colour)) in zip((-1, 1), SERIES.items(), strict=True):
            data = cells[cell]["models"][model]
            values = [100 * r["median_rel_error"] for r in data["by_range"]]
            positions = [i + offset * (width / 2 + gap / 2) for i in range(len(values))]
            ax.bar(
                positions,
                values,
                width=width,
                color=colour,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=name,
            )
        overall = {m: 100 * cells[cell]["models"][m]["point"]["dist_rel_median"] for m in SERIES}
        ax.set_title(
            f"{title}: {overall['baseline']:.1f}% → {overall['fusion']:.1f}% overall",
            loc="left",
            fontsize=11,
            color=INK,
            pad=10,
        )
        ax.set_xticks(range(len(labels)), labels, color=INK_2, fontsize=9)
        ax.tick_params(axis="y", colors=INK_2, labelsize=9, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("median distance error (% of true distance)", color=INK_2, fontsize=9)
    axes[0].set_ylim(0, 14)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.suptitle(
        "Distance error by true distance of the object (lower is better)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.01,
        0.01,
        f"nuScenes validation split, 6,019 frames. Night: 15 scenes, {int(min(night_counts))} to "
        f"{int(max(night_counts))} matched objects per bar.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
