"""Generate the three primary figures for the UniGIR paper draft.

The values in this script are copied from the verified G0 archive and the
paper-table draft. This script only plots archived evaluation values; it does
not load data, checkpoints, or training code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SEEDS = ["42", "7", "123"]
BASELINE_COLOR = "#8C96A0"
UNIGIR_COLOR = "#2F6F9F"
TRAIN_COLOR = "#B07AA1"
FIXED_COLOR = "#D95F02"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax: plt.Axes, bars: object, values: list[float], fmt: str = ".3f") -> None:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        offset = 0.006 if height >= 0 else -0.012
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            format(value, fmt),
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_synergy_auc(output_dir: Path) -> None:
    baseline = [0.763, 0.776, 0.780]
    unigir = [0.790, 0.813, 0.794]
    deltas = [u - b for b, u in zip(baseline, unigir)]

    x = np.arange(len(SEEDS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars_baseline = ax.bar(x - width / 2, baseline, width, label="Baseline", color=BASELINE_COLOR)
    bars_unigir = ax.bar(x + width / 2, unigir, width, label="UniGIR", color=UNIGIR_COLOR)
    add_value_labels(ax, bars_baseline, baseline)
    add_value_labels(ax, bars_unigir, unigir)

    for index, delta in enumerate(deltas):
        ax.text(index, max(baseline[index], unigir[index]) + 0.025, f"Δ {delta:+.3f}", ha="center", fontsize=8)

    ax.set_title("G0 synergy ROC AUC by model seed")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
    ax.set_ylim(0.70, 0.86)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    save_figure(fig, output_dir, "figure1_g0_synergy_auc")


def plot_unique2_fixed_eval(output_dir: Path) -> None:
    baseline = [0.769815, 0.829853, 0.826531]
    unigir = [0.719506, 0.791172, 0.710014]
    deltas = [u - b for b, u in zip(baseline, unigir)]

    x = np.arange(len(SEEDS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars_baseline = ax.bar(x - width / 2, baseline, width, label="Baseline", color=BASELINE_COLOR)
    bars_unigir = ax.bar(x + width / 2, unigir, width, label="UniGIR", color=UNIGIR_COLOR)
    add_value_labels(ax, bars_baseline, baseline, ".3f")
    add_value_labels(ax, bars_unigir, unigir, ".3f")

    for index, delta in enumerate(deltas):
        ax.text(index, max(baseline[index], unigir[index]) + 0.025, f"Δ {delta:+.3f}", ha="center", fontsize=8)

    ax.set_title("G0 unique2 fixed-transform evaluation")
    ax.set_ylabel("acc@1")
    ax.set_xticks(x, [f"seed={seed}" for seed in SEEDS])
    ax.set_ylim(0.65, 0.89)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    save_figure(fig, output_dir, "figure2_g0_unique2_fixed_eval")


def plot_texture_deltas(output_dir: Path) -> None:
    textures = [
        "solid",
        "stripes",
        "grid",
        "hexgrid",
        "dots",
        "noise",
        "triangles",
        "zigzags",
        "rain",
        "pluses",
    ]
    train_end = [0.010, -0.043, -0.075, -0.023, -0.087, -0.010, -0.088, -0.059, -0.058, -0.092]
    fixed_eval = [0.008, -0.082, -0.077, -0.022, -0.065, -0.008, -0.060, -0.056, -0.039, -0.123]

    x = np.arange(len(textures))
    width = 0.37
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bars_train = ax.bar(x - width / 2, train_end, width, label="Training-end class accuracy", color=TRAIN_COLOR)
    bars_fixed = ax.bar(x + width / 2, fixed_eval, width, label="Fixed-transform recall", color=FIXED_COLOR)
    for bar, value in zip(bars_fixed, fixed_eval):
        height = bar.get_height()
        if height >= 0:
            y = height + 0.004
            va = "bottom"
        else:
            y = height - 0.004
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format(value, "+.3f"),
            ha="center",
            va=va,
            fontsize=7,
            rotation=90,
        )

    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("G0 unique2 texture-level changes")
    ax.set_ylabel("UniGIR - Baseline")
    ax.set_xticks(x, textures, rotation=35, ha="right")
    ax.set_ylim(-0.15, 0.035)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, output_dir, "figure3_g0_texture_deltas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ProjectDocs/figures"),
        help="Directory for generated PNG files.",
    )
    args = parser.parse_args()

    configure_plot_style()
    plot_synergy_auc(args.output_dir)
    plot_unique2_fixed_eval(args.output_dir)
    plot_texture_deltas(args.output_dir)
    print(f"Generated 3 figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
