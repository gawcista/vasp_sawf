#!/usr/bin/env python3
"""Redraw symmetry residual plots from an existing comparison.json.

Edit the plotting parameters below, then run:
    python plot_symmetry.py comparison.json --output figures

This script plots existing values without recomputing symmetry or validating the model.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (11.2, 4.7)
FONT_SIZE = 11
FONT_FAMILY = ["DejaVu Sans"]
COLORS = ("#3677A8", "#D16A35")
BAR_WIDTH = 0.33
BAR_OFFSET = 0.35
YLIMS = ((1e-9, 4e-4), (1e-10, 4e-5))
DPI = 200
FIG_FORMATS = ("png", "pdf")
# Preserve legacy comparison.json operation keys.
OPERATIONS = ("C4z", "C3_111", "\u955c\u9762z", "\u53cd\u6f14", "\u65f6\u95f4\u53cd\u6f14")
OPERATION_LABELS = (r"$C_{4z}$", r"$C_{3,[111]}$", r"$m_z$", r"$P$", r"$\Theta$")
MODEL_LABELS = ("Wannier (aligned)", "SAWF")
TITLES = (
    "Gauge covariance over 216 k points",
    "Nearest-neighbor hopping covariance",
)
YLABELS = ("Maximum gauge residual", "Maximum matrix-element difference (eV)")
SUPTITLE = r"SrVO$_3$: symmetry residuals in the target representation"
FOOTER = (
    "Ordinary Wannier: cell and constant-basis alignment only. "
    "Both models retain time-reversal symmetry."
)


def plot_symmetry(input_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(input_file).read_text(encoding="utf-8"))["metrics"]
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    positions = np.arange(len(OPERATIONS))
    fields = (("selected_operations", "gauge"), ("nearest_hopping_relations", "max_abs_eV"))

    for panel, (axis, (section, field)) in enumerate(zip(axes, fields)):
        for model_index, model in enumerate(("ordinary_aligned", "sawf")):
            values = [metrics[model][section][operation][field] for operation in OPERATIONS]
            axis.bar(
                positions + (model_index - 0.5) * BAR_OFFSET,
                values,
                width=BAR_WIDTH,
                color=COLORS[model_index],
                label=MODEL_LABELS[model_index],
                zorder=3,
            )
        axis.set_yscale("log")
        axis.set_ylim(*YLIMS[panel])
        axis.set_xticks(positions, OPERATION_LABELS)
        axis.set_ylabel(YLABELS[panel])
        axis.set_title(TITLES[panel])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", which="major", alpha=0.2, zorder=0)

    axes[0].legend(loc="upper right", fontsize=10, frameon=False)
    figure.suptitle(SUPTITLE, fontsize=15, y=0.99)
    figure.text(0.5, 0.035, FOOTER, ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.065, 1, 0.95), w_pad=2)
    for extension in FIG_FORMATS:
        output_file = output_dir / f"symmetry_comparison.{extension}"
        figure.savefig(output_file, dpi=DPI)
        print(output_file)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing comparison.json")
    parser.add_argument("--output", type=Path, default=Path("figures"), help="Figure directory")
    args = parser.parse_args()
    plot_symmetry(args.input, args.output)


if __name__ == "__main__":
    main()
