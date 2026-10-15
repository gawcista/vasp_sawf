#!/usr/bin/env python3
"""Redraw symmetry residual plots from an existing comparison.json.

Edit the plotting parameters below, then run:
    sawf-plot-symmetry comparison.json --output figures

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


def _operation_label(operation):
    rotation = np.asarray(operation['rotation'])
    translation = np.asarray(operation['translation'])
    if np.array_equal(rotation, np.eye(3)):
        label = 'E'
    elif np.array_equal(rotation, -np.eye(3)):
        label = 'P'
    else:
        label = f"g_{{{operation['index']}}}"
        for i, axis in enumerate('xyz'):
            diagonal = -np.ones(3)
            diagonal[i] = 1
            if np.array_equal(rotation, np.diag(diagonal)):
                label = f'C_{{2{axis}}}'
    if not np.allclose(translation, 0, rtol=0, atol=1e-12):
        label = r'\{' + label + r'|\mathbf{t}\}'
    if operation['antiunitary']:
        label = r'\Theta' if label == 'E' else label + r'\Theta'
    return '$' + label + '$'


def _plot_model_comparison(data, output_dir):
    operations = data['operations']
    indices = [i for i, op in enumerate(operations)
               if op['antiunitary'] or not np.array_equal(op['rotation'], np.eye(3))
               or not np.allclose(op['translation'], 0, rtol=0, atol=1e-12)]
    if not indices:
        indices = list(range(len(operations)))
    positions = np.arange(len(indices))
    labels = [_operation_label(operations[i]) for i in indices]
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fields = ('gauge_entry_max', 'hamiltonian_spectral_max_ev')
    for panel, (axis, field) in enumerate(zip(axes, fields, strict=True)):
        positive = []
        for j, model in enumerate(('ordinary_aligned', 'sawf')):
            values = np.asarray(data['metrics'][model][field])[indices] * (1000 if panel else 1)
            positive.extend(values[values > 0])
            axis.bar(positions + (j-.5)*BAR_OFFSET, values, width=BAR_WIDTH,
                     color=COLORS[j], label=MODEL_LABELS[j], zorder=3)
        if positive:
            axis.set_yscale('log')
            axis.set_ylim(min(positive)*.2, max(positive)*5)
        axis.set_xticks(positions, labels)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', which='major', alpha=.2, zorder=0)
    axes[0].set_title(f"Gauge covariance over {data['nk']} k points")
    axes[0].set_ylabel('Maximum gauge residual')
    axes[1].set_title('Interpolated Hamiltonian covariance')
    axes[1].set_ylabel('Maximum sampled spectral norm (meV)')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .925),
                  ncol=2, fontsize=10, frameon=False)
    figure.suptitle('Wannier and SAWF symmetry comparison', fontsize=15, y=.99)
    figure.text(.5, .035, 'Ordinary Wannier: column, cell and constant unitary basis alignment.',
                ha='center', fontsize=10)
    figure.tight_layout(rect=(0, .065, 1, .86), w_pad=2)
    for extension in FIG_FORMATS:
        path = output_dir/f'symmetry_comparison.{extension}'
        figure.savefig(path, dpi=DPI)
        print(path)
    plt.close(figure)


def plot_symmetry(input_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(input_file).read_text(encoding="utf-8"))
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "pdf.fonttype": 42,
    })
    if data.get('schema') == 'sawf-model-comparison-v1':
        return _plot_model_comparison(data, output_dir)
    metrics = data['metrics']
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
