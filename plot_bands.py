#!/usr/bin/env python3
"""Plot DFT, ordinary Wannier, and SAWF bands separately from bands.npz.

Edit the parameters below, then run:
    python plot_bands.py bands.npz --output figures

Energy arrays are in eV before reference subtraction; segment_slices use half-open intervals.
This script only plots data; it neither recomputes nor validates the model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (4, 4)
FONT_SIZE = 18
XTICK_FONT_SIZE = 14
YLABEL_FONT_SIZE = 24
FONT_FAMILY = ["Arial", "Liberation Sans", "DejaVu Sans"]
BORDER_WIDTH = 3
TICK_PAD = 8
TITLE_PAD = 15
LINEWIDTH = 0.5
YLIM = (-1.2, 1.8)
COLORS = {"dft": "tab:blue", "wannier": "tab:green", "sawf": "tab:orange"}
TITLES = {"dft": "DFT", "wannier": "Wannier", "sawf": "SAWF"}
FIG_FORMATS = ("png", "pdf")
DPI = 600
TRANSPARENT = True


def plot_bands(input_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": FONT_SIZE, "font.family": FONT_FAMILY,
        "xtick.major.pad": TICK_PAD, "ytick.major.pad": TICK_PAD,
        "figure.autolayout": True, "pdf.fonttype": 42,
    })

    with np.load(input_file, allow_pickle=False) as bands:
        distance = bands["distance"]
        segments = bands["segment_slices"]
        ticks = bands["tick_positions"]
        labels = bands["tick_labels"].tolist()
        energy_reference = float(bands["energy_reference_ev"])

        for name in ("dft", "wannier", "sawf"):
            if name not in bands:
                continue
            energies = bands[name] - energy_reference
            figure, axis = plt.subplots(figsize=FIGSIZE)
            for start, stop in segments:
                axis.plot(
                    distance[start:stop], energies[start:stop],
                    color=COLORS[name], linewidth=LINEWIDTH,
                )

            if TITLES[name]:
                axis.set_title(TITLES[name], pad=TITLE_PAD)
            axis.set_ylabel("Energy (eV)", fontsize=YLABEL_FONT_SIZE)
            axis.set_xticks(ticks, labels)
            axis.set_xlim(distance[0], distance[-1])
            axis.set_ylim(*YLIM)
            for spine in axis.spines.values():
                spine.set_linewidth(BORDER_WIDTH)
            axis.tick_params(direction="in", width=BORDER_WIDTH)
            axis.tick_params(axis="x", labelsize=XTICK_FONT_SIZE)
            axis.axhline(0, color="gray", linewidth=1, linestyle="--", zorder=0)
            for position in ticks:
                if distance[0] < position < distance[-1]:
                    axis.axvline(position, color="gray", linewidth=1,
                                 linestyle="--", zorder=0)

            figure.tight_layout()

            for extension in FIG_FORMATS:
                output_file = output_dir / f"band_{name}.{extension}"
                figure.savefig(output_file, dpi=DPI, transparent=TRANSPARENT)
                print(output_file)
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing bands.npz")
    parser.add_argument("--output", type=Path, default=Path("figures"), help="Figure directory")
    args = parser.parse_args()
    plot_bands(args.input, args.output)


if __name__ == "__main__":
    main()
