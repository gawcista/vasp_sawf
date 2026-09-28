#!/usr/bin/env python3
"""Extract full space-group and TR representations of the target subspace on the WAVECAR host."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=Path, help='Common interface file prefix without an extension')
    parser.add_argument('--wavecar', required=True, type=Path, help='WAVECAR saved at the end of the same interface calculation')
    parser.add_argument('--outcar', required=True, type=Path, help='OUTCAR from the same interface calculation')
    parser.add_argument('--output', required=True, type=Path, help='New symmetry package output directory')
    args = parser.parse_args()
    from core.symmetry import export_symmetry
    try:
        report = export_symmetry(args.seed, args.wavecar, args.outcar, args.output)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Symmetry extraction failed: {error}', file=sys.stderr)
        return 1
    print(f"Symmetry package: {args.output.resolve()}; status: {report['status']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
