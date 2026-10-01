#!/usr/bin/env python3
"""Extract full space-group and TR representations of the target subspace on the WAVECAR host."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=Path('wannier90'), type=Path, help='Common interface file prefix without an extension')
    parser.add_argument('--wavecar', default=Path('WAVECAR'), type=Path, help='WAVECAR saved at the end of the same interface calculation')
    parser.add_argument('--outcar', default=Path('OUTCAR'), type=Path, help='OUTCAR from the same interface calculation')
    parser.add_argument('--output', required=True, type=Path, help='New symmetry package output directory')
    parser.add_argument('--workers', type=int, default=None, help='Maximum worker processes; automatic from the Slurm CPU allocation, otherwise one')
    parser.add_argument('--memory-gb', type=float, default=None, help='Optional memory budget in GiB; capped by available memory')
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('NUMBA_CACHE_DIR', str(args.output.resolve().parent / '.cache' / 'numba'))
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve().parent / '.cache' / 'matplotlib'))
    sys.dont_write_bytecode = True
    from vasp_sawf.symmetry import export_symmetry
    try:
        report = export_symmetry(args.seed, args.wavecar, args.outcar, args.output,
                                 workers=args.workers, memory_gb=args.memory_gb)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Symmetry extraction failed: {error}', file=sys.stderr)
        return 1
    print(f"Symmetry package: {args.output.resolve()}; status: {report['status']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
