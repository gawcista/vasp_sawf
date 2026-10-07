#!/usr/bin/env python3
"""Extract spatial and antiunitary representations of the selected target subspace on the WAVECAR host."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=Path('wannier90'), type=Path, help='Common interface file prefix without an extension')
    parser.add_argument('--wavecar', default=Path('WAVECAR'), type=Path, help='WAVECAR matching the source interfaces; input symbolic links are supported')
    parser.add_argument('--outcar', default=Path('OUTCAR'), type=Path, help='OUTCAR from the same interface calculation')
    parser.add_argument('--output', default=Path('symmetry'), type=Path, help='Symmetry package directory; reuse an existing directory and warn before replacing generated files')
    parser.add_argument('--workers', type=int, default=None, help='Maximum worker processes; automatic from the Slurm CPU allocation, otherwise one')
    parser.add_argument('--memory-gb', type=float, default=None, help='Optional memory budget in GiB; capped by available memory')
    parser.add_argument('--spin-channel', type=int, choices=(1, 2), default=None, help='For non-SOC ISPIN=2: selected up/down channel; inferred from WIN spin or the standard .1/.2 seed suffix')
    parser.add_argument('--tol', type=float, default=1e-5, help='Absolute WAVECAR/WIN lattice tolerance in angstroms; does not change any symmetry or localization tolerance')
    parser.add_argument('--energy-tol', type=float, default=1e-8, help='Absolute WAVECAR/EIG energy tolerance in eV; does not change energy covariance or other symmetry/localization tolerances')
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    cache_root = Path(os.environ.get('XDG_CACHE_HOME', ''))
    if not cache_root.is_absolute():
        cache_root = Path.home() / '.cache'
    cache_root = cache_root / 'vasp_sawf'
    os.environ.setdefault('NUMBA_CACHE_DIR', str(cache_root / 'numba'))
    os.environ.setdefault('MPLCONFIGDIR', str(cache_root / 'matplotlib'))
    sys.dont_write_bytecode = True
    from vasp_sawf.symmetry import export_symmetry
    try:
        report = export_symmetry(args.seed, args.wavecar, args.outcar, args.output,
                                 workers=args.workers, memory_gb=args.memory_gb, spin_channel=args.spin_channel,
                                 tol=args.tol, energy_tol=args.energy_tol)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Symmetry extraction failed: {error}', file=sys.stderr)
        return 1
    print(f"Symmetry package: {args.output.resolve()}; status: {report['status']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
