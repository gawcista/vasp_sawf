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
    parser.add_argument('--diagnose-kpoints', nargs='+', type=int, default=None, metavar='K',
                        help='Diagnose selected 1-based stored WAVECAR k points, adding Gamma automatically; write diagnostic.json only, without a SAWF-ready export')
    parser.add_argument('--from-diagnostic', type=Path, metavar='DIRECTORY',
                        help='Reuse a completed, reviewed diagnostic without reading WAVECAR; requires --accept-approximate')
    parser.add_argument('--accept-approximate', action='store_true',
                        help='Accept an approximate model under explicit numerical budgets; strict checks remain unchanged')
    parser.add_argument('--matrix-tol', '--mmn-tol', dest='mmn_tol', type=float,
                        help='Explicit dimensionless entrywise budget for native MMN covariance, action correction and independent sewing agreement')
    parser.add_argument('--hamiltonian-tol', type=float,
                        help='Explicit Hamiltonian symmetry budget in eV; checked again on the continuous interpolation domain')
    args = parser.parse_args()
    has_budgets = args.mmn_tol is not None or args.hamiltonian_tol is not None
    if args.from_diagnostic is not None and not args.accept_approximate:
        parser.error('--from-diagnostic requires --accept-approximate')
    if args.diagnose_kpoints is not None and (args.from_diagnostic is not None or args.accept_approximate):
        parser.error('--diagnose-kpoints cannot be combined with approximate extraction')
    if has_budgets and not args.accept_approximate:
        parser.error('Numerical budgets require --accept-approximate')
    if ((args.accept_approximate and args.from_diagnostic is None) or has_budgets):
        if args.mmn_tol is None or args.hamiltonian_tol is None:
            parser.error('Fresh approximate extraction requires both --matrix-tol and --hamiltonian-tol')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    cache_root = Path(os.environ.get('XDG_CACHE_HOME', ''))
    if not cache_root.is_absolute():
        cache_root = Path.home() / '.cache'
    cache_root = cache_root / 'vasp_sawf'
    os.environ.setdefault('NUMBA_CACHE_DIR', str(cache_root / 'numba'))
    os.environ.setdefault('MPLCONFIGDIR', str(cache_root / 'matplotlib'))
    sys.dont_write_bytecode = True
    try:
        if args.from_diagnostic is not None:
            from vasp_sawf.approximate import export_approximate
            budgets = dict(mmn_tol=args.mmn_tol, hamiltonian_tol=args.hamiltonian_tol) if has_budgets else {}
            report = export_approximate(args.seed, args.from_diagnostic, args.output, **budgets)
        elif args.accept_approximate:
            from vasp_sawf.approximate import extract_fresh_approximate
            report = extract_fresh_approximate(args.seed, args.wavecar, args.outcar, args.output,
                mmn_tol=args.mmn_tol, hamiltonian_tol=args.hamiltonian_tol, workers=args.workers,
                memory_gb=args.memory_gb, spin_channel=args.spin_channel, tol=args.tol, energy_tol=args.energy_tol)
        else:
            from vasp_sawf.symmetry import export_symmetry
            report = export_symmetry(args.seed, args.wavecar, args.outcar, args.output,
                                     workers=args.workers, memory_gb=args.memory_gb, spin_channel=args.spin_channel,
                                     tol=args.tol, energy_tol=args.energy_tol, diagnose_kpoints=args.diagnose_kpoints)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Symmetry extraction failed: {error}', file=sys.stderr)
        return 1
    destination = args.output.resolve()
    label = 'Symmetry package'
    if args.diagnose_kpoints is not None:
        destination = destination / 'diagnostic.json'
        label = 'Diagnostic report'
    print(f"{label}: {destination}; status: {report['status']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
