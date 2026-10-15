#!/usr/bin/env python3
"""Localize original SCDM inputs or analyze saved trials/models without WAVECAR."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, help='Original WIN/MMN/AMN/EIG prefix; localization defaults to wannier90, analysis to the saved source')
    parser.add_argument('--symmetry', type=Path, help='Symmetry package directory (localization default: symmetry)')
    parser.add_argument('--center', action='append', nargs=3, type=float, metavar=('X','Y','Z'), help='Fractional representative of a target orbit; repeat with --orbital for independent orbits')
    parser.add_argument('--orbital', action='append', help='Target local representation, such as t2g; one per --center, in matching order')
    parser.add_argument('--center-tol', type=float, help='Automatic center matching tolerance in angstroms (default: 0.01); not a symmetry residual tolerance')
    parser.add_argument('--output', required=True, type=Path, help='New result directory')
    parser.add_argument('--num-iter', type=int, help='Maximum localization iterations (default: 1000)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--trial', action='store_true', help='Use diagnostic.json for an explicitly unaccepted numerical trial; never export model.npz, hr or WANPROJ')
    mode.add_argument('--analyze-trial', type=Path, metavar='DIRECTORY', help='Analyze saved trial.npz and summary.json without repeating SAWF')
    mode.add_argument('--analyze-model', type=Path, metavar='DIRECTORY', help='Analyze saved approximate model.npz and summary.json without repeating either localization')
    parser.add_argument('--points-per-segment', type=int, help='Analysis path sampling per WIN segment (default: 101)')
    parser.add_argument('--dft-eigenval', type=Path, nargs='+', help='Optional DFT path EIGENVAL; analysis also accepts split files in path order')
    parser.add_argument('--energy-reference-ev', type=float, default=0., help='Shared plot energy reference; defaults to absolute energies')
    args = parser.parse_args()
    if args.analyze_trial is not None or args.analyze_model is not None:
        mode_name = '--analyze-model' if args.analyze_model is not None else '--analyze-trial'
        if any(value is not None for value in (args.symmetry, args.center, args.orbital, args.num_iter, args.center_tol)):
            parser.error(f'{mode_name} uses the saved gauge and target; do not provide localization options')
        if args.analyze_model is not None:
            from vasp_sawf.model_analysis import analyze_model as analyze
            source = args.analyze_model
        else:
            from vasp_sawf.analysis import analyze_trial as analyze
            source = args.analyze_trial
        try:
            report = analyze(source, args.output, seed=args.seed,
                points_per_segment=101 if args.points_per_segment is None else args.points_per_segment,
                dft_eigenval=args.dft_eigenval, energy_reference_ev=args.energy_reference_ev)
        except (ValueError, OSError, RuntimeError) as error:
            print(f'Analysis failed: {error}', file=sys.stderr)
            return 1
        if args.analyze_model is not None:
            print(f"Analysis: {args.output.resolve()}; status: {report['status']}; "
                  f"source model: {report['source_model_status']}; localization repeated: {report['localization_repeated']}")
        else:
            print(f"Analysis: {args.output.resolve()}; status: {report['status']}; accepted: {report['accepted']}")
        return 0
    if (args.center is None) != (args.orbital is None):
        parser.error('Provide --center and --orbital together, or omit both for automatic targets')
    args.seed = Path('wannier90') if args.seed is None else args.seed
    args.symmetry = Path('symmetry') if args.symmetry is None else args.symmetry
    if args.points_per_segment is not None:
        parser.error('--points-per-segment requires --analyze-trial or --analyze-model')
    if args.dft_eigenval is not None and len(args.dft_eigenval) != 1:
        parser.error('Split DFT files require --analyze-trial or --analyze-model')
    approximate = False
    try:
        if args.trial:
            from vasp_sawf.trial import run_trial as run_sawf
        else:
            package_report = args.symmetry / 'report.json'
            if package_report.is_file():
                metadata = json.loads(package_report.read_text())
                approximate = (isinstance(metadata, dict)
                               and metadata.get('schema') == 'sawf-bridge-approximate-bloch-v1')
            if approximate:
                from vasp_sawf.approximate import run_approximate as run_sawf
            else:
                from vasp_sawf.localize import run_sawf
        report = run_sawf(args.seed, args.symmetry, args.output,
                          center=args.center, orbital=args.orbital, num_iter=1000 if args.num_iter is None else args.num_iter,
                          center_tol=.01 if args.center_tol is None else args.center_tol,
                          dft_eigenval=args.dft_eigenval[0] if args.dft_eigenval else None,
                          energy_reference_ev=args.energy_reference_ev)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'SAWF failed: {error}', file=sys.stderr)
        return 1
    label = 'Numerical trial (not accepted)' if args.trial else ('Accepted approximate model' if approximate else 'Model')
    print(f"{label}: {args.output.resolve()}; status: {report['status']}; converged: {report['converged']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
