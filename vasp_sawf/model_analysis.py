"""Analyze saved approximate models without repeating either localization."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from .analysis import _adjoint, _analyze, _covariance, _digest, _json
from .bands import evaluate_bands


def _saved_system(vectors, matrices):
    return SimpleNamespace(num_wann=matrices.shape[1], rvec=SimpleNamespace(iRvec=vectors),
                           get_R_mat=lambda name: matrices)


def _check_saved_system(model, system, centers, spreads, prefix, report):
    from .localize import _gate

    vectors, matrices = model[prefix+'R'], model[prefix+'H_R']
    if not np.array_equal(vectors, system.rvec.iRvec):
        raise ValueError(f'Saved {prefix}R disagrees with the reconstructed Fourier vectors')
    expected = system.get_R_mat('Ham')
    if matrices.shape != expected.shape or not np.isfinite(matrices).all():
        raise ValueError(f'Invalid saved {prefix}H_R')
    tolerance = 256*np.finfo(float).eps*max(1, len(vectors))*max(1., float(abs(expected).max()))
    _gate(report, prefix+'H_R_recompute_max_ev', abs(matrices-expected).max(), tolerance)
    for suffix, values in (('centers_angstrom', centers), ('spreads_angstrom2', spreads)):
        saved = model[prefix+suffix]
        if saved.shape != values.shape or not np.isfinite(saved).all():
            raise ValueError(f'Invalid saved {prefix+suffix}')
        _gate(report, prefix+suffix+'_recompute_max', abs(saved-values).max())
    return _saved_system(vectors, matrices)


def _aligned_fourier(system, order, shifts, q):
    """Apply column, integer-cell and constant-unitary changes to the existing Fourier series."""
    hr = system.get_R_mat('Ham')[:, order][:, :, order]
    coefficients = {}
    nw = system.num_wann
    for r, matrix in zip(system.rvec.iRvec, hr, strict=True):
        for m in range(nw):
            for n in range(nw):
                vector = tuple(r+shifts[n]-shifts[m])
                coefficients.setdefault(vector, np.zeros((nw, nw), complex))[m, n] += matrix[m, n]
    vectors = np.array(list(coefficients), dtype=int)
    matrices = _adjoint(q) @ np.array(list(coefficients.values())) @ q
    return _saved_system(vectors, matrices)


def _comparison(bundle, saved, ordinary, ordinary_centers, ordinary_system, system, group, sym, target, sets):
    from .localize import _gate, align_scdm_initial_gauge

    trial_ledger = saved['trial_summary']['initial_alignment']
    target_centers = np.asarray(saved['trial_summary']['target']['column_centers_fractional']) @ bundle.lattice
    _, alignment = align_scdm_initial_gauge(ordinary, ordinary, ordinary_centers, bundle.lattice,
        bundle.kpoints, sym, target_centers=target_centers)
    order = np.asarray(alignment['permutation_source_columns'])
    shifts = np.asarray(alignment['cell_shifts'])
    q = np.asarray(alignment['Q_real']) + 1j*np.asarray(alignment['Q_imag'])
    phase = np.exp(2j*np.pi*bundle.kpoints @ shifts.T)
    aligned = (ordinary[:, :, order]*phase[:, None, :]) @ q
    recovered = ((aligned @ _adjoint(q))*phase.conj()[:, None, :])[:, :, np.argsort(order)]
    _gate(alignment, 'saved_ordinary_inverse_transform_max', abs(recovered-ordinary).max(),
          256*np.finfo(float).eps*ordinary.shape[-1])
    alignment.update(ordinary_provenance='Saved ordinary_U and centers; no localization repeated',
        formula='ordinary_U(k) P exp(+2*pi*i*k.cell_shifts) Q',
        trial_ledger_centers_difference_max_angstrom=float(abs(
            np.asarray(trial_ledger['ordinary_centers_angstrom'])-ordinary_centers).max()),
        trial_ledger_reused=False,
        role='Recomputed reversible comparison basis from saved ordinary gauge and centers; no symmetry averaging')
    aligned_system = _aligned_fourier(ordinary_system, order, shifts, q)
    points = np.concatenate([points for name, points in sets.items() if name != 'mesh'])
    scale = 256*np.finfo(float).eps*max(len(aligned_system.rvec.iRvec), len(ordinary_system.rvec.iRvec))*max(
        1., float(abs(ordinary_system.get_R_mat('Ham')).max()))
    _gate(alignment, 'offgrid_spectrum_preservation_max_ev', abs(
        evaluate_bands(aligned_system, points)-evaluate_bands(ordinary_system, points)).max(), scale)
    result = dict(schema='sawf-model-comparison-v1', nk=len(bundle.kpoints),
        operations=[dict(index=g, antiunitary=bool(op.time_reversal), rotation=op.rotation.tolist(),
                         translation=op.translation.tolist()) for g, op in enumerate(group.symmetries)],
        sampling_scope='Gauge: every original mesh point. Hamiltonian and spectrum: finite WIN path, grid and random samples; no supremum claim.',
        sampling_counts={name: len(points) for name, points in sets.items()}, alignment=alignment,
        gauge_definition='max abs(B_g C_g(U(k))-U(gk) D_g(k)); reconstructed Bloch action and the same target representation',
        hamiltonian_definition='spectral norm of H(gk)-D_g(k) C_g(H(k)) D_g(k)^dagger; original Fourier series with reversible ordinary basis alignment',
        metrics={}, per_sampling_set={})
    for label, gauge, model in (('ordinary_aligned', aligned, aligned_system), ('sawf', saved['U'], system)):
        per_set = {name: _covariance(model, points, group.symmetries, target)
                   for name, points in sets.items() if name != 'mesh'}
        result['per_sampling_set'][label] = per_set
        gauge_error = [float(abs(saved['reconstructed_d'][g] @ (gauge.conj() if anti else gauge)
            - gauge[saved['kmap'][g]] @ saved['target_D'][g]).max()) for g, anti in enumerate(saved['anti'])]
        result['metrics'][label] = dict(gauge_entry_max=gauge_error,
            hamiltonian_spectral_max_ev=[max(rows[g]['spectral_max_ev'] for rows in per_set.values())
                                        for g in range(group.size)],
            spectrum_difference_max_ev=[max(rows[g]['spectrum_difference_max_ev'] for rows in per_set.values())
                                        for g in range(group.size)])
    return result


def _read_model(folder):
    from .approximate import _json as parse_json

    report = _json(folder/'summary.json')
    flags = dict(accepted=True, sawf_ready=True, converged=True, numerical_converged=True,
                 strict_ready=False, strict_checks_passed=False, source_full_closure_certified=False,
                 symmetrize_hr=False)
    if (report.get('schema') != 'sawf-bridge-approximate-model-v1'
            or report.get('status') != 'accepted_approximate'
            or any(report.get(key) is not value for key, value in flags.items())):
        raise ValueError('Analysis requires an accepted approximate saved model with unchanged strict limitations')
    if _digest(folder/'model.npz') != report.get('output_sha256', {}).get('model.npz'):
        raise ValueError('Saved model hash disagrees with its summary')
    with np.load(folder/'model.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    for key, value in arrays.items():
        if value.dtype.hasobject or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
            raise ValueError(f'Invalid saved model array: {key}')
    for key in ('trial_archive', 'trial_summary_utf8', 'diagnostic_utf8'):
        value = arrays.get(key)
        if value is None or value.dtype != np.uint8 or value.ndim != 1 or not len(value):
            raise ValueError(f'Model lacks exact embedded evidence: {key}')
    trial = parse_json(arrays['trial_summary_utf8'].tobytes())
    if hashlib.sha256(arrays['trial_archive'].tobytes()).hexdigest() != trial.get('trial_sha256'):
        raise ValueError('Embedded trial hash disagrees with its summary')
    diagnostic = arrays['diagnostic_utf8'].tobytes()
    if (hashlib.sha256(diagnostic).hexdigest() != report.get('source_diagnostic_sha256')
            or report.get('source_diagnostic_sha256') != trial.get('source_diagnostic_sha256')
            or parse_json(diagnostic) != trial.get('source_diagnostic')):
        raise ValueError('Embedded source diagnostic disagrees with the saved model')
    decision = report.get('acceptance', {})
    if (report.get('source_hashes') != trial.get('source_hashes')
            or decision.get('source_hashes') != trial.get('source_hashes')
            or decision.get('diagnostic_sha256') != trial.get('source_diagnostic_sha256')
            or report.get('target') != trial.get('target')
            or decision.get('hamiltonian_budget_ev') != report.get('hamiltonian_budget_ev')):
        raise ValueError('Model acceptance, target or source binding disagrees with embedded evidence')
    arrays['trial_summary'] = trial
    return report, arrays


def _revalidate_acceptance(seed, source, arrays, folder):
    from .approximate import EXPLICIT_POLICY, _budgets, _check_target, _reconstruct
    from .localize import _gate

    recorded = source['acceptance']
    explicit = recorded.get('policy') == EXPLICIT_POLICY
    budgets = _budgets(recorded.get('matrix_tolerance'), recorded.get('hamiltonian_budget_ev')) if explicit else None
    (folder/'diagnostic.json').write_bytes(arrays['diagnostic_utf8'].tobytes())
    details, rebuilt, decision = _reconstruct(seed, folder, budgets=budgets)
    if recorded != decision:
        raise ValueError('Saved model acceptance policy disagrees with recomputed source evidence')
    for key in ('source_hashes', 'source_num_bands', 'shape_nk_nb_nw', 'mesh', 'bands_vasp_1based', 'spin'):
        if source.get(key) != details[key]:
            raise ValueError(f'Saved model source metadata disagrees with recomputed evidence: {key}')
    checks = {}
    for key, expected in rebuilt.items():
        actual = arrays.get(key)
        if actual is None or actual.shape != expected.shape:
            raise ValueError(f'Saved model array dimensions disagree with source evidence: {key}')
        if key in ('raw_d', 'reconstructed_d'):
            _gate(checks, key+'_recomputed', abs(actual-expected).max())
        elif not np.array_equal(actual, expected):
            raise ValueError(f'Saved model array disagrees with source evidence: {key}')
    if not explicit:
        orbits = source['target']['orbits']
        _check_target([row['position_fractional'] for row in orbits],
                      [row['orbital'] for row in orbits], decision)
    details['saved_model_array_checks'] = checks
    return details


def analyze_model(model_dir, output_dir, *, seed=None, points_per_segment=101,
                  energy_reference_ev=0., dft_eigenval=None):
    """Revalidate and analyze saved gauges, preserving the source approximation scope."""
    from .approximate import _check_budget
    from .localize import _new_output

    if type(points_per_segment) is not int or points_per_segment < 2 or not np.isfinite(energy_reference_ev):
        raise ValueError('Provide at least two points per segment and a finite energy reference')
    folder = Path(model_dir).resolve()
    source, arrays = _read_model(folder)
    seed = Path(source['source_seed'] if seed is None else seed).resolve()
    dft_files = [] if dft_eigenval is None else ([dft_eigenval] if isinstance(dft_eigenval, (str, Path)) else list(dft_eigenval))
    dft_files = [Path(path).resolve() for path in dft_files]
    inputs = [folder/'summary.json', folder/'model.npz',
              *[Path(f'{seed}.{ext}') for ext in ('win', 'amn', 'mmn', 'eig')], *dft_files]
    output = _new_output(output_dir, inputs)
    report = dict(schema='sawf-bridge-model-analysis-v1', status='not_ready', accepted=False, sawf_ready=False,
        strict_ready=False, source_full_closure_certified=False, symmetrize_hr=False, wavefunction_bytes_read=0,
        localization_repeated=False, source_seed=str(seed), source_model=str(folder),
        source_model_status=source['status'], source_model_accepted=source['accepted'],
        source_acceptance=source['acceptance'], physical_acceptance_status=source['physical_acceptance_status'],
        analysis_role='Postprocessing of an existing accepted approximate model; no new model or source certification')
    started, transcript = time.perf_counter(), io.StringIO()
    try:
        hashes = {str(path): _digest(path) for path in inputs}
        report['input_sha256'] = hashes
        with tempfile.TemporaryDirectory(prefix='sawf-model-analysis-') as temporary, redirect_stdout(transcript):
            trial = Path(temporary)
            (trial/'trial.npz').write_bytes(arrays['trial_archive'].tobytes())
            (trial/'summary.json').write_bytes(arrays['trial_summary_utf8'].tobytes())
            report['source_approximate_revalidation'] = _revalidate_acceptance(seed, source, arrays, trial)
            _analyze(trial, seed, output, report, points_per_segment, energy_reference_ev, dft_files,
                     saved_model=arrays, ordinary_report=source['analysis']['ordinary_localization'])
        report['hamiltonian_uniform_upper_ev'] = _check_budget(report, source['acceptance'])
        report['hamiltonian_budget_ev'] = source['acceptance']['hamiltonian_budget_ev']
        if any(_digest(path) != digest for path, digest in hashes.items()):
            raise ValueError('Model analysis inputs changed during execution')
        report.update(status='model_analysis_complete', arrays_readback_exact=True,
            output_sha256={path.name: _digest(path) for path in output.iterdir() if path.name != 'summary.json'})
    except Exception as error:
        report.update(error=str(error), last_output_lines=transcript.getvalue().splitlines()[-15:])
        for name in ('analysis.npz', 'bands.npz', 'dft_bands.npz', 'comparison.json'):
            (output/name).unlink(missing_ok=True)
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
