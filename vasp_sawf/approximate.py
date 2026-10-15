"""Export and localize approximate symmetry data under explicit numerical budgets."""

import hashlib
import io
import json
import sys
import tempfile
import time
from numbers import Real
from pathlib import Path

import numpy as np


SCHEMA = 'sawf-bridge-approximate-bloch-v1'
EXPLICIT_POLICY = 'explicit_numerical_budgets_v1'


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _json(content):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    def finite(value):
        result = float(value)
        return result if np.isfinite(result) else invalid(value)
    return json.loads(content, parse_constant=invalid, parse_float=finite)


def _decision(diagnostic_sha256, hashes):
    registry = _json(Path(__file__).with_name('accepted_approximate.json').read_bytes())
    candidates = [row for row in registry['decisions']
                  if row['diagnostic_sha256'] == diagnostic_sha256 and row['source_hashes'] == hashes]
    if len(candidates) != 1:
        raise ValueError('No reviewed approximate decision matches these diagnostic and interface hashes')
    return candidates[0]


def _budgets(mmn_tol, hamiltonian_tol):
    for name, value in (('matrix tolerance', mmn_tol), ('Hamiltonian budget', hamiltonian_tol)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be a positive finite real number')
    return dict(matrix_tolerance=float(mmn_tol), hamiltonian_budget_ev=float(hamiltonian_tol))


def _explicit_decision(digest, bundle, group, budgets):
    return dict(policy=EXPLICIT_POLICY, diagnostic_sha256=digest, source_hashes=bundle.hashes,
        operation_count=group.size, active_axes=np.flatnonzero(bundle.mesh > 1).tolist(), **budgets,
        scope='Explicit numerical budgets for an approximate single-particle model; '
              'full source coefficient closure, exact symmetry and DFT interpolation accuracy are not certified',
        target_policy='Infer from the current SCDM localization or validate an explicit target; no material registry')


def _source_metrics(source, raw, corrected):
    from .localize import _gate

    measurements = dict(coefficient_closure_relative_max=0., raw_vs_measured_entry_max=0.,
        corrected_vs_measured_entry_max=0., source_unitarity_entry_max=0.,
        source_energy_covariance_spectral_max_ev=0., independent_lstsq_difference_max=0.)
    for point in source['diagnostic_kpoints']:
        k = point['interface_kpoint_1based'] - 1
        operations = point['operation_indices_0based']
        measured = np.asarray(point['matrices_real']) + 1j * np.asarray(point['matrices_imag'])
        rows = point.get('operations', [])
        if [row.get('operation_index_0based') for row in rows] != operations:
            raise ValueError('Independent source operation coverage is incomplete')
        singular = np.linalg.svd(measured, compute_uv=False)
        if np.min(singular) <= np.finfo(float).eps * measured.shape[-1] * np.max(singular):
            raise ValueError('Independent source sewing matrix loses rank')
        measurements['coefficient_closure_relative_max'] = max(
            measurements['coefficient_closure_relative_max'], point['coefficient_closure_relative_max'])
        for label, action in (('raw', raw), ('corrected', corrected)):
            key = label + '_vs_measured_entry_max'
            measurements[key] = max(measurements[key], float(abs(action[operations, k] - measured).max()))
        measurements['source_unitarity_entry_max'] = max(measurements['source_unitarity_entry_max'],
            float(abs(measured.conj().swapaxes(-1, -2) @ measured - np.eye(measured.shape[-1])).max()))
        energies = np.asarray(point['energies_ev'])
        energy = (energies[:, None] - energies[None, :]) * measured
        measurements['source_energy_covariance_spectral_max_ev'] = max(
            measurements['source_energy_covariance_spectral_max_ev'],
            float(np.linalg.norm(energy, ord=2, axis=(-2, -1)).max()))
        for row in rows:
            field = 'independent_lstsq_difference_max'
            value = row.get(field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f'Invalid independent source measurement: {field}')
            measurements[field] = max(measurements[field], float(value))
    checks = {}
    _gate(checks, 'independent_lstsq_difference_max', measurements['independent_lstsq_difference_max'])
    return measurements


def _check_explicit_metrics(report, decision):
    from .localize import _gate

    checks = report.setdefault('approximate_checks', {})
    matrix_tol = decision['matrix_tolerance']
    energy_tol = decision['hamiltonian_budget_ev']
    for name in ('raw_action', 'reconstructed_action'):
        action = report[name]
        _gate(checks, name + '_mmn_max', max(row['mmn_covariance_entry_max']
            for row in action['per_operation']), matrix_tol)
        _gate(checks, name + '_energy_spectral_max_ev', max(row['energy_covariance_spectral_max_ev']
            for row in action['per_operation']), energy_tol)
    _gate(checks, 'raw_unitarity_max', report['raw_action']['unitarity_entry_max'], matrix_tol)
    _gate(checks, 'action_change_entry_max', report['action_change_entry_max'], matrix_tol)
    for key in ('raw_vs_measured_entry_max', 'corrected_vs_measured_entry_max', 'source_unitarity_entry_max'):
        _gate(checks, key, report['source_checks'][key], matrix_tol)
    _gate(checks, 'source_energy_covariance_spectral_max_ev',
          report['source_checks']['source_energy_covariance_spectral_max_ev'], energy_tol)


def _check_target(center, orbital, decision):
    from .localize import _normalize_target_orbits

    centers, orbitals = _normalize_target_orbits(center, orbital)
    expected = decision['target']['orbits']
    if len(centers) != len(expected) or orbitals != [row['orbital'] for row in expected]:
        raise ValueError('Requested target orbit order or local representations differ from the reviewed target')
    delta = np.asarray(centers) - np.asarray([row['position_fractional'] for row in expected])
    if not np.isfinite(delta).all() or np.max(abs(delta - np.rint(delta))) > 1e-12:
        raise ValueError('Requested target centers differ from the reviewed target modulo lattice translations')
    return centers, orbitals


def _check_budget(analysis, decision):
    rows = analysis.get('uniform_covariance_bounds', [])
    if [row.get('operation_index') for row in rows] != list(range(decision['operation_count'])):
        raise ValueError('Hamiltonian bound operation coverage is incomplete')
    axes = decision['active_axes']
    fixed = {str(i): 0. for i in range(3) if i not in axes}
    if any(row.get('active_axes') != axes or row.get('fixed_inactive_coordinates') != fixed for row in rows):
        raise ValueError('Hamiltonian bound coordinate plane differs from the configured scope')
    values = [row.get('uniform_upper_ev') for row in rows]
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not np.isfinite(value) or value < 0 for value in values):
        raise ValueError('Hamiltonian bound is missing, negative or nonfinite')
    maximum = float(max(values))
    budget = decision['hamiltonian_budget_ev']
    if (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Real)
            or not np.isfinite(budget) or budget <= 0):
        raise ValueError('Hamiltonian budget must be a positive finite real number')
    if maximum > budget:
        raise ValueError(f'Hamiltonian bound {maximum} eV exceeds the configured budget {budget} eV')
    return maximum


def _prepare_output(output_dir, inputs, *, warn=True, preserved_inputs=()):
    requested = Path(output_dir)
    output = requested.resolve()
    protected = {Path(path).resolve() for path in inputs}
    preserved = {Path(path).resolve() for path in preserved_inputs}
    if (requested.is_symlink() or (output.exists() and not output.is_dir())
            or any(output.is_relative_to(path) or path.is_relative_to(output)
                   for path in protected - preserved)):
        raise ValueError('Output must be a real directory separate from protected input file paths')
    targets = [output/'bloch.npz', output/'report.json']
    existing = []
    for target in targets:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f'Output target {target.name} must be a regular file')
        if (any(target.resolve().is_relative_to(path) or path.is_relative_to(target.resolve()) for path in protected)
                or (target.exists() and any(path.exists() and target.samefile(path) for path in protected))):
            raise ValueError(f'Output target {target.name} overlaps a protected input file')
        if target.exists():
            existing.append(target)
    output.mkdir(parents=True, exist_ok=True)
    if existing and warn:
        print('Warning: overwriting existing symmetry files: '+', '.join(path.name for path in existing),
              file=sys.stderr, flush=True)
    for path in existing:
        path.unlink()
    return output


def _new_model_output(output_dir, inputs):
    requested = Path(output_dir)
    output = requested.resolve()
    protected = {Path(path).resolve() for path in inputs}
    if (requested.is_symlink() or output.exists()
            or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected)):
        raise ValueError('Model output must be a new real directory separate from protected input file paths')
    output.mkdir(parents=True, exist_ok=False)
    return output


def _base_report(schema):
    return dict(schema=schema, status='not_ready', accepted=False, sawf_ready=False,
        strict_ready=False, strict_checks_passed=False, numerical_checks_passed=False,
        source_full_closure_certified=False, symmetrize_hr=False, wavefunction_bytes_read=0,
        physical_acceptance_status='not_assessed',
        limitation='Explicit numerical approximation only. Full source coefficient closure, exact symmetry, '
                   'and DFT interpolation accuracy are not certified; original MMN and EIG remain unchanged.')


def _reconstruct(seed, folder, *, budgets=None):
    from .localize import _gate
    from .representation import reconstruct_representation
    from .symmetry import mmn_translation_phases, transport_sewing
    from .trial import _action_metrics, _read_diagnostic

    bundle, source, digest, spin, group, kmap, edges, gamma, anchors, binding = _read_diagnostic(seed, folder)
    decision = (_decision(digest, bundle.hashes) if budgets is None
                else _explicit_decision(digest, bundle, group, budgets))
    if source['evaluated_wavecar_kpoints_1based'] != list(range(1, source['wavecar']['num_kpoints'] + 1)):
        raise ValueError('Approximate extraction requires complete stored-k diagnostic coverage')
    if group.size != decision['operation_count']:
        raise ValueError('Reviewed operation count differs from the recomputed group')
    anti = np.array([operation.time_reversal for operation in group.symmetries], dtype=bool)
    phases = mmn_translation_phases(bundle, group)
    raw, transport = transport_sewing(bundle.mmn, bundle.neighbor_indices, kmap, edges, anti,
                                     anchors, anchor_k=gamma, edge_phases=phases)
    product, translations, factors = map(np.asarray, group.get_product_table(get_diff=True))
    cocycle = factors[:, :, None] * np.exp(-2j*np.pi*np.einsum(
        'ghki,ghi->ghk', bundle.kpoints[kmap[product]], translations))
    corrected, construction = reconstruct_representation(raw, kmap, anti, product, cocycle)
    raw_metrics = _action_metrics(raw, bundle, group, kmap, edges, phases)
    metrics = _action_metrics(corrected, bundle, group, kmap, edges, phases)
    checks = {}
    _gate(checks, 'reconstructed_unitarity_max', metrics['unitarity_entry_max'])
    _gate(checks, 'reconstructed_group_composition_max', metrics['group']['group_composition_max'])
    report = dict(source_hashes=bundle.hashes, source_num_bands=source['source_num_bands'],
        shape_nk_nb_nw=list(bundle.amn.shape), mesh=bundle.mesh.tolist(),
        bands_vasp_1based=bundle.bands_vasp_1based.tolist(), spin=spin,
        source_diagnostic_sha256=digest, diagnostic_input_binding=binding,
        acceptance=decision, transport=transport, reconstruction=construction,
        raw_action=raw_metrics, reconstructed_action=metrics, structural_checks=checks,
        action_change_entry_max=float(abs(corrected-raw).max()),
        source_coefficient_coverage=dict(scope='All stored-k little groups; generic cross-k source closure is not certified',
            checked_stored_kpoints=source['evaluated_wavecar_kpoints_1based'],
            full_closure_certified=False, stored_kpoint_count=source['wavecar']['num_kpoints']))
    if budgets is not None:
        report['source_checks'] = _source_metrics(source, raw, corrected)
        try:
            _check_explicit_metrics(report, decision)
        except ValueError as error:
            error.report = report
            raise
    arrays = dict(raw_d=raw, reconstructed_d=corrected, kmap=kmap, anti=anti,
        kpoints=bundle.kpoints, eigenvalues_eV=bundle.eig, bands_vasp_1based=bundle.bands_vasp_1based)
    return report, arrays, decision


def export_approximate(seed, diagnostic_dir, output_dir, *, mmn_tol=None, hamiltonian_tol=None,
                       _warn_overwrite=True):
    """Reuse compact evidence under reviewed or explicit budgets, without strict-source certification."""
    budgets = None if mmn_tol is None and hamiltonian_tol is None else _budgets(mmn_tol, hamiltonian_tol)
    seed, folder = Path(seed).resolve(), Path(diagnostic_dir).resolve()
    files = [Path(f'{seed}.{ext}') for ext in ('win', 'amn', 'mmn', 'eig')]
    diagnostic = folder/'diagnostic.json'
    preserved = [diagnostic] if diagnostic.parent == Path(output_dir).resolve() else []
    output = _prepare_output(output_dir, [seed, *files, diagnostic], warn=_warn_overwrite,
                             preserved_inputs=preserved)
    report = _base_report(SCHEMA)
    path = output/'report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    started = time.perf_counter()
    try:
        content = diagnostic.read_bytes()
        preliminary = _json(content)
        if budgets is None:
            _decision(_digest(content), preliminary.get('source_hashes'))
        details, arrays, decision = _reconstruct(seed, folder, budgets=budgets)
        if _digest(content) != details['source_diagnostic_sha256'] or diagnostic.read_bytes() != content:
            raise ValueError('Diagnostic evidence changed during approximate extraction')
        if any(_digest(file.read_bytes()) != details['source_hashes'][file.suffix[1:]] for file in files):
            raise ValueError('Original interface inputs changed during approximate extraction')
        arrays['diagnostic_utf8'] = np.frombuffer(content, dtype=np.uint8)
        np.savez_compressed(output/'bloch.npz', **arrays)
        with np.load(output/'bloch.npz', allow_pickle=False) as saved:
            if any(not np.array_equal(value, saved[key]) for key, value in arrays.items()):
                raise ValueError('Approximate package array save/readback failed')
        package = (output/'bloch.npz').read_bytes()
        report.update(details, source_seed=str(seed), source_diagnostic_path=str(diagnostic),
            bloch_sha256=_digest(package), bloch_bytes=len(package),
            status='accepted_approximate', accepted=True, sawf_ready=True,
            physical_acceptance_status='accepted_approximate_single_particle_model',
            acceptance_scope=decision['scope'], diagnostic_embedded_exact=True)
    except Exception as error:
        if isinstance(getattr(error, 'report', None), dict):
            report.update(error.report)
        report.update(error=str(error), status='not_ready', accepted=False, sawf_ready=False)
        (output/'bloch.npz').unlink(missing_ok=True)
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def extract_fresh_approximate(seed, wavecar, outcar, output_dir, *, mmn_tol, hamiltonian_tol,
                              workers=None, memory_gb=None, spin_channel=None, tol=1e-5, energy_tol=1e-8):
    """Measure every stored k point before applying explicitly supplied numerical budgets."""
    from .symmetry import export_symmetry

    budgets = _budgets(mmn_tol, hamiltonian_tol)
    seed, wavecar, outcar = map(lambda path: Path(path).resolve(), (seed, wavecar, outcar))
    inputs = [seed, wavecar, outcar, *(Path(f'{seed}.{ext}') for ext in ('win', 'amn', 'mmn', 'eig'))]
    output = _prepare_output(output_dir, inputs)
    report = _base_report(SCHEMA)
    report.update(requested_budgets=budgets, extraction_mode='fresh_all_stored_kpoints',
                  wavefunction_bytes_read=None)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    started = time.perf_counter()
    diagnostic = None
    try:
        diagnostic = export_symmetry(seed, wavecar, outcar, output, workers=workers, memory_gb=memory_gb,
            spin_channel=spin_channel, tol=tol, energy_tol=energy_tol, diagnose_all=True)
        report = export_approximate(seed, output, output, mmn_tol=mmn_tol,
                                    hamiltonian_tol=hamiltonian_tol, _warn_overwrite=False)
    except Exception as error:
        current = _json((output/'report.json').read_bytes())
        report.update(current, error=str(error), status='not_ready', accepted=False, sawf_ready=False)
        raise
    finally:
        report.update(extraction_mode='fresh_all_stored_kpoints', requested_budgets=budgets)
        if diagnostic is not None:
            report.update(wavefunction_bytes_read=sum(row['bytes_returned']
                for row in diagnostic.get('read_ledger', []) if row['kind'] == 'coefficients'),
                source_read_accounting={key: diagnostic.get(key) for key in
                    ('read_bytes', 'metadata_read_bytes', 'total_logical_read_bytes', 'kpoint_resources',
                     'execution', 'memory_estimate', 'wall_seconds', 'peak_rss_kib', 'read_accounting_complete')})
        report['fresh_extraction_seconds'] = time.perf_counter() - started
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def _read_package(seed, folder):
    from .localize import _gate

    folder = Path(folder)
    report = _json((folder/'report.json').read_bytes())
    flags = dict(accepted=True, sawf_ready=True, strict_ready=False, strict_checks_passed=False,
                 numerical_checks_passed=False,
                 source_full_closure_certified=False, symmetrize_hr=False)
    if (report.get('schema') != SCHEMA or report.get('status') != 'accepted_approximate'
            or any(report.get(key) is not value for key, value in flags.items())):
        raise ValueError('Symmetry package is not an explicitly accepted approximate package')
    content = (folder/'bloch.npz').read_bytes()
    if len(content) != report.get('bloch_bytes') or _digest(content) != report.get('bloch_sha256'):
        raise ValueError('Approximate package hash or size disagrees with its report')
    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    for key, value in arrays.items():
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
            raise ValueError(f'Unsafe approximate array: {key}')
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite approximate array: {key}')
    encoded = arrays.get('diagnostic_utf8')
    if encoded is None or encoded.dtype != np.uint8 or encoded.ndim != 1:
        raise ValueError('Approximate package lacks exact diagnostic UTF-8 bytes')
    diagnostic = encoded.tobytes()
    recorded = report.get('acceptance', {})
    explicit = recorded.get('policy') == EXPLICIT_POLICY
    budgets = _budgets(recorded.get('matrix_tolerance'), recorded.get('hamiltonian_budget_ev')) if explicit else None
    decision = recorded if explicit else _decision(_digest(diagnostic), report.get('source_hashes'))
    if report.get('acceptance') != decision or report.get('source_diagnostic_sha256') != _digest(diagnostic):
        raise ValueError('Approximate acceptance record differs from the reviewed registry')
    with tempfile.TemporaryDirectory(prefix='sawf-approximate-check-') as temporary:
        Path(temporary, 'diagnostic.json').write_bytes(diagnostic)
        details, rebuilt, decision = _reconstruct(seed, Path(temporary), budgets=budgets)
    if recorded != decision:
        raise ValueError('Approximate acceptance policy disagrees with recomputed input evidence')
    for key in ('source_hashes', 'source_num_bands', 'shape_nk_nb_nw', 'mesh', 'bands_vasp_1based', 'spin'):
        if report.get(key) != details[key]:
            raise ValueError(f'Approximate source metadata differs from recomputed evidence: {key}')
    checks = {}
    for key, expected in rebuilt.items():
        actual = arrays.get(key)
        if actual is None or actual.shape != expected.shape:
            raise ValueError(f'Approximate array dimensions differ from recomputed evidence: {key}')
        if key in ('raw_d', 'reconstructed_d'):
            _gate(checks, key+'_recomputed', np.max(abs(actual-expected)))
        elif not np.array_equal(actual, expected):
            raise ValueError(f'Approximate array differs from recomputed evidence: {key}')
    return diagnostic, decision, details


def _clean_staging(value, root):
    if isinstance(value, str):
        return value.replace(str(root), 'embedded-stage:')
    if isinstance(value, list):
        return [_clean_staging(item, root) for item in value]
    if isinstance(value, dict):
        return {_clean_staging(key, root): _clean_staging(item, root) for key, item in value.items()}
    return value


def run_approximate(seed, symmetry_dir, output_dir, *, center=None, orbital=None, num_iter=1000,
                    dft_eigenval=None, energy_reference_ev=0., points_per_segment=101, center_tol=1e-2):
    """Localize unchanged inputs and enforce the configured Fourier error budget."""
    from .analysis import analyze_trial
    from .trial import run_trial

    seed, folder = Path(seed).resolve(), Path(symmetry_dir).resolve()
    dft = [] if dft_eigenval is None else ([dft_eigenval] if isinstance(dft_eigenval, (str, Path)) else list(dft_eigenval))
    files = [Path(f'{seed}.{ext}') for ext in ('win', 'amn', 'mmn', 'eig')]
    files += [folder/'bloch.npz', folder/'report.json', *map(Path, dft)]
    output = _new_model_output(output_dir, [seed, *files])
    report = _base_report('sawf-bridge-approximate-model-v1')
    report.update(converged=False, source_seed=str(seed), source_symmetry=str(folder))
    started = time.perf_counter()
    try:
        hashes = {str(path.resolve()): _digest(path.read_bytes()) for path in files}
        diagnostic, decision, details = _read_package(seed, folder)
        report.update(details, input_sha256=hashes, hamiltonian_budget_ev=decision['hamiltonian_budget_ev'])
        explicit = decision.get('policy') == EXPLICIT_POLICY
        if not explicit and center is not None:
            center, orbital = _check_target(center, orbital, decision)
        with tempfile.TemporaryDirectory(prefix='sawf-approximate-run-') as temporary:
            temporary = Path(temporary)
            evidence = temporary/'source'
            evidence.mkdir()
            (evidence/'diagnostic.json').write_bytes(diagnostic)
            try:
                trial = run_trial(seed, evidence, temporary/'trial', center=center, orbital=orbital,
                                  num_iter=num_iter, center_tol=center_tol)
            except Exception:
                failed = temporary/'trial/summary.json'
                if failed.exists():
                    report['trial_evidence'] = _clean_staging(_json(failed.read_bytes()), temporary)
                raise
            report.update(target=trial['target'], localisation=trial['localisation'],
                converged=trial['converged'], numerical_converged=trial['numerical_converged'],
                trial_evidence=_clean_staging(trial, temporary))
            if not explicit:
                _check_target([row['position_fractional'] for row in trial['target']['orbits']],
                              [row['orbital'] for row in trial['target']['orbits']], decision)
            try:
                analysis = analyze_trial(temporary/'trial', temporary/'analysis', seed=seed,
                    points_per_segment=points_per_segment, dft_eigenval=dft or None,
                    energy_reference_ev=energy_reference_ev)
            except Exception:
                failed = temporary/'analysis/summary.json'
                if failed.exists():
                    report['analysis'] = _clean_staging(_json(failed.read_bytes()), temporary)
                raise
            report['analysis'] = _clean_staging(analysis, temporary)
            maximum = _check_budget(analysis, decision)
            with np.load(temporary/'analysis/analysis.npz', allow_pickle=False) as saved:
                model = {key: saved[key].copy() for key in saved.files}
            model['diagnostic_utf8'] = np.frombuffer(diagnostic, dtype=np.uint8)
            model['trial_archive'] = np.frombuffer((temporary/'trial/trial.npz').read_bytes(), dtype=np.uint8)
            model['trial_summary_utf8'] = np.frombuffer((temporary/'trial/summary.json').read_bytes(), dtype=np.uint8)
            np.savez_compressed(output/'model.npz', **model)
            with np.load(output/'model.npz', allow_pickle=False) as saved:
                if any(not np.array_equal(value, saved[key]) for key, value in model.items()):
                    raise ValueError('Approximate model array save/readback failed')
            for name in ('bands.npz', 'dft_bands.npz'):
                source = temporary/'analysis'/name
                if source.exists():
                    (output/name).write_bytes(source.read_bytes())
            report.update(details, target=trial['target'], localisation=trial['localisation'],
                trial_evidence=_clean_staging(trial, temporary), analysis=_clean_staging(analysis, temporary),
                stage_evidence_convention='Embedded diagnostic stages remain marked unaccepted in their original schemas; '
                    'only this enclosing report applies the explicit reviewed decision and model budget. '
                    'embedded-stage: paths identify retained trial_archive, trial_summary_utf8 and model arrays. '
                    'Exact trial_summary_utf8 bytes preserve the historical stage hash; any paths inside those '
                    'bytes are historical provenance, never required external files.',
                hamiltonian_uniform_upper_ev=maximum, hamiltonian_budget_ev=decision['hamiltonian_budget_ev'])
        from .wanproj import write_wanproj
        if report['spin']['source_ispin'] == 2:
            report['wanproj'] = dict(status='not_exported', reason='A standalone collinear spin-channel '
                'model is not a complete two-channel VASP WANPROJ; both channel gauges are required')
        else:
            report['wanproj'] = write_wanproj(output/'WANPROJ', U=model['U'], kpoints=model['kpoints'],
                bands_vasp_1based=model['bands_vasp_1based'], source_nb=report['source_num_bands'],
                mesh=model['mesh'])
        if any(_digest(Path(path).read_bytes()) != digest for path, digest in hashes.items()):
            raise ValueError('Approximate model inputs changed during execution')
        report.update(status='accepted_approximate', accepted=True, sawf_ready=True, converged=True,
            physical_acceptance_status='accepted_approximate_single_particle_model',
            input_sha256=hashes, output_sha256={path.name: _digest(path.read_bytes()) for path in output.iterdir()
                if path.suffix == '.npz' or path.name == 'WANPROJ'})
    except Exception as error:
        report.update(error=str(error), accepted=False, sawf_ready=False, status='not_ready')
        if isinstance(getattr(error, 'report', None), dict):
            report['localisation'] = error.report
        for name in ('model.npz', 'bands.npz', 'dft_bands.npz', 'WANPROJ'):
            (output/name).unlink(missing_ok=True)
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
