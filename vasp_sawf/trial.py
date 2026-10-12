"""Run unaccepted representation/localization diagnostics from existing compact evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


def _index(value, upper, label):
    if type(value) is not int or not 1 <= value <= upper:
        raise ValueError(f'Invalid {label} one-based index')
    return value - 1


def _matrices(record, shape, label):
    real = np.asarray(record.get('matrices_real'), dtype=float)
    imag = np.asarray(record.get('matrices_imag'), dtype=float)
    if real.shape != shape or imag.shape != shape or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError(f'Invalid {label} matrix dimensions or values')
    return real + 1j * imag


def _read_diagnostic(seed, folder):
    from .inputs import read_inputs
    from .localize import _validate_spin_metadata
    from .spin import _spacegroup_from_context
    from .symmetry import _cell_from_win, _validate_grey_group, build_symmetry_maps

    content = (folder / 'diagnostic.json').read_bytes()
    def invalid_constant(value):
        raise ValueError(f'Nonfinite diagnostic JSON value: {value}')
    def finite_float(value):
        parsed = float(value)
        if not np.isfinite(parsed):
            invalid_constant(value)
        return parsed
    source = json.loads(content, parse_constant=invalid_constant, parse_float=finite_float)
    if (source.get('schema') != 'sawf-bridge-diagnostic-v1'
            or source.get('status') != 'diagnostic_complete'
            or source.get('diagnostic_only') is not True or source.get('sawf_ready') is not False
            or source.get('read_accounting_complete') is not True):
        raise ValueError('Trial input must be a completed, explicitly unaccepted diagnostic report')
    source_nb = source.get('source_num_bands')
    if type(source_nb) is not int or source_nb < 1:
        raise ValueError('Diagnostic report lacks the original source band count')
    energy_tolerance = source.get('wavecar_interface_energy_tolerance_ev')
    if (type(energy_tolerance) not in (int, float) or not np.isfinite(energy_tolerance)
            or energy_tolerance <= 0):
        raise ValueError('Diagnostic source energy tolerance must be recorded, positive and finite')
    bundle = read_inputs(seed, source_nb=source_nb)
    if source.get('source_hashes') != bundle.hashes:
        raise ValueError('Diagnostic report does not match the original interface hashes')
    for key, expected in (('shape_nk_nb_nw', list(bundle.amn.shape)), ('mesh', bundle.mesh.tolist()),
                          ('bands_vasp_1based', bundle.bands_vasp_1based.tolist())):
        if source.get(key) != expected:
            raise ValueError(f'Diagnostic report disagrees with original interface {key}')
    spin_record = source.get('spin')
    if not isinstance(spin_record, dict):
        raise ValueError('Diagnostic report lacks explicit spin semantics')
    arrays = {key: np.asarray(value) for key, value in spin_record.items()}
    arrays['spacegroup_spinor'] = np.asarray(spin_record.get('spinor'))
    win = Path(f'{seed}.win').read_text()
    spin = _validate_spin_metadata(arrays, source, win_text=win)
    context = deepcopy(source.get('outcar', {}).get('spin_context', {}))
    if (context.get('spinor') is not spin['spinor']
            or type(context.get('source_ispin')) is not int
            or context['source_ispin'] != spin['source_ispin']):
        raise ValueError('Diagnostic spin context disagrees with the selected source spin mode')
    nk, nb, nw = bundle.amn.shape
    if nb != nw or (spin['spinor'] and nb % 2):
        raise ValueError('Trial localization requires all bands in a square target subspace')
    positions, types, _ = _cell_from_win(win, bundle.lattice)
    group = _spacegroup_from_context(bundle, positions, types, context)
    _validate_grey_group(group)
    serialized = source.get('symmetry_operations', {})
    operations = serialized.get('operations', [])
    if (serialized.get('total') != group.size or len(operations) != group.size
            or serialized.get('antiunitary') != sum(bool(op.time_reversal) for op in group.symmetries)):
        raise ValueError('Diagnostic symmetry operation counts disagree with the reconstructed group')
    spinor_recompute_difference = 0.
    for i, (stored, actual) in enumerate(zip(operations, group.symmetries, strict=True)):
        if type(stored.get('index_0based')) is not int or stored['index_0based'] != i:
            raise ValueError('Diagnostic symmetry operation indices are not canonical')
        fields = dict(rotation=actual.rotation, translation=actual.translation)
        if (type(stored.get('time_reversal')) is not bool
                or stored['time_reversal'] != bool(actual.time_reversal)
                or any(not np.array_equal(np.asarray(stored.get(key)), value) for key, value in fields.items())):
            raise ValueError('Diagnostic symmetry operations disagree with WIN and stored spin context')
        for key, value in (('spinor_rotation_real', actual.spinor_rotation.real),
                           ('spinor_rotation_imag', actual.spinor_rotation.imag)):
            saved = np.asarray(stored.get(key), dtype=float)
            if (saved.shape != value.shape or not np.isfinite(saved).all()
                    or np.max(abs(saved - value)) > 32 * np.finfo(float).eps):
                raise ValueError('Diagnostic spinor operations disagree beyond floating-point recomputation')
            spinor_recompute_difference = max(spinor_recompute_difference, float(np.max(abs(saved - value))))
    kmap, edges = build_symmetry_maps(bundle, group)
    gamma_candidates = np.flatnonzero(np.max(abs(bundle.kpoints - np.rint(bundle.kpoints)), axis=1) < 1e-12)
    anchor = source.get('gamma_anchor', {})
    gamma = _index(anchor.get('interface_kpoint_1based'), nk, 'Gamma interface k point')
    if len(gamma_candidates) != 1 or gamma != gamma_candidates[0] or not np.all(kmap[:, gamma] == gamma):
        raise ValueError('Diagnostic Gamma anchor does not identify the unique full-group fixed Gamma')
    all_indices = list(range(group.size))
    if anchor.get('operation_indices_0based') != all_indices:
        raise ValueError('Diagnostic Gamma anchor does not cover every operation in canonical order')
    anchors = _matrices(anchor, (group.size, nb, nb), 'Gamma anchor')
    stored_nk = source.get('wavecar', {}).get('num_kpoints')
    if type(stored_nk) is not int or stored_nk < 1:
        raise ValueError('Diagnostic report lacks the stored source k-point count')
    gamma_stored = _index(anchor.get('wavecar_kpoint_1based'), stored_nk, 'Gamma stored k point')
    points = source.get('diagnostic_kpoints')
    if not isinstance(points, list) or not points:
        raise ValueError('Diagnostic report has no completed source k-point checks')
    seen, seen_stored, gamma_rows = set(), set(), 0
    folding, energy_difference = [], 0.
    for point in points:
        k = _index(point.get('interface_kpoint_1based'), nk, 'diagnostic interface k point')
        stored = _index(point.get('wavecar_kpoint_1based'), stored_nk, 'diagnostic stored k point')
        if k in seen or stored in seen_stored:
            raise ValueError('Diagnostic point mapping is duplicated')
        seen.add(k)
        seen_stored.add(stored)
        coordinate = np.asarray(point.get('kpoint_fractional'), dtype=float)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ValueError('Diagnostic point coordinates disagree with the interface mapping')
        shift = np.rint(coordinate - bundle.kpoints[k])
        if np.max(abs(coordinate - bundle.kpoints[k] - shift)) > 1e-8:
            raise ValueError('Diagnostic point coordinates disagree with the interface mapping')
        folding.append(shift.astype(int).tolist())
        little = np.flatnonzero(kmap[:, k] == k).tolist()
        if point.get('operation_indices_0based') != little:
            raise ValueError('Diagnostic point operations do not equal its complete little group')
        matrices = _matrices(point, (len(little), nb, nb), 'selected little group')
        energies = np.asarray(point.get('energies_ev'), dtype=float)
        if (energies.shape != (nb,) or not np.isfinite(energies).all()
                or np.max(abs(energies - bundle.eig[k])) > energy_tolerance):
            raise ValueError('Diagnostic selected energies disagree with the original EIG')
        energy_difference = max(energy_difference, float(np.max(abs(energies - bundle.eig[k]))))
        closure = point.get('coefficient_closure_relative_max')
        if type(closure) not in (int, float) or not np.isfinite(closure) or closure < 0:
            raise ValueError('Diagnostic selected coefficient residual is invalid')
        if k == gamma:
            gamma_rows += 1
            if (stored != gamma_stored or not np.array_equal(matrices, anchors)
                    or closure != anchor.get('coefficient_closure_relative_max')):
                raise ValueError('Duplicate diagnostic Gamma evidence disagrees')
    completed = sorted(k + 1 for k in seen_stored)
    if (gamma_rows != 1 or source.get('evaluated_wavecar_kpoints_1based') != completed
            or source.get('planned_wavecar_kpoints_1based') != completed):
        raise ValueError('Diagnostic selected source coverage is incomplete or inconsistent')
    binding = dict(selected_kpoint_integer_folding=folding,
                   folding_convention='Stored diagnostic k minus WIN k, rounded to its reciprocal lattice integer',
                   spinor_operation_recompute_max_abs=spinor_recompute_difference,
                   selected_energy_difference_max_ev=energy_difference,
                   recorded_source_energy_tolerance_ev=energy_tolerance)
    return bundle, source, hashlib.sha256(content).hexdigest(), spin, group, kmap, edges, gamma, anchors, binding


def _action_metrics(action, bundle, group, kmap, edges, phases):
    from .symmetry import group_residuals

    product, translations, factors = group.get_product_table(get_diff=True)
    anti = np.array([op.time_reversal for op in group.symmetries], dtype=bool)
    rows = []
    for g in range(group.size):
        covariance = (action[g, :, None].conj().swapaxes(-1, -2)
                      @ bundle.mmn[kmap[g, :, None], edges[g]] @ action[g, bundle.neighbor_indices]
                      - phases[g, :, :, None, None] * (bundle.mmn.conj() if anti[g] else bundle.mmn))
        energy = bundle.eig[kmap[g], :, None] * action[g] - action[g] * bundle.eig[:, None, :]
        rows.append(dict(operation_index=g, antiunitary=bool(anti[g]),
            mmn_covariance_entry_max=float(abs(covariance).max()),
            mmn_covariance_spectral_max=float(np.linalg.norm(covariance, ord=2, axis=(-2, -1)).max()),
            energy_covariance_entry_max_ev=float(abs(energy).max()),
            energy_covariance_spectral_max_ev=float(np.linalg.norm(energy, ord=2, axis=(-2, -1)).max())))
    return dict(unitarity_entry_max=float(abs(action.conj().swapaxes(-1, -2) @ action
                                             - np.eye(bundle.eig.shape[-1])).max()),
                group=group_residuals(action, kmap, anti, product, factors,
                    kpoints=bundle.kpoints, translations_diff=translations), per_operation=rows)


def run_trial(seed, symmetry_dir, output_dir, *, center, orbital, num_iter=1000,
              dft_eigenval=None, energy_reference_ev=0.):
    """Localize a diagnostic action without accepting its source or exporting a production model."""
    from .localize import _new_output, _normalize_target_orbits

    if dft_eigenval is not None:
        raise ValueError('DFT path export is unavailable for unaccepted diagnostic trials')
    if not np.isfinite(energy_reference_ev):
        raise ValueError('The energy reference must be finite')
    center, orbital = _normalize_target_orbits(center, orbital)
    seed, folder = Path(seed).resolve(), Path(symmetry_dir).resolve()
    files = [Path(f'{seed}.{ext}') for ext in ('win', 'amn', 'mmn', 'eig')]
    output = _new_output(output_dir, [*files, folder / 'diagnostic.json'])
    report = dict(schema='sawf-bridge-diagnostic-trial-v1', status='not_ready', diagnostic_only=True,
                  accepted=False, sawf_ready=False, converged=False, numerical_converged=False,
                  source_seed=str(seed), wavefunction_bytes_read=0, symmetrize_hr=False,
                  physical_acceptance_status='not_assessed',
                  limitation='Reconstruction changes the approximate symmetry action; numerical convergence '
                             'does not establish source closure, a physical error budget, or DFT interpolation accuracy')
    started, transcript = time.perf_counter(), io.StringIO()
    try:
        with redirect_stdout(transcript):
            _run_trial(seed, folder, output, report, center, orbital, num_iter)
    except Exception as error:
        report.update(error=str(error), last_output_lines=transcript.getvalue().splitlines()[-15:])
        if isinstance(getattr(error, 'report', None), dict):
            report['localisation'] = error.report
        (output / 'trial.npz').unlink(missing_ok=True)
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def _run_trial(seed, folder, output, report, center, orbital, num_iter):
    from .inputs import load_wannier_data
    from .localize import (_ordinary_initial_gauge, align_scdm_initial_gauge, _target_representation,
                           _target_matrix, wannierise_full_mesh, _gate)
    from .representation import reconstruct_representation
    from .symmetry import mmn_translation_phases, transport_sewing, make_symmetrizer, group_residuals

    bundle, source, digest, spin, group, kmap, edges, gamma, anchors, binding = _read_diagnostic(seed, folder)
    report.update(source_hashes=bundle.hashes, source_num_bands=source['source_num_bands'],
                  bands_vasp_1based=bundle.bands_vasp_1based.tolist(), mesh=bundle.mesh.tolist(),
                  shape_nk_nb_nw=list(bundle.amn.shape), spin=spin,
                  source_diagnostic_sha256=digest, source_diagnostic=source, diagnostic_input_binding=binding,
                  source_coefficient_coverage=dict(scope='Selected stored-k little groups only',
                      full_closure_certified=False, checked_stored_kpoints=source['evaluated_wavecar_kpoints_1based'],
                      stored_kpoint_count=source['wavecar']['num_kpoints'], full_mesh_kpoint_count=len(bundle.kpoints)))
    anti = np.array([op.time_reversal for op in group.symmetries], dtype=bool)
    phases = mmn_translation_phases(bundle, group)
    raw, transport = transport_sewing(bundle.mmn, bundle.neighbor_indices, kmap, edges, anti,
                                     anchors, anchor_k=gamma, edge_phases=phases)
    product, translations, factors = map(np.asarray, group.get_product_table(get_diff=True))
    cocycle = factors[:, :, None] * np.exp(-2j * np.pi * np.einsum(
        'ghki,ghi->ghk', bundle.kpoints[kmap[product]], translations))
    corrected, construction = reconstruct_representation(raw, kmap, anti, product, cocycle)
    report.update(transport=transport, reconstruction=construction,
                  action_change_entry_max=float(abs(corrected - raw).max()),
                  raw_action=_action_metrics(raw, bundle, group, kmap, edges, phases),
                  reconstructed_action=_action_metrics(corrected, bundle, group, kmap, edges, phases))
    report['direct_source_comparison'] = [dict(interface_kpoint_1based=point['interface_kpoint_1based'],
        operation_index=g, used_as_gamma_anchor=point['interface_kpoint_1based'] - 1 == gamma,
        raw_difference_entry_max=float(abs(raw[g, point['interface_kpoint_1based'] - 1] - matrix).max()),
        reconstructed_difference_entry_max=float(abs(corrected[g, point['interface_kpoint_1based'] - 1] - matrix).max()))
        for point in source['diagnostic_kpoints']
        for g, matrix in zip(point['operation_indices_0based'],
            _matrices(point, (len(point['operation_indices_0based']), bundle.eig.shape[-1], bundle.eig.shape[-1]),
                      'selected little group'), strict=True)]
    sym = make_symmetrizer(bundle, group, corrected)
    blocks, positions, target_report = _target_representation(sym, center, orbital)
    if sym.num_wann != bundle.eig.shape[-1]:
        raise ValueError('Target representation does not span every original target band')
    target = np.array([[_target_matrix(blocks, k, bundle.kpoints[kmap[g, i]], g)
                        for i, k in enumerate(bundle.kpoints)] for g in range(group.size)])
    target_product = group_residuals(target, kmap, anti, product, factors,
                                    kpoints=bundle.kpoints, translations_diff=translations)
    _gate(report, 'target_group_composition_max', target_product['group_composition_max'])
    _gate(report, 'target_unitarity_max', abs(target.conj().swapaxes(-1, -2) @ target
                                             - np.eye(bundle.eig.shape[-1])).max())
    target_report['basis_role'] = ('User-supplied center orbits and local matrix representations; '
                                 'orbital labels alone do not establish atomic orbital character')
    data, _ = load_wannier_data(seed, source_nb=source['source_num_bands'])
    ordinary, centers, ordinary_report = _ordinary_initial_gauge(data, num_iter=num_iter)
    initial, alignment = align_scdm_initial_gauge(bundle.amn, ordinary, centers, bundle.lattice,
        bundle.kpoints, sym, target_centers=positions @ bundle.lattice)
    alignment['ordinary_provenance'] = 'Unconstrained localization from the bound original AMN in this process'
    report.update(target=target_report, ordinary_initialization=ordinary_report, initial_alignment=alignment)
    report['localisation'] = wannierise_full_mesh(data, sym, target, initial_gauge=initial, num_iter=num_iter)
    u = np.array([data.chk.v_matrix[k] for k in range(sym.NK)])
    gauge_covariance = max(float(abs(corrected[g] @ (u.conj() if anti[g] else u)
                                      - u[kmap[g]] @ target[g]).max()) for g in range(group.size))
    _gate(report, 'all_k_gauge_covariance_max', gauge_covariance)
    centers, spreads = data.chk.get_wannier_centers(data.bkvec, data.mmn, spreads=True)
    hk = u.conj().swapaxes(-1, -2) @ (bundle.eig[:, :, None] * u)
    hcovariance = []
    for g in range(group.size):
        predicted = target[g] @ (hk.conj() if anti[g] else hk) @ target[g].conj().swapaxes(-1, -2)
        hcovariance.append(float(np.linalg.norm(hk[kmap[g]] - predicted, ord=2, axis=(-2, -1)).max()))
    report.update(hamiltonian_covariance_spectral_max_ev=max(hcovariance),
                  hamiltonian_covariance_spectral_per_operation_ev=hcovariance,
                  target_center_covariance_max_abs_angstrom=float(abs(sym.symmetrize_WCC(centers) - centers).max()),
                  centers_angstrom=centers.tolist(), spreads_angstrom2=spreads.tolist(),
                  total_spread_angstrom2=float(spreads.sum()))
    for ext, digest in bundle.hashes.items():
        if hashlib.sha256(Path(f'{seed}.{ext}').read_bytes()).hexdigest() != digest:
            raise ValueError('Original interface inputs changed during the diagnostic trial')
    if hashlib.sha256((folder / 'diagnostic.json').read_bytes()).hexdigest() != report['source_diagnostic_sha256']:
        raise ValueError('Source diagnostic evidence changed during the trial')
    values = dict(U=u, kpoints=bundle.kpoints, eigenvalues_eV=bundle.eig, lattice=bundle.lattice,
                  centers_angstrom=centers, spreads_angstrom2=spreads, raw_d=raw,
                  reconstructed_d=corrected, target_D=target, kmap=kmap, anti=anti)
    np.savez_compressed(output / 'trial.npz', **values)
    with np.load(output / 'trial.npz', allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], value) for key, value in values.items()):
            raise ValueError('Trial array save/readback changed the numerical result')
    report.update(status='numerical_trial', converged=True, numerical_converged=True,
                  trial_file='trial.npz', trial_sha256=hashlib.sha256((output / 'trial.npz').read_bytes()).hexdigest())
