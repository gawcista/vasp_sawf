"""Read-only, unsymmetrized Hamiltonian analysis of a saved diagnostic trial."""

from contextlib import redirect_stdout
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .bands import evaluate_bands, evaluate_hamiltonian, path_geometry, read_win_path


def _adjoint(value):
    return value.conj().swapaxes(-1, -2)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite analysis input JSON value: {value}')
    def finite(value):
        number = float(value)
        return number if np.isfinite(number) else invalid(value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid, parse_float=finite)


def _difference(delta):
    return dict(max_abs_ev=float(abs(delta).max()), rms_ev=float(np.sqrt(np.mean(abs(delta)**2))))


def _covariance(system, points, operations, target_matrix, sample_arrays=None):
    """Measure H(gk) - D_g(k) H(k)^(*) D_g(k)^dagger without changing H."""
    h = evaluate_hamiltonian(system, points)
    energies = evaluate_bands(system, points)
    rows, entries, spectra, spectral_differences = [], [], [], []
    for g, operation in enumerate(operations):
        image = np.array([operation.transform_k(k) for k in points])
        target = np.array([target_matrix(k, kg, g) for k, kg in zip(points, image, strict=True)])
        error = evaluate_hamiltonian(system, image) - target @ (
            h.conj() if operation.time_reversal else h) @ _adjoint(target)
        entry = abs(error).max(axis=(-2, -1))
        spectral = np.linalg.norm(error, ord=2, axis=(-2, -1))
        difference = abs(evaluate_bands(system, image) - energies).max(axis=-1)
        imax, smax, emax = int(entry.argmax()), int(spectral.argmax()), int(difference.argmax())
        rows.append(dict(operation_index=g, antiunitary=bool(operation.time_reversal),
            entry_max_ev=float(entry[imax]), spectral_max_ev=float(spectral[smax]),
            entry_worst_kpoint=points[imax].tolist(), spectral_worst_kpoint=points[smax].tolist(),
            spectrum_difference_max_ev=float(difference[emax]), spectrum_worst_kpoint=points[emax].tolist()))
        entries.append(entry)
        spectra.append(spectral)
        spectral_differences.append(difference)
    if sample_arrays is not None:
        sample_arrays.update(H_covariance_entry_ev=np.array(entries), H_covariance_spectral_ev=np.array(spectra),
                             spectrum_difference_ev=np.array(spectral_differences))
    return rows


def _sample_path(segments, points_per_segment):
    points = []
    for _, start, _, end in segments:
        segment = np.linspace(start, end, points_per_segment)
        if points and np.array_equal(points[-1], segment[0]):
            segment = segment[1:]
        points.extend(segment)
    return np.asarray(points)


def _target_fourier_parts(blocks, operation):
    """Exact site-permutation matrices and integer-cell phases in the saved target basis."""
    dimension, offset, pieces = sum(block.num_wann for block in blocks), 0, {}
    for block in blocks:
        for ip in range(block.num_points):
            translation = np.asarray(block.T[ip, operation])
            integer = np.rint(translation).astype(int)
            if np.max(abs(translation-integer)) > 1e-7:
                raise ValueError('Target representation contains a noninteger cell phase')
            matrix = pieces.setdefault(tuple(integer), np.zeros((dimension, dimension), complex))
            jp, width = block.atommap[ip, operation], block.num_orbitals
            matrix[offset+jp*width:offset+(jp+1)*width, offset+ip*width:offset+(ip+1)*width] += block.rot_orb[ip, operation]
        offset += block.num_wann
    return np.array(list(pieces)), np.array(list(pieces.values()))


def _antiunitary_pairing(points, energies, operations, target_matrix):
    """Spectral lower bounds implied by fixed-k antiunitary operations squaring to -I."""
    energies = np.sort(energies, axis=1)
    if energies.shape[1] % 2:
        return []
    rows = []
    for g, operation in enumerate(operations):
        if not operation.time_reversal:
            continue
        indices, square_errors = [], []
        for i, k in enumerate(points):
            shift = operation.transform_k(k)-k
            if np.max(abs(shift-np.rint(shift))) > 1e-8:
                continue
            d = target_matrix(k, k, g)
            error = float(abs(d@d.conj()+np.eye(len(d))).max())
            if error <= 1e-6:
                indices.append(i)
                square_errors.append(error)
        if not indices:
            continue
        splittings = energies[indices, 1::2]-energies[indices, ::2]
        worst, _ = np.unravel_index(splittings.argmax(), splittings.shape)
        maximum = float(splittings.max())
        rows.append(dict(operation_index=g, fixed_kpoint_indices_0based=indices,
            fixed_kpoints=np.asarray(points)[indices].tolist(), pair_splittings_ev=splittings.tolist(),
            max_pair_splitting_ev=maximum, minimum_required_max_energy_change_ev=maximum/2,
            worst_kpoint=points[indices[worst]].tolist(), square_plus_identity_entry_max=max(square_errors)))
    return rows


def _fourier_bounds(system, group, blocks, active, values, report):
    from .covariance import covariance_coefficients, certify_covariance
    from .localize import _gate, _target_matrix

    points = np.array([[.137, -.239, .317], [2.137, -1.239, 3.317], [-.313, .217, -.119]])
    h = evaluate_hamiltonian(system, points)
    rows, transforms = [], []
    for g, operation in enumerate(group.symmetries):
        q = np.array([operation.transform_k(e) for e in np.eye(3)])
        translations, matrices = _target_fourier_parts(blocks, g)
        images = points@q
        direct_d = np.array([_target_matrix(blocks, k, kg, g) for k, kg in zip(points, images, strict=True)])
        fourier_d = np.einsum('kt,tmn->kmn', np.exp(2j*np.pi*images@translations.T), matrices)
        _gate(report, f'target_fourier_g{g}_recompute_entry_max', abs(direct_d-fourier_d).max())
        frequencies, coefficients = covariance_coefficients(system.rvec.iRvec, system.get_R_mat('Ham'), q,
            bool(operation.time_reversal), translations, matrices)
        direct_delta = evaluate_hamiltonian(system, images)-direct_d @ (
            h.conj() if operation.time_reversal else h) @ _adjoint(direct_d)
        fourier_delta = np.einsum('kr,rnm->knm', np.exp(2j*np.pi*points@frequencies.T), coefficients)
        _gate(report, f'covariance_fourier_g{g}_recompute_entry_max_ev', abs(direct_delta-fourier_delta).max())
        rows.append(dict(operation_index=g, antiunitary=bool(operation.time_reversal),
            **certify_covariance(frequencies, coefficients, active_axes=active, grid_side=161 if len(active) <= 2 else 17)))
        values.update({f'covariance_frequencies_g{g}': frequencies, f'covariance_coefficients_g{g}': coefficients,
                       f'target_translations_g{g}': translations, f'target_matrices_g{g}': matrices})
        transforms.append(q)
    values['operation_k_transforms'] = np.array(transforms)
    report.update(uniform_covariance_bounds=rows,
        target_fourier_convention='gk=k@Q_g; D_g(k)=sum_t exp(+2*pi*i*(k@Q_g).t)*P_gt; saved integer translations, P matrices and Q transforms retain complete Seitz cell phases',
        covariance_fourier_convention='Delta_g(k)=sum_n exp(+2*pi*i*k.n)*delta_g[n] equals H(gk)-D_g(k) H(k)^(*) D_g(k)^dagger; no coefficient filtering or Hamiltonian averaging')


def _point_sets(bundle, path):
    active = np.flatnonzero(bundle.mesh > 1)
    # Bound the total deterministic grid size in three dimensions.
    side = 41 if len(active) <= 2 else 17
    grid = np.zeros((side**len(active), 3))
    grid[:, active] = np.array(list(itertools.product(np.linspace(-.5, .5, side), repeat=len(active))))
    random = np.zeros((127, 3))
    random[:, active] = np.random.default_rng(7).uniform(-.5, .5, (127, len(active)))
    trim = np.zeros((2**len(active), 3))
    trim[:, active] = np.array(list(itertools.product((0., .5), repeat=len(active))))
    return dict(mesh=np.rint(bundle.kpoints*bundle.mesh)/bundle.mesh, path=path, grid=grid, random=random), trim, active


def _load_trial(folder, seed, report):
    from .localize import _gate, _target_matrix, _target_representation
    from .representation import reconstruct_representation
    from .symmetry import group_residuals, make_symmetrizer, mmn_translation_phases, transport_sewing
    from .trial import _action_metrics, _read_diagnostic

    summary = _json(folder / 'summary.json')
    if (summary.get('schema') != 'sawf-bridge-diagnostic-trial-v1' or summary.get('status') != 'numerical_trial'
            or any(summary.get(key) is not value for key, value in dict(
                diagnostic_only=True, accepted=False, sawf_ready=False, converged=True, numerical_converged=True,
                symmetrize_hr=False).items()) or summary.get('trial_file') != 'trial.npz'):
        raise ValueError('Analysis requires a converged, explicitly unaccepted saved diagnostic trial')
    if _digest(folder / 'trial.npz') != summary.get('trial_sha256'):
        raise ValueError('Saved trial arrays do not match the summary hash')
    # Reuse the complete diagnostic validator on embedded evidence; never read WAVECAR.
    with tempfile.TemporaryDirectory(prefix='sawf-analysis-') as temporary:
        Path(temporary, 'diagnostic.json').write_text(json.dumps(summary['source_diagnostic'], allow_nan=False))
        bundle, source, _, spin, group, kmap, edges, gamma, anchors, binding = _read_diagnostic(seed, Path(temporary))
    for key, value in dict(source_hashes=bundle.hashes, source_num_bands=source['source_num_bands'], spin=spin,
                            mesh=bundle.mesh.tolist(), shape_nk_nb_nw=list(bundle.amn.shape),
                            bands_vasp_1based=bundle.bands_vasp_1based.tolist()).items():
        if summary.get(key) != value:
            raise ValueError(f'Trial summary disagrees with original inputs: {key}')
    coverage = dict(scope='Selected stored-k little groups only', full_closure_certified=False,
        checked_stored_kpoints=source['evaluated_wavecar_kpoints_1based'],
        stored_kpoint_count=source['wavecar']['num_kpoints'], full_mesh_kpoint_count=len(bundle.kpoints))
    if summary.get('source_coefficient_coverage') != coverage:
        raise ValueError('Trial source coefficient coverage disagrees with validated diagnostic evidence')
    with np.load(folder / 'trial.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    nk, nb, nw = bundle.amn.shape
    anti = np.array([op.time_reversal for op in group.symmetries], dtype=bool)
    shapes = dict(U=(nk, nb, nw), kpoints=(nk, 3), eigenvalues_eV=(nk, nb), lattice=(3, 3),
                  centers_angstrom=(nw, 3), spreads_angstrom2=(nw,), kmap=(group.size, nk), anti=(group.size,))
    shapes.update({key: (group.size, nk, nb, nb) for key in ('raw_d', 'reconstructed_d', 'target_D')})
    for key, shape in shapes.items():
        value = arrays.get(key)
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid saved trial dimensions or finite values: {key}')
    for key, expected in dict(kpoints=bundle.kpoints, eigenvalues_eV=bundle.eig, lattice=bundle.lattice,
                              kmap=kmap, anti=anti).items():
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f'Saved trial disagrees with original inputs: {key}')
    for key in ('centers_angstrom', 'spreads_angstrom2'):
        if not np.array_equal(arrays[key], summary.get(key)):
            raise ValueError(f'Saved trial disagrees with summary: {key}')
    phases = mmn_translation_phases(bundle, group)
    raw, _ = transport_sewing(bundle.mmn, bundle.neighbor_indices, kmap, edges, anti, anchors,
                             anchor_k=gamma, edge_phases=phases)
    product, translations, factors = map(np.asarray, group.get_product_table(get_diff=True))
    cocycle = factors[:, :, None] * np.exp(-2j*np.pi*np.einsum(
        'ghki,ghi->ghk', bundle.kpoints[kmap[product]], translations))
    corrected, _ = reconstruct_representation(raw, kmap, anti, product, cocycle)
    for key, rebuilt in (('raw_d', raw), ('reconstructed_d', corrected)):
        _gate(report, key+'_recompute_entry_max', abs(arrays[key]-rebuilt).max())
    sym = make_symmetrizer(bundle, group, arrays['reconstructed_d'])
    orbits = summary['target']['orbits']
    blocks, positions, target_report = _target_representation(sym,
        [orbit['position_fractional'] for orbit in orbits], [orbit['orbital'] for orbit in orbits])
    if sym.num_wann != nw:
        raise ValueError('Target does not retain every original target band')
    if not np.array_equal(positions, summary['target']['column_centers_fractional']):
        raise ValueError('Target column order or positions disagree with reconstructed representation')
    target = np.array([[_target_matrix(blocks, k, bundle.kpoints[kmap[g, i]], g)
                       for i, k in enumerate(bundle.kpoints)] for g in range(group.size)])
    _gate(report, 'target_recompute_entry_max', abs(target-arrays['target_D']).max())
    composition = group_residuals(arrays['target_D'], kmap, anti, product, factors,
                                  kpoints=bundle.kpoints, translations_diff=translations)
    _gate(report, 'target_group_composition_max', composition['group_composition_max'])
    for key in ('U', 'target_D', 'reconstructed_d'):
        _gate(report, key+'_unitarity_max', abs(_adjoint(arrays[key])@arrays[key]-np.eye(nw)).max())
    u = arrays['U']
    for name in ('raw_d', 'reconstructed_d'):
        report[name+'_gauge_covariance_entry_max'] = max(float(abs(arrays[name][g] @ (
            u.conj() if anti[g] else u) - u[kmap[g]] @ target[g]).max()) for g in range(group.size))
    _gate(report, 'all_k_gauge_covariance_max', report['reconstructed_d_gauge_covariance_entry_max'])
    report.update(source_hashes=bundle.hashes, source_num_bands=source['source_num_bands'],
        bands_vasp_1based=bundle.bands_vasp_1based.tolist(), mesh=bundle.mesh.tolist(), shape_nk_nb_nw=list(bundle.amn.shape),
        spin=spin, target=target_report, symmetry_operations=source['symmetry_operations'], diagnostic_input_binding=binding,
        source_diagnostic_sha256=summary['source_diagnostic_sha256'], source_diagnostic=source,
        source_diagnostic_binding='Embedded evidence revalidated against original interface and reconstructed group; original diagnostic file hash retained as historical provenance, file not reread',
        source_coefficient_coverage=coverage,
        raw_action=_action_metrics(arrays['raw_d'], bundle, group, kmap, edges, phases),
        reconstructed_action=_action_metrics(arrays['reconstructed_d'], bundle, group, kmap, edges, phases),
        action_change_entry_max=float(abs(arrays['reconstructed_d']-arrays['raw_d']).max()))
    return bundle, arrays, spin, group, blocks, sym


def _system(data, gauge, spinor):
    from wannierberri.system import System_w90

    data.chk.v_matrix = dict(enumerate(gauge))
    centers, spreads = data.chk.get_wannier_centers(data.bkvec, data.mmn, spreads=True)
    data.chk.wannier_centers_cart, data.chk.wannier_spreads = centers, spreads
    data.irreducible, data.wannierised = False, True
    system = System_w90(data, symmetrize=False, fftlib='numpy', spinor=spinor,
                        berry=False, morb=False, spin=False, wannier_centers_from_chk=True)
    return system, centers, spreads


def _locality(system, centers, lattice):
    vectors, hr = np.asarray(system.rvec.iRvec), system.get_R_mat('Ham')
    distances = np.linalg.norm((vectors@lattice)[:, None, None, :] + centers[None, None, :, :] - centers[None, :, None, :], axis=-1)
    weight = abs(hr)**2
    order = distances.ravel().argsort()
    def weighted_metrics(weights):
        total = float(weights.sum())
        radius = 0. if total == 0 else float(distances.ravel()[order][np.searchsorted(np.cumsum(weights.ravel()[order]), .95*total)])
        return dict(hopping_weight_ev2=total,
                    hopping_weighted_distance_rms_angstrom=0. if total == 0 else float(np.sqrt(np.sum(weights*distances**2)/total)),
                    hopping_weight_95_percent_radius_angstrom=radius)
    hopping = weight.copy()
    hopping[np.all(vectors == 0, axis=1)] *= 1-np.eye(len(centers))
    return dict(real_space_vector_count=len(vectors), **weighted_metrics(weight),
                excluding_onsite_diagonal=dict(**weighted_metrics(hopping),
                    definition='Exclude only R=0, m=n; invariant under a common additive energy reference'),
                maximum_pair_distance_angstrom=float(distances.max()),
                definition='Weights are abs(H_R[m,n])**2, distance is norm(R@lattice + center[n] - center[m]); includes onsite entries'), weight.sum(axis=(1, 2)), distances


def _band_arrays(points, segments, lattice, energy_reference, systems):
    geometry = path_geometry(points, segments, lattice)
    result = dict(kpoints=points, distance=geometry['distance_inv_angstrom'],
        segment_slices=np.array(geometry['segment_slices']), tick_positions=geometry['tick_positions_inv_angstrom'],
        tick_labels=geometry['tick_labels'], energy_reference_ev=energy_reference)
    result.update({key: evaluate_bands(system, points) for key, system in systems.items()})
    return result


def _save(path, arrays):
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as restored:
        if any(not np.array_equal(restored[key], value) for key, value in arrays.items()):
            raise ValueError('Analysis array readback changed the result')


def analyze_trial(trial_dir, output_dir, *, seed=None, points_per_segment=101,
                  energy_reference_ev=0., dft_eigenval=None):
    """Export diagnostic Fourier arrays and measurements; never promote the trial."""
    from .localize import _new_output

    if type(points_per_segment) is not int or points_per_segment < 2 or not np.isfinite(energy_reference_ev):
        raise ValueError('Provide at least two points per segment and a finite energy reference')
    folder = Path(trial_dir).resolve()
    initial = _json(folder / 'summary.json')
    seed = Path(seed if seed is not None else initial['source_seed']).resolve()
    dft_files = [] if dft_eigenval is None else ([dft_eigenval] if isinstance(dft_eigenval, (str, Path)) else list(dft_eigenval))
    dft_files = [Path(path).resolve() for path in dft_files]
    inputs = [folder / 'summary.json', folder / 'trial.npz', *[Path(f'{seed}.{ext}') for ext in ('win', 'amn', 'mmn', 'eig')], *dft_files]
    output = _new_output(output_dir, inputs)
    report = dict(schema='sawf-bridge-diagnostic-analysis-v1', status='not_ready', diagnostic_only=True,
        accepted=False, sawf_ready=False, symmetrize_hr=False, wavefunction_bytes_read=0,
        source_seed=str(seed), source_trial=str(folder), physical_acceptance_status='not_assessed',
        limitations=['Diagnostic arrays do not establish source closure or physical acceptance.',
                     'Finite path, grid and random samples are not a supremum bound.',
                     'Ordinary versus SAWF differences are not DFT interpolation errors.',
                     'Raw Bloch-action errors, reconstructed-action errors, gauge covariance and H covariance are distinct.'])
    started, transcript = time.perf_counter(), io.StringIO()
    try:
        hashes = {str(path): _digest(path) for path in inputs}
        report['input_sha256'] = hashes
        with redirect_stdout(transcript):
            _analyze(folder, seed, output, report, points_per_segment, energy_reference_ev, dft_files)
        if any(_digest(path) != digest for path, digest in hashes.items()):
            raise ValueError('Analysis inputs changed during execution')
        report.update(status='diagnostic_analysis_complete', arrays_readback_exact=True,
                      output_sha256={path.name: _digest(path) for path in output.glob('*.npz')})
    except Exception as error:
        report.update(error=str(error), last_output_lines=transcript.getvalue().splitlines()[-15:])
        for name in ('analysis.npz', 'bands.npz', 'dft_bands.npz'):
            (output / name).unlink(missing_ok=True)
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def _analyze(folder, seed, output, report, points_per_segment, reference, dft_files,
             *, saved_model=None, ordinary_report=None):
    from .inputs import load_wannier_data
    from .localize import _gate, _ordinary_initial_gauge, _target_matrix

    bundle, saved, spin, group, blocks, sym = _load_trial(folder, seed, report)
    data, _ = load_wannier_data(seed, source_nb=report['source_num_bands'])
    if saved_model is None:
        ordinary, _, ordinary_report = _ordinary_initial_gauge(data)
    else:
        for key, value in saved.items():
            if key not in saved_model or not np.array_equal(value, saved_model[key]):
                raise ValueError(f'Saved model disagrees with validated embedded trial: {key}')
        required = ('ordinary_U', 'R', 'H_R', 'ordinary_R', 'ordinary_H_R',
                    'ordinary_centers_angstrom', 'ordinary_spreads_angstrom2')
        if any(key not in saved_model for key in required):
            raise ValueError('Saved model is missing ordinary gauge or Fourier analysis arrays')
        ordinary = saved_model['ordinary_U']
        if ordinary.shape != saved['U'].shape or not np.isfinite(ordinary).all():
            raise ValueError('Invalid saved ordinary gauge shape or values')
        _gate(report, 'saved_ordinary_unitarity_max', abs(_adjoint(ordinary)@ordinary-np.eye(ordinary.shape[-1])).max(),
              256*np.finfo(float).eps*ordinary.shape[-1])
        if not isinstance(ordinary_report, dict) or ordinary_report.get('converged') is not True:
            raise ValueError('Saved ordinary localization lacks convergence evidence')
    ordinary_system, ordinary_centers, ordinary_spreads = _system(data, ordinary, spin['spinor'])
    data, _ = load_wannier_data(seed, source_nb=report['source_num_bands'])
    system, centers, spreads = _system(data, saved['U'], spin['spinor'])
    _gate(report, 'centers_recompute_max_angstrom', abs(centers-saved['centers_angstrom']).max())
    _gate(report, 'spreads_recompute_max_angstrom2', abs(spreads-saved['spreads_angstrom2']).max())
    if saved_model is not None:
        from .model_analysis import _check_saved_system
        ordinary_system = _check_saved_system(saved_model, ordinary_system, ordinary_centers, ordinary_spreads,
                                              'ordinary_', report)
        system = _check_saved_system(saved_model, system, centers, spreads, '', report)
        ordinary_centers, ordinary_spreads = (saved_model['ordinary_centers_angstrom'],
                                              saved_model['ordinary_spreads_angstrom2'])
        centers, spreads = saved['centers_angstrom'], saved['spreads_angstrom2']
    report.update(ordinary_localization=ordinary_report, centers_angstrom=centers.tolist(), spreads_angstrom2=spreads.tolist(),
        total_spread_angstrom2=float(spreads.sum()), ordinary_centers_angstrom=ordinary_centers.tolist(),
        ordinary_spreads_angstrom2=ordinary_spreads.tolist(),
        target_center_covariance_max_abs_angstrom=float(abs(sym.symmetrize_WCC(centers)-centers).max()),
        system_construction='WannierBerri System_w90, symmetrize=False, full native mesh and original EIG; no spatial/TR averaging',
        fourier_convention='H(k)=sum_R exp(+2*pi*i*k.R)*H_R; reduced reciprocal k, integer R; H_R shape (NR,NW,NW) already includes pair-dependent WS weights; no extra degeneracy division; eV and angstrom units')
    segments = read_win_path(Path(f'{seed}.win'))
    path = _sample_path(segments, points_per_segment)
    systems = dict(wannier=ordinary_system, sawf=system)
    bands = _band_arrays(path, segments, bundle.lattice, reference, systems)
    sets, trim, active = _point_sets(bundle, path)
    report.update(sampling=dict(active_axes_0based=active.tolist(), grid_points_per_axis=41 if len(active) <= 2 else 17,
        random_seed=7, random_count=127, points_per_segment=points_per_segment, counts={key: len(points) for key, points in sets.items()},
        inactive_axes='Fixed at zero reduced coordinates'), mesh_checks={}, covariance={}, ordinary_vs_sawf={}, locality={})
    values = dict(saved, R=system.rvec.iRvec, H_R=system.get_R_mat('Ham'), ordinary_U=ordinary,
        ordinary_R=ordinary_system.rvec.iRvec, ordinary_H_R=ordinary_system.get_R_mat('Ham'),
        ordinary_centers_angstrom=ordinary_centers, ordinary_spreads_angstrom2=ordinary_spreads,
        bands_vasp_1based=bundle.bands_vasp_1based, mesh=bundle.mesh)
    for label, model, gauge, model_centers in (('sawf', system, saved['U'], centers),
                                              ('wannier', ordinary_system, ordinary, ordinary_centers)):
        raw_h = _adjoint(gauge) @ (bundle.eig[:, :, None]*gauge)
        mesh_h = evaluate_hamiltonian(model, sets['mesh'])
        metrics = dict(H_roundtrip_max_ev=float(abs(mesh_h-raw_h).max()),
            spectrum_error_max_ev=float(abs(evaluate_bands(model, sets['mesh'])-np.sort(bundle.eig, axis=1)).max()),
            raw_H_hermiticity_max_ev=float(abs(raw_h-_adjoint(raw_h)).max()))
        _gate(metrics, 'H_roundtrip_max_ev', metrics['H_roundtrip_max_ev'])
        _gate(metrics, 'spectrum_error_max_ev', metrics['spectrum_error_max_ev'])
        report['mesh_checks'][label] = metrics
        report['locality'][label], weights, distances = _locality(model, model_centers, bundle.lattice)
        values[label+'_hopping_weight_per_R_ev2'], values[label+'_pair_distances_angstrom'] = weights, distances
    target = lambda k, image, g: _target_matrix(blocks, k, image, g)
    sampled, offset, slices = {}, 0, []
    report['hermiticity_max_ev'] = {}
    report['antiunitary_pairing'] = dict(source_mesh=_antiunitary_pairing(
        bundle.kpoints, bundle.eig, group.symmetries, target))
    report['antiunitary_pairing_definition'] = ('Only fixed k modulo integers with D(k) D(k)* = -I within the existing '
        '1e-6 matrix check are included. Half the largest adjacent sorted pair splitting is a lower bound on the '
        'maximum eigenvalue change required for exact pairing, not an acceptance test.')
    for name, points in sets.items():
        measured = {}
        report['covariance'][name] = _covariance(system, points, group.symmetries, target, measured)
        for key, value in measured.items():
            sampled.setdefault(key, []).append(value)
        slices.append((offset, offset+len(points)))
        offset += len(points)
        report['ordinary_vs_sawf'][name] = _difference(evaluate_bands(system, points)-evaluate_bands(ordinary_system, points))
        report['antiunitary_pairing'][name] = _antiunitary_pairing(points, evaluate_bands(system, points), group.symmetries, target)
        h = evaluate_hamiltonian(system, points)
        report['hermiticity_max_ev'][name] = float(abs(h-_adjoint(h)).max())
    values.update({key: np.concatenate(value, axis=1) for key, value in sampled.items()})
    values.update(check_kpoints=np.concatenate(list(sets.values())), check_set_names=np.array(list(sets)),
                  check_set_slices=np.array(slices), trim_kpoints=trim, trim_energies_ev=evaluate_bands(system, trim))
    if spin['spinor']:
        splits = values['trim_energies_ev'][:, 1::2]-values['trim_energies_ev'][:, ::2]
        values['trim_pair_splittings_ev'] = splits
        report['trim'] = dict(kpoints=trim.tolist(), pair_splittings_ev=splits.tolist(), max_abs_split_ev=float(abs(splits).max()))
    if dft_files:
        from .bands import read_dft_eigenval
        records = [read_dft_eigenval(path, bundle.bands_vasp_1based.tolist(), spin_channel=spin['spin_channel'],
                                    source_ispin=spin['source_ispin']) for path in dft_files]
        dft_points = np.concatenate([record['kpoints'] for record in records])
        dft = _band_arrays(dft_points, segments, bundle.lattice, reference, systems)
        dft['dft'] = np.concatenate([record['eigenvalues_eV'] for record in records])
        _save(output / 'dft_bands.npz', dft)
        report['dft_comparison'] = {key: _difference(dft[key]-np.sort(dft['dft'], axis=1)) for key in systems}
        report['dft_reference_role'] = 'comparison_only_not_same_settings_validation'
        report['dft_reference_files'] = [str(path) for path in dft_files]
        gamma = np.flatnonzero(np.max(abs(dft_points-np.rint(dft_points)), axis=1) <= 1e-8)
        if len(gamma):
            first = int(gamma[0])
            dft_reference = float(dft['dft'][first, -1])
            alignment = dict(gamma_kpoint_indices_0based=gamma.tolist(), reference_index_0based=first,
                band_vasp_1based=int(bundle.bands_vasp_1based[-1]), dft_reference_ev=dft_reference,
                definition='Align the highest selected band at the first Gamma record; shift_ev is added to model energies for comparison only; original arrays and Hamiltonians are unchanged',
                models={})
            for key in systems:
                model_reference = float(dft[key][first, -1])
                shift = dft_reference-model_reference
                alignment['models'][key] = dict(model_reference_ev=model_reference, shift_ev=shift,
                    **_difference(dft[key]+shift-np.sort(dft['dft'], axis=1)))
            report['gamma_upper_band_alignment'] = alignment
    _fourier_bounds(system, group, blocks, active, values, report)
    if saved_model is not None:
        from .model_analysis import _comparison
        comparison = _comparison(bundle, saved_model, ordinary, ordinary_centers, ordinary_system,
                                 system, group, sym, target, sets)
        comparison.update(source_model=report['source_model'], source_acceptance=report['source_acceptance'])
        (output/'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')
    _save(output / 'analysis.npz', values)
    _save(output / 'bands.npz', bands)
