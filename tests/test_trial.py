"""Diagnostic localization does not confer source acceptance or read wavefunctions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _diagnostic_inputs(tmp_path):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.spin import _outcar_spin_context, _spacegroup_from_context
    from vasp_sawf.symmetry import _cell_from_win

    seed = _atomic_inputs(tmp_path / 'inputs', 1, 1)
    (seed.parent / 'WAVECAR').unlink()
    bundle = read_inputs(seed, source_nb=1)
    positions, types, _ = _cell_from_win(Path(f'{seed}.win').read_text(), bundle.lattice)
    outcar = (seed.parent / 'OUTCAR').read_bytes()
    context = _outcar_spin_context(outcar.decode(), len(positions))
    group = _spacegroup_from_context(bundle, positions, types, context)
    operations = [dict(index_0based=i, rotation=op.rotation.tolist(), translation=op.translation.tolist(),
                       time_reversal=bool(op.time_reversal), spinor_rotation_real=op.spinor_rotation.real.tolist(),
                       spinor_rotation_imag=op.spinor_rotation.imag.tolist())
                  for i, op in enumerate(group.symmetries)]
    matrices = np.full((group.size, 1, 1), 1.001)
    indices = list(range(group.size))
    point = dict(wavecar_kpoint_1based=1, interface_kpoint_1based=1, operation_indices_0based=indices,
                 matrices_real=matrices.tolist(), matrices_imag=np.zeros_like(matrices).tolist(),
                 kpoint_fractional=[0., 0., 0.], energies_ev=[0.], coefficient_closure_relative_max=.02,
                 operations=[], group={})
    diagnostic = dict(schema='sawf-bridge-diagnostic-v1', status='diagnostic_complete',
        diagnostic_only=True, sawf_ready=False, numerical_checks_passed=False,
        gauge_status='diagnosed_not_certified', read_accounting_complete=True,
        wavecar_interface_energy_tolerance_ev=1e-6,
        source_hashes=bundle.hashes, source_num_bands=1, shape_nk_nb_nw=[1, 1, 1], mesh=[1, 1, 1],
        bands_vasp_1based=[1], spin=dict(spinor=False, source_ispin=1, spin_channel=1,
            antiunitary_kind='orbital_complex_conjugation', time_reversal_square=1),
        outcar=dict(spin_context=context, sha256=hashlib.sha256(outcar).hexdigest()),
        wavecar=dict(spin_channels=1, num_kpoints=1),
        symmetry_operations=dict(total=group.size, antiunitary=sum(op.time_reversal for op in group.symmetries),
                                 operations=operations),
        gamma_anchor={key: value for key, value in point.items() if key not in
                      ('kpoint_fractional', 'energies_ev', 'operations', 'group')},
        diagnostic_kpoints=[point], evaluated_wavecar_kpoints_1based=[1],
        planned_wavecar_kpoints_1based=[1], residuals={'source_raw_marker': .02})
    folder = tmp_path / 'symmetry'
    folder.mkdir()
    (folder / 'diagnostic.json').write_text(json.dumps(diagnostic))
    return seed, folder, diagnostic


@pytest.mark.parametrize('change', ['hash', 'incomplete', 'ready', 'gamma', 'operation', 'shape',
                                   'bands', 'point', 'spin', 'spinor_operation', 'seed'])
def test_trial_rejects_unbound_or_inconsistent_source_evidence(tmp_path, change):
    from vasp_sawf.trial import run_trial

    seed, folder, record = _diagnostic_inputs(tmp_path)
    if change == 'hash':
        record['source_hashes']['mmn'] = 'different'
    elif change == 'incomplete':
        record['read_accounting_complete'] = False
    elif change == 'ready':
        record['sawf_ready'] = True
    elif change == 'gamma':
        record['gamma_anchor']['interface_kpoint_1based'] = 2
    elif change == 'operation':
        record['symmetry_operations']['operations'][0]['translation'][0] = .25
    elif change == 'shape':
        record['shape_nk_nb_nw'] = [1, 2, 2]
    elif change == 'bands':
        record['bands_vasp_1based'] = [2]
    elif change == 'point':
        record['diagnostic_kpoints'][0]['kpoint_fractional'] = [.25, 0., 0.]
    elif change == 'spin':
        record['spin']['source_ispin'] = 2
    elif change == 'spinor_operation':
        record['symmetry_operations']['operations'][0]['spinor_rotation_imag'][0][0] += 1e-8
    else:
        with Path(f'{seed}.win').open('a') as handle:
            handle.write('\n! changed source bytes\n')
    (folder / 'diagnostic.json').write_text(json.dumps(record))
    output = tmp_path / 'trial'
    with pytest.raises(ValueError):
        run_trial(seed, folder, output, center=[0, 0, 0], orbital='s', num_iter=8)
    report = json.loads((output / 'summary.json').read_text())
    assert report['accepted'] is False and report['sawf_ready'] is False
    assert report['converged'] is False and report['error']
    assert list(output.iterdir()) == [output / 'summary.json']


def test_trial_numerical_success_preserves_raw_errors_and_cannot_export_ready_model(tmp_path):
    from vasp_sawf.trial import run_trial

    seed, folder, record = _diagnostic_inputs(tmp_path)
    before = {p: p.read_bytes() for p in seed.parent.iterdir()}
    diagnostic_before = (folder / 'diagnostic.json').read_bytes()
    output = tmp_path / 'trial'
    report = run_trial(seed, folder, output, center=[0, 0, 0], orbital='s', num_iter=8)
    assert report['status'] == 'numerical_trial'
    assert report['numerical_converged'] and report['converged']
    assert report['diagnostic_only'] and not report['accepted'] and not report['sawf_ready']
    assert report['wavefunction_bytes_read'] == 0
    assert report['source_coefficient_coverage']['full_closure_certified'] is False
    assert report['source_diagnostic_sha256'] == hashlib.sha256(diagnostic_before).hexdigest()
    assert report['raw_action']['unitarity_entry_max'] > 1e-3
    assert report['reconstructed_action']['unitarity_entry_max'] < 1e-12
    assert report['source_diagnostic']['residuals'] == record['residuals']
    assert {p.name for p in output.iterdir()} == {'trial.npz', 'summary.json'}
    with np.load(output / 'trial.npz', allow_pickle=False) as arrays:
        assert set(arrays.files) == {'U', 'kpoints', 'eigenvalues_eV', 'lattice', 'centers_angstrom',
            'spreads_angstrom2', 'raw_d', 'reconstructed_d', 'target_D', 'kmap', 'anti'}
        np.testing.assert_allclose(arrays['U'].conj().swapaxes(-1, -2) @ arrays['U'], 1., atol=1e-12)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (folder / 'diagnostic.json').read_bytes() == diagnostic_before


def test_trial_output_protections_and_no_dft_export(tmp_path):
    from vasp_sawf.trial import run_trial

    seed, folder, _ = _diagnostic_inputs(tmp_path)
    existing = tmp_path / 'existing'
    existing.mkdir()
    marker = existing / 'keep'
    marker.write_text('unchanged')
    for output in (existing, seed.parent / 'nested', folder / 'nested'):
        with pytest.raises(ValueError, match='output|directory|protected|overwriting'):
            run_trial(seed, folder, output, center=[0, 0, 0], orbital='s')
    assert marker.read_text() == 'unchanged'
    with pytest.raises(ValueError, match='DFT|dft'):
        run_trial(seed, folder, tmp_path / 'dft', center=[0, 0, 0], orbital='s',
                  dft_eigenval=tmp_path / 'EIGENVAL')


def test_trial_optimizer_failure_leaves_only_unaccepted_summary(tmp_path, monkeypatch):
    import vasp_sawf.localize as localize
    from vasp_sawf.fullmesh import FullMeshConvergenceError
    from vasp_sawf.trial import run_trial

    seed, folder, _ = _diagnostic_inputs(tmp_path)
    def fail(*args, **kwargs):
        raise FullMeshConvergenceError({'numerical_converged': False,
            'final_fresh_chart_gradient_max': .01, 'parameters': {'gradient_tol': 1e-8}})
    monkeypatch.setattr(localize, 'wannierise_full_mesh', fail)
    output = tmp_path / 'trial'
    with pytest.raises(RuntimeError, match='converge'):
        run_trial(seed, folder, output, center=[0, 0, 0], orbital='s', num_iter=8)
    report = json.loads((output / 'summary.json').read_text())
    assert not report['accepted'] and not report['sawf_ready'] and not report['converged']
    assert report['localisation']['final_fresh_chart_gradient_max'] == .01
    assert {p.name for p in output.iterdir()} == {'summary.json'}


def test_trial_loader_uses_recorded_source_energy_tolerance_and_periodic_coordinates(tmp_path):
    from vasp_sawf.trial import _read_diagnostic

    seed, folder, record = _diagnostic_inputs(tmp_path)
    record['wavecar_interface_energy_tolerance_ev'] = 2e-4
    record['diagnostic_kpoints'][0]['energies_ev'] = [1e-4]
    record['diagnostic_kpoints'][0]['kpoint_fractional'] = [1., -2., 3.]
    record['symmetry_operations']['operations'][0]['spinor_rotation_imag'][0][0] += 1e-17
    (folder / 'diagnostic.json').write_text(json.dumps(record))
    loaded = _read_diagnostic(seed, folder)
    assert loaded[0].kpoints.tolist() == [[0., 0., 0.]]
    assert loaded[-1]['selected_kpoint_integer_folding'] == [[1, -2, 3]]
    assert loaded[-1]['selected_energy_difference_max_ev'] == 1e-4


@pytest.mark.parametrize('tolerance', [None, 0., -1., True, float('inf')])
def test_trial_loader_rejects_invalid_recorded_energy_tolerance(tmp_path, tolerance):
    from vasp_sawf.trial import _read_diagnostic

    seed, folder, record = _diagnostic_inputs(tmp_path)
    record['wavecar_interface_energy_tolerance_ev'] = tolerance
    (folder / 'diagnostic.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        _read_diagnostic(seed, folder)


def test_trial_rejects_target_group_inconsistency_before_localization(tmp_path, monkeypatch):
    import vasp_sawf.localize as localize
    from vasp_sawf.trial import run_trial

    seed, folder, _ = _diagnostic_inputs(tmp_path)
    original = localize._target_matrix
    monkeypatch.setattr(localize, '_target_matrix', lambda blocks, k, image, g:
                        1j * original(blocks, k, image, g) if g == 0 else original(blocks, k, image, g))
    with pytest.raises(ValueError, match='target_group'):
        run_trial(seed, folder, tmp_path / 'trial', center=[0, 0, 0], orbital='s', num_iter=8)


def test_trial_rechecks_final_complete_group_gauge_covariance(tmp_path, monkeypatch):
    import vasp_sawf.localize as localize
    from vasp_sawf.trial import run_trial

    seed, folder, _ = _diagnostic_inputs(tmp_path)
    original = localize.wannierise_full_mesh
    def altered(data, *args, **kwargs):
        report = original(data, *args, **kwargs)
        data.chk.v_matrix[0] *= 1j
        return report
    monkeypatch.setattr(localize, 'wannierise_full_mesh', altered)
    with pytest.raises(ValueError, match='all_k_gauge_covariance'):
        run_trial(seed, folder, tmp_path / 'trial', center=[0, 0, 0], orbital='s', num_iter=8)
    assert not (tmp_path / 'trial' / 'trial.npz').exists()


def test_trial_rejects_overflowing_unused_json_metric_before_writing_arrays(tmp_path):
    from vasp_sawf.trial import run_trial

    seed, folder, record = _diagnostic_inputs(tmp_path)
    record['unused_nested'] = {'metric': 'OVERFLOW'}
    text = json.dumps(record).replace('"OVERFLOW"', '1e999')
    (folder / 'diagnostic.json').write_text(text)
    output = tmp_path / 'trial'
    with pytest.raises(ValueError):
        run_trial(seed, folder, output, center=[0, 0, 0], orbital='s', num_iter=8)
    assert {p.name for p in output.iterdir()} == {'summary.json'}
    report = json.loads((output / 'summary.json').read_text())
    assert report['status'] == 'not_ready'
    assert not report['accepted'] and not report['sawf_ready'] and not report['converged']
    assert 'Nonfinite diagnostic JSON' in report['error']
    assert (folder / 'diagnostic.json').read_text() == text
