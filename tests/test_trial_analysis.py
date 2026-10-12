"""Independent Fourier checks and the diagnostic analysis trust boundary."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _trial(tmp_path):
    from test_trial import _diagnostic_inputs
    from vasp_sawf.trial import run_trial

    seed, folder, record = _diagnostic_inputs(tmp_path)
    win = Path(f'{seed}.win')
    with win.open('a') as handle:
        handle.write('begin kpoint_path\nG 0 0 0 X .5 0 0\nY 0 .5 0 G 0 0 0\nend kpoint_path\n')
    record['source_hashes']['win'] = hashlib.sha256(win.read_bytes()).hexdigest()
    (folder / 'diagnostic.json').write_text(json.dumps(record))
    trial = tmp_path / 'trial'
    run_trial(seed, folder, trial, center=[0, 0, 0], orbital='s', num_iter=8)
    return seed, trial


def test_fourier_covariance_retains_antiunitary_and_integer_cell_phase():
    from vasp_sawf.analysis import _covariance
    from vasp_sawf.bands import evaluate_hamiltonian

    # Two sites exchanged by a half translation followed by complex conjugation.
    # H_01(k) = 1 + exp(-2 pi i k); D_01(k) = exp(+2 pi i k).
    hr = np.array([[[0, 1], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [1, 0]]], complex)
    system = SimpleNamespace(num_wann=2, rvec=SimpleNamespace(iRvec=np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])),
                             get_R_mat=lambda name: hr)
    points = np.array([[.25, 0, 0], [.75, 0, 0], [1.25, 0, 0]])
    actual = evaluate_hamiltonian(system, points)
    np.testing.assert_allclose(actual[0], [[0, 1-1j], [1+1j, 0]], atol=1e-14)
    np.testing.assert_allclose(actual[0], actual[2], atol=1e-14)
    operation = SimpleNamespace(time_reversal=True, transform_k=lambda k: -k)
    target = lambda k, image, g: np.array([[0, np.exp(2j*np.pi*k[0])], [1, 0]])
    row = _covariance(system, points, [operation], target)[0]
    assert row['entry_max_ev'] < 1e-14 and row['spectral_max_ev'] < 1e-14
    assert row['spectrum_difference_max_ev'] < 1e-14
    wrong = _covariance(system, points, [operation], lambda k, image, g: np.array([[0, 1], [1, 0]]))[0]
    assert wrong['entry_max_ev'] == pytest.approx(2.)


def test_analysis_preserves_inputs_exports_standalone_arrays_and_stays_unaccepted(tmp_path):
    from vasp_sawf.analysis import analyze_trial

    seed, trial = _trial(tmp_path)
    before = {p: p.read_bytes() for folder in (seed.parent, trial) for p in folder.iterdir()}
    output = tmp_path / 'analysis'
    report = analyze_trial(trial, output, points_per_segment=5, energy_reference_ev=.25)
    assert report['status'] == 'diagnostic_analysis_complete'
    assert report['diagnostic_only'] and not report['accepted'] and not report['sawf_ready']
    assert not report['symmetrize_hr'] and report['wavefunction_bytes_read'] == 0
    assert {p.name for p in output.iterdir()} == {'analysis.npz', 'bands.npz', 'summary.json'}
    assert report['raw_action']['unitarity_entry_max'] > 1e-3
    assert report['reconstructed_action']['unitarity_entry_max'] < 1e-12
    assert report['mesh_checks']['sawf']['spectrum_error_max_ev'] < 1e-12
    assert report['uniform_covariance_bounds']
    assert all(row['uniform_upper_ev'] < 1e-8 for row in report['uniform_covariance_bounds'])
    with np.load(output / 'analysis.npz', allow_pickle=False) as arrays:
        for key in ('U', 'R', 'H_R', 'kpoints', 'eigenvalues_eV', 'lattice', 'centers_angstrom',
                    'spreads_angstrom2', 'raw_d', 'reconstructed_d', 'target_D', 'kmap', 'anti',
                    'ordinary_U', 'ordinary_R', 'ordinary_H_R'):
            assert key in arrays
        assert arrays['H_R'].shape == (len(arrays['R']), 1, 1)
        np.testing.assert_allclose(arrays['H_R'], 0., atol=1e-14)
    with np.load(output / 'bands.npz', allow_pickle=False) as bands:
        assert bands['segment_slices'].tolist() == [[0, 5], [5, 10]]
        assert bands['tick_labels'].tolist() == ['G', 'X|Y', 'G']
        assert 'dft' not in bands
        assert float(bands['energy_reference_ev']) == .25
        assert bands['sawf'].shape == (10, 1)
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize('change', ['source', 'trial', 'status', 'finite', 'shape', 'kmap', 'target', 'action', 'gauge'])
def test_analysis_rejects_mutations_without_exporting_model(tmp_path, change):
    from vasp_sawf.analysis import analyze_trial

    seed, trial = _trial(tmp_path)
    summary = json.loads((trial / 'summary.json').read_text())
    if change == 'source':
        with Path(f'{seed}.eig').open('a') as handle:
            handle.write('\n')
    elif change == 'trial':
        with (trial / 'trial.npz').open('ab') as handle:
            handle.write(b'mutation')
    elif change == 'status':
        summary['accepted'] = True
    else:
        with np.load(trial / 'trial.npz', allow_pickle=False) as archive:
            arrays = dict(archive)
        if change == 'finite':
            arrays['U'][0, 0, 0] = np.nan
        elif change == 'shape':
            arrays['U'] = arrays['U'][:, :0]
        elif change == 'kmap':
            arrays['kmap'][0, 0] = 10
        elif change == 'target':
            arrays['target_D'][0] *= 1j
        elif change == 'action':
            arrays['raw_d'][0] *= 1j
        else:
            arrays['U'] *= 1j
        np.savez_compressed(trial / 'trial.npz', **arrays)
        summary['trial_sha256'] = hashlib.sha256((trial / 'trial.npz').read_bytes()).hexdigest()
    (trial / 'summary.json').write_text(json.dumps(summary))
    output = tmp_path / 'analysis'
    with pytest.raises(ValueError):
        analyze_trial(trial, output)
    if output.exists():
        assert {p.name for p in output.iterdir()} == {'summary.json'}
        report = json.loads((output / 'summary.json').read_text())
        assert report['accepted'] is False and report['sawf_ready'] is False


def test_analysis_rejects_protected_or_existing_output(tmp_path):
    from vasp_sawf.analysis import analyze_trial

    seed, trial = _trial(tmp_path)
    for output in (trial / 'child', seed.parent / 'child', trial):
        with pytest.raises(ValueError, match='output|directory|protected|overwriting'):
            analyze_trial(trial, output)


@pytest.mark.parametrize('change', ['certification', 'indices', 'count'])
def test_analysis_rejects_forged_source_coverage_before_export(tmp_path, change):
    from vasp_sawf.analysis import analyze_trial

    _, trial = _trial(tmp_path)
    summary = json.loads((trial / 'summary.json').read_text())
    coverage = summary['source_coefficient_coverage']
    if change == 'certification':
        coverage['full_closure_certified'] = True
    elif change == 'indices':
        coverage['checked_stored_kpoints'] = [2]
    else:
        coverage['full_mesh_kpoint_count'] = 2
    (trial / 'summary.json').write_text(json.dumps(summary))
    output = tmp_path / 'analysis'
    with pytest.raises(ValueError, match='coverage'):
        analyze_trial(trial, output, points_per_segment=3)
    assert {p.name for p in output.iterdir()} == {'summary.json'}
    report = json.loads((output / 'summary.json').read_text())
    assert report['accepted'] is False and report['sawf_ready'] is False


def test_analysis_retains_actual_split_dft_path_without_resampling(tmp_path):
    from vasp_sawf.analysis import analyze_trial

    seed, trial = _trial(tmp_path)
    references = []
    for index, points in enumerate(([[0, 0, 0], [.5, 0, 0]], [[0, .5, 0], [0, 0, 0]])):
        path = seed.parent / f'EIGENVAL.{index}'
        path.write_text('1 1 1 1\nheader\nheader\nheader\nheader\n2 2 1\n\n' + ''.join(
            f' {k[0]} {k[1]} {k[2]} 1\n1 {index+1}.0 1\n\n' for k in points))
        references.append(path)
    output = tmp_path / 'analysis'
    report = analyze_trial(trial, output, points_per_segment=7, dft_eigenval=references)
    with np.load(output / 'dft_bands.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['dft'], [[1.], [1.], [2.], [2.]])
        assert arrays['segment_slices'].tolist() == [[0, 2], [2, 4]]
        np.testing.assert_allclose(arrays['sawf'], 0., atol=1e-14)
    with np.load(output / 'bands.npz', allow_pickle=False) as arrays:
        assert arrays['kpoints'].shape == (14, 3)
        assert 'dft' not in arrays
    assert report['dft_comparison']['sawf']['max_abs_ev'] == pytest.approx(2.)
    assert report['dft_reference_role'] == 'comparison_only_not_same_settings_validation'
    assert not report['accepted'] and not report['sawf_ready']


def test_analysis_removes_exports_if_source_changes_during_analysis(tmp_path, monkeypatch):
    from vasp_sawf.analysis import analyze_trial
    import vasp_sawf.localize as localize

    seed, trial = _trial(tmp_path)
    original = localize._ordinary_initial_gauge
    def changing_source(*args, **kwargs):
        result = original(*args, **kwargs)
        with Path(f'{seed}.amn').open('a') as handle:
            handle.write('\n')
        return result
    monkeypatch.setattr(localize, '_ordinary_initial_gauge', changing_source)
    output = tmp_path / 'analysis'
    with pytest.raises(ValueError, match='changed during'):
        analyze_trial(trial, output, points_per_segment=3)
    assert {p.name for p in output.iterdir()} == {'summary.json'}
    assert not json.loads((output / 'summary.json').read_text())['accepted']


def test_sampling_uses_all_active_directions_and_preserves_connected_segments():
    from vasp_sawf.analysis import _point_sets, _sample_path

    segments = [('G', [0, 0, 0], 'X', [.5, 0, 0]), ('X', [.5, 0, 0], 'M', [.5, 0, .5])]
    path = _sample_path(segments, 3)
    np.testing.assert_array_equal(path, [[0, 0, 0], [.25, 0, 0], [.5, 0, 0], [.5, 0, .25], [.5, 0, .5]])
    bundle = SimpleNamespace(mesh=np.array([6, 1, 6]), kpoints=np.zeros((1, 3)))
    sets, trim, active = _point_sets(bundle, path)
    assert active.tolist() == [0, 2]
    assert sets['grid'].shape == (1681, 3)
    np.testing.assert_array_equal(trim, [[0, 0, 0], [0, 0, .5], [.5, 0, 0], [.5, 0, .5]])
    assert np.all(sets['random'][:, 1] == 0.)
    assert np.ptp(sets['random'][:, 2]) > .9


def test_hopping_locality_excludes_onsite_diagonal_energy_reference():
    from vasp_sawf.analysis import _locality

    hr = np.array([[[1.]], [[100.]], [[1.]]])
    system = SimpleNamespace(rvec=SimpleNamespace(iRvec=np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])),
                             get_R_mat=lambda name: hr)
    result, _, _ = _locality(system, np.zeros((1, 3)), np.eye(3))
    hopping = result['excluding_onsite_diagonal']
    assert hopping['hopping_weight_ev2'] == 2.
    assert hopping['hopping_weighted_distance_rms_angstrom'] == 1.
    assert hopping['hopping_weight_95_percent_radius_angstrom'] == 1.
    hr[1, 0, 0] = -300.
    changed, _, _ = _locality(system, np.zeros((1, 3)), np.eye(3))
    assert changed['excluding_onsite_diagonal'] == hopping


def test_target_fourier_parts_preserve_cell_phases_and_block_column_order():
    from vasp_sawf.analysis import _target_fourier_parts

    exchange = SimpleNamespace(num_points=2, num_orbitals=1, num_wann=2,
        atommap=np.array([[1], [0]]), T=np.array([[[0, 0, 0]], [[-1, 0, 0]]]),
        rot_orb=np.ones((2, 1, 1, 1)))
    local = SimpleNamespace(num_points=1, num_orbitals=1, num_wann=1,
        atommap=np.array([[0]]), T=np.array([[[0, 1, 0]]]), rot_orb=np.array([[[[1j]]]]))
    translations, matrices = _target_fourier_parts([exchange, local], 0)
    q = np.array([.25, .125, 0])
    result = np.einsum('t,tmn->mn', np.exp(2j*np.pi*translations@(-q)), matrices)
    expected = np.array([[0, 1j, 0], [1, 0, 0], [0, 0, (1+1j)/np.sqrt(2)]])
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_antiunitary_pairing_reports_required_energy_changes_and_excludes_positive_square():
    from vasp_sawf.analysis import _antiunitary_pairing

    points = np.array([[0, 0, 0], [.5, 0, 0], [.25, 0, 0]])
    energies = np.array([[0., .00002], [1., 1.00004], [2., 2.00008]])
    anti = SimpleNamespace(time_reversal=True, transform_k=lambda k: -k)
    target = lambda k, image, g: np.array([[0., 1.], [-1., 0.]]) if g == 0 else np.eye(2)
    rows = _antiunitary_pairing(points, energies, [anti, anti], target)
    assert len(rows) == 1 and rows[0]['operation_index'] == 0
    assert rows[0]['fixed_kpoint_indices_0based'] == [0, 1]
    assert rows[0]['max_pair_splitting_ev'] == pytest.approx(.00004)
    assert rows[0]['minimum_required_max_energy_change_ev'] == pytest.approx(.00002)
    assert rows[0]['worst_kpoint'] == [.5, 0, 0]
