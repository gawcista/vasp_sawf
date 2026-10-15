"""Saved-model analysis preserves both gauges and their original interpolants."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def saved_model(tmp_path):
    from test_trial import _diagnostic_inputs
    from vasp_sawf.approximate import export_approximate, run_approximate

    seed, diagnostic_dir, diagnostic = _diagnostic_inputs(tmp_path)
    win = Path(f'{seed}.win')
    with win.open('a') as handle:
        handle.write('begin kpoint_path\nG 0 0 0 X .5 0 0\nY 0 .5 0 G 0 0 0\nend kpoint_path\n')
    diagnostic['source_hashes']['win'] = hashlib.sha256(win.read_bytes()).hexdigest()
    for point in diagnostic['diagnostic_kpoints']:
        point['operations'] = [dict(operation_index_0based=g, independent_lstsq_difference_max=0.)
                               for g in point['operation_indices_0based']]
    (diagnostic_dir/'diagnostic.json').write_text(json.dumps(diagnostic))
    package = tmp_path/'package'
    export_approximate(seed, diagnostic_dir, package, mmn_tol=.01, hamiltonian_tol=1e-4)
    model = tmp_path/'model'
    run_approximate(seed, package, model, center=[0, 0, 0], orbital='s', num_iter=8, points_per_segment=3)
    return seed, model


def test_saved_model_reuses_both_gauges_with_migrated_inputs(saved_model, tmp_path, monkeypatch):
    from vasp_sawf.model_analysis import analyze_model
    import vasp_sawf.localize as localize

    seed, model = saved_model
    relocated = tmp_path/'relocated'
    seed.parent.rename(relocated)
    seed = relocated/seed.name
    before = {path: path.read_bytes() for folder in (model, relocated) for path in folder.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-model analysis must not localize either gauge')
    monkeypatch.setattr(localize, '_ordinary_initial_gauge', forbidden)
    monkeypatch.setattr(localize, 'wannierise_full_mesh', forbidden)
    output = tmp_path/'analysis'
    report = analyze_model(model, output, seed=seed, points_per_segment=7, energy_reference_ev=.25)
    assert report['status'] == 'model_analysis_complete'
    assert report['source_model_status'] == 'accepted_approximate'
    assert 'full source coefficient closure' in report['source_acceptance']['scope']
    assert report['strict_ready'] is False and report['source_full_closure_certified'] is False
    assert report['wavefunction_bytes_read'] == 0 and not report['symmetrize_hr']
    assert report['localization_repeated'] is False
    with np.load(model/'model.npz', allow_pickle=False) as original, np.load(output/'analysis.npz', allow_pickle=False) as saved:
        for key in ('U', 'ordinary_U', 'R', 'H_R', 'ordinary_R', 'ordinary_H_R'):
            np.testing.assert_array_equal(saved[key], original[key])
    with np.load(output/'bands.npz', allow_pickle=False) as bands:
        assert bands['sawf'].shape == (14, 1)
        assert float(bands['energy_reference_ev']) == .25
    comparison = json.loads((output/'comparison.json').read_text())
    assert comparison['schema'] == 'sawf-model-comparison-v1'
    assert comparison['nk'] == 1
    count = len(comparison['operations'])
    assert any(row['antiunitary'] for row in comparison['operations'])
    for name in ('ordinary_aligned', 'sawf'):
        assert len(comparison['metrics'][name]['gauge_entry_max']) == count
        assert max(comparison['metrics'][name]['hamiltonian_spectral_max_ev']) < 1e-12
    assert report['ordinary_vs_sawf']['path']['max_abs_ev'] < 1e-12
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('change', ['status', 'model_hash', 'trial_hash', 'input', 'fourier', 'ordinary_centers'])
def test_saved_model_rejects_inconsistent_evidence(saved_model, tmp_path, change):
    from vasp_sawf.model_analysis import analyze_model

    seed, model = saved_model
    report = json.loads((model/'summary.json').read_text())
    if change == 'status':
        report['strict_ready'] = True
    elif change == 'model_hash':
        report['output_sha256']['model.npz'] = '0'*64
    elif change == 'input':
        Path(f'{seed}.eig').write_text('1 1 0.1\n')
    else:
        with np.load(model/'model.npz', allow_pickle=False) as saved:
            arrays = dict(saved)
        if change == 'trial_hash':
            arrays['trial_archive'] = np.append(arrays['trial_archive'], np.uint8(0))
        elif change == 'fourier':
            arrays['ordinary_H_R'] = arrays['ordinary_H_R'] + .01
        else:
            arrays['ordinary_centers_angstrom'] = arrays['ordinary_centers_angstrom'] + .1
        np.savez_compressed(model/'model.npz', **arrays)
        report['output_sha256']['model.npz'] = hashlib.sha256((model/'model.npz').read_bytes()).hexdigest()
    (model/'summary.json').write_text(json.dumps(report))
    output = tmp_path/'analysis'
    with pytest.raises(ValueError):
        analyze_model(model, output, seed=seed, points_per_segment=3)
    if output.exists():
        assert not list(output.glob('*.npz')) and not (output/'comparison.json').exists()


@pytest.mark.parametrize('change', ['unknown_policy', 'negative_matrix_budget', 'tight_matrix_budget',
                                   'extra_target', 'scope', 'independent_source_check'])
def test_saved_model_revalidates_complete_acceptance(saved_model, tmp_path, change):
    from vasp_sawf.model_analysis import analyze_model

    seed, model = saved_model
    report = json.loads((model/'summary.json').read_text())
    decision = report['acceptance']
    if change == 'unknown_policy':
        decision['policy'] = 'unknown_policy'
    elif change == 'negative_matrix_budget':
        decision['matrix_tolerance'] = -1.
    elif change == 'tight_matrix_budget':
        decision['matrix_tolerance'] = 1e-5
    elif change == 'extra_target':
        decision['target'] = dict(orbits=[dict(position_fractional=[.25, 0, 0], orbital='s')])
    elif change == 'scope':
        decision['scope'] = 'Full coefficient closure and exact symmetry certified'
    else:
        with np.load(model/'model.npz', allow_pickle=False) as saved:
            arrays = dict(saved)
        diagnostic = json.loads(arrays['diagnostic_utf8'].tobytes())
        diagnostic['diagnostic_kpoints'][0]['operations'][0]['independent_lstsq_difference_max'] = .1
        encoded = json.dumps(diagnostic).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        trial = json.loads(arrays['trial_summary_utf8'].tobytes())
        trial.update(source_diagnostic=diagnostic, source_diagnostic_sha256=digest)
        arrays['diagnostic_utf8'] = np.frombuffer(encoded, dtype=np.uint8)
        arrays['trial_summary_utf8'] = np.frombuffer(json.dumps(trial).encode(), dtype=np.uint8)
        np.savez_compressed(model/'model.npz', **arrays)
        report['output_sha256']['model.npz'] = hashlib.sha256((model/'model.npz').read_bytes()).hexdigest()
        report['source_diagnostic_sha256'] = decision['diagnostic_sha256'] = digest
    (model/'summary.json').write_text(json.dumps(report))
    output = tmp_path/'analysis'
    with pytest.raises(ValueError):
        analyze_model(model, output, seed=seed, points_per_segment=3)
    if output.exists():
        assert not list(output.glob('*.npz')) and not (output/'comparison.json').exists()


@pytest.mark.parametrize('target_shift', [1., .25])
def test_saved_model_checks_reviewed_target_modulo_integer_cells(saved_model, tmp_path, monkeypatch, target_shift):
    from vasp_sawf import approximate
    from vasp_sawf.model_analysis import analyze_model

    seed, model = saved_model
    report = json.loads((model/'summary.json').read_text())
    decision = {key: value for key, value in report['acceptance'].items()
                if key not in ('policy', 'matrix_tolerance', 'target_policy')}
    decision.update(id='reviewed-fixture', target=dict(orbits=[dict(
        position_fractional=[target_shift, 0, 0], orbital='s')]))
    def reviewed(digest, hashes):
        if digest != decision['diagnostic_sha256'] or hashes != decision['source_hashes']:
            raise ValueError('No reviewed fixture matches these inputs')
        return decision
    monkeypatch.setattr(approximate, '_decision', reviewed)
    report['acceptance'] = decision
    (model/'summary.json').write_text(json.dumps(report))
    output = tmp_path/'analysis'
    if target_shift == 1.:
        result = analyze_model(model, output, seed=seed, points_per_segment=3)
        assert result['source_acceptance'] == decision
        assert result['status'] == 'model_analysis_complete'
    else:
        with pytest.raises(ValueError, match='target'):
            analyze_model(model, output, seed=seed, points_per_segment=3)


def test_aligned_fourier_keeps_integer_cell_phases_and_offgrid_spectra():
    from vasp_sawf.model_analysis import _aligned_fourier
    from vasp_sawf.bands import evaluate_bands, evaluate_hamiltonian

    hr = np.array([[[1., 2.+1j], [2.-1j, 4.]]])
    system = SimpleNamespace(num_wann=2, rvec=SimpleNamespace(iRvec=np.array([[0, 0, 0]])),
                             get_R_mat=lambda name: hr)
    q = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
    aligned = _aligned_fourier(system, np.array([1, 0]), np.array([[1, 0, 0], [0, 0, 0]]), q)
    points = np.array([[.25, 0, 0], [-.137, .2, 0], [1.25, 0, 0]])
    expected_quarter = q.conj().T @ np.array([[4, -1-2j], [-1+2j, 1]]) @ q
    np.testing.assert_allclose(evaluate_hamiltonian(aligned, points)[0], expected_quarter, atol=1e-14)
    np.testing.assert_allclose(evaluate_bands(aligned, points), evaluate_bands(system, points), atol=1e-14)
    np.testing.assert_array_equal(system.get_R_mat('Ham'), hr)


def test_saved_model_cli_accepts_existing_analysis_options(saved_model, tmp_path, monkeypatch, capsys):
    from vasp_sawf.run_sawf import main
    seed, model = saved_model
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--analyze-model', str(model), '--seed', str(seed),
        '--output', str(tmp_path/'analysis'), '--points-per-segment', '3', '--energy-reference-ev', '.2'])
    assert main() == 0
    assert (tmp_path/'analysis/comparison.json').is_file()
    output = capsys.readouterr().out
    assert 'source model: accepted_approximate' in output
    assert 'localization repeated: False' in output
    assert 'accepted: False' not in output


def test_saved_model_dft_gamma_alignment_keeps_original_energies(saved_model, tmp_path):
    from vasp_sawf.model_analysis import analyze_model

    seed, model = saved_model
    files = []
    for i, (points, energies) in enumerate((([[0, 0, 0], [.5, 0, 0]], [1., 1.2]),
                                           ([[0, .5, 0], [0, 0, 0]], [1.4, 1.]))):
        path = seed.parent/f'EIGENVAL.{i}'
        path.write_text('1 1 1 1\nheader\nheader\nheader\nheader\n2 2 1\n\n'+''.join(
            f' {k[0]} {k[1]} {k[2]} 1\n1 {energy} 1\n\n'
            for k, energy in zip(points, energies, strict=True)))
        files.append(path)
    report = analyze_model(model, tmp_path/'analysis', seed=seed, points_per_segment=3,
                           dft_eigenval=files, energy_reference_ev=.25)
    aligned = report['gamma_upper_band_alignment']
    assert aligned['dft_reference_ev'] == 1.
    assert aligned['band_vasp_1based'] == 1
    assert aligned['gamma_kpoint_indices_0based'] == [0, 3]
    for name in ('wannier', 'sawf'):
        assert aligned['models'][name]['shift_ev'] == pytest.approx(1.)
        assert aligned['models'][name]['max_abs_ev'] == pytest.approx(.4)
        assert aligned['models'][name]['rms_ev'] == pytest.approx(np.sqrt(.05))
        assert report['dft_comparison'][name]['max_abs_ev'] == pytest.approx(1.4)
    with np.load(tmp_path/'analysis/dft_bands.npz', allow_pickle=False) as bands:
        np.testing.assert_array_equal(bands['dft'][:, 0], [1., 1.2, 1.4, 1.])
        np.testing.assert_allclose(bands['sawf'], 0., atol=1e-14)


@pytest.mark.parametrize('option', [['--trial'], ['--analyze-trial', 'trial'], ['--num-iter', '4'],
                                    ['--center', '0', '0', '0'], ['--symmetry', 'symmetry']])
def test_saved_model_cli_rejects_localization_options(monkeypatch, option):
    from vasp_sawf.run_sawf import main
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--analyze-model', 'model', '--output', 'output', *option])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
