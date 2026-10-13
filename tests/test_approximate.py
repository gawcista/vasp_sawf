"""Explicit approximate acceptance never changes strict scientific checks."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_unreviewed_diagnostic_is_rejected():
    from vasp_sawf.approximate import _decision
    with pytest.raises(ValueError, match='reviewed'):
        _decision('0' * 64, {})


def test_target_order_is_bound_and_integer_cell_choice_is_allowed():
    from vasp_sawf.approximate import _check_target
    decision = {'target': {'orbits': [
        {'position_fractional': [.2, .3, .4], 'orbital': 's'},
        {'position_fractional': [.7, .8, .9], 'orbital': 's'}]}}
    _check_target([[.2, .3, .4], [.7, .8, .9]], ['s', 's'], decision)
    _check_target([[1.2, .3, .4], [.7, -1.2, .9]], ['s', 's'], decision)
    for centers, orbitals in [([[.7, .8, .9], [.2, .3, .4]], ['s', 's']),
                              ([[.21, .3, .4], [.7, .8, .9]], ['s', 's']),
                              ([[.2, .3, .4], [.7, .8, .9]], ['pz', 's'])]:
        with pytest.raises(ValueError, match='target'):
            _check_target(centers, orbitals, decision)


@pytest.mark.parametrize('value', [1.00001e-4, float('nan'), float('inf'), -1.])
def test_budget_never_accepts_larger_or_invalid_bound(value):
    from vasp_sawf.approximate import _check_budget
    with pytest.raises(ValueError, match='bound'):
        _check_budget({'uniform_covariance_bounds': [dict(operation_index=0,
            uniform_upper_ev=value, active_axes=[0, 1], fixed_inactive_coordinates={'2': 0.})]},
            {'hamiltonian_budget_ev': 1e-4, 'active_axes': [0, 1], 'operation_count': 1})


def test_budget_requires_complete_operation_coverage_and_plane():
    from vasp_sawf.approximate import _check_budget
    decision = {'hamiltonian_budget_ev': 1e-4, 'active_axes': [0, 1], 'operation_count': 2}
    row = dict(operation_index=0, uniform_upper_ev=9e-5, active_axes=[0, 1],
               fixed_inactive_coordinates={'2': 0.})
    with pytest.raises(ValueError, match='coverage'):
        _check_budget({'uniform_covariance_bounds': [row]}, decision)
    with pytest.raises(ValueError, match='plane'):
        _check_budget({'uniform_covariance_bounds': [row, dict(row, operation_index=1,
                      fixed_inactive_coordinates={'2': .5})]}, decision)


def test_budget_accepts_native_numpy_real_measurements():
    from vasp_sawf.approximate import _check_budget
    analysis = {'uniform_covariance_bounds': [dict(operation_index=0,
        uniform_upper_ev=np.float64(9e-5), active_axes=[0, 1], fixed_inactive_coordinates={'2': 0.})]}
    decision = {'hamiltonian_budget_ev': np.float64(1e-4), 'active_axes': [0, 1], 'operation_count': 1}
    assert _check_budget(analysis, decision) == 9e-5


@pytest.mark.parametrize('value', [False, np.bool_(False)])
def test_budget_rejects_boolean_measurements(value):
    from vasp_sawf.approximate import _check_budget
    analysis = {'uniform_covariance_bounds': [dict(operation_index=0,
        uniform_upper_ev=value, active_axes=[0, 1], fixed_inactive_coordinates={'2': 0.})]}
    with pytest.raises(ValueError, match='bound'):
        _check_budget(analysis, {'hamiltonian_budget_ev': 1e-4, 'active_axes': [0, 1], 'operation_count': 1})


@pytest.mark.parametrize('budget', [True, np.bool_(True), 0., -1., float('nan'), float('inf'), '0.0001'])
def test_budget_itself_must_be_a_positive_finite_real(budget):
    from vasp_sawf.approximate import _check_budget
    analysis = {'uniform_covariance_bounds': [dict(operation_index=0,
        uniform_upper_ev=9e-5, active_axes=[0, 1], fixed_inactive_coordinates={'2': 0.})]}
    with pytest.raises(ValueError, match='budget'):
        _check_budget(analysis, {'hamiltonian_budget_ev': budget, 'active_axes': [0, 1], 'operation_count': 1})


def test_existing_output_is_reused_and_failed_rerun_invalidates_ready(tmp_path, capsys):
    from vasp_sawf.approximate import export_approximate
    source = tmp_path / 'inputs'
    source.mkdir()
    diagnostic = tmp_path / 'diagnostic'
    diagnostic.mkdir()
    (diagnostic / 'diagnostic.json').write_text('{}')
    output = tmp_path / 'symmetry'
    output.mkdir()
    (output / 'report.json').write_text('{"status":"accepted_approximate"}')
    (output / 'bloch.npz').write_bytes(b'old')
    (output / 'keep.txt').write_text('keep')
    with pytest.raises(ValueError):
        export_approximate(source/'wannier90', diagnostic, output)
    assert not (output/'bloch.npz').exists()
    assert json.loads((output/'report.json').read_text())['accepted'] is False
    assert (output/'keep.txt').read_text() == 'keep'
    assert 'report.json' in capsys.readouterr().err


@pytest.mark.parametrize('name', ['bloch.npz', 'report.json'])
@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'directory', 'fifo'])
def test_output_aliases_and_nonregular_targets_are_protected(tmp_path, name, kind):
    from vasp_sawf.approximate import export_approximate
    source = tmp_path / 'inputs'
    source.mkdir()
    protected = source/'wannier90.win'
    protected.write_text('original')
    diagnostic = tmp_path/'diagnostic'
    diagnostic.mkdir()
    output = tmp_path/'symmetry'
    output.mkdir()
    target = output/name
    if kind == 'symlink':
        target.symlink_to(protected)
    elif kind == 'hardlink':
        os.link(protected, target)
    elif kind == 'directory':
        target.mkdir()
    else:
        os.mkfifo(target)
    with pytest.raises(ValueError, match='Output|output'):
        export_approximate(source/'wannier90', diagnostic, output)
    assert protected.read_text() == 'original'


def test_existing_empty_output_has_no_overwrite_warning(tmp_path, capsys):
    from vasp_sawf.approximate import export_approximate
    output = tmp_path/'symmetry'
    output.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        export_approximate(tmp_path/'inputs/wannier90', tmp_path/'diagnostic', output)
    assert not capsys.readouterr().err


def test_new_model_output_may_be_a_child_of_the_calculation_directory(tmp_path):
    from vasp_sawf.approximate import _new_model_output
    source = tmp_path/'calculation'
    source.mkdir()
    protected = source/'wannier90.win'
    protected.write_bytes(b'original WIN')
    output = _new_model_output(source/'sawf', [protected])
    assert output == source/'sawf' and output.is_dir()
    assert protected.read_bytes() == b'original WIN'


def test_new_model_output_may_be_a_sibling_of_the_calculation_directory(tmp_path):
    from vasp_sawf.approximate import _new_model_output
    source = tmp_path/'calculation'
    source.mkdir()
    protected = source/'wannier90.win'
    protected.write_bytes(b'original WIN')
    assert _new_model_output(tmp_path/'sawf', [protected]).is_dir()
    assert protected.read_bytes() == b'original WIN'


@pytest.mark.parametrize('kind', ['existing_calculation', 'existing_result', 'input_file',
    'input_child', 'missing_input', 'missing_input_child', 'symlink', 'dangling_symlink'])
def test_new_model_output_rejects_aliases_and_nonnew_paths(tmp_path, kind):
    from vasp_sawf.approximate import _new_model_output
    source = tmp_path/'calculation'
    source.mkdir()
    protected = source/'wannier90.win'
    protected.write_bytes(b'original WIN')
    missing = source/'wannier90.eig'
    output = source/'sawf'
    if kind == 'existing_calculation':
        output = source
    elif kind == 'existing_result':
        output.mkdir()
        (output/'keep.txt').write_text('keep')
    elif kind == 'input_file':
        output = protected
    elif kind == 'input_child':
        output = protected/'result'
    elif kind == 'missing_input':
        output = missing
    elif kind == 'missing_input_child':
        output = missing/'result'
    elif kind == 'symlink':
        output.symlink_to(source, target_is_directory=True)
    else:
        output.symlink_to(tmp_path/'nonexistent', target_is_directory=True)
    with pytest.raises(ValueError, match='Output|output'):
        _new_model_output(output, [protected, missing])
    assert protected.read_bytes() == b'original WIN'
    assert not missing.exists()
    if kind == 'existing_result':
        assert (output/'keep.txt').read_text() == 'keep'


def test_new_model_output_rejects_missing_ancestor_of_future_input(tmp_path):
    from vasp_sawf.approximate import _new_model_output
    output = tmp_path/'future_calculation'
    with pytest.raises(ValueError, match='Output|output'):
        _new_model_output(output, [output/'wannier90.win'])
    assert not output.exists()


@pytest.mark.skipif(not os.environ.get('SAWF_APPROXIMATE_SEED'), reason='Real reviewed inputs are optional')
def test_real_reviewed_diagnostic_package_and_tamper(tmp_path):
    from vasp_sawf.approximate import export_approximate, _read_package
    seed = Path(os.environ['SAWF_APPROXIMATE_SEED'])
    diagnostic = Path(os.environ['SAWF_APPROXIMATE_DIAGNOSTIC'])
    output = tmp_path/'symmetry'
    report = export_approximate(seed, diagnostic, output)
    assert report['accepted'] and not report['strict_checks_passed']
    assert report['wavefunction_bytes_read'] == 0
    assert report['source_full_closure_certified'] is False
    _read_package(seed, output)
    for key, value in [('numerical_checks_passed', True), ('mesh', [3, 3, 1])]:
        modified = dict(report, **{key: value})
        (output/'report.json').write_text(json.dumps(modified))
        with pytest.raises(ValueError, match='package|metadata'):
            _read_package(seed, output)
    (output/'report.json').write_text(json.dumps(report))
    with np.load(output/'bloch.npz', allow_pickle=False) as saved:
        arrays = dict(saved)
    arrays['raw_d'] = arrays['raw_d'] + .1
    np.savez_compressed(output/'bloch.npz', **arrays)
    content = (output/'bloch.npz').read_bytes()
    report.update(bloch_sha256=hashlib.sha256(content).hexdigest(), bloch_bytes=len(content))
    (output/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='recomputed'):
        _read_package(seed, output)


@pytest.mark.skipif(not os.environ.get('SAWF_APPROXIMATE_FULL'), reason='Full real-data localization is opt-in')
def test_real_end_to_end_model_and_bound(tmp_path):
    from vasp_sawf.approximate import export_approximate, run_approximate
    seed = Path(os.environ['SAWF_APPROXIMATE_SEED'])
    diagnostic = Path(os.environ['SAWF_APPROXIMATE_DIAGNOSTIC'])
    package = export_approximate(seed, diagnostic, tmp_path/'symmetry')
    orbits = package['acceptance']['target']['orbits']
    report = run_approximate(seed, tmp_path/'symmetry', tmp_path/'model',
        center=[row['position_fractional'] for row in orbits], orbital=[row['orbital'] for row in orbits])
    assert report['accepted'] and report['converged'] and not report['strict_ready']
    assert report['hamiltonian_uniform_upper_ev'] <= 1e-4
    assert report['source_full_closure_certified'] is False
    for name, digest in report['output_sha256'].items():
        assert hashlib.sha256((tmp_path/'model'/name).read_bytes()).hexdigest() == digest
    with np.load(tmp_path/'model/model.npz', allow_pickle=False) as model:
        trial = json.loads(model['trial_summary_utf8'].tobytes())
        assert hashlib.sha256(model['trial_archive'].tobytes()).hexdigest() == trial['trial_sha256']
        assert hashlib.sha256(model['diagnostic_utf8'].tobytes()).hexdigest() == package['source_diagnostic_sha256']
        assert model['U'].shape == (36, 8, 8)
        assert not np.array_equal(model['H_R'], np.zeros_like(model['H_R']))
    assert not list((tmp_path/'model').glob('*hr*'))
