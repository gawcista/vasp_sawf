import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


def _package(tmp_path):
    folder = tmp_path / 'package'
    folder.mkdir()
    np.savez(folder / 'bloch.npz', d=np.ones((1, 1, 1, 1)))
    content = (folder / 'bloch.npz').read_bytes()
    report = dict(schema='sawf-bridge-bloch-v1', status='ready', sawf_ready=True,
                  source_hashes={'win': 'test'}, numerical_checks_passed=True,
                  residuals={'ibz_coefficient_closure_relative_max': 1e-7},
                  bloch_sha256=hashlib.sha256(content).hexdigest(),
                  bloch_bytes=len(content))
    (folder / 'report.json').write_text(json.dumps(report))
    return folder, report


def test_package_rejects_changed_matrix_even_if_it_remains_unitary(tmp_path):
    from vasp_sawf.localize import _read_bound_package
    folder, _ = _package(tmp_path)
    np.savez(folder / 'bloch.npz', d=-np.ones((1, 1, 1, 1)))
    with pytest.raises(ValueError, match='SHA256'):
        _read_bound_package(folder, {'win': 'test'})


@pytest.mark.parametrize('change', ['not_ready', 'different_input', 'nonfinite'])
def test_package_rejects_unready_foreign_or_nonfinite_data(tmp_path, change):
    from vasp_sawf.localize import _read_bound_package
    folder, report = _package(tmp_path)
    if change == 'not_ready':
        report['sawf_ready'] = False
    elif change == 'different_input':
        report['source_hashes'] = {'win': 'other'}
    else:
        np.savez(folder / 'bloch.npz', d=np.full((1, 1, 1, 1), np.nan))
        content = (folder / 'bloch.npz').read_bytes()
        report.update(bloch_bytes=len(content), bloch_sha256=hashlib.sha256(content).hexdigest())
    (folder / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _read_bound_package(folder, {'win': 'test'})


@pytest.mark.parametrize('value', [np.nan, np.inf, 1.0001e-6])
def test_numerical_gate_rejects_invalid_residuals(value):
    from vasp_sawf.localize import _gate
    with pytest.raises(ValueError):
        _gate({}, 'test', value)


def test_unreviewed_trial_package_cannot_inherit_this_decision(tmp_path):
    from vasp_sawf.localize import _read_bound_package
    folder, report = _package(tmp_path)
    report.update(status='numerical_trial', sawf_ready=False, numerical_checks_passed=True,
                  physical_acceptance_status='pending_coefficient_closure')
    report['residuals']['ibz_coefficient_closure_relative_max'] = 2.589629272055618e-6
    (folder / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _read_bound_package(folder, {'win': 'test'})


@pytest.mark.parametrize('alteration', ['pending', 'false_acceptance', 'nan', 'failed_checks'])
def test_ready_flags_cannot_replace_actual_acceptance(tmp_path, alteration):
    from vasp_sawf.localize import _read_bound_package
    folder, report = _package(tmp_path)
    if alteration == 'pending':
        report['physical_acceptance_status'] = 'pending_coefficient_closure'
    elif alteration == 'false_acceptance':
        report['physical_acceptance_status'] = 'accepted_for_single_particle_model'
        report['coefficient_closure'] = {'acceptance_id':'srvo3-closure-20260924'}
        report['residuals']['ibz_coefficient_closure_relative_max'] = 2.589629272055618e-6
    elif alteration == 'nan':
        report['residuals']['ibz_coefficient_closure_relative_max'] = float('nan')
    else:
        report['numerical_checks_passed'] = False
    (folder / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _read_bound_package(folder, {'win':'test'})


@pytest.mark.real_data
def test_reviewed_real_legacy_package_loads_without_override_and_is_not_rewritten():
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.localize import _read_bound_package
    folder, seed = os.environ.get('SAWF_SRVO3_SYMMETRY'), os.environ.get('SAWF_SRVO3_SEED')
    if not folder or not seed:
        pytest.skip('The reviewed SrVO3 symmetry package and original interfaces were not specified')
    folder = Path(folder)
    original = (folder/'report.json').read_bytes()
    assert json.loads(original)['status'] == 'numerical_trial'
    bundle = read_inputs(seed,source_nb=72)
    arrays, report = _read_bound_package(folder,bundle.hashes)
    assert report['status'] == 'ready' and report['sawf_ready'] is True
    assert report['physical_acceptance_status'] == 'accepted_for_single_particle_model'
    assert report['source_report_status'] == 'numerical_trial'
    assert arrays['d'].shape == (96,216,6,6)
    assert (folder/'report.json').read_bytes() == original


@pytest.mark.real_data
@pytest.mark.parametrize('alteration', ['failed_checks','greater_residual','substituted_payload'])
def test_legacy_acceptance_cannot_cover_altered_evidence(tmp_path, alteration):
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.localize import _read_bound_package
    folder, seed = os.environ.get('SAWF_SRVO3_SYMMETRY'), os.environ.get('SAWF_SRVO3_SEED')
    if not folder or not seed:
        pytest.skip('The reviewed SrVO3 symmetry package and original interfaces were not specified')
    report = json.loads((Path(folder)/'report.json').read_text())
    if alteration == 'failed_checks':
        report['numerical_checks_passed'] = False
    elif alteration == 'greater_residual':
        report['residuals']['ibz_coefficient_closure_relative_max'] = 4e-6
    if alteration == 'substituted_payload':
        np.savez(tmp_path/'bloch.npz',d=np.eye(6))
        content = (tmp_path/'bloch.npz').read_bytes()
        report.update(bloch_sha256=hashlib.sha256(content).hexdigest(),bloch_bytes=len(content))
    else:
        (tmp_path/'bloch.npz').symlink_to(Path(folder)/'bloch.npz')
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _read_bound_package(tmp_path,read_inputs(seed,source_nb=72).hashes)


def test_ordinary_initialization_needs_only_existing_interface_matrices():
    from vasp_sawf.localize import _ordinary_initial_gauge
    from test_sawf import _data
    data = _data()
    before = data.amn.data[0].copy()
    gauge, centers, report = _ordinary_initial_gauge(data, num_iter=8)
    assert report['converged'] is True
    assert gauge.shape == (1, 6, 6)
    np.testing.assert_allclose(gauge[0], np.eye(6), atol=1e-14)
    np.testing.assert_allclose(centers, np.zeros((6, 3)), atol=1e-14)
    np.testing.assert_array_equal(data.amn.data[0], before)


def test_ordinary_initialization_rejects_exhausted_iteration_budget():
    from vasp_sawf.localize import _ordinary_initial_gauge
    from test_sawf import _data
    with pytest.raises(ValueError, match='converge'):
        _ordinary_initial_gauge(_data(), num_iter=1)
