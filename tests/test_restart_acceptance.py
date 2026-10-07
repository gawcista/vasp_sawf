import numpy as np
import pytest


def _report():
    return {'residuals': {}, 'source_hashes': {
        'win': 'f426ca2971c69da60dc24e15227e1520f69ac14de4433f44c6586f56c4a81368',
        'amn': '69592c97ee9645e0e8c1a2536fd916398918fbb6956e4204a6198caafb7a2e1c',
        'eig': '5e4c34ca9e91e9676e8162d70ec758cdaecdb93d88516e052de490be1cd27d50',
        'mmn': 'e2c8c5a8baf0f9acae18952bbfc092df2b24d607ddf56b64084e36a1bd6ece13',
    }}


@pytest.mark.parametrize('value', [1.5343413566685595e-6, 1.534345e-6])
def test_reviewed_restart_closure_is_accepted_only_with_its_bound_interfaces(value):
    from vasp_sawf.symmetry import _check_coefficient_closure

    report = _report()
    decision = _check_coefficient_closure(report, value)
    assert decision['id'] == 'srvo3-restart-closure-20261007'
    assert decision['observed_relative'] == 1.5343413566685595e-6
    assert report['physical_acceptance_status'] == 'accepted_for_single_particle_model'
    assert report['coefficient_closure']['acceptance_id'] == decision['id']
    assert report['coefficient_closure']['accepted_max_relative'] == 1.534345e-6
    assert report['coefficient_closure']['reference_tolerance'] == 1e-6
    assert report['coefficient_closure']['reference_status'] == 'above_reference'
    assert report['residuals']['ibz_coefficient_closure_relative_max'] == value


@pytest.mark.parametrize('changed', ['win', 'amn', 'eig', 'mmn'])
def test_restart_acceptance_does_not_transfer_to_different_interfaces(changed):
    from vasp_sawf.symmetry import _check_coefficient_closure

    report = _report()
    report['source_hashes'][changed] = 'different source'
    with pytest.raises(ValueError, match='not covered'):
        _check_coefficient_closure(report, 1.5343413566685595e-6)
    assert report['physical_acceptance_status'] == 'not_assessed'


@pytest.mark.parametrize('value', [np.nextafter(1.534345e-6, np.inf), 2.589629272055618e-6])
def test_restart_acceptance_rejects_larger_error_even_below_old_dataset_limit(value):
    from vasp_sawf.symmetry import _check_coefficient_closure

    with pytest.raises(ValueError, match='not covered'):
        _check_coefficient_closure(_report(), value)


def test_restart_decision_preserves_other_matrix_gates():
    from vasp_sawf.symmetry import _check, _check_coefficient_closure

    report = _report()
    _check_coefficient_closure(report, 1.5343413566685595e-6)
    with pytest.raises(ValueError, match='mmn_covariance_max'):
        _check(report['residuals'], 'mmn_covariance_max', 1.1e-6)
