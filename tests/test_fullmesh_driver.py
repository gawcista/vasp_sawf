"""Full-grid adapter and strict raw-action acceptance regressions."""

import numpy as np
import pytest


def _identity_case():
    from test_sawf import _data, _symmetrizer
    data, sym = _data(nb=8), _symmetrizer(time_reversal=True, nb=8)
    target = np.array([np.eye(8), np.kron(np.eye(4), [[0, 1], [-1, 0]])], complex)[:, None]
    return data, sym, target


def test_fullmesh_driver_preserves_inputs_and_populates_complete_checkpoint():
    from vasp_sawf.localize import wannierise_full_mesh
    data, sym, target = _identity_case()
    originals = {n: data.get_file(n).data[0].copy() for n in ('amn', 'mmn', 'eig')}
    result = wannierise_full_mesh(data, sym, target, num_iter=8)
    assert result['converged'] and result['objective_scope'] == 'all_native_mmn_edges'
    assert result['shape_nk_nb_nw'] == [1, 8, 8] and result['excluded_bands'] == []
    assert data.wannierised and set(data.chk.v_matrix) == {0}
    np.testing.assert_allclose(data.chk.v_matrix[0], np.eye(8), atol=1e-12)
    np.testing.assert_allclose(result['spreads_angstrom2'], 0, atol=1e-12)
    for n in originals:
        np.testing.assert_array_equal(data.get_file(n).data[0], originals[n])


def test_failed_optimizer_does_not_publish_checkpoint(monkeypatch):
    from vasp_sawf.localize import wannierise_full_mesh
    import vasp_sawf.fullmesh as kernel
    data, sym, target = _identity_case()
    def fail(*args, **kwargs):
        raise RuntimeError('Fresh gradient did not converge')
    monkeypatch.setattr(kernel, 'optimize_full_mesh', fail)
    with pytest.raises(RuntimeError, match='Fresh gradient'):
        wannierise_full_mesh(data, sym, target, num_iter=8)
    assert not data.wannierised


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5])
def test_fullmesh_driver_rejects_invalid_iteration_budget(invalid):
    from vasp_sawf.localize import wannierise_full_mesh
    data, sym, target = _identity_case()
    with pytest.raises(ValueError, match='iteration'):
        wannierise_full_mesh(data, sym, target, num_iter=invalid)


def test_strict_reconstruction_checks_raw_before_repair(monkeypatch):
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.localize import _checked_bloch_action
    import vasp_sawf.representation as representation
    bundle, sg, mmn, d = _two_site_model()
    bundle.mmn, bundle.eig = mmn, np.zeros((4, 4))
    def forbidden(*args, **kwargs):
        raise AssertionError('Failed raw inputs must never reach reconstruction')
    monkeypatch.setattr(representation, 'reconstruct_representation', forbidden)
    with pytest.raises(ValueError, match='bloch_'):
        _checked_bloch_action(bundle, sg, d*.999, {})


def test_strict_reconstruction_rechecks_native_matrices_and_preserves_raw():
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.localize import _checked_bloch_action
    bundle, sg, mmn, d = _two_site_model()
    bundle.mmn, bundle.eig = mmn, np.zeros((4, 4))
    raw = d*(1+1e-8)
    before = raw.copy()
    report = {}
    corrected = _checked_bloch_action(bundle, sg, raw, report)
    np.testing.assert_array_equal(before, raw)
    np.testing.assert_allclose(corrected, d, atol=1e-12, rtol=0)
    assert report['representation_reconstruction']['correction_entry_max'] < 1e-6
    assert report['bloch_unitarity_max'] > 1e-9
    assert report['reconstructed_bloch_unitarity_max'] < 1e-12


def test_trial_cli_is_explicit_and_dispatches_without_strict_export(monkeypatch, tmp_path):
    import sys
    import vasp_sawf.run_sawf as cli
    import vasp_sawf.localize as localize
    import vasp_sawf.trial as trial
    called = []
    def forbidden(*args, **kwargs):
        raise AssertionError('Trial must not enter strict model export')
    def fake(*args, **kwargs):
        called.append((args, kwargs))
        return dict(status='numerical_trial', converged=True)
    monkeypatch.setattr(localize, 'run_sawf', forbidden)
    monkeypatch.setattr(trial, 'run_trial', fake)
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--seed', 'wannier90', '--symmetry', 'symmetry',
        '--center', '0', '0', '0', '--orbital', 's', '--output', str(tmp_path/'trial'), '--trial'])
    assert cli.main() == 0
    assert len(called) == 1


def test_zero_spread_roundoff_is_reported_without_clipping():
    from vasp_sawf.localize import _check_spread_nonnegativity
    spreads = np.full(3, -6.524368435795835e-16)
    before = spreads.copy()
    report = {}
    _check_spread_nonnegativity(spreads, np.zeros((3, 3)), np.ones(6)/8,
                                np.tile(np.eye(3), (1, 6, 1, 1)), report)
    np.testing.assert_array_equal(spreads, before)
    assert report['spread_nonnegativity']['minimum_raw_angstrom2'] < 0
    assert report['spread_nonnegativity']['values_modified'] is False


@pytest.mark.parametrize('negative', [-1e-5, -1, float('nan')])
def test_significant_negative_or_nonfinite_spread_remains_rejected(negative):
    import json
    from vasp_sawf.localize import _check_spread_nonnegativity
    report = {}
    with pytest.raises(ValueError, match='spread'):
        _check_spread_nonnegativity(np.full(3, negative), np.zeros((3, 3)), np.ones(6)/8,
                                    np.tile(np.eye(3), (1, 6, 1, 1)), report)
    json.dumps(report, allow_nan=False)
