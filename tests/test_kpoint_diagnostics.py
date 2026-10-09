"""Analytic coefficient fixtures exercise the real IrRep transformation and diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest


def _evaluate(monkeypatch, coefficients, energies, *, spinor, diagnostics=True, gamma=False):
    from irrep.kpoint import Kpoint
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.extraction import evaluate_kpoint
    from vasp_sawf import wavecar

    coefficients = np.asarray(coefficients, dtype=np.complex64)
    nb, ng, _ = coefficients.shape
    g = np.zeros((ng, 3), dtype=int)
    if ng == 2:
        g[:, 0] = [-1, 1]
    ig = np.column_stack((g, np.arange(ng), np.zeros(ng, int), np.full(ng, ng)))
    point = Kpoint(ik=0, num_bands=nb, RecLattice=np.eye(3), spinor=spinor,
                   kpt=np.zeros(3), WF=coefficients.copy(), Energy=np.asarray(energies),
                   ig=ig, upper=None, normalize=False)
    original = point.WF.copy()
    stored = SimpleNamespace(kpoints=[point], read_ledger=[])
    monkeypatch.setattr(wavecar, 'read_selected_wavecar', lambda *args, **kwargs: stored)
    group = SpaceGroup(Lattice=np.eye(3), spinor=spinor,
                       rotations=[np.eye(3, dtype=int)] * 2,
                       translations=np.zeros((2, 3)), time_reversals=[False, True],
                       spinor_rotations=[np.eye(2, dtype=complex)] * 2)
    result = evaluate_kpoint('unused', list(range(1, nb + 1)), np.eye(3), 1,
                             group.as_dict(), [0, 1], None,
                             diagnostics=diagnostics, gamma=gamma)
    np.testing.assert_array_equal(point.WF, original)
    assert point.WF.dtype == np.complex64
    return result


@pytest.mark.parametrize('spinor', [False, True])
def test_exact_antiunitary_diagnostics_keep_complex_phase_and_source(monkeypatch, spinor):
    if spinor:
        wf = [[[1, 0]], [[0, 1j]]]
        expected = [[0, -1j], [1j, 0]]
    else:
        wf = [[[1], [0]], [[0], [1j]]]
        expected = [[0, -1j], [-1j, 0]]
    result = _evaluate(monkeypatch, wf, [2., 5.], spinor=spinor)
    np.testing.assert_allclose(result['matrices'][1], expected, atol=1e-14)
    identity, anti = result['operation_diagnostics']
    assert [identity['operation_index_0based'], anti['operation_index_0based']] == [0, 1]
    for row in (identity, anti):
        assert row['coefficient_closure_relative_fro'] < 1e-14
        np.testing.assert_allclose(row['per_band_relative_closure'], [0., 0.], atol=1e-14)
        assert row['unitarity_max_abs'] < 1e-14
        np.testing.assert_allclose(row['singular_values'], [1., 1.], atol=1e-14)
        assert row['independent_lstsq_difference_max'] < 1e-14
    assert identity['energy_intertwining_max_abs_ev'] == 0.
    assert anti['energy_intertwining_max_abs_ev'] == pytest.approx(3.)
    assert anti['energy_intertwining_norm2_ev'] == pytest.approx(3.)
    assert result['kpoint_energies_ev'] == [2., 5.]
    assert result['gamma_lstsq'] == 0.


def test_incomplete_scalar_subspace_reports_known_leakage_without_unitarizing(monkeypatch):
    result = _evaluate(monkeypatch, [[[1], [2j]]], [-4.], spinor=False)
    row = result['operation_diagnostics'][1]
    np.testing.assert_allclose(result['matrices'][1], [[-.8j]], atol=1e-14)
    assert row['coefficient_closure_relative_fro'] == pytest.approx(.6)
    assert row['per_band_relative_closure'] == pytest.approx([.6])
    assert row['singular_values'] == pytest.approx([.8])
    assert row['unitarity_max_abs'] == pytest.approx(.36)
    assert row['energy_intertwining_norm2_ev'] == 0.
    assert row['independent_lstsq_difference_max'] < 1e-14
    assert result['closure'] == pytest.approx(.6)


def test_energy_intertwining_is_invariant_to_energy_reference(monkeypatch):
    wf = [[[1, 0]], [[0, 1j]]]
    baseline = _evaluate(monkeypatch, wf, [2., 5.], spinor=True)
    shifted = _evaluate(monkeypatch, wf, [2. + 2**30, 5. + 2**30], spinor=True)
    for key in ('energy_intertwining_max_abs_ev', 'energy_intertwining_norm2_ev'):
        assert shifted['operation_diagnostics'][1][key] == baseline['operation_diagnostics'][1][key]


def test_optional_diagnostics_preserve_normal_results(monkeypatch):
    wf = [[[1], [2j]]]
    normal = _evaluate(monkeypatch, wf, [-4.], spinor=False, diagnostics=False, gamma=True)
    detailed = _evaluate(monkeypatch, wf, [-4.], spinor=False, diagnostics=True, gamma=True)
    assert 'operation_diagnostics' not in normal
    assert 'kpoint_energies_ev' not in normal
    for key in ('closure', 'gamma_lstsq', 'little_indices', 'read_ledger', 'kpoint_1based'):
        assert normal[key] == detailed[key]
    np.testing.assert_array_equal(normal['matrices'], detailed['matrices'])


@pytest.mark.parametrize('energies', [[1.], [1., np.nan], [1., np.inf]])
def test_diagnostics_reject_invalid_energy_vectors(monkeypatch, energies):
    with pytest.raises(ValueError, match='energies'):
        _evaluate(monkeypatch, [[[1, 0]], [[0, 1j]]], energies, spinor=True)


@pytest.mark.parametrize('workers', [1, 2])
def test_scheduled_workers_return_requested_diagnostics(tmp_path, workers):
    from irrep.spacegroup import SpaceGroup
    from test_extraction import _worker_wavecar
    from vasp_sawf.extraction import evaluate_kpoints
    from vasp_sawf.wavecar import inspect_selected_wavecar

    path, lattice = _worker_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[1, 2], lattice=lattice)
    group = SpaceGroup(Lattice=lattice, spinor=True, rotations=[np.eye(3, dtype=int)],
                       translations=np.zeros((1, 3)), time_reversals=[False],
                       spinor_rotations=[np.eye(2, dtype=complex)])
    tasks = [(path, [1, 2], lattice, ik, group.as_dict(), [0], metadata.source_identity) for ik in (1, 2)]
    results = list(evaluate_kpoints(tasks, workers=workers, diagnostics=True))
    assert len(results) == 2
    for result in results:
        row = result['operation_diagnostics'][0]
        assert row['coefficient_closure_relative_fro'] < 1e-11
        assert row['independent_lstsq_difference_max'] < 1e-11
        assert row['unitarity_max_abs'] < 1e-11
