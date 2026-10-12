"""Independent full-grid spread, derivative, and constraint regressions."""

import numpy as np
import pytest
from scipy.linalg import expm


def _problem(*, shifted=False):
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.symmetry import build_symmetry_maps, make_symmetrizer

    bundle, group, mmn, bloch = _two_site_model()
    bundle.eig = np.zeros((4, 4))
    sym = make_symmetrizer(bundle, group, bloch)
    kmap, _ = build_symmetry_maps(bundle, group)
    target = bloch.copy()
    initial = np.tile(np.eye(4, dtype=complex), (4, 1, 1))
    if shifted:
        # Change the second site's integer cell, retaining the boundary phase.
        phases = np.exp(2j*np.pi*bundle.kpoints[:, 1])
        initial[:, 2, 2] = initial[:, 3, 3] = phases
        for g, anti in enumerate(sym.time_reversals):
            target[g] = (initial[kmap[g]].conj().swapaxes(-1, -2) @ bloch[g]
                         @ (initial.conj() if anti else initial))
    sym.set_D_wann(target[:, sym.kptirr].transpose(1, 0, 2, 3))
    weights = np.array([.5, .5])
    bvectors = np.array([[0., 1., 0.], [0., -1., 0.]])
    return initial, sym, target, mmn, bundle.neighbor_indices, weights, bvectors, bloch, kmap


def _literal_spread(u, mmn, neighbors, weights, bvectors, sources=None):
    """Direct scalar loops, intentionally separate from the vectorized kernel."""
    sources = range(len(u)) if sources is None else sources
    centers = np.zeros((u.shape[-1], 3))
    second = np.zeros(u.shape[-1])
    for k in sources:
        for b, j in enumerate(neighbors[k]):
            for n in range(u.shape[-1]):
                z = u[k, :, n].conj() @ mmn[k, b] @ u[j, :, n]
                phase = np.angle(z)
                centers[n] -= weights[b]*bvectors[b]*phase / len(sources)
                second[n] += weights[b]*(1-abs(z)**2+phase**2) / len(sources)
    return float(np.sum(second-np.sum(centers**2, axis=1)))


def test_native_full_mesh_objective_detects_nonrepresentative_edge_perturbation():
    from vasp_sawf.fullmesh import _objective

    u, sym, _, mmn, nn, w, b, _, _ = _problem()
    nonrepresentative = next(k for k in range(len(u)) if k not in sym.kptirr)
    changed = mmn.copy()
    changed[nonrepresentative, 0, 0, 0] *= .8*np.exp(.3j)
    before = _objective(u, mmn, nn, w, b, gradient=False)[0]
    after = _objective(u, changed, nn, w, b, gradient=False)[0]
    assert abs(after-before) > .01
    assert after == pytest.approx(_literal_spread(u, changed, nn, w, b), abs=1e-14)
    assert _literal_spread(u, changed, nn, w, b, sym.kptirr) == pytest.approx(
        _literal_spread(u, mmn, nn, w, b, sym.kptirr), abs=1e-14)


@pytest.mark.parametrize('stable', [False, True])
def test_ambient_gradient_includes_incoming_edges_and_completeness_correction(stable):
    from vasp_sawf.fullmesh import _objective

    u, _, _, mmn, nn, w, b, _, _ = _problem()
    rng = np.random.default_rng(188)
    u = u + .05*(rng.normal(size=u.shape)+1j*rng.normal(size=u.shape))
    mmn = mmn + .02*(rng.normal(size=mmn.shape)+1j*rng.normal(size=mmn.shape))
    b = b[:, 1, None]*np.array([.31, 1.07, -.23])  # Imperfect shell, all Cartesian axes.
    _, gradient = _objective(u, mmn, nn, w, b, stable=stable)
    direction = np.zeros_like(u)
    direction[3] = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    direction /= np.linalg.norm(direction)
    step = 1e-6
    finite_difference = (
        _objective(u+step*direction, mmn, nn, w, b, gradient=False, stable=stable)[0]
        - _objective(u-step*direction, mmn, nn, w, b, gradient=False, stable=stable)[0]
    )/(2*step)
    assert 2*np.vdot(gradient, direction).real == pytest.approx(finite_difference, abs=2e-9)


def test_stable_objective_differs_only_by_full_mesh_invariant_on_unitaries():
    from vasp_sawf.fullmesh import _objective

    u, _, _, mmn, nn, w, b, _, _ = _problem()
    rng = np.random.default_rng(29)
    for k in range(len(u)):
        x = .15*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        u[k] = expm(x-x.conj().T)
    mmn = mmn*.93
    b = b*1.2
    full = _literal_spread(u, mmn, nn, w, b)
    stable = _objective(u, mmn, nn, w, b, gradient=False, stable=True)[0]
    invariant = sum(w[e]*(4-np.linalg.norm(mmn[k, e])**2)
                    for k in range(4) for e in range(2))/4
    assert stable+invariant == pytest.approx(full, abs=1e-13)


def test_cayley_gradient_and_full_grid_constraints_away_from_origin():
    from vasp_sawf.fullmesh import _ConstrainedChart

    u, sym, target, mmn, nn, w, b, bloch, kmap = _problem(shifted=True)
    chart = _ConstrainedChart(u[sym.kptirr], sym, target, mmn, nn, w, b)
    rng = np.random.default_rng(71)
    x = rng.normal(size=chart.ndof)*.12
    direction = rng.normal(size=chart.ndof)
    direction /= np.linalg.norm(direction)
    full, _ = chart.matrices(x)
    for g, anti in enumerate(sym.time_reversals):
        np.testing.assert_allclose(bloch[g] @ (full.conj() if anti else full),
                                   full[kmap[g]] @ target[g], atol=2e-13)
    np.testing.assert_allclose(full.conj().swapaxes(-1, -2) @ full,
                               np.tile(np.eye(4), (4, 1, 1)), atol=2e-13)
    finite_difference = (chart(x+1e-6*direction)[0]-chart(x-1e-6*direction)[0])/2e-6
    assert chart(x)[1] @ direction == pytest.approx(finite_difference, abs=2e-9)
    assert max(np.max(abs(bloch[g] @ full-full[kmap[g]] @ target[g]))
               for g in np.flatnonzero(sym.time_reversals)) > .1


@pytest.mark.parametrize('shifted', [False, True])
def test_optimizer_preserves_full_dimension_periodic_constraints_and_inputs(shifted):
    from vasp_sawf.fullmesh import optimize_full_mesh

    args = _problem(shifted=shifted)
    initial, sym, target, mmn, nn, w, b, bloch, kmap = args
    originals = [a.copy() for a in (initial, target, mmn, nn, w, b)]
    full, report = optimize_full_mesh(*args[:7])
    assert full.shape == (4, 4, 4)
    assert report['numerical_converged'] is True
    assert report['final_fresh_chart_gradient_max'] < 1e-8
    assert report['full_grid_total_spread_angstrom2'] == pytest.approx(
        _literal_spread(full, mmn, nn, w, b), abs=1e-13)
    expected_centers = np.zeros((4, 3))
    expected_centers[2:, 1] = (-1 if shifted else 1)*np.pi/4
    np.testing.assert_allclose(report['centers_angstrom'], expected_centers, atol=1e-13)
    for g, anti in enumerate(sym.time_reversals):
        np.testing.assert_allclose(bloch[g] @ (full.conj() if anti else full),
                                   full[kmap[g]] @ target[g], atol=2e-13)
    for before, after in zip(originals, (initial, target, mmn, nn, w, b), strict=True):
        np.testing.assert_array_equal(before, after)


def test_optimizer_lowers_full_mesh_spread_from_nonstationary_initial_gauge():
    from vasp_sawf.fullmesh import optimize_full_mesh

    args = list(_problem()[:7])
    rng = np.random.default_rng(88)
    for k in range(len(args[0])):
        x = .06*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        args[0][k] = expm(x-x.conj().T)
    full, report = optimize_full_mesh(*args, maxiter=300)
    assert report['final_fresh_chart_gradient_max'] < 1e-8
    assert report['full_grid_total_spread_angstrom2'] < report['initial_full_grid_total_spread_angstrom2']-1e-5
    assert _literal_spread(full, *args[3:]) < 1e-10


def test_iteration_exhaustion_raises_with_fresh_gradient_report():
    from vasp_sawf.fullmesh import FullMeshConvergenceError, optimize_full_mesh

    args = list(_problem()[:7])
    rng = np.random.default_rng(126)
    args[3] += .07*(rng.normal(size=args[3].shape)+1j*rng.normal(size=args[3].shape))
    with pytest.raises(FullMeshConvergenceError) as failure:
        optimize_full_mesh(*args, maxiter=1, gradient_tol=1e-14)
    report = failure.value.report
    assert report['numerical_converged'] is False
    assert report['final_fresh_chart_gradient_max'] >= 1e-14
    assert report['optimization_blocks']
    assert report['iterations'] <= 1


def test_scipy_ftol_success_cannot_substitute_for_fresh_stationarity(monkeypatch):
    from types import SimpleNamespace
    from vasp_sawf import fullmesh

    args = list(_problem()[:7])
    rng = np.random.default_rng(523)
    args[3] += .03*(rng.normal(size=args[3].shape)+1j*rng.normal(size=args[3].shape))

    def stopped_before_stationarity(fun, x, **kwargs):
        value, _ = fun(x)
        return SimpleNamespace(x=x, nit=1, nfev=1, fun=value, success=True,
                               message='CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH')

    monkeypatch.setattr(fullmesh, 'minimize', stopped_before_stationarity)
    with pytest.raises(fullmesh.FullMeshConvergenceError) as failure:
        fullmesh.optimize_full_mesh(*args, maxiter=1)
    assert failure.value.report['final_fresh_chart_gradient_max'] > 1e-8
    assert failure.value.report['numerical_converged'] is False


def test_scalar_time_reversal_with_no_continuous_gauge_freedom():
    from wannierberri.symmetry.sawf import SymmetrizerSAWF
    from vasp_sawf.fullmesh import optimize_full_mesh

    sym = SymmetrizerSAWF()
    sym._NB = sym.num_wann = sym._NK = sym.NKirr = 1
    sym.Nsym = 2
    sym.grid = np.ones(3, int)
    sym.kptirr = np.array([0])
    sym.kptirr2kpt = np.array([[0, 0]])
    sym.time_reversals = np.array([False, True])
    sym.d_band_block_indices = [np.array([[0, 1]])]
    sym.d_band_blocks = [[[np.ones((1, 1), complex)] for _ in range(2)]]
    target = np.ones((2, 1, 1, 1), complex)
    sym.set_D_wann(target.transpose(1, 0, 2, 3))
    full, report = optimize_full_mesh(np.ones((1, 1, 1)), sym, target,
                                     np.full((1, 2, 1, 1), .9), np.zeros((1, 2), int),
                                     np.array([.5, .5]), np.array([[1., 0, 0], [-1., 0, 0]]))
    np.testing.assert_array_equal(full, np.ones((1, 1, 1)))
    assert report['real_parameters'] == 0
    assert report['numerical_converged'] is True
    assert report['full_grid_total_spread_angstrom2'] == pytest.approx(.19)


def test_dense_tangent_workspace_limit_fails_before_basis_allocation(monkeypatch):
    from vasp_sawf import fullmesh

    args = _problem()[:7]
    monkeypatch.setattr(fullmesh, '_TANGENT_MAX_BYTES', 1, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Large dense tangent basis must not be allocated')

    monkeypatch.setattr(fullmesh, '_commutant_basis', forbidden)
    with pytest.raises(ValueError, match='workspace'):
        fullmesh.optimize_full_mesh(*args)


@pytest.mark.parametrize('corruption', [
    'partial_grid', 'rectangular', 'nonfinite', 'neighbors', 'weights', 'vectors',
    'target', 'target_mismatch', 'nonunitary_bloch', 'rank_loss', 'maxiter', 'tolerance',
])
def test_invalid_or_incompatible_inputs_fail_closed(corruption):
    from vasp_sawf.fullmesh import optimize_full_mesh

    args = list(_problem()[:7])
    kwargs = {}
    if corruption == 'partial_grid':
        args[0] = args[0][:-1]
    elif corruption == 'rectangular':
        args[0] = args[0][:, :, :-1]
    elif corruption == 'nonfinite':
        args[3][0, 0, 0, 0] = np.nan
    elif corruption == 'neighbors':
        args[4] = args[4].astype(float)
    elif corruption == 'weights':
        args[5][0] = np.inf
    elif corruption == 'vectors':
        args[6] = args[6][:, :2]
    elif corruption == 'target':
        args[2][1, 0] *= .99
    elif corruption == 'target_mismatch':
        args[2][1, 0] *= -1
    elif corruption == 'nonunitary_bloch':
        args[1].d_band_blocks[0][1][0] *= .99
    elif corruption == 'rank_loss':
        args[0][0] = 1j*np.eye(4)
    elif corruption == 'maxiter':
        kwargs['maxiter'] = 0
    elif corruption == 'tolerance':
        kwargs['gradient_tol'] = 0
    with pytest.raises(ValueError):
        optimize_full_mesh(*args, **kwargs)
