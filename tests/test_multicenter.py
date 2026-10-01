import numpy as np
import pytest

from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.sawf import SymmetrizerSAWF


def _group(kind):
    rotation = np.diag([-1, 1, -1]) if kind == 'screw' else np.diag([1, -1, 1])
    translation = [0, .5, 0] if kind == 'screw' else [.5, 0, 0]
    return SpaceGroup(Lattice=np.diag([2., 3., 4.]), spinor=True,
                      rotations=[np.eye(3, dtype=int), rotation]*2,
                      translations=np.array([[0, 0, 0], translation]*2),
                      time_reversals=[False, False, True, True])


def _target(kind='screw', centers=((.2, .17, .3),), orbitals=('s',)):
    from vasp_sawf.localize import _target_representation

    sym = SymmetrizerSAWF()
    sym.set_spacegroup(_group(kind))
    sym.kpoints_all = np.array([[0., 0., 0.], [0., .5, 0.]])
    if kind == 'glide':
        sym.kpoints_all[1] = [.5, 0., 0.]
    sym._NK = sym.NKirr = 2
    sym.kptirr = np.arange(2)
    sym.kptirr2kpt = np.repeat(np.arange(2)[:, None], 4, axis=1)
    sym.kpt2kptirr = np.arange(2)
    blocks, positions, report = _target_representation(sym, centers, orbitals)
    sym._NB = sym.num_wann
    sym.d_band_block_indices = [np.array([[0, sym.NB]])]*2
    from vasp_sawf.localize import _target_matrix
    sym.d_band_blocks = [
        [[_target_matrix(blocks, k, k, s)] for s in range(4)]
        for k in sym.kpoints_all]
    return sym, blocks, positions, report


@pytest.mark.parametrize('kind', ['screw', 'glide'])
def test_target_orbit_has_cell_phases_and_spinor_square_at_boundary(kind):
    from vasp_sawf.localize import _target_matrix

    sym, blocks, positions, report = _target(kind)
    assert positions.shape == (4, 3)
    assert report['orbits'][0]['orbital'] == 's'
    for k, sign in zip(sym.kpoints_all, [-1, 1]):
        action = _target_matrix(blocks, k, k, 1)
        np.testing.assert_allclose(action @ action, sign*np.eye(4), atol=1e-14)
        np.testing.assert_allclose(action[:2, :2], 0, atol=1e-14)
        np.testing.assert_allclose(action[2:, 2:], 0, atol=1e-14)
    from vasp_sawf.symmetry import group_residuals
    products, differences, factors = sym.spacegroup.get_product_table(get_diff=True)
    matrices = np.array([[_target_matrix(blocks, k, k, s) for k in sym.kpoints_all]
                         for s in range(4)])
    residuals = group_residuals(matrices, sym.kptirr2kpt.T, sym.time_reversals,
                                products, factors, kpoints=sym.kpoints_all,
                                translations_diff=differences)
    assert residuals['group_composition_max'] < 1e-14


def test_multiple_inequivalent_orbits_use_all_blocks_and_declared_order():
    from vasp_sawf.localize import _target_matrix

    centers = [[.2, .17, .3], [.1, .29, .43]]
    sym, blocks, positions, report = _target(centers=centers, orbitals=['s', 'pz'])
    assert sym.num_wann == 8
    assert positions.shape == (8, 3)
    assert len(blocks) == 2
    np.testing.assert_allclose(positions[0], centers[0])
    np.testing.assert_allclose(positions[4], centers[1])
    for k in sym.kpoints_all:
        matrix = _target_matrix(blocks, k, k, 1)
        np.testing.assert_allclose(matrix[:4, 4:], 0)
        np.testing.assert_allclose(matrix[4:, :4], 0)


def test_general_center_constraint_allows_free_coordinates_but_rejects_broken_orbit():
    from vasp_sawf.localize import _center_covariance

    sym, _, positions, _ = _target()
    cartesian = positions @ sym.spacegroup.lattice
    cartesian[:, 1] += .23
    assert _center_covariance(sym, cartesian) < 1e-14
    cartesian[0, 0] += .01
    assert _center_covariance(sym, cartesian) > 1e-3


@pytest.mark.parametrize('kind', ['screw', 'glide'])
def test_multicenter_alignment_records_permutation_and_reversible_cell_shifts(kind):
    from vasp_sawf.localize import align_scdm_initial_gauge

    sym, _, positions, _ = _target(kind)
    nb = sym.NB
    lattice, kpoints = sym.spacegroup.lattice, sym.kpoints_all
    permutation = np.array([2, 3, 0, 1])
    shifts = np.array([[1, 0, 0], [1, 0, 0], [0, -1, 0], [0, -1, 0]])
    ordinary = np.repeat(np.eye(nb, dtype=complex)[None], 2, axis=0)[:, :, permutation]
    ordinary *= np.exp(-2j*np.pi*kpoints @ shifts.T)[:, None, :]
    centers = (positions[permutation]+shifts) @ lattice
    amn = ordinary @ np.diag([.7, .8, .9, 1.1])
    before = amn.copy()
    initial, report = align_scdm_initial_gauge(
        amn, ordinary, centers, lattice, kpoints, sym,
        target_centers=positions @ lattice)
    np.testing.assert_array_equal(amn, before)
    np.testing.assert_array_equal(report['permutation_source_columns'], permutation)
    q = np.asarray(report['Q_real'])+1j*np.asarray(report['Q_imag'])
    cell = np.asarray(report['cell_shifts'])
    restored_order = (initial @ q.conj().T)*np.exp(-2j*np.pi*kpoints @ cell.T)[:, None, :]
    restored = restored_order[:, :, np.argsort(report['permutation_source_columns'])]
    np.testing.assert_allclose(restored, ordinary, atol=1e-13)
    np.testing.assert_allclose(initial, np.repeat(np.eye(nb)[None], 2, axis=0), atol=1e-13)


def test_center_assignment_uses_cartesian_metric_on_skew_lattice():
    from vasp_sawf.localize import _match_center_columns

    lattice = np.array([[1., 0, 0], [.95, .1, 0], [0, 0, 2.]])
    displacement = np.array([[.49, .49, 0.]])
    order, shifts, distance = _match_center_columns(displacement @ lattice, np.zeros((1, 3)), lattice)
    candidates = np.array([[i, j, 0] for i in range(-3, 4) for j in range(-3, 4)])
    distances = np.linalg.norm((displacement-candidates) @ lattice, axis=1)
    assert distance == pytest.approx(distances.min())
    assert np.any(shifts != np.rint(displacement))
    np.testing.assert_array_equal(order, [0])


@pytest.mark.parametrize('centers,orbitals', [([[0, 0, 0], [1, 0, 0]], ['s']),
                                            ([0, np.nan, 0], 's'), ([0, 0, 0], '')])
def test_target_orbit_input_rejects_mismatched_or_nonfinite_values(centers, orbitals):
    from vasp_sawf.localize import _normalize_target_orbits

    with pytest.raises(ValueError):
        _normalize_target_orbits(centers, orbitals)


@pytest.mark.parametrize('kind', ['screw', 'glide'])
def test_atomic_limit_sawf_keeps_all_multicenter_spinors_and_native_matrices(kind):
    from types import SimpleNamespace
    from wannierberri.w90files.chk import CheckPoint
    from vasp_sawf.localize import wannierise_strict, _target_representation
    from vasp_sawf.symmetry import make_symmetrizer

    group = _group(kind)
    axis = 1 if kind == 'screw' else 0
    kpoints = np.zeros((4, 3))
    kpoints[:, axis] = np.arange(4)/4
    mesh = np.ones(3, dtype=int)
    mesh[axis] = 4
    tr = np.array([[0, 1], [-1, 0]])
    sewing = np.empty((4, 4, 4, 4), dtype=complex)
    for s, operation in enumerate(group.symmetries):
        for i, k in enumerate(kpoints):
            site = np.eye(2)
            if s % 2:
                site = np.array([[0, np.exp(-2j*np.pi*operation.transform_k(k)[axis])], [1, 0]])
            spin = operation.spinor_rotation
            if operation.time_reversal:
                spin = tr @ spin.conj()
            sewing[s, i] = np.kron(site, spin)
    sym = make_symmetrizer(SimpleNamespace(kpoints=kpoints, mesh=mesh, eig=np.zeros((4, 4))), group, sewing)
    _, positions, _ = _target_representation(sym, [[0, 0, 0]], ['s'])
    nb, nk = sym.NB, sym.NK
    assert sym.NKirr < nk
    lattice = sym.spacegroup.lattice
    b = np.r_[np.diag(1/sym.grid), -np.diag(1/sym.grid)]
    bk = 2*np.pi*b @ np.linalg.inv(lattice).T
    weights = 1/(2*np.sum(bk**2, axis=1))
    neighbors = {}
    for i, k in enumerate(sym.kpoints_all):
        neighbors[i] = np.array([np.argmin(np.linalg.norm(
            (sym.kpoints_all-k-shift)-np.rint(sym.kpoints_all-k-shift), axis=1)) for shift in b])
    # Exact position eigenstates: overlaps have the known point-orbital Bloch phase.
    overlap = np.array([np.diag(np.exp(-2j*np.pi*positions @ shift)) for shift in b])
    matrices = {
        'amn': SimpleNamespace(data={k: np.eye(nb, dtype=complex) for k in range(nk)}, NK=nk, NB=nb, NW=nb),
        'mmn': SimpleNamespace(data={k: overlap.copy() for k in range(nk)}, NK=nk, NB=nb),
        'eig': SimpleNamespace(data={k: np.zeros(nb) for k in range(nk)}, NK=nk, NB=nb),
    }
    data = SimpleNamespace(**matrices, irreducible=False, wannierised=False,
        chk=CheckPoint(real_lattice=lattice, kpt_latt=sym.kpoints_all, num_wann=nb,
                       num_bands=nb, num_kpts=nk, mp_grid=sym.grid), mp_grid=sym.grid,
        bkvec=SimpleNamespace(neighbours=neighbors, wk=weights, bk_cart=bk, NNB=6))
    data.get_file = matrices.__getitem__
    data.set_symmetrizer = lambda symmetrizer, **kwargs: setattr(data, 'symmetrizer', symmetrizer)
    report = wannierise_strict(data, sym, frozen_all=True, num_iter=8)
    assert report['converged']
    assert report['excluded_bands'] == []
    np.testing.assert_allclose(report['centers_angstrom'], positions @ lattice, atol=1e-13)
    np.testing.assert_allclose(report['spreads_angstrom2'], 0, atol=1e-13)
    for k in range(nk):
        np.testing.assert_array_equal(data.amn.data[k], np.eye(nb))
        np.testing.assert_array_equal(data.mmn.data[k], overlap)
        np.testing.assert_allclose(data.chk.v_matrix[k], np.eye(nb), atol=1e-13)
