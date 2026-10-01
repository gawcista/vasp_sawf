from types import SimpleNamespace

import numpy as np
import pytest


def _two_site_model(rotation=None, translation=None, spin=None):
    from irrep.spacegroup import SpaceGroup

    rotation = np.diag([-1, 1, -1]) if rotation is None else rotation
    translation = np.array([0., .5, 0.]) if translation is None else translation
    spin = np.array([[0., -1.], [1., 0.]]) if spin is None else spin
    identity = np.eye(3, dtype=int)
    group = SpaceGroup(Lattice=np.eye(3), spinor=True,
                       rotations=np.array([identity, rotation, identity, rotation]),
                       translations=np.array([np.zeros(3), translation, np.zeros(3), translation]),
                       time_reversals=[False, False, True, True],
                       spinor_rotations=np.array([np.eye(2), spin, np.eye(2), spin], dtype=complex))
    kpoints = np.array([[0., j / 4, 0.] for j in range(4)])
    neighbors = np.array([[(j + 1) % 4, (j - 1) % 4] for j in range(4)])
    shifts = np.zeros((4, 2, 3), int)
    shifts[3, 0, 1], shifts[0, 1, 1] = 1, -1
    bundle = SimpleNamespace(kpoints=kpoints, mesh=np.array([1, 4, 1]),
                             neighbor_indices=neighbors, neighbor_shifts=shifts)
    # Exact overlaps of point orbitals at y=0 and y=1/2, with both spins.
    plus = np.diag([1., 1., np.exp(-.25j * np.pi), np.exp(-.25j * np.pi)])
    mmn = np.array([[plus, plus.conj()] for _ in range(4)])
    tr = np.array([[0., 1.], [-1., 0.]])
    exact = np.empty((4, 4, 4, 4), complex)
    for j in range(4):
        # The second site crosses a unit cell; this fixes the Bloch phase directly.
        site = np.array([[0., np.exp(-2j * np.pi * j / 4)], [1., 0.]])
        exact[0, j] = np.eye(4)
        exact[1, j] = np.kron(site, spin)
        exact[2, j] = np.kron(np.eye(2), tr)
        exact[3, j] = np.kron(site.conj(), tr @ spin.conj())
    return bundle, group, mmn, exact


@pytest.mark.parametrize('kind', ['screw', 'glide', 'centering'])
def test_full_edge_translation_phase_recovers_analytic_sewing(kind):
    from vasp_sawf.symmetry import build_symmetry_maps, mmn_translation_phases, transport_sewing

    args = dict(screw={}, glide=dict(rotation=np.diag([-1, 1, 1]),
                                    spin=np.array([[0., -1j], [-1j, 0.]])),
                centering=dict(rotation=np.eye(3, dtype=int), spin=np.eye(2)))[kind]
    bundle, group, mmn, exact = _two_site_model(**args)
    kmap, emap = build_symmetry_maps(bundle, group)
    phases = mmn_translation_phases(bundle, group)
    # Even at k=3/4 -> 0 the edge is +1/4, not -3/4.
    np.testing.assert_allclose(phases[1, :, 0], np.exp(-.25j * np.pi), atol=1e-15)
    np.testing.assert_allclose(phases[3, :, 0], np.exp(.25j * np.pi), atol=1e-15)
    d, report = transport_sewing(mmn, bundle.neighbor_indices, kmap, emap,
                                [False, False, True, True], exact[:, 0], edge_phases=phases)
    np.testing.assert_allclose(d, exact, atol=1e-14)
    assert report['mmn_covariance_max'] < 1e-14
    reverse, _ = transport_sewing(mmn, bundle.neighbor_indices, kmap, emap,
                                 [False, False, True, True], exact[:, 0],
                                 edge_phases=phases, reverse_edges=True)
    np.testing.assert_allclose(reverse, exact, atol=1e-14)
    without, _ = transport_sewing(mmn, bundle.neighbor_indices, kmap, emap,
                                 [False, False, True, True], exact[:, 0])
    assert np.max(abs(without - exact)) > 1.


def test_group_product_includes_lattice_translation_and_antiunitary_conjugation():
    from vasp_sawf.symmetry import build_symmetry_maps, group_residuals

    bundle, group, _, exact = _two_site_model()
    kmap, _ = build_symmetry_maps(bundle, group)
    products, differences, factors = group.get_product_table(get_diff=True)
    assert products[1, 1] == 0
    np.testing.assert_array_equal(differences[1, 1], [0, 1, 0])
    assert factors[1, 1] == -1
    np.testing.assert_allclose(exact[1, 2] @ exact[1, 2], np.eye(4), atol=1e-14)
    np.testing.assert_allclose(exact[1, 1] @ exact[1, 1], 1j * np.eye(4), atol=1e-14)
    report = group_residuals(exact, kmap, [False, False, True, True], products, factors,
                             kpoints=bundle.kpoints, translations_diff=differences)
    assert report['group_composition_max'] < 1e-14
    assert group_residuals(exact, kmap, [False, False, True, True], products, factors
                           )['group_composition_max'] > 1.
    broken = group_residuals(exact, kmap, [False] * 4, products, factors,
                             kpoints=bundle.kpoints, translations_diff=differences)
    assert broken['group_composition_max'] > 1.


def test_shifted_origin_seitz_transport_uses_all_three_reciprocal_components():
    from vasp_sawf.symmetry import build_symmetry_maps, mmn_translation_phases, transport_sewing

    translation = np.array([.26, .5, .14])
    _, group, _, axis_exact = _two_site_model(translation=translation)
    mesh = np.array([2, 4, 2])
    addresses = np.array(list(np.ndindex(*mesh)))
    kpoints = addresses / mesh
    steps = np.concatenate([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    lookup = {tuple(v): i for i, v in enumerate(addresses)}
    endpoints = addresses[:, None] + steps[None, :]
    neighbors = np.array([[lookup[tuple(v % mesh)] for v in row] for row in endpoints])
    shifts = endpoints // mesh
    bundle = SimpleNamespace(kpoints=kpoints, mesh=mesh, neighbor_indices=neighbors,
                             neighbor_shifts=shifts)
    # Direct overlaps for the two-site orbit {0,t}; S*t+t is exactly the y lattice vector.
    edges = [np.diag([1., 1., np.exp(-2j*np.pi*np.dot(step/mesh, translation)),
                     np.exp(-2j*np.pi*np.dot(step/mesh, translation))]) for step in steps]
    mmn = np.tile(np.array(edges)[None], (len(kpoints), 1, 1, 1))
    exact = axis_exact[:, addresses[:, 1]]
    kmap, emap = build_symmetry_maps(bundle, group)
    phases = mmn_translation_phases(bundle, group)
    for g in [1, 3]:
        sign = 1 if g == 1 else -1
        np.testing.assert_allclose(phases[g, :, 0], np.exp(sign * .26j*np.pi), atol=1e-14)
        np.testing.assert_allclose(phases[g, :, 2], np.exp(sign * .14j*np.pi), atol=1e-14)
    d, report = transport_sewing(mmn, neighbors, kmap, emap, [False, False, True, True],
                                exact[:, 0], edge_phases=phases)
    np.testing.assert_allclose(d, exact, atol=1e-14)
    assert report['mmn_covariance_max'] < 1e-14


@pytest.mark.parametrize('anti', [False, True])
def test_plane_wave_translation_at_boundary_matches_irrep_and_literal_phase(anti):
    from irrep.symmetry_operation import SymmetryOperation
    from vasp_sawf.symmetry import _independent_transform

    operation = SymmetryOperation(rot=np.diag([-1, 1, -1]),
                                  trans=np.array([.25, .5, .125]), Lattice=np.eye(3),
                                  spinor=True, time_reversal=anti)
    g = np.array([[x, y, z] for x in [-1, 0, 1] for y in [-1, 0] for z in [-1, 0, 1]])
    k = np.array([0., .5, 0.])
    rng = np.random.default_rng(182)
    wf = rng.normal(size=(2, len(g), 2)) + 1j * rng.normal(size=(2, len(g), 2))
    point = SimpleNamespace(k=k, ig=g, WF=wf)
    transformed = _independent_transform(point, operation)
    _, official, official_g = operation.transform_WF(k, wf, g, k_new=k)
    reference_order = [next(i for i, row in enumerate(official_g) if np.array_equal(row, v)) for v in g]
    np.testing.assert_allclose(transformed, official[:, reference_order], atol=1e-14)
    zero_trans = SymmetryOperation(rot=operation.rotation, trans=np.zeros(3), Lattice=np.eye(3),
                                   spinor=True, time_reversal=anti,
                                   spinor_rotation=operation.spinor_rotation)
    without = _independent_transform(point, zero_trans)
    # Direct action of translation on exp[2*pi*i*(k+G).r], in the target order.
    literal = np.array([np.exp(-2j * np.pi * (.25*x + .5*(y+.5) + .125*z)) for x,y,z in g])
    np.testing.assert_allclose(transformed, without * literal[None, :, None], atol=1e-14)
    assert np.max(abs(transformed - without)) > .5


def test_grey_group_pairs_translations_and_accepts_centering():
    from vasp_sawf.symmetry import _validate_grey_group

    _, group, _, _ = _two_site_model(rotation=np.eye(3, dtype=int), spin=np.eye(2))
    assert _validate_grey_group(group) == 2
    group.symmetries[3].translation = np.array([0., .25, 0.])
    with pytest.raises(ValueError, match='grey group|one-to-one'):
        _validate_grey_group(group)


def test_grey_group_requires_pure_tr_not_just_rotation_identity():
    from vasp_sawf.symmetry import _validate_grey_group

    _, group, _, _ = _two_site_model()
    group.symmetries[0].translation = np.array([0., .25, 0.])
    group.symmetries[2].translation = np.array([0., .25, 0.])
    with pytest.raises(ValueError, match='pure TR|identity'):
        _validate_grey_group(group)


def test_nonzero_translations_survive_spacegroup_serialization():
    from vasp_sawf.symmetry import _spacegroup_arrays

    _, group, _, _ = _two_site_model(translation=np.array([.25, .5, .125]))
    arrays = _spacegroup_arrays(group)
    np.testing.assert_array_equal(arrays['translations'][1], [.25, .5, .125])


def test_transport_rejects_nonunit_modulus_translation_phases():
    from vasp_sawf.symmetry import build_symmetry_maps, transport_sewing

    bundle, group, mmn, exact = _two_site_model()
    kmap, emap = build_symmetry_maps(bundle, group)
    with pytest.raises(ValueError, match='phase'):
        transport_sewing(mmn, bundle.neighbor_indices, kmap, emap,
                         [False, False, True, True], exact[:, 0],
                         edge_phases=np.ones((4, 4, 2)) * 2)


@pytest.mark.parametrize('shift', [np.nan, .25])
def test_group_product_rejects_invalid_lattice_difference(shift):
    from vasp_sawf.symmetry import build_symmetry_maps, group_residuals

    bundle, group, _, exact = _two_site_model()
    kmap, _ = build_symmetry_maps(bundle, group)
    products, differences, factors = group.get_product_table(get_diff=True)
    differences = differences.astype(float)
    differences[1, 1, 1] = shift
    with pytest.raises(ValueError, match='translation differences'):
        group_residuals(exact, kmap, [False, False, True, True], products, factors,
                        kpoints=bundle.kpoints, translations_diff=differences)
