"""Automatic target geometry must preserve metric periodicity and local representations."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from irrep.spacegroup import SpaceGroup


def _group(kind='screw', *, lattice=None, spinor=True, tr=True):
    lattice = np.diag([2., 3., 4.]) if lattice is None else np.asarray(lattice)
    rotation, translation = {
        'identity': (np.eye(3, dtype=int), [0., 0., 0.]),
        'screw': (np.diag([-1, 1, -1]), [.26, .5, .14]),
        'glide': (np.diag([1, -1, 1]), [.5, .26, 0.]),
        'centering': (np.eye(3, dtype=int), [.5, .5, 0.]),
        'rotation': (np.diag([-1, -1, 1]), [0., 0., 0.]),
    }[kind]
    rotations = [np.eye(3, dtype=int)]
    translations = [[0., 0., 0.]]
    if kind != 'identity':
        rotations.append(rotation)
        translations.append(translation)
    count = len(rotations)
    return SpaceGroup(Lattice=lattice, spinor=spinor,
        rotations=rotations * (2 if tr else 1),
        translations=np.array(translations * (2 if tr else 1)),
        time_reversals=[False] * count + ([True] * count if tr else []))


@pytest.mark.parametrize('kind,partner', [
    ('screw', [.06, .67, .84]),
    ('glide', [.7, .09, .3]),
    ('centering', [.7, .67, .3]),
])
def test_unordered_pairs_and_integer_images_recover_full_seitz_orbit(kind, partner):
    from vasp_sawf.targets import infer_target

    group = _group(kind)
    first = np.array([.2, .17, .3])
    partner = np.array(partner)
    shifts = np.array([[2, -1, 0], [0, 0, -2], [-3, 1, 2], [1, 0, 0]])
    fractional = np.array([first, partner, partner, first]) + shifts
    untouched = fractional.copy()
    centers, orbitals, report = infer_target(fractional @ group.lattice, group.lattice, group)

    assert centers.shape == (1, 3)
    assert orbitals == ['s']
    assert report['local_representation'] == 'local_kramers_doublet'
    assert report['atomic_orbital_assignment'] is False
    assert report['num_wann'] == 4
    assert sorted(sorted(v) for v in report['site_input_columns_0based']) == [[0, 3], [1, 2]]
    assert report['max_center_adjustment_angstrom'] < 1e-12
    np.testing.assert_array_equal(fractional, untouched)
    expected = sorted([first.tolist(), partner.tolist()])[0]
    np.testing.assert_allclose(centers[0], expected, atol=1e-14)
    positions = np.asarray(report['projected_site_centers_fractional'])
    permutations = np.asarray(report['site_permutations'])
    cell_shifts = np.asarray(report['site_cell_shifts'])
    for g, op in enumerate(group.symmetries[:2]):
        np.testing.assert_allclose(positions @ op.rotation.T + op.translation,
            positions[permutations[g]] + cell_shifts[g], atol=1e-14)


def test_pairing_and_averaging_use_nearest_cartesian_image_on_skew_cell():
    from vasp_sawf.targets import infer_target

    lattice = np.array([[1., 0., 0.], [.95, .1, 0.], [0., 0., 2.]])
    group = _group('identity', lattice=lattice)
    fractional = np.array([[0., 0., .2], [.49, .49, .2]])
    centers, _, report = infer_target(fractional @ lattice, lattice, group, center_tol=.08)
    # The closest second image is (.49, -.51, .2), rather than (.49, .49, .2).
    np.testing.assert_allclose(centers, [[.245, .745, .2]], atol=1e-14)
    assert report['max_center_adjustment_angstrom'] < .03


def test_geometric_projection_preserves_free_coordinate_and_reports_actual_shift():
    from vasp_sawf.targets import infer_target

    group = _group('screw')
    fractional = np.array([[.2, .17, .3], [.2, .17, .3],
                           [.0604, .6706, .8402], [.0604, .6706, .8402]])
    centers, _, report = infer_target(fractional @ group.lattice, group.lattice,
                                     group, center_tol=.01)
    np.testing.assert_allclose(centers, [[.0602, .6703, .8401]], atol=1e-14)
    expected = np.linalg.norm(np.array([.0002, .0003, .0001]) @ group.lattice)
    assert report['max_center_adjustment_angstrom'] == pytest.approx(expected)
    assert report['center_tolerance_angstrom'] == .01
    assert report['max_orbit_mismatch_angstrom'] == pytest.approx(2 * expected)
    assert report['projected_covariance_angstrom'] < 1e-12


def test_multiple_orbits_are_deterministic_under_column_reordering():
    from vasp_sawf.targets import infer_target

    group = _group('screw')
    sites = np.array([[.2, .17, .3], [.06, .67, .84], [.4, .22, .1], [.86, .72, .04]])
    fractional = np.repeat(sites, 2, axis=0)
    expected = [[.06, .67, .84], [.4, .22, .1]]
    for order in [np.arange(8), np.array([7, 2, 5, 0, 1, 6, 3, 4])]:
        centers, orbitals, report = infer_target(fractional[order] @ group.lattice,
                                                 group.lattice, group)
        np.testing.assert_allclose(centers, expected, atol=1e-14)
        assert orbitals == ['s', 's']
        assert sorted(report['orbit_site_counts']) == [2, 2]
        assert report['num_wann'] == 8


def test_inferred_metadata_is_json_serializable_for_runtime_reports():
    from vasp_sawf.targets import infer_target

    group = _group('screw')
    sites = np.array([[.2, .17, .3], [.06, .67, .84], [.4, .22, .1], [.86, .72, .04]])
    _, _, metadata = infer_target(np.repeat(sites, 2, axis=0) @ group.lattice,
                                   group.lattice, group)
    restored = json.loads(json.dumps(metadata, allow_nan=False))
    assert sorted(index for orbit in restored['orbits']
                  for index in orbit['site_indices_0based']) == [0, 1, 2, 3]
    assert sorted(column for site in restored['site_input_columns_0based']
                  for column in site) == list(range(8))


def test_nearby_unpaired_columns_are_not_silently_dropped():
    from vasp_sawf.targets import infer_target

    group = _group('identity')
    for fractional in [np.zeros((3, 3)), [[0., 0., 0.], [.1, 0., 0.]]]:
        with pytest.raises(ValueError, match='pair|even|Kramers'):
            infer_target(np.asarray(fractional) @ group.lattice, group.lattice, group)


def test_ambiguous_overlapping_kramers_pairs_require_explicit_target():
    from vasp_sawf.targets import infer_target

    group = _group('identity')
    with pytest.raises(ValueError, match='[Aa]mbiguous|multiplicity'):
        infer_target(np.zeros((4, 3)), group.lattice, group)


def test_unmatched_symmetry_partner_reports_distance_without_enlarging_tolerance():
    from vasp_sawf.targets import infer_target

    group = _group('screw')
    fractional = np.repeat([[.2, .17, .3], [.06, .69, .84]], 2, axis=0)
    with pytest.raises(ValueError, match='closest.*0.06|0.06.*closest'):
        infer_target(fractional @ group.lattice, group.lattice, group, center_tol=.01)


def test_nontrivial_site_symmetry_requires_local_representation_instead_of_s_guess():
    from vasp_sawf.targets import infer_target

    group = _group('rotation')
    with pytest.raises(ValueError, match='site symmetry.*explicit local representation'):
        infer_target(np.repeat([[0., 0., .2]], 2, axis=0) @ group.lattice,
                     group.lattice, group)


def test_spinor_without_pure_time_reversal_is_not_assigned_kramers_pairs():
    from vasp_sawf.targets import infer_target

    group = _group('identity', tr=False)
    with pytest.raises(ValueError, match='pure TR|time reversal|grey group'):
        infer_target(np.zeros((2, 3)), group.lattice, group)


def test_scalar_general_positions_keep_one_state_per_center():
    from vasp_sawf.targets import infer_target

    group = _group('glide', spinor=False)
    fractional = np.array([[.2, .17, .3], [.7, .09, .3]])
    centers, orbitals, report = infer_target(fractional @ group.lattice, group.lattice, group)
    np.testing.assert_allclose(centers, [[.2, .17, .3]], atol=1e-14)
    assert orbitals == ['s'] and report['num_wann'] == 2
    assert report['local_representation'] == 'local_scalar_state'


@pytest.mark.parametrize('kind,partner,axis,spin_square', [
    ('screw', [.06, .67, .84], 1, -1),
    ('glide', [.7, .09, .3], 0, -1),
    ('centering', [.7, .67, .3], 0, 1),
])
def test_inferred_target_keeps_boundary_translation_phase_and_spin_square(kind, partner, axis, spin_square):
    from vasp_sawf.targets import infer_target
    from vasp_sawf.localize import _target_matrix, _target_representation
    from wannierberri.symmetry.sawf import SymmetrizerSAWF

    group = _group(kind)
    raw = np.repeat([[.2, .17, .3], partner], 2, axis=0) @ group.lattice
    centers, orbitals, _ = infer_target(raw, group.lattice, group)
    sym = SymmetrizerSAWF()
    sym.set_spacegroup(group)
    sym.kpoints_all = np.zeros((2, 3))
    sym.kpoints_all[1, axis] = .5
    sym._NK = sym.NKirr = 2
    sym.kptirr = np.arange(2)
    sym.kptirr2kpt = np.repeat(np.arange(2)[:, None], 4, axis=1)
    sym.kpt2kptirr = np.arange(2)
    blocks, positions, _ = _target_representation(sym, centers, orbitals)
    assert positions.shape == (4, 3)
    for point, sign in zip(sym.kpoints_all, [spin_square, -spin_square]):
        action = _target_matrix(blocks, point, point, 1)
        np.testing.assert_allclose(action @ action, sign * np.eye(4), atol=1e-13)
        tr = _target_matrix(blocks, point, -point, 2)
        np.testing.assert_allclose(tr @ tr.conj(), -np.eye(4), atol=1e-13)


@pytest.mark.parametrize('tol', [0., -1., np.nan, np.inf, True])
def test_invalid_geometry_tolerance_is_rejected(tol):
    from vasp_sawf.targets import infer_target

    group = _group('identity')
    with pytest.raises(ValueError, match='tolerance'):
        infer_target(np.zeros((2, 3)), group.lattice, group, center_tol=tol)


def test_wout_reads_only_complete_final_centers_with_fortran_exponents(tmp_path):
    from vasp_sawf.targets import read_wout_centers

    path = tmp_path / 'wannier90.wout'
    path.write_text('Initial State\n WF centre and spread 1 ( 9.,9.,9. ) 1.\n'
        'Final State\n WF centre and spread 2 ( -1.2D+00, 3.5E-01, 4. ) 2.\n'
        ' WF centre and spread 1 ( 0.1, 2.3, -4.5 ) 3.\n'
        ' Sum of centres and spreads ( -1.1, 2.65, -.5 ) 5.\n')
    np.testing.assert_allclose(read_wout_centers(path, 2), [[.1, 2.3, -4.5], [-1.2, .35, 4.]])


@pytest.mark.parametrize('contents', [
    'Initial State\n WF centre and spread 1 ( 0.,0.,0. ) 1.\n',
    'Final State\n WF centre and spread 1 ( 0.,0.,0. ) 1.\n',
    'Final State\n WF centre and spread 1 ( 0.,0.,0. ) 1.\n'
    ' WF centre and spread 1 ( 1.,0.,0. ) 1.\n',
    'Final State\n WF centre and spread 1 ( 0.,0.,0. ) 1.\n'
    ' WF centre and spread 2 ( 1.,0.,0. ) 1.\n Final State\n',
])
def test_wout_rejects_missing_partial_duplicate_or_last_incomplete_final_state(tmp_path, contents):
    from vasp_sawf.targets import read_wout_centers

    path = tmp_path / 'wannier90.wout'
    path.write_text(contents)
    with pytest.raises(ValueError, match='[Ff]inal|centers|indices'):
        read_wout_centers(path, 2)


@pytest.mark.real_data
def test_real_wannier90_centers_are_read_without_requiring_checkpoint():
    from vasp_sawf.targets import read_wout_centers

    path = os.environ.get('SAWF_TARGET_WOUT')
    if not path:
        pytest.skip('SAWF_TARGET_WOUT is unset; original WOUT is read only when supplied')
    centers = read_wout_centers(Path(path), 8)
    assert centers.shape == (8, 3)
    # Independent, native Wannier90 final output, in its unwrapped Angstrom convention.
    np.testing.assert_allclose(centers[0], [-9.437193, -11.302436, 4.873747], atol=5e-7)
