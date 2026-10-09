"""Independent matrix invariants and analytic little-group diagnostics."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _overlaps():
    mmn = np.tile(np.diag([.9, .7])[None, None], (2, 2, 1, 1)).astype(complex)
    mmn[1, 0, 0, 0] += .004
    bundle = SimpleNamespace(
        mmn=mmn, eig=np.array([[1., 2.], [1.001, 2.]]),
        kpoints=np.array([[0., 0., 0.], [.5, 0., 0.]]),
        neighbor_indices=np.array([[1, 1], [0, 0]]),
        neighbor_shifts=np.array([[[0, 0, 0], [-1, 0, 0]],
                                  [[1, 0, 0], [0, 0, 0]]]))
    return bundle, np.array([[0, 1], [1, 0]]), np.tile([0, 1], (2, 2, 1))


def test_mmn_invariants_give_unitary_lower_bound_and_full_edge_identity():
    from vasp_sawf.diagnostics import mmn_invariant_report

    bundle, kmap, emap = _overlaps()
    before = bundle.mmn.copy()
    report = mmn_invariant_report(bundle, kmap, emap)
    json.dumps(report, allow_nan=False)
    assert report['kpoints_checked'] == 2
    assert report['edges_checked_per_operation'] == 4
    assert report['num_bands'] == 2
    assert report['singular_value_difference_max'] == pytest.approx(.004)
    assert report['unitary_covariance_entrywise_lower_bound'] == pytest.approx(.002)
    assert report['per_operation'][0]['singular_value_difference_max'] == 0
    mapped = report['per_operation'][1]
    assert mapped['energy_spectrum_difference_max_ev'] == pytest.approx(.001)
    edge = mapped['worst_edge']
    assert edge['source_k_1based'] == 1
    assert edge['source_neighbor_k_1based'] == 2
    assert edge['source_G'] == [0, 0, 0]
    assert edge['mapped_source_k_1based'] == 2
    assert edge['mapped_neighbor_k_1based'] == 1
    assert edge['mapped_G'] == [1, 0, 0]
    np.testing.assert_array_equal(bundle.mmn, before)


def test_mmn_singular_values_ignore_endpoint_gauges_and_edge_phase():
    from vasp_sawf.diagnostics import mmn_invariant_report

    bundle, kmap, emap = _overlaps()
    reference = mmn_invariant_report(bundle, kmap, emap)
    rng = np.random.default_rng(78)
    gauges = [np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))[0]
              for _ in range(2)]
    for k in range(2):
        for edge, neighbor in enumerate(bundle.neighbor_indices[k]):
            bundle.mmn[k, edge] = (np.exp(.41j*(k + edge)) * gauges[k].conj().T
                                   @ bundle.mmn[k, edge].conj() @ gauges[neighbor])
    result = mmn_invariant_report(bundle, kmap, emap)
    assert result['singular_value_difference_max'] == pytest.approx(
        reference['singular_value_difference_max'], abs=1e-14)


@pytest.mark.parametrize('damage', ['nonfinite', 'noninteger_map', 'endpoint'])
def test_invariants_reject_invalid_inputs(damage):
    from vasp_sawf.diagnostics import mmn_invariant_report

    bundle, kmap, emap = _overlaps()
    if damage == 'nonfinite':
        bundle.mmn[0, 0, 0, 0] = np.nan
    elif damage == 'noninteger_map':
        kmap = kmap.astype(float)
    else:
        bundle.neighbor_indices[0, 0] = 0
    with pytest.raises(ValueError):
        mmn_invariant_report(bundle, kmap, emap)


@pytest.mark.parametrize('j,indices', [(0, [0, 1, 2, 3]), (1, [0, 1]), (2, [0, 1, 2, 3])])
def test_little_group_preserves_screw_translation_and_antiunitary_products(j, indices):
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.diagnostics import little_group_report

    bundle, group, _, exact = _two_site_model()
    matrices = exact[indices, j].copy()
    report = little_group_report(matrices, indices, bundle.kpoints[j], group)
    json.dumps(report, allow_nan=False)
    assert report['group_composition_max'] < 1e-14
    assert report['unitarity_max'] < 1e-14
    assert report['operation_indices'] == indices
    if j == 1:
        assert report['tr_square_max'] is None
    else:
        assert report['tr_square_max'] < 1e-14
        assert report['tr_square_expected_sign'] == -1
    np.testing.assert_array_equal(matrices, exact[indices, j])


def test_little_group_detects_missing_screw_phase_at_boundary():
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.diagnostics import little_group_report

    bundle, group, _, exact = _two_site_model()
    report = little_group_report(exact[:, 0], range(4), bundle.kpoints[2], group)
    assert report['group_composition_max'] == pytest.approx(2.)
    assert report['worst_product']['left_operation_index'] in (1, 3)


def test_scalar_antiunitary_square_uses_conjugation_and_plus_sign():
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.diagnostics import little_group_report

    group = SpaceGroup(Lattice=np.eye(3), spinor=False,
                       rotations=np.array([np.eye(3, dtype=int)]*2),
                       translations=np.zeros((2, 3)), time_reversals=[False, True])
    matrices = np.array([[[1.]], [[1j]]])
    report = little_group_report(matrices, [0, 1], np.zeros(3), group)
    assert report['group_composition_max'] < 1e-14
    assert report['tr_square_max'] < 1e-14
    assert report['tr_square_expected_sign'] == 1


@pytest.mark.parametrize('indices,j', [([0, 2], 1), ([1], 0), ([0, 0], 0)])
def test_little_group_rejects_nonstabilizers_nonclosure_and_duplicate_indices(indices, j):
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.diagnostics import little_group_report

    bundle, group, _, exact = _two_site_model()
    with pytest.raises(ValueError):
        little_group_report(exact[indices, j], indices, bundle.kpoints[j], group)


def test_little_group_reports_nonunitarity_without_repair_or_acceptance():
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.diagnostics import little_group_report

    bundle, group, _, exact = _two_site_model()
    matrices = exact[:, 0].copy()
    matrices[2] *= 1.01
    before = matrices.copy()
    report = little_group_report(matrices, range(4), bundle.kpoints[0], group)
    assert report['unitarity_max'] == pytest.approx(.0201)
    assert report['tr_square_max'] == pytest.approx(.0201)
    assert 'accepted' not in report
    np.testing.assert_array_equal(matrices, before)


def test_direct_mmn_check_uses_independent_endpoint_matrices_and_seitz_phases():
    from test_nonsymmorphic import _two_site_model
    from vasp_sawf.diagnostics import direct_mmn_covariance_report
    from vasp_sawf.symmetry import build_symmetry_maps, mmn_translation_phases

    bundle, group, mmn, exact = _two_site_model()
    bundle.mmn = mmn
    km, em = build_symmetry_maps(bundle, group)
    phases = mmn_translation_phases(bundle, group)
    local = {k: {g: exact[g, k] for g in (0, 1)} for k in range(4)}
    report = direct_mmn_covariance_report(bundle, km, em, [False, False, True, True], phases, local)
    assert report['edges_checked'] == 16
    assert report['mmn_covariance_max'] < 1e-14
    assert report['per_operation'][1]['edges_checked'] == 8
    assert report['per_operation'][2]['mmn_covariance_max'] is None
    broken = direct_mmn_covariance_report(bundle, km, em, [False, False, True, True],
                                          np.ones_like(phases), local)
    assert broken['mmn_covariance_max'] > .7


def test_direct_scalar_antiunitary_edge_check_needs_complex_conjugation():
    from vasp_sawf.diagnostics import direct_mmn_covariance_report

    phases = np.array([.31, .72])
    neighbors = np.array([[1], [0]])
    mmn = np.exp(1j*(phases[neighbors] - phases[:, None]))[:, :, None, None]
    bundle = SimpleNamespace(mmn=mmn, neighbor_indices=neighbors,
                             neighbor_shifts=np.zeros((2, 1, 3), int))
    km, em = np.array([[0, 1]]), np.zeros((1, 2, 1), int)
    local = {k: {0: np.array([[np.exp(-2j*phases[k])]])} for k in range(2)}
    report = direct_mmn_covariance_report(bundle, km, em, [True], np.ones((1, 2, 1)), local)
    assert report['mmn_covariance_max'] < 1e-14
    assert direct_mmn_covariance_report(bundle, km, em, [False],
                                        np.ones((1, 2, 1)), local)['mmn_covariance_max'] > .7


def test_direct_mmn_check_does_not_treat_missing_coverage_as_zero_residual():
    from vasp_sawf.diagnostics import direct_mmn_covariance_report

    bundle, km, em = _overlaps()
    result = direct_mmn_covariance_report(bundle, km, em, [False, False],
                                          np.ones((2, 2, 2)), {0: {0: np.eye(2)}})
    assert result['edges_checked'] == 0
    assert result['mmn_covariance_max'] is None
    assert all(row['mmn_covariance_max'] is None for row in result['per_operation'])


def test_direct_mmn_check_rejects_matrix_for_nonstabilizing_operation():
    from vasp_sawf.diagnostics import direct_mmn_covariance_report

    bundle, km, em = _overlaps()
    with pytest.raises(ValueError, match='stabiliz'):
        direct_mmn_covariance_report(bundle, km, em, [False, False],
                                     np.ones((2, 2, 2)), {0: {1: np.eye(2)}})


@pytest.mark.real_data
def test_real_native_mmn_tr_invariant_matches_independent_raw_record_calculation():
    from vasp_sawf.diagnostics import mmn_invariant_report
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.symmetry import build_symmetry_maps

    value = os.environ.get('SAWF_DIAGNOSTIC_SEED')
    if not value:
        pytest.skip('Set SAWF_DIAGNOSTIC_SEED for read-only native MMN regression')
    bundle = read_inputs(Path(value))
    group = SimpleNamespace(symmetries=[SimpleNamespace(transform_k=lambda k: -np.asarray(k))])
    km, em = build_symmetry_maps(bundle, group)
    report = mmn_invariant_report(bundle, km, em)
    # Read the native indexed records independently, without the project MMN parser.
    records = {}
    with Path(f'{value}.mmn').open() as stream:
        next(stream)
        nb, nk, ne = map(int, next(stream).split())
        for _ in range(nk*ne):
            k, l, *shift = map(int, next(stream).split())
            values = [complex(*map(float, next(stream).split())) for _ in range(nb*nb)]
            records[k - 1, l - 1, tuple(shift)] = np.array(values).reshape(nb, nb).T
    mapping = []
    for k in bundle.kpoints:
        delta = bundle.kpoints + k
        matches = np.flatnonzero(np.max(abs(delta - np.rint(delta)), axis=1) < 1e-8)
        assert len(matches) == 1
        mapping.append(int(matches[0]))
    maximum = 0.
    for (k, l, shift), matrix in records.items():
        full_edge = bundle.kpoints[l] + shift - bundle.kpoints[k]
        mapped_shift = -full_edge - bundle.kpoints[mapping[l]] + bundle.kpoints[mapping[k]]
        np.testing.assert_allclose(mapped_shift, np.rint(mapped_shift), atol=1e-8)
        mapped = records[mapping[k], mapping[l], tuple(np.rint(mapped_shift).astype(int))]
        error = np.max(abs(np.linalg.svd(matrix, compute_uv=False)
                           - np.linalg.svd(mapped, compute_uv=False)))
        maximum = max(maximum, float(error))
    assert report['singular_value_difference_max'] == pytest.approx(maximum, abs=1e-14)
    assert report['unitary_covariance_entrywise_lower_bound'] == pytest.approx(maximum/nb, abs=1e-14)
