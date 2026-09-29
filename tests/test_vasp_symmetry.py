import os
from pathlib import Path

import numpy as np
import pytest


_OUTCAR = '''
 Subroutine IBZKPT returns following result:
 Found 2 irreducible k-points:
 Following reciprocal coordinates:
 Coordinates Weight
 0.000000 0.000000 0.000000 1.000000
 0.250000 0.000000 0.000000 2.000000
 Following cartesian coordinates:
 Subroutine IBZKPT_HF returns following result:
 Found 3 k-points in 1st BZ
 the following 3 k-points will be used (e.g. in the exchange kernel)
 Following reciprocal coordinates:   # in IRBZ
 0.000000 0.000000 0.000000 0.33333333 1 t-inv F
 0.250000 0.000000 0.000000 0.33333333 2 t-inv F
 -0.250000 0.000000 0.000000 0.33333333 2 t-inv F
'''


def test_source_map_preserves_output_coordinates_and_records_no_gauge_claim():
    from vasp_sawf.symmetry import parse_outcar_kpoint_map

    result = parse_outcar_kpoint_map(_OUTCAR)
    np.testing.assert_array_equal(result.source_ibz, [0, 1, 1])
    np.testing.assert_array_equal(result.full_kpoints[:, 0], [0, .25, -.25])
    np.testing.assert_array_equal(result.time_reversal, [False] * 3)
    assert result.gauge_status == 'unproven'
    assert not result.operator_table_present


def test_interface_mapping_tracks_permutation_and_integer_folding():
    from vasp_sawf.symmetry import parse_outcar_kpoint_map

    result = parse_outcar_kpoint_map(_OUTCAR)
    indices, shifts = result.match_interface(np.array([[.75, 0, 0], [0, 0, 0], [.25, 0, 0]]))
    np.testing.assert_array_equal(indices, [2, 0, 1])
    np.testing.assert_array_equal(shifts, [[1, 0, 0], [0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize('bad', [
    _OUTCAR + _OUTCAR,
    _OUTCAR.replace('2 t-inv F', '3 t-inv F'),
    _OUTCAR.replace('-0.250000', '0.250000'),
    _OUTCAR.replace('Found 3 k-points', 'Found 4 k-points'),
    _OUTCAR.replace('0.33333333 2 t-inv F', '0.33333333 2 t-inv Q'),
    _OUTCAR.replace('2.000000', '3.000000'),
])
def test_ambiguous_or_inconsistent_outcar_is_rejected(bad):
    from vasp_sawf.symmetry import VaspSymmetryError, parse_outcar_kpoint_map

    with pytest.raises(VaspSymmetryError):
        parse_outcar_kpoint_map(bad)


def test_geometric_candidates_do_not_select_a_wavefunction_operation():
    from vasp_sawf.symmetry import parse_outcar_kpoint_map

    result = parse_outcar_kpoint_map(_OUTCAR)
    rotations = np.array([np.eye(3), -np.eye(3), np.diag([-1, 1, 1])], dtype=int)
    candidates = result.geometric_candidates(rotations)
    assert candidates[2] == (1, 2)
    assert result.gauge_status == 'unproven'


@pytest.mark.real_data
def test_real_srvo3_outcar_matches_interface_and_independent_cubic_group():
    fixture = os.environ.get('SAWF_SRVO3_WANNIER')
    if fixture is None:
        pytest.skip('SAWF_SRVO3_WANNIER was not explicitly set for the small-system fixture')
    import itertools
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.symmetry import parse_outcar_kpoint_map

    fixture = Path(fixture)
    result = parse_outcar_kpoint_map((fixture / 'OUTCAR').read_text())
    bundle = read_inputs(fixture / 'wannier90', source_nb=72)
    assert result.ibz_kpoints.shape == (20, 3)
    assert result.full_kpoints.shape == (216, 3)
    assert not result.time_reversal.any()
    assert not result.operator_table_present
    indices, shifts = result.match_interface(bundle.kpoints)
    np.testing.assert_array_equal(indices, np.arange(216))
    np.testing.assert_array_equal(shifts, np.zeros((216, 3), dtype=int))
    rotations = np.array([np.eye(3, dtype=int)[list(p)] * np.array(s)[:, None]
                          for p in itertools.permutations(range(3))
                          for s in itertools.product((-1, 1), repeat=3)])
    candidates = result.geometric_candidates(rotations)
    assert all(len(c) >= 2 for c in candidates)
    counts = np.bincount(result.source_ibz, minlength=20)
    np.testing.assert_array_equal(counts, result.ibz_weights)
    assert set(counts) == {1, 3, 6, 8, 12, 24}


@pytest.mark.real_data
def test_real_output_order_does_not_determine_actual_rotation():
    fixture = os.environ.get('SAWF_SRVO3_WANNIER')
    if fixture is None:
        pytest.skip('SAWF_SRVO3_WANNIER was not explicitly set for the small-system fixture')
    import itertools
    from vasp_sawf.symmetry import parse_outcar_kpoint_map
    table = parse_outcar_kpoint_map((Path(fixture) / 'OUTCAR').read_text())
    rotations = np.array([np.eye(3, dtype=int)[list(p)] * np.array(s)[:, None]
                          for p in itertools.permutations(range(3))
                          for s in itertools.product((-1, 1), repeat=3)])
    # Independent geometric witnesses, not VASP operation IDs or licensed source.
    orders = [
        [7, 0, 39, 32, 31, 24, 19, 20, 11, 12, 43, 44, 1, 6, 33, 35, 38, 25, 30, 15,
         8, 21, 18, 13, 9, 10, 37, 14, 28, 27, 29, 26, 34, 36, 45, 42, 4, 2, 3, 5,
         22, 16, 17, 23, 46, 41, 40, 47],
        [7, 0, 39, 32, 31, 24, 19, 20, 11, 12, 43, 44, 1, 6, 33, 35, 38, 25, 30, 15,
         8, 21, 18, 13, 9, 10, 45, 14, 28, 27, 26, 29, 34, 36, 37, 42, 4, 2, 3, 5,
         22, 16, 17, 23, 46, 41, 40, 47],
    ]
    assert len(orders) == 2
    generated, selected = [], []
    # Replay independently from the constraint solver, comparing raw signed coordinates.
    for order in orders:
        assert sorted(order) == list(range(48))
        points = list(table.ibz_kpoints.copy())
        choices = [-1] * 20
        for k in table.ibz_kpoints:
            for op in order:
                new = rotations[op] @ k
                diff = new - np.array(points)
                if np.any(np.max(np.abs(diff - np.rint(diff)), axis=1) < 1e-6):
                    continue
                points.append(new)
                choices.append(op)
        generated.append(np.array(points))
        selected.append(np.array(choices))
    np.testing.assert_allclose(generated[0], table.full_kpoints, rtol=0, atol=1e-12)
    np.testing.assert_allclose(generated[1], table.full_kpoints, rtol=0, atol=1e-12)
    assert np.any(selected[0] != selected[1])
