from pathlib import Path
import hashlib
import os

import numpy as np
import pytest


WIN = """num_bands = 2
num_wann = 2
exclude_bands = 1, 4-5
spinors = .true.
begin unit_cell_cart
2 0 0
0 3 0
0 0 4
end unit_cell_cart
mp_grid = 2 1 1
begin kpoints
0 0 0
0.5 0 0
end kpoints
"""

AMN = """handwritten asymmetric indexed amplitudes
2 2 2
2 2 2 8 -8
1 1 1 1 -1
2 1 1 2 -2
1 2 1 3 -3
2 2 1 4 -4
1 1 2 5 -5
2 1 2 6 -6
1 2 2 7 -7
"""

EIG = """2 2 4.5
1 1 1.5
1 2 3.5
2 1 2.5
"""

MMN = """handwritten interleaved source indices; column-major matrices
2 2 2
2 1 0 0 0
1 -1
3 -3
2 -2
4 -4
1 2 0 0 0
1 1
2 2
3 3
4 4
2 1 1 0 0
5 -5
7 -7
6 -6
8 -8
1 2 -1 0 0
5 5
6 6
7 7
8 8
"""


@pytest.fixture
def seed(tmp_path):
    seed = tmp_path / "case.with.dots"
    for suffix, text in {"win": WIN, "amn": AMN, "eig": EIG, "mmn": MMN}.items():
        Path(f"{seed}.{suffix}").write_text(text)
    return seed


def read_inputs(*args, **kwargs):
    from vasp_sawf.inputs import read_inputs as read
    return read(*args, **kwargs)


def change(seed, suffix, old, new):
    path = Path(f"{seed}.{suffix}")
    original = path.read_text()
    assert old in original
    path.write_text(original.replace(old, new, 1))


def test_reads_explicit_indices_column_major_mmn_without_changing_files(seed):
    before = {p.name: p.read_bytes() for p in seed.parent.iterdir()}
    result = read_inputs(seed, source_nb=5)
    np.testing.assert_array_equal(result.lattice, np.diag([2, 3, 4]))
    np.testing.assert_array_equal(result.kpoints, [[0, 0, 0], [.5, 0, 0]])
    np.testing.assert_array_equal(result.mesh, [2, 1, 1])
    np.testing.assert_array_equal(result.bands_vasp_1based, [2, 3])
    np.testing.assert_array_equal(result.amn, np.array([[[1, 3], [2, 4]], [[5, 7], [6, 8]]]) * (1-1j))
    np.testing.assert_array_equal(result.eig, [[1.5, 2.5], [3.5, 4.5]])
    np.testing.assert_array_equal(result.mmn[0, 0], np.array([[1, 3], [2, 4]]) * (1+1j))
    np.testing.assert_array_equal(result.mmn[1, 0], result.mmn[0, 0].conj().T)
    np.testing.assert_array_equal(result.neighbor_indices, [[1, 1], [0, 0]])
    np.testing.assert_array_equal(result.neighbor_shifts, [[[0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [1, 0, 0]]])
    assert result.mmn_reverse_max_abs == 0
    assert result.gauge_status == result.source_status == "unproven"
    assert result.warnings
    assert result.hashes == {suffix: hashlib.sha256(before[f"{seed.name}.{suffix}"]).hexdigest() for suffix in ("win", "amn", "eig", "mmn")}
    assert before == {p.name: p.read_bytes() for p in seed.parent.iterdir()}


def test_unknown_source_band_count_does_not_guess_exclude_endpoint(seed):
    change(seed, "win", "exclude_bands = 1, 4-5", "exclude_bands = 1")
    result = read_inputs(seed)
    assert result.bands_vasp_1based is None
    np.testing.assert_array_equal(read_inputs(seed, source_nb=3).bands_vasp_1based, [2, 3])


@pytest.mark.parametrize("source_nb", [2, 4, 6, 0, -1, 5.5, True])
def test_inconsistent_source_band_count_is_rejected(seed, source_nb):
    with pytest.raises(ValueError):
        read_inputs(seed, source_nb=source_nb)


@pytest.mark.parametrize("suffix,old,new", [
    ("amn", "1 2 2 7 -7", "1 1 1 7 -7"),
    ("amn", "1 2 2 7 -7\n", ""),
    ("amn", "1 2 2 7 -7", "1.5 2 2 7 -7"),
    ("amn", "1 2 2 7 -7", "3 2 2 7 -7"),
    ("amn", "1 2 2 7 -7", "1 2 2 nan -7"),
    ("eig", "1 2 3.5", "2 2 3.5"),
    ("eig", "1 2 3.5\n", ""),
    ("eig", "1 2 3.5", "1 3 3.5"),
    ("eig", "1 2 3.5", "1 2 inf"),
    ("amn", "2 2 2\n", "3 2 2\n"),
    ("mmn", "2 2 2\n", "2 3 2\n"),
])
def test_rejects_incomplete_duplicate_out_of_range_or_nonfinite_interface(seed, suffix, old, new):
    change(seed, suffix, old, new)
    with pytest.raises(ValueError):
        read_inputs(seed)


@pytest.mark.parametrize("old,new", [
    ("0.5 0 0", "1 0 0"),
    ("0.5 0 0", "0.4 0 0"),
    ("mp_grid = 2 1 1", "mp_grid = 3 1 1"),
    ("mp_grid = 2 1 1", "mp_grid = 2 0 1"),
    ("0.5 0 0", "nan 0 0"),
    ("0 0 4", "0 0 0"),
    ("num_bands = 2", "num_bands = 2\nnum_bands = 2"),
])
def test_rejects_inconsistent_win_grid_and_lattice(seed, old, new):
    change(seed, "win", old, new)
    with pytest.raises(ValueError):
        read_inputs(seed)


def test_uniform_shifted_mesh_is_supported(seed):
    change(seed, "win", "0 0 0\n0.5 0 0", "0.25 0 0\n0.75 0 0")
    np.testing.assert_array_equal(read_inputs(seed).kpoints, [[.25, 0, 0], [.75, 0, 0]])


def test_swapping_only_win_order_is_detected_by_mmn_geometry(seed):
    change(seed, "win", "0 0 0\n0.5 0 0", "0.5 0 0\n0 0 0")
    with pytest.raises(ValueError):
        read_inputs(seed)


@pytest.mark.parametrize("old,new", [
    ("1 2 -1 0 0", "1 2 -2 0 0"),
    ("1 2 -1 0 0", "1 2 0 0 0"),
    ("1 2 -1 0 0", "1 3 -1 0 0"),
    ("1 2 -1 0 0", "1 2 -1.5 0 0"),
    ("8 8\n", ""),
    ("8 8\n", "nan 8\n"),
])
def test_rejects_wrong_mmn_shift_duplicate_edge_or_truncated_matrix(seed, old, new):
    change(seed, "mmn", old, new)
    with pytest.raises(ValueError):
        read_inputs(seed)


def test_missing_reverse_edge_is_rejected_even_for_uniform_neighbor_set(seed):
    mmn = """one-way uniform links
2 2 1
1 2 0 0 0
1 0
0 0
0 0
1 0
2 1 1 0 0
1 0
0 0
0 0
1 0
"""
    Path(f"{seed}.mmn").write_text(mmn)
    with pytest.raises(ValueError):
        read_inputs(seed)


def test_reverse_matrix_difference_is_reported_without_claiming_physical_pass(seed):
    change(seed, "mmn", "1 -1", "2 -1")
    result = read_inputs(seed)
    assert result.mmn_reverse_max_abs == 1
    assert result.gauge_status == "unproven"


@pytest.mark.parametrize("unit,factor", [("ang", 1.0), ("bohr", 0.529177210544)])
def test_lattice_units_are_converted_to_angstrom(seed, unit, factor):
    change(seed, "win", "begin unit_cell_cart\n", f"begin unit_cell_cart\n{unit}\n")
    np.testing.assert_allclose(read_inputs(seed).lattice, np.diag([2, 3, 4]) * factor, rtol=1e-12)


def test_unknown_lattice_unit_is_rejected(seed):
    change(seed, "win", "begin unit_cell_cart\n", "begin unit_cell_cart\nfurlong\n")
    with pytest.raises(ValueError):
        read_inputs(seed)


def test_only_absolute_seed_is_accepted(seed):
    with pytest.raises(ValueError):
        read_inputs(Path(seed.name))


@pytest.mark.real_data
@pytest.mark.parametrize("env_name,source_nb,shape,mesh,bands", [
    ("SAWF_SRVO3_SEED", 72, (216, 6, 6), [6, 6, 6], [33, 34, 35, 36, 37, 38]),
    ("SAWF_TSNS_SEED", 4320, (36, 8, 8), [6, 6, 1], list(range(3633, 3641))),
])
def test_explicit_external_fixtures(env_name, source_nb, shape, mesh, bands):
    if env_name not in os.environ:
        pytest.skip(f"Set {env_name} explicitly to read the external fixture")
    result = read_inputs(os.environ[env_name], source_nb=source_nb)
    assert result.amn.shape == shape
    assert result.eig.shape == shape[:2]
    assert result.mmn.shape == (shape[0], 6, shape[1], shape[1])
    np.testing.assert_array_equal(result.mesh, mesh)
    np.testing.assert_array_equal(result.bands_vasp_1based, bands)
    assert result.mmn_reverse_max_abs < 2e-12
    assert result.source_status == result.gauge_status == "unproven"
