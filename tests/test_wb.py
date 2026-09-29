import os
from pathlib import Path

import numpy as np
import pytest


def _tiny_seed(tmp_path):
    seed = tmp_path / "tiny"
    seed.with_suffix(".win").write_text(
        "num_bands=1\nnum_wann=1\nspinors=true\nmp_grid=1 1 1\n"
        "begin unit_cell_cart\n3 0 0\n0 3 0\n0 0 3\nend unit_cell_cart\n"
        "begin atoms_frac\nH 0 0 0\nend atoms_frac\n"
        "begin kpoints\n0 0 0\nend kpoints\n"
    )
    seed.with_suffix(".amn").write_text("test\n1 1 1\n1 1 1 0.3 0.4\n")
    seed.with_suffix(".eig").write_text("1 1 2.5\n")
    directions = [(0, 0, -1), (1, 0, 0), (0, -1, 0),
                  (0, 0, 1), (-1, 0, 0), (0, 1, 0)]
    text = "test\n1 1 6\n"
    for x, y, z in directions:
        text += f"1 1 {x} {y} {z}\n0.9 {0.01*(x+y+z):.2f}\n"
    seed.with_suffix(".mmn").write_text(text)
    return seed


def test_external_amn_and_exact_mmn_edges_survive_without_nnkp(tmp_path):
    from vasp_sawf.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    data, report = load_wannier_data(seed, source_nb=1)
    assert (data.mmn.NB, data.mmn.NK, data.amn.NW) == (1, 1, 1)
    np.testing.assert_array_equal(data.amn.data[0], [[0.3+0.4j]])
    assert report["amn_max_abs_difference"] == 0
    assert report["mmn_max_abs_difference"] == 0
    assert report["gauge_status"] == "unproven"
    for ib, shift in enumerate(data.bkvec.G[0]):
        assert data.bkvec.neighbours[0][ib] == 0
        assert data.mmn.data[0][ib, 0, 0] == 0.9 + 0.01j*sum(shift)
    np.testing.assert_allclose(data.bkvec.wk, 9/(8*np.pi**2), rtol=1e-13)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_bohr_uses_one_explicit_conversion_convention(tmp_path):
    from vasp_sawf.inputs import load_wannier_data
    from vasp_sawf.inputs import read_inputs

    seed = _tiny_seed(tmp_path)
    win = seed.with_suffix(".win")
    win.write_text(win.read_text().replace("begin unit_cell_cart\n", "begin unit_cell_cart\nbohr\n"))
    data, _ = load_wannier_data(seed)
    np.testing.assert_array_equal(data.chk.real_lattice, read_inputs(seed).lattice)
    np.testing.assert_allclose(data.chk.real_lattice, np.eye(3)*3*0.529177210544, rtol=0, atol=0)


def test_unrelated_nnkp_is_not_an_implicit_input(tmp_path):
    from vasp_sawf.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    seed.with_suffix(".nnkp").write_text("unrelated invalid cache must not be read\n")
    data, _ = load_wannier_data(seed)
    assert data.bkvec.NNB == 6


def test_shifted_grid_has_explicit_unsupported_error(tmp_path):
    from vasp_sawf.inputs import load_wannier_data
    from vasp_sawf.inputs import InputValidationError

    seed = _tiny_seed(tmp_path)
    win = seed.with_suffix(".win")
    win.write_text(win.read_text().replace("begin kpoints\n0 0 0", "begin kpoints\n0.5 0 0"))
    with pytest.raises(InputValidationError, match="shifted"):
        load_wannier_data(seed)


@pytest.mark.real_data
@pytest.mark.parametrize("env_name,source_nb,shape", [
    ("SAWF_SRVO3_SEED", 72, (216, 6, 6)),
    ("SAWF_TSNS_SEED", 4320, (36, 8, 8)),
])
def test_real_compact_interfaces_keep_all_bands_and_neighbor_shifts(env_name, source_nb, shape):
    seed = os.environ.get(env_name)
    if seed is None:
        pytest.skip(f"{env_name} is unset; local data are not used implicitly")
    from vasp_sawf.inputs import load_wannier_data

    root = Path(seed).parent
    before_names = {p.name for p in root.iterdir()}
    data, report = load_wannier_data(seed, source_nb=source_nb)
    assert (data.mmn.NK, data.mmn.NB, data.amn.NW) == shape
    assert report["amn_max_abs_difference"] == 0
    assert report["eig_max_abs_difference"] == 0
    assert report["mmn_max_abs_difference"] == 0
    assert len(report["neighbor_permutation_wb_to_input"]) == shape[0]
    assert report["source_hashes_before"] == report["source_hashes_after"]
    assert {p.name for p in root.iterdir()} == before_names
