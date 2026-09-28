import os
from pathlib import Path
import re

import numpy as np
import pytest

from test_wb import _tiny_seed


def _set_edges(seed, edges):
    seed.with_suffix(".mmn").write_text(
        f"handwritten geometry fixture\n1 1 {len(edges)}\n" +
        "".join(f"1 1 {x} {y} {z}\n0.9 {0.01*(x+y+z):.2f}\n" for x, y, z in edges)
    )


def test_anisotropic_cell_uses_only_supplied_mmn_edges(tmp_path):
    from core.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    win = seed.with_suffix(".win")
    win.write_text(win.read_text().replace("3 0 0\n0 3 0\n0 0 3", "6 0 0\n0 5 0\n0 0 1"))
    data, report = load_wannier_data(seed)
    assert report["neighbor_source"] == "mmn_explicit_edges_with_win_geometry"
    assert data.bkvec.NNB == 6
    for vector, weight in zip(data.bkvec.bk_latt, data.bkvec.wk):
        axis = np.flatnonzero(vector)[0]
        assert np.count_nonzero(vector) == 1
        assert abs(vector[axis]) == 1
        np.testing.assert_allclose(weight, np.array([6, 5, 1])[axis]**2/(8*np.pi**2), rtol=2e-14)
    assert report["weight_solver"]["rank"] == 3
    assert report["weight_solver"]["shell_count"] == 3
    assert report["weight_completeness_max_abs"] < 1e-14


def test_incomplete_existing_edges_are_rejected(tmp_path):
    from core.inputs import InputValidationError
    from core.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    _set_edges(seed, [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)])
    with pytest.raises(InputValidationError, match="completeness"):
        load_wannier_data(seed)


def test_redundant_shells_are_rejected_before_unstable_inverse(tmp_path):
    from core.inputs import InputValidationError
    from core.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    win = seed.with_suffix(".win")
    win.write_text(win.read_text().replace("3 0 0\n0 3 0\n0 0 3", "3 0 0\n0 4 0\n0 0 5"))
    _set_edges(seed, [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                      (0, 0, 1), (0, 0, -1), (2, 0, 0), (-2, 0, 0)])
    with pytest.raises(InputValidationError, match="rank deficient"):
        load_wannier_data(seed)


def test_zero_displacement_is_not_silently_dropped(tmp_path):
    from core.inputs import InputValidationError
    from core.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    _set_edges(seed, [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0),
                      (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    with pytest.raises(InputValidationError, match="zero displacement"):
        load_wannier_data(seed)


def test_official_shell_filter_cannot_discard_a_supplied_edge(tmp_path):
    from core.inputs import InputValidationError
    from core.inputs import load_wannier_data

    seed = _tiny_seed(tmp_path)
    win = seed.with_suffix(".win")
    win.write_text(win.read_text().replace("3 0 0\n0 3 0\n0 0 3", "1000000000 0 0\n0 3 0\n0 0 3"))
    with pytest.raises(InputValidationError, match="lost"):
        load_wannier_data(seed)


@pytest.mark.real_data
def test_tsns_weights_match_independent_original_wannier90_output():
    from core.inputs import load_wannier_data

    seed = os.environ.get("SAWF_TSNS_SEED")
    if seed is None:
        pytest.skip("SAWF_TSNS_SEED is unset; local WOUT files are not read implicitly")
    wout = Path(seed+".wout").read_text()
    table = wout.split("b_k Vectors (Ang^-1) and Weights (Ang^2)", 1)[1]
    rows = re.findall(r"\|\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+\|", table.split("b_k Directions", 1)[0])
    expected = sorted(tuple(map(float, row)) for row in rows)
    assert len(expected) == 6
    data, report = load_wannier_data(seed, source_nb=4320)
    actual = sorted(tuple(np.round(np.r_[vector, weight], 6)) for vector, weight in zip(data.bkvec.bk_cart, data.bkvec.wk))
    assert actual == expected
    assert report["weight_solver"]["physical_acceptance_threshold_set"] is False
