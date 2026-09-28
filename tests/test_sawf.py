from types import SimpleNamespace
import os

import numpy as np
import pytest

from wannierberri.symmetry.sawf import SymmetrizerSAWF
from wannierberri.symmetry.sawf_kirr import Symmetrizer_Uirr as OfficialSymmetrizerUirr
from wannierberri.w90files.chk import CheckPoint
from core.localize import StrictSymmetrizerUirr as Symmetrizer_Uirr


def _symmetrizer(*, invalid_upper=False, time_reversal=False, nb=6):
    sym = SymmetrizerSAWF()
    sym._NB = sym.num_wann = nb
    sym._NK = sym.NKirr = 1
    sym.Nsym = 2
    sym.grid = np.ones(3, dtype=int)
    sym.kpoints_all = np.zeros((1, 3))
    sym.kptirr = np.array([0])
    sym.kptirr2kpt = np.array([[0, 0]])
    sym.kpt2kptirr = np.array([0])
    sym.time_reversals = np.array([False, time_reversal])
    identity = np.eye(nb, dtype=complex)
    spin_tr = np.eye(nb, dtype=complex)
    for i in range(0, nb - 1, 2):
        spin_tr[i:i + 2, i:i + 2] = [[0, 1], [-1, 0]]
    rotation = spin_tr.copy() if time_reversal else identity.copy()
    if invalid_upper:
        rotation[2:, 2:] *= 0.8
    sym.d_band_block_indices = [np.array([[0, 2], [2, nb]])]
    sym.d_band_blocks = [[[identity[:2, :2], identity[2:, 2:]],
                         [rotation[:2, :2], rotation[2:, 2:]]]]
    target = spin_tr if time_reversal else identity
    sym.set_D_wann(np.array([[identity, target]]))
    sym.symmetrize_WCC = lambda x: x
    sym.symmetrize_spreads = lambda x: x
    return sym


def test_upper_four_bands_must_fail_instead_of_being_excluded():
    # Replacing the strict class with the official class reproduces silent loss.
    with pytest.raises(RuntimeError):
        Symmetrizer_Uirr(_symmetrizer(invalid_upper=True), 0)


def test_official_default_characterization_excludes_the_upper_four_bands(capsys):
    sym = OfficialSymmetrizerUirr(_symmetrizer(invalid_upper=True), 0)
    np.testing.assert_array_equal(sym.include_bands, [True, True, False, False, False, False])
    assert "did not converge" in capsys.readouterr().out


def test_local_iteration_exhaustion_must_raise_instead_of_warning(capsys):
    sym = Symmetrizer_Uirr(_symmetrizer(time_reversal=True), 0)
    rng = np.random.default_rng(123)
    u = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
    with pytest.raises(RuntimeError):
        sym(u, maxiter=1)


def test_valid_tr_action_preserves_all_six_bands():
    sym = Symmetrizer_Uirr(_symmetrizer(time_reversal=True), 0)
    before = np.eye(6, dtype=complex)
    result = sym(before)
    np.testing.assert_allclose(result, before, atol=1e-14, rtol=0)
    assert sym.include_bands.all()


def test_complex_trial_obeys_antiunitary_conjugation_after_symmetrisation():
    sym = Symmetrizer_Uirr(_symmetrizer(time_reversal=True), 0)
    rng = np.random.default_rng(147)
    trial = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
    result = sym(trial)
    spin_tr = np.kron(np.eye(3), [[0, 1], [-1, 0]])
    np.testing.assert_allclose(spin_tr @ result.conj() @ (-spin_tr), result, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.conj().T @ result, np.eye(6), atol=1e-12, rtol=0)


def _data(nb=6):
    identity = np.eye(nb, dtype=complex)
    matrices = {
        "amn": SimpleNamespace(data={0: identity.copy()}, NK=1, NB=nb, NW=nb),
        "mmn": SimpleNamespace(data={0: np.repeat(identity[None], 6, axis=0)}, NK=1, NB=nb),
        "eig": SimpleNamespace(data={0: np.arange(nb, dtype=float)//2}, NK=1, NB=nb),
    }
    data = SimpleNamespace(**matrices, irreducible=False, wannierised=False,
                           chk=CheckPoint(kpt_latt=np.zeros((1, 3)), num_wann=nb, num_bands=nb,
                                          num_kpts=1, mp_grid=np.ones(3, dtype=int)),
                           mp_grid=np.ones(3, dtype=int),
                           bkvec=SimpleNamespace(neighbours={0: np.zeros(6, dtype=int)},
                               wk=np.ones(6)/2, NNB=6,
                               bk_cart=np.r_[np.eye(3), -np.eye(3)]))
    data.get_file = matrices.__getitem__
    data.set_symmetrizer = lambda symmetrizer, **kwargs: setattr(data, "symmetrizer", symmetrizer)
    return data


def test_eight_dimensional_spinor_action_requires_conjugation_and_retains_all_bands():
    local = Symmetrizer_Uirr(_symmetrizer(time_reversal=True, nb=8), 0)
    rng = np.random.default_rng(5487)
    trial, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8)))
    result = local(trial)
    spin_tr = np.kron(np.eye(4), np.array([[0., 1.], [-1., 0.]]))
    np.testing.assert_allclose(spin_tr @ result.conj() @ spin_tr.T, result,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.conj().T @ result, np.eye(8), atol=1e-12, rtol=0)
    assert np.max(abs(spin_tr @ result @ spin_tr.T - result)) > 1e-2
    assert local.include_bands.all()


def test_eight_band_driver_identity_neighbours_have_exact_zero_spread():
    from core.localize import wannierise_strict

    data = _data(nb=8)
    report = wannierise_strict(data, _symmetrizer(time_reversal=True, nb=8),
                               num_iter=8, frozen_all=True)
    np.testing.assert_allclose(data.chk.v_matrix[0], np.eye(8), atol=1e-13, rtol=0)
    np.testing.assert_allclose(report["centers_angstrom"], np.zeros((8, 3)), atol=1e-13)
    np.testing.assert_allclose(report["spreads_angstrom2"], np.zeros(8), atol=1e-13)
    np.testing.assert_array_equal(data.amn.data[0], np.eye(8))
    assert report["shape_nk_nb_nw"] == [1, 8, 8]
    assert report["parameters"]["no_exclude_bands"] == 8
    assert report["parameters"]["frozen_bands"] == 8
    assert report["excluded_bands"] == []


@pytest.mark.parametrize("corruption", ["odd", "non_square", "eigenvalue_metadata", "checkpoint", "band_block"])
def test_dimension_changes_are_rejected_before_localisation(corruption):
    from core.localize import wannierise_strict

    nb = 7 if corruption == "odd" else 6
    data, sym = _data(nb=nb), _symmetrizer(time_reversal=True, nb=nb)
    if corruption == "non_square":
        data.amn.NW = 4
    elif corruption == "eigenvalue_metadata":
        data.eig.NB = 8
    elif corruption == "checkpoint":
        data.chk.num_bands = 8
    elif corruption == "band_block":
        sym.d_band_block_indices[0][-1, 1] = 4
    with pytest.raises(ValueError):
        wannierise_strict(data, sym, num_iter=8)
    assert not data.wannierised


def test_driver_preserves_external_amn_and_exports_complete_square_gauge():
    from core.localize import wannierise_strict

    data = _data()
    originals = {name: data.get_file(name).data[0].copy() for name in ("amn", "mmn", "eig")}
    report = wannierise_strict(data, _symmetrizer(time_reversal=True), num_iter=8)
    assert report["converged"] is True
    assert report["parameters"]["init"] == "amn"
    assert report["parameters"]["irreducible"] is False
    assert report["parameters"]["symmetrize_hr"] is False
    assert set(data.chk.v_matrix) == {0}
    np.testing.assert_allclose(data.chk.v_matrix[0].conj().T @ data.chk.v_matrix[0], np.eye(6), atol=1e-13)
    for name, before in originals.items():
        np.testing.assert_array_equal(data.get_file(name).data[0], before)


def test_final_centers_are_checked_from_the_complete_final_gauge():
    from core.localize import wannierise_strict

    data, sym = _data(), _symmetrizer(time_reversal=True)
    sym.symmetrize_WCC = lambda centers: centers + 0.1
    with pytest.raises(RuntimeError, match="recomputation"):
        wannierise_strict(data, sym, num_iter=8)
    assert not data.wannierised
    assert not hasattr(data.chk, "v_matrix")


def test_driver_global_iteration_exhaustion_leaves_no_success_state():
    from core.localize import wannierise_strict

    data = _data()
    with pytest.raises(RuntimeError, match="converge"):
        wannierise_strict(data, _symmetrizer(time_reversal=True), num_iter=1)
    assert not data.wannierised
    assert not hasattr(data.chk, "v_matrix")


@pytest.mark.parametrize("corruption", ["irreducible", "missing_band", "k_order", "nonfinite", "missing_tr"])
def test_driver_rejects_inputs_before_localisation(corruption):
    from core.localize import wannierise_strict

    data, sym = _data(), _symmetrizer(time_reversal=True)
    if corruption == "irreducible":
        data.irreducible = True
    elif corruption == "missing_band":
        data.amn.data[0] = data.amn.data[0][:, :5]
    elif corruption == "k_order":
        sym.kpoints_all[0, 0] = 0.5
    elif corruption == "nonfinite":
        data.mmn.data[0][0, 0, 0] = np.nan
    else:
        sym.time_reversals[:] = False
    with pytest.raises(ValueError):
        wannierise_strict(data, sym, num_iter=8)
    assert not data.wannierised


@pytest.mark.real_data
def test_real_srvo3_loader_neighbour_mapping_passes_strict_shape_guard():
    from core.localize import _validated_inputs
    from core.inputs import load_wannier_data

    seed = os.environ.get("SAWF_SRVO3_SEED")
    if not seed:
        pytest.skip("Read-only paths to the real SrVO3 interfaces were not set")
    data, _ = load_wannier_data(seed, source_nb=72)
    geometry = SimpleNamespace(NB=6, num_wann=6, NK=data.mmn.NK,
        kpoints_all=data.chk.kpt_latt.copy(), grid=data.mp_grid,
        time_reversals=np.array([False, True]),
        d_band_block_indices=[np.array([[0, 6]])])
    originals, neighbours = _validated_inputs(data, geometry)
    assert neighbours.shape == (216, 6)
    for ik in range(216):
        np.testing.assert_array_equal(neighbours[ik], data.bkvec.neighbours[ik])
        for name in ("amn", "mmn", "eig"):
            np.testing.assert_array_equal(originals[name][ik], data.get_file(name).data[ik])


def _alignment_fixture(nb=6):
    sym = _symmetrizer(time_reversal=True, nb=nb)
    kpoints = np.array([[0., 0., 0.], [.5, 0., 0.]])
    sym._NK = sym.NKirr = 2
    sym.kpoints_all = kpoints.copy()
    sym.kptirr = np.array([0, 1])
    sym.kptirr2kpt = np.array([[0, 0], [1, 1]])
    sym.kpt2kptirr = np.array([0, 1])
    sym.d_band_block_indices = sym.d_band_block_indices * 2
    sym.d_band_blocks = sym.d_band_blocks * 2
    sym.D_wann_blocks = sym.D_wann_blocks * 2
    sym.symmetrize_WCC = lambda centers: np.zeros_like(centers)
    lattice = np.diag([2., 3., 4.])
    shifts = np.array([[-1, -1, 0], [-1, -1, 0], [0, -1, 0], [0, -1, 0],
                       [-1, -1, -1], [-1, -1, -1], [1, 0, 0], [1, 0, 0]])[:nb]
    phases = np.array([.2, .8, -.4, .1, .3, 1.1, .7, -.2])[:nb]
    rotation = np.diag(np.exp(1j*phases))
    cell_phases = np.exp(-2j*np.pi*kpoints @ shifts.T)
    ordinary = rotation[None] * cell_phases[:, None, :]
    amn = ordinary @ np.diag(np.array([.7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4])[:nb])
    return amn, ordinary, shifts @ lattice, lattice, kpoints, sym, shifts


def test_eight_band_alignment_keeps_reversible_column_and_cell_ledger():
    from core.localize import align_scdm_initial_gauge

    amn, ordinary, centers, lattice, kpoints, sym, shifts = _alignment_fixture(nb=8)
    before = amn.copy()
    initial, report = align_scdm_initial_gauge(amn, ordinary, centers, lattice, kpoints, sym)
    np.testing.assert_array_equal(amn, before)
    np.testing.assert_array_equal(report["cell_shifts"], shifts)
    q = np.array(report["Q_real"]) + 1j*np.array(report["Q_imag"])
    recovered = (initial @ q.conj().T) * np.exp(-2j*np.pi*kpoints @ shifts.T)[:, None, :]
    np.testing.assert_allclose(recovered, ordinary, atol=1e-13, rtol=0)
    np.testing.assert_allclose(initial[0], initial[1], atol=1e-13, rtol=0)


def test_alignment_does_not_claim_support_for_multiple_target_centers():
    from core.localize import align_scdm_initial_gauge

    amn, ordinary, centers, lattice, kpoints, sym, _ = _alignment_fixture(nb=8)
    target = np.zeros((8, 3))
    target[4:, 0] = 0.5
    sym.symmetrize_WCC = lambda unused: target.copy()
    with pytest.raises(ValueError, match="same canonical target center"):
        align_scdm_initial_gauge(amn, ordinary, centers, lattice, kpoints, sym)


def test_alignment_preserves_raw_amn_removes_cell_phases_and_uses_tr_conjugation():
    from core.localize import align_scdm_initial_gauge

    amn, ordinary, centers, lattice, kpoints, sym, shifts = _alignment_fixture()
    before = amn.copy()
    d_before = np.array(sym.D_wann_blocks).copy()
    initial, report = align_scdm_initial_gauge(amn, ordinary, centers, lattice, kpoints, sym)
    np.testing.assert_array_equal(amn, before)
    np.testing.assert_array_equal(np.array(sym.D_wann_blocks), d_before)
    np.testing.assert_array_equal(report["cell_shifts"], shifts)
    np.testing.assert_allclose(initial[1], initial[0], rtol=0, atol=1e-13)
    j = np.kron(np.eye(3), [[0, 1], [-1, 0]])
    np.testing.assert_allclose(j @ initial[0].conj(), initial[0] @ j, rtol=0, atol=1e-13)
    q = np.array(report["Q_real"]) + 1j*np.array(report["Q_imag"])
    undo = q.conj().T[None] * np.exp(-2j*np.pi*kpoints @ shifts.T)[:, None, :]
    np.testing.assert_allclose(initial @ undo, ordinary, rtol=0, atol=1e-13)
    assert report["original_amn_unchanged"] is True


@pytest.mark.parametrize("corruption", ["center", "ordinary_nonunitary", "amn_singular", "no_gamma", "nonfinite", "lattice", "intertwiner_rank"])
def test_alignment_rejects_unproven_or_singular_transformations(corruption):
    from core.localize import align_scdm_initial_gauge

    amn, ordinary, centers, lattice, kpoints, sym, _ = _alignment_fixture()
    if corruption == "center":
        centers[0, 0] += 0.01
    elif corruption == "ordinary_nonunitary":
        ordinary[0, 0, 0] *= 0.5
    elif corruption == "amn_singular":
        amn[0, :, -1] = 0
    elif corruption == "no_gamma":
        kpoints[:, 2] = .25
        sym.kpoints_all = kpoints.copy()
    elif corruption == "nonfinite":
        amn[1, 0, 0] = np.nan
    elif corruption == "lattice":
        lattice[0] = 0
    else:
        ordinary[:] = 1j*np.eye(6)
    with pytest.raises(ValueError):
        align_scdm_initial_gauge(amn, ordinary, centers, lattice, kpoints, sym)


def test_driver_uses_provided_initial_gauge_without_overwriting_amn():
    from core.localize import wannierise_strict

    data = _data()
    initial = np.diag(np.exp(1j*np.array([.2, -.2, .3, -.3, .4, -.4])))[None]
    before = data.amn.data[0].copy()
    report = wannierise_strict(data, _symmetrizer(time_reversal=True), initial_gauge=initial, num_iter=8)
    np.testing.assert_allclose(data.chk.v_matrix[0], initial[0], atol=1e-13, rtol=0)
    np.testing.assert_array_equal(data.amn.data[0], before)
    assert report["parameters"]["init"] == "provided_initial_gauge"


@pytest.mark.parametrize("initial", [np.zeros((1, 6, 6)), np.full((1, 6, 6), np.nan), np.eye(6)])
def test_driver_rejects_invalid_provided_initial_gauge(initial):
    from core.localize import wannierise_strict

    data = _data()
    with pytest.raises(ValueError):
        wannierise_strict(data, _symmetrizer(time_reversal=True), initial_gauge=initial, num_iter=8)
    assert not data.wannierised
