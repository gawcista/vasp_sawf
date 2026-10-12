"""Finite-group reconstruction checks independent of material-specific inputs."""

import json
from itertools import permutations

import numpy as np
import pytest

from test_nonsymmorphic import _two_site_model


def _reconstruct(*args):
    from vasp_sawf.representation import reconstruct_representation

    return reconstruct_representation(*args)


def _adjoint(value):
    return value.conj().swapaxes(-1, -2)


def _metrics(matrices, kmap, anti, product, cocycle):
    group = 0.0
    for g, h in np.ndindex(product.shape):
        composed = matrices[g, kmap[h]] @ (matrices[h].conj() if anti[g] else matrices[h])
        group = max(group, float(np.max(abs(composed - cocycle[g, h, :, None, None]
                                           * matrices[product[g, h]]))))
    unitary = float(np.max(abs(_adjoint(matrices) @ matrices - np.eye(matrices.shape[-1]))))
    return group, unitary


def _nonsymmorphic(kind):
    from vasp_sawf.symmetry import build_symmetry_maps

    options = {
        "screw": {},
        "glide": {"rotation": np.diag([-1, 1, 1]), "spin": np.array([[0., -1j], [-1j, 0.]])},
        "centering": {"rotation": np.eye(3, dtype=int), "spin": np.eye(2)},
    }[kind]
    bundle, group, _, exact = _two_site_model(**options)
    kmap, _ = build_symmetry_maps(bundle, group)
    product, translations, factors = group.get_product_table(get_diff=True)
    cocycle = np.array([[factors[g, h] * np.exp(-2j * np.pi * (
        bundle.kpoints[kmap[product[g, h]]] @ translations[g, h]))
        for h in range(group.size)] for g in range(group.size)])
    anti = np.array([op.time_reversal for op in group.symmetries])
    return exact, kmap, anti, product, cocycle


def _d3_with_scalar_tr():
    # D3 is the full permutation group of three vertices. Its faithful real
    # regular representation is constructed from permutation composition.
    elements = list(permutations(range(3)))
    spatial_product = np.array([[elements.index(tuple(p[q[i]] for i in range(3)))
                                 for q in elements] for p in elements])
    assert spatial_product[1, 2] != spatial_product[2, 1]
    regular = np.zeros((6, 6, 6))
    for g in range(6):
        regular[g, spatial_product[g], np.arange(6)] = 1
    anti = np.repeat([False, True], 6)
    product = np.array([[spatial_product[g % 6, h % 6] + 6 * (anti[g] ^ anti[h])
                         for h in range(12)] for g in range(12)])
    # Two momentum layers make TR act nontrivially on k, commuting with D3.
    kmap = np.array([[elements[g % 6][k % 3] + 3 * ((k // 3) ^ int(anti[g]))
                     for k in range(6)] for g in range(12)])
    rng = np.random.default_rng(34928)
    gauge = np.linalg.qr(rng.normal(size=(6, 6, 6)) + 1j * rng.normal(size=(6, 6, 6)))[0]
    exact = np.array([_adjoint(gauge[kmap[g]]) @ regular[g % 6]
                      @ (gauge.conj() if anti[g] else gauge) for g in range(12)])
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(12, 6)))
    phases[0] = 1
    exact *= phases[:, :, None, None]
    cocycle = np.array([[phases[g, kmap[h]] * (phases[h].conj() if anti[g] else phases[h])
                        / phases[product[g, h]] for h in range(12)] for g in range(12)])
    return exact, kmap, anti, product, cocycle


@pytest.mark.parametrize("kind", ["screw", "glide", "centering"])
def test_exact_spinor_seitz_representation_is_fixed(kind):
    args = _nonsymmorphic(kind)
    corrected, report = _reconstruct(*args)
    np.testing.assert_allclose(corrected, args[0], atol=3e-13, rtol=0)
    assert report["projector_gap_min"] > .99
    assert report["selected_cluster_min"] > .99
    assert report["discarded_cluster_max"] < 1e-12


@pytest.mark.parametrize("kind", ["screw", "glide", "centering"])
def test_noisy_spinor_seitz_representation_recovers_complete_algebra(kind):
    exact, kmap, anti, product, cocycle = _nonsymmorphic(kind)
    rng = np.random.default_rng(49)
    raw = exact + 1e-4 * (rng.normal(size=exact.shape) + 1j * rng.normal(size=exact.shape))
    args = raw, kmap, anti, product, cocycle
    originals = [value.copy() for value in args]
    for value in args:
        value.flags.writeable = False
    corrected, report = _reconstruct(*args)
    assert corrected.shape == raw.shape
    raw_group, raw_unitary = _metrics(*args)
    group, unitary = _metrics(corrected, *args[1:])
    assert raw_group > 1e-4 and raw_unitary > 1e-4
    assert group < 3e-12 and unitary < 3e-12
    np.testing.assert_allclose(corrected, exact, atol=1e-3, rtol=0)
    assert report["raw"]["group_entry_max"] == pytest.approx(raw_group)
    assert report["raw"]["unitarity_entry_max"] == pytest.approx(raw_unitary)
    assert report["corrected"]["group_entry_max"] == pytest.approx(group)
    assert report["corrected"]["unitarity_entry_max"] == pytest.approx(unitary)
    assert report["correction_entry_max"] == pytest.approx(np.max(abs(corrected - raw)))
    assert report["invariant_projector_covariance_max"] < 3e-12
    assert report["embedding_isometry_max"] < 3e-12
    assert report["diagnostic_only"] is True
    assert report["physical_acceptance"] == "not_assessed"
    json.dumps(report, allow_nan=False)
    for value, original in zip(args, originals):
        np.testing.assert_array_equal(value, original)


def test_independent_nonabelian_d3_scalar_tr_with_complex_cocycle():
    exact, kmap, anti, product, cocycle = _d3_with_scalar_tr()
    assert _metrics(exact, kmap, anti, product, cocycle)[0] < 1e-12
    corrected, report = _reconstruct(exact, kmap, anti, product, cocycle)
    np.testing.assert_allclose(corrected, exact, atol=2e-12, rtol=0)
    assert report["projector_gap_min"] > .99
    rng = np.random.default_rng(171)
    raw = exact + 1e-3 * (rng.normal(size=exact.shape) + 1j * rng.normal(size=exact.shape))
    corrected, report = _reconstruct(raw, kmap, anti, product, cocycle)
    assert _metrics(raw, kmap, anti, product, cocycle)[0] > 1e-3
    group, unitary = _metrics(corrected, kmap, anti, product, cocycle)
    assert group < 3e-12 and unitary < 3e-12
    np.testing.assert_allclose(corrected, exact, atol=2e-3, rtol=0)


@pytest.mark.parametrize("example", ["screw", "d3"])
def test_reconstruction_is_covariant_under_arbitrary_bloch_basis_changes(example):
    exact, kmap, anti, product, cocycle = _d3_with_scalar_tr() if example == "d3" else _nonsymmorphic(example)
    rng = np.random.default_rng(23)
    raw = exact + 1e-4 * (rng.normal(size=exact.shape) + 1j * rng.normal(size=exact.shape))
    gauge = np.linalg.qr(rng.normal(size=exact.shape[1:]) + 1j * rng.normal(size=exact.shape[1:]))[0]

    def change(matrices):
        return np.array([_adjoint(gauge[kmap[g]]) @ matrices[g]
                         @ (gauge.conj() if anti[g] else gauge) for g in range(len(anti))])

    base, _ = _reconstruct(raw, kmap, anti, product, cocycle)
    changed, _ = _reconstruct(change(raw), kmap, anti, product, cocycle)
    np.testing.assert_allclose(changed, change(base), atol=5e-12, rtol=0)


def test_operation_order_does_not_select_a_preferred_identity_index():
    exact, kmap, anti, product, cocycle = _d3_with_scalar_tr()
    order = np.roll(np.arange(len(anti)), 4)
    inverse = np.argsort(order)
    corrected, _ = _reconstruct(exact[order], kmap[order], anti[order],
                                inverse[product[order][:, order]], cocycle[order][:, order])
    np.testing.assert_allclose(corrected, exact[order], atol=3e-12, rtol=0)


def test_trivial_group_retains_all_bands_and_has_no_discarded_cluster():
    raw = np.array([[[[2., .1j], [.2, 1.]]]], complex)
    corrected, report = _reconstruct(raw, [[0]], [False], [[0]], [[[1.]]])
    np.testing.assert_allclose(corrected, np.eye(2)[None, None], atol=2e-14)
    assert report["discarded_cluster_max"] is None
    assert report["projector_gap_min"] == 1.


def test_scalar_non_normalized_identity_cocycle_is_preserved():
    phase = np.exp(.7j)
    exact = np.array([[[[phase]]]])
    corrected, _ = _reconstruct(exact, [[0]], [False], [[0]], [[[phase]]])
    np.testing.assert_allclose(corrected, exact, atol=1e-14)


@pytest.mark.parametrize("invalid", [
    "raw_shape", "raw_nonsquare", "raw_empty", "raw_nan", "cocycle_shape", "cocycle_nan",
    "cocycle_modulus", "cocycle_relation", "kmap_shape", "kmap_fractional", "kmap_negative",
    "kmap_duplicate", "kmap_nan", "product_shape", "product_fractional", "product_range",
    "product_duplicate", "anti_shape", "anti_non_boolean", "anti_nan", "anti_product",
])
def test_malformed_inputs_raise_value_error(invalid):
    raw, kmap, anti, product, cocycle = _nonsymmorphic("screw")
    if invalid == "raw_shape":
        raw = raw[0]
    elif invalid == "raw_nonsquare":
        raw = raw[..., :2]
    elif invalid == "raw_empty":
        raw = raw[:0]
    elif invalid == "raw_nan":
        raw[0, 0, 0, 0] = np.nan
    elif invalid == "cocycle_shape":
        cocycle = cocycle[0]
    elif invalid == "cocycle_nan":
        cocycle[0, 0, 0] = np.nan
    elif invalid == "cocycle_modulus":
        cocycle[0, 0, 0] = .5
    elif invalid == "cocycle_relation":
        cocycle[1, 2, 0] *= np.exp(.2j)
    elif invalid == "kmap_shape":
        kmap = kmap[0]
    elif invalid == "kmap_fractional":
        kmap = kmap.astype(float) + .25
    elif invalid == "kmap_negative":
        kmap[0, 0] = -1
    elif invalid == "kmap_duplicate":
        kmap[0, 1] = kmap[0, 0]
    elif invalid == "kmap_nan":
        kmap = kmap.astype(float)
        kmap[0, 0] = np.nan
    elif invalid == "product_shape":
        product = product[0]
    elif invalid == "product_fractional":
        product = product.astype(float) + .25
    elif invalid == "product_range":
        product[0, 0] = len(anti)
    elif invalid == "product_duplicate":
        product[0, 1] = product[0, 0]
    elif invalid == "anti_shape":
        anti = anti[:, None]
    elif invalid == "anti_non_boolean":
        anti = anti.astype(int)
        anti[0] = 2
    elif invalid == "anti_nan":
        anti = anti.astype(float)
        anti[0] = np.nan
    elif invalid == "anti_product":
        anti[0] = True
    with pytest.raises(ValueError):
        _reconstruct(raw, kmap, anti, product, cocycle)


def test_nonassociative_latin_table_with_identity_is_rejected():
    product = np.array([[0, 1, 2, 3, 4], [1, 0, 3, 4, 2], [2, 3, 4, 0, 1],
                        [3, 4, 1, 2, 0], [4, 2, 0, 1, 3]])
    with pytest.raises(ValueError, match="associat"):
        _reconstruct(np.ones((5, 1, 1, 1)), np.zeros((5, 1), int), np.zeros(5, bool),
                     product, np.ones((5, 5, 1)))


def test_table_without_two_sided_identity_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        _reconstruct(np.ones((2, 1, 1, 1)), np.zeros((2, 1), int), [False, False],
                     [[0, 1], [0, 1]], np.ones((2, 2, 1)))


def test_valid_permutation_that_disagrees_with_group_action_is_rejected():
    raw, kmap, anti, product, cocycle = _nonsymmorphic("screw")
    kmap[0] = np.roll(kmap[0], 1)
    with pytest.raises(ValueError, match="k.point|action"):
        _reconstruct(raw, kmap, anti, product, cocycle)


def test_singular_embedding_is_rejected_without_dropping_bands():
    raw, *args = _nonsymmorphic("screw")
    raw[..., -1, :] = 0
    with pytest.raises(ValueError, match="rank"):
        _reconstruct(raw, *args)


def test_absent_rank_nb_spectral_gap_is_rejected():
    # Averaging |e><e| over a two-element regular action gives I/2.
    with pytest.raises(ValueError, match="gap"):
        _reconstruct(np.array([1., 0.]).reshape(2, 1, 1, 1), [[0], [0]],
                     [False, False], [[0, 1], [1, 0]], np.ones((2, 2, 1)))
