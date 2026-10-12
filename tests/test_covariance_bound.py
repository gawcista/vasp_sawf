"""Independent finite-Fourier checks; no DFT accuracy assertion is involved."""

import numpy as np
import pytest


def _fourier(points, vectors, matrices):
    return np.einsum('kr,rij->kij', np.exp(2j*np.pi*np.asarray(points) @ vectors.T), matrices)


def test_spinless_tr_sine_residual_and_global_bound():
    from vasp_sawf.covariance import covariance_coefficients, certify_covariance

    # H(k) = sin(2 pi kx); spinless TR error is -2 sin(2 pi kx).
    r = np.array([[1, 0, 0], [-1, 0, 0]])
    h = np.array([-0.5j, 0.5j])[:, None, None]
    before = h.copy()
    n, delta = covariance_coefficients(r, h, -np.eye(3), True, [[0, 0, 0]], [[[1]]])
    points = np.array([[.125, 0, 0], [.25, 0, 0], [-.25, 0, 0]])
    np.testing.assert_allclose(_fourier(points, n, delta)[:, 0, 0], [-np.sqrt(2), -2, 2], atol=1e-14)
    np.testing.assert_array_equal(h, before)
    report = certify_covariance(n, delta, active_axes=[0], grid_side=9)
    assert report['triangle_upper_ev'] == pytest.approx(2)
    assert report['derivative_bounds_ev'] == pytest.approx([4*np.pi, 0, 0])
    assert report['spectral_grid_max_ev'] == pytest.approx(2*np.cos(np.pi/18))
    assert 2 <= report['uniform_upper_ev'] < 2+1e-10
    assert report['grid_points'] == 9
    assert report['floating_point_allowance_ev'] > 0


@pytest.mark.parametrize('anti', [False, True])
def test_cell_phase_permutation_matches_direct_matrix_residual(anti):
    from vasp_sawf.covariance import covariance_coefficients, certify_covariance

    rng = np.random.default_rng(107)
    r = np.array([[0, 0, 0], [1, -2, 3], [-1, 2, -3]])
    hopping = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
    onsite = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
    h = np.array([onsite+onsite.conj().T, hopping, hopping.conj().T])
    q = np.array([[0, -1, 0], [1, 1, 0], [0, 0, -1]])
    translations = np.array([[0, 0, 0], [1, -2, 3]])
    parts = np.array([[[0, 1j], [0, 0]], [[0, 0], [1, 0]]])
    n, delta = covariance_coefficients(r, h, q, anti, translations, parts)
    k = rng.uniform(-.5, .5, (71, 3))
    kg = k @ q
    # Direct site permutation, with its full cell phase written independently.
    d = np.zeros((len(k), 2, 2), complex)
    d[:, 0, 1] = 1j
    d[:, 1, 0] = np.exp(2j*np.pi*(kg[:, 0]-2*kg[:, 1]+3*kg[:, 2]))
    source = _fourier(k, r, h)
    expected = _fourier(kg, r, h)-d @ (source.conj() if anti else source) @ d.conj().swapaxes(-1, -2)
    np.testing.assert_allclose(_fourier(k, n, delta), expected, atol=1e-12, rtol=0)
    report = certify_covariance(n, delta, active_axes=[0, 1, 2], grid_side=7, chunk_size=23)
    assert np.linalg.norm(expected, ord=2, axis=(-2, -1)).max() <= report['uniform_upper_ev']
    assert report['grid_points'] == 343
    assert len(set(map(tuple, n))) == len(n)


def test_plane_bound_keeps_z_terms_and_all_numerically_small_coefficients():
    from vasp_sawf.covariance import covariance_coefficients, certify_covariance

    n, delta = covariance_coefficients([[0, 0, 1]], [[[1e-30j]]], -np.eye(3), True,
                                      [[0, 0, 0]], [[[1]]])
    assert (np.abs(n[:, 2]) == 1).any()
    assert np.abs(delta).sum() == pytest.approx(2e-30, abs=1e-45)
    # Distinct z harmonics cancel identically on the z=0 plane only.
    frequencies = np.array([[1, 0, 7], [1, 0, -4]])
    coefficients = np.array([1, -1], complex)[:, None, None]
    report = certify_covariance(frequencies, coefficients, active_axes=[0, 1], grid_side=11)
    assert report['spectral_grid_max_ev'] == 0
    assert report['derivative_bounds_ev'][2] == pytest.approx(22*np.pi)
    assert report['grid_lipschitz_upper_ev'] == pytest.approx(2*np.pi/11)
    assert report['fixed_inactive_coordinates'] == {'2': 0.0}


def test_grid_lipschitz_bound_encloses_dense_grid_including_periodic_boundary():
    from vasp_sawf.covariance import certify_covariance

    n = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    coefficients = np.array([1, 0.3j, -0.3j])[:, None, None]
    report = certify_covariance(n, coefficients, active_axes=[0], grid_side=17, chunk_size=3)
    dense = np.zeros((10001, 3))
    dense[:, 0] = np.linspace(-.5, .5, len(dense))
    maximum = np.abs(_fourier(dense, n, coefficients)).max()
    assert report['spectral_grid_max_ev'] <= maximum <= report['uniform_upper_ev']
    assert report['grid_lipschitz_upper_ev'] == pytest.approx(
        report['spectral_grid_max_ev']+report['derivative_bounds_ev'][0]/34)


@pytest.mark.parametrize('field,value', [
    ('R', [[.1, 0, 0]]), ('Q', np.diag([1, 1, 2])), ('anti', 1),
    ('translations', [[0, 0, np.nan]]), ('matrices', [[[np.inf]]]),
    ('H_R', [[[np.nan]]]), ('H_R', np.ones((1, 2, 1))),
])
def test_rejects_invalid_fourier_inputs(field, value):
    from vasp_sawf.covariance import covariance_coefficients

    arguments = dict(R=[[0, 0, 0]], H_R=[[[1]]], Q=np.eye(3), anti=False,
                     translations=[[0, 0, 0]], matrices=[[[1]]])
    arguments[field] = value
    with pytest.raises(ValueError):
        covariance_coefficients(**arguments)


@pytest.mark.parametrize('options', [dict(active_axes=[0, 0]), dict(active_axes=[3]),
    dict(active_axes=[False]), dict(active_axes=[0], grid_side=0),
    dict(active_axes=[0], grid_side=1.5), dict(active_axes=[0, 1, 2], grid_side=100),
    dict(active_axes=[0], chunk_size=2049), dict(active_axes=[0], max_points=300001)])
def test_rejects_unbounded_or_invalid_grid(options):
    from vasp_sawf.covariance import certify_covariance

    with pytest.raises(ValueError):
        certify_covariance([[0, 0, 0]], [[[1]]], **options)
