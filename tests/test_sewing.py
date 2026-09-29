import importlib.util

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('vasp_sawf.symmetry') is not None
    from vasp_sawf import symmetry as sewing
    return sewing


def connection():
    rng = np.random.default_rng(28)
    q = np.stack([np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))[0] for _ in range(6)])
    neighbors = np.array([[(k+1)%6, (k-1)%6] for k in range(6)])
    mmn = np.array([[q[k].conj().T @ q[j]*0.9 for j in neighbors[k]] for k in range(6)])
    kmap = np.array([np.arange(6), (-np.arange(6))%6])
    emap = np.array([np.tile([0, 1], (6, 1)), np.tile([1, 0], (6, 1))])
    spin = np.array([[0, 1], [-1, 0]])
    exact = np.array([np.tile(np.eye(2), (6, 1, 1)),
                      [q[kmap[1,k]].conj().T @ spin @ q[k].conj() for k in range(6)]])
    return mmn, neighbors, kmap, emap, exact


def test_antiunitary_transport_matches_independent_known_gauges():
    m, n, kmap, emap, expected = connection()
    d, report = module().transport_sewing(m, n, kmap, emap, [False, True], expected[:, 0])
    np.testing.assert_allclose(d, expected, atol=3e-14)
    assert report['mmn_covariance_max'] < 3e-14
    assert report['unitarity_max'] < 3e-14


def test_omitting_conjugation_is_detected_against_known_antiunitary():
    m, n, kmap, emap, expected = connection()
    wrong, _ = module().transport_sewing(m, n, kmap, emap, [False, False], expected[:, 0])
    assert np.max(abs(wrong[1]-expected[1])) > 0.1


def test_reverse_spanning_tree_agrees():
    m, n, kmap, emap, expected = connection()
    forward, _ = module().transport_sewing(m,n,kmap,emap,[False,True],expected[:,0])
    reverse, _ = module().transport_sewing(m,n,kmap,emap,[False,True],expected[:,0],reverse_edges=True)
    np.testing.assert_allclose(reverse,forward,atol=3e-14)


def test_singular_connection_fails_without_regularization():
    m,n,kmap,emap,expected=connection()
    m[0,0]=0
    with pytest.raises(ValueError,match='singular'):
        module().transport_sewing(m,n,kmap,emap,[False,True],expected[:,0])


def test_non_tree_corruption_cannot_be_hidden_by_successful_transport():
    m,n,kmap,emap,expected=connection()
    m[3,0]*=1.1
    _,report=module().transport_sewing(m,n,kmap,emap,[False,True],expected[:,0])
    assert report['mmn_covariance_max'] > 0.01


def test_wrong_edge_target_is_rejected():
    m,n,kmap,emap,expected=connection()
    emap[1,2,0]=0
    with pytest.raises(ValueError,match='endpoint'):
        module().transport_sewing(m,n,kmap,emap,[False,True],expected[:,0])


@pytest.mark.parametrize('field', ['d', 'factor'])
def test_group_residual_rejects_nan_instead_of_reporting_zero(field):
    d = np.ones((1, 1, 1, 1), complex)
    factors = np.ones((1, 1))
    (d if field == 'd' else factors).flat[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        module().group_residuals(d, np.array([[0]]), [False], np.array([[0]]), factors)
