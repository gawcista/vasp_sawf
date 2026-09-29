import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _analytic_system():
    vectors = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    matrices = np.zeros((5, 2, 2), dtype=complex)
    matrices[0] = [[1.2, 0.3], [0.3, -0.7]]
    matrices[1] = [[0.4, 0.2j], [0, -0.1]]
    matrices[2] = matrices[1].conj().T
    matrices[3] = [[0.1, 0], [0, 0.2]]
    matrices[4] = matrices[3].conj().T
    return SimpleNamespace(rvec=SimpleNamespace(iRvec=vectors), num_wann=2,
                           get_R_mat=lambda key: matrices)


def test_fourier_matches_independent_analytic_hamiltonian_and_eigenvalues():
    from vasp_sawf.bands import evaluate_hamiltonian, evaluate_bands

    k = np.array([[0.173, -0.219, 0.312], [0.41, 0.227, -0.29], [0, 0, 0]])
    x, y = (2*np.pi*k[:, axis] for axis in (0, 1))
    a = 1.2 + 0.8*np.cos(x) + 0.2*np.cos(y)
    d = -0.7 - 0.2*np.cos(x) + 0.4*np.cos(y)
    b = 0.3 + 0.2j*np.exp(1j*x)
    expected = np.array([[a, b], [b.conj(), d]]).transpose(2, 0, 1)
    split = np.sqrt((a-d)**2 + 4*abs(b)**2)/2
    expected_e = np.array([(a+d)/2-split, (a+d)/2+split]).T
    system = _analytic_system()
    before = system.get_R_mat("Ham").copy()
    np.testing.assert_allclose(evaluate_hamiltonian(system, k), expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(evaluate_bands(system, k), expected_e, atol=1e-14, rtol=0)
    np.testing.assert_allclose(evaluate_hamiltonian(system, k+[2, -3, 1]), expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(system.get_R_mat("Ham"), before)


@pytest.mark.parametrize("kpoints", [np.zeros(3), np.ones((2, 2)), [[float("nan"), 0, 0]]])
def test_invalid_path_rejected(kpoints):
    from vasp_sawf.bands import evaluate_hamiltonian

    with pytest.raises(ValueError, match="k points"):
        evaluate_hamiltonian(_analytic_system(), kpoints)


def test_nonhermitian_input_is_rejected_without_averaging():
    from vasp_sawf.bands import evaluate_bands

    system = _analytic_system()
    system.get_R_mat("Ham")[0, 0, 1] += 0.01
    with pytest.raises(ValueError, match="Hermiticity"):
        evaluate_bands(system, [[0.13, 0.17, 0]])


def test_fourier_matches_official_single_k_fft_without_hermitian_projection():
    from vasp_sawf.bands import evaluate_hamiltonian
    from wannierberri.fourier.rvectors import Rvectors

    system = _analytic_system()
    rvec = Rvectors(lattice=np.eye(3), iRvec=system.rvec.iRvec)
    kpoints = [[0.17, 0.31, -0.13], [0.49, 0.27, 0.4]]
    reference = []
    for k in kpoints:
        rvec.set_fft_R_to_k(NK=(1, 1, 1), num_wann=2, fftlib="numpy", dK=k)
        reference.append(rvec.R_to_k(rvec.apply_expdK(system.get_R_mat("Ham").copy()), hermitian=False)[0])
    np.testing.assert_allclose(evaluate_hamiltonian(system, kpoints), reference, rtol=0, atol=1e-14)


def test_workflow_rejects_output_inside_protected_input_tree(tmp_path):
    from vasp_sawf.localize import _new_output
    source = tmp_path / 'inputs'
    source.mkdir()
    rejected = source / "protected_input"
    with pytest.raises(ValueError, match="Run output"):
        _new_output(rejected, [source / 'wannier90.mmn'])
    assert not rejected.exists()
