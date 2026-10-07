"""The configurable lattice tolerance must never become a matrix acceptance threshold."""

import json
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('arguments,expected', [([], 1e-5), (['--tol=1e-6'], 1e-6)])
def test_cli_forwards_default_and_explicit_lattice_tolerance(tmp_path, monkeypatch, arguments, expected):
    from vasp_sawf.extract_symmetry import main
    import vasp_sawf.symmetry as symmetry

    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return {'status': 'ready'}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(symmetry, 'export_symmetry', capture)
    monkeypatch.setattr(sys, 'argv', ['sawf-extract', *arguments])
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'numba'))
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    assert main() == 0
    assert len(calls) == 1
    assert calls[0]['tol'] == expected


@pytest.mark.parametrize('tol', [np.nan, np.inf, -np.inf, 0, -1e-5, True, False, np.bool_(True)])
def test_invalid_lattice_tolerance_is_rejected_before_wavecar_headers(tmp_path, monkeypatch, tol):
    import vasp_sawf.symmetry as symmetry

    def forbid_read(*args, **kwargs):
        raise AssertionError('Invalid tolerance must be rejected before WAVECAR reading')

    monkeypatch.setattr(symmetry, 'inspect_wavecar', forbid_read)
    with pytest.raises(ValueError, match='tol|tolerance'):
        symmetry.export_symmetry(tmp_path / 'wannier90', tmp_path / 'WAVECAR',
                                  tmp_path / 'OUTCAR', tmp_path / 'symmetry', tol=tol)


def test_larger_lattice_tolerance_does_not_relax_mmn_matrix_gate(tmp_path, monkeypatch):
    from test_scalar_extraction import _atomic_inputs
    import vasp_sawf.symmetry as symmetry

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, 1, 1)
    original = symmetry.transport_sewing

    def measured_matrix_failure(*args, **kwargs):
        sewing, report = original(*args, **kwargs)
        report['mmn_covariance_max'] = 2e-6
        return sewing, report

    monkeypatch.setattr(symmetry, 'transport_sewing', measured_matrix_failure)
    output = tmp_path / 'symmetry'
    with pytest.raises(ValueError, match='mmn_covariance_max failed'):
        symmetry.export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                                  workers=1, memory_gb=4, tol=1e-3)
    report = json.loads((output / 'report.json').read_text())
    assert report['numerical_gate'] == 1e-6
    assert report['residuals']['mmn_covariance_max'] == 2e-6
    assert not report['sawf_ready']
    assert not (output / 'bloch.npz').exists()
