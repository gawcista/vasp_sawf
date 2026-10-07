"""The selected WAVECAR/EIG energy tolerance is explicit and independent of matrix checks."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('arguments,expected', [([], 1e-8), (['--energy-tol=1e-4'], 1e-4)])
def test_cli_forwards_default_and_explicit_energy_tolerance(tmp_path, monkeypatch, arguments, expected):
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
    assert len(calls) == 1 and calls[0]['energy_tol'] == expected


@pytest.mark.parametrize('energy_tol', [np.nan, np.inf, -np.inf, 0, -1e-8, True, False, np.bool_(True)])
def test_invalid_energy_tolerance_fails_before_header_read(tmp_path, monkeypatch, energy_tol):
    import vasp_sawf.symmetry as symmetry

    def forbid_read(*args, **kwargs):
        raise AssertionError('Invalid energy tolerance must be rejected before WAVECAR reading')

    monkeypatch.setattr(symmetry, 'inspect_wavecar', forbid_read)
    with pytest.raises(ValueError, match='energy.*tol|energy.*tolerance'):
        symmetry.export_symmetry(tmp_path / 'wannier90', tmp_path / 'WAVECAR',
            tmp_path / 'OUTCAR', tmp_path / 'symmetry', energy_tol=energy_tol)


@pytest.mark.parametrize('delta,energy_tol,accepted', [
    (5e-9, None, True), (5e-5, None, False), (5e-5, 1e-4, True), (5e-9, 1e-10, False),
])
def test_energy_tolerance_controls_only_recorded_energy_difference(tmp_path, monkeypatch, delta, energy_tol, accepted):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.symmetry import export_symmetry
    import vasp_sawf.wavecar as wavecar

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, 1, 1)
    Path(f'{seed}.eig').write_text('1 1 0.000000000000\n')
    eig_before = Path(f'{seed}.eig').read_bytes()
    with (source / 'WAVECAR').open('r+b') as handle:
        handle.seek(2 * 512 + 4 * 8)
        np.array([delta], dtype='<f8').tofile(handle)
    options = {} if energy_tol is None else {'energy_tol': energy_tol}
    output = tmp_path / 'symmetry'
    if accepted:
        report = export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                                  workers=1, memory_gb=4, **options)
        assert report['sawf_ready'] and (output / 'bloch.npz').is_file()
    else:
        def forbid_coefficients(*args, **kwargs):
            raise AssertionError('Energy mismatch must fail before coefficient reading')

        monkeypatch.setattr(wavecar, 'read_selected_wavecar', forbid_coefficients)
        with pytest.raises(ValueError, match='wavecar_interface_energy_max_ev'):
            export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                            workers=1, memory_gb=4, **options)
        report = json.loads((output / 'report.json').read_text())
        assert not report['sawf_ready'] and not (output / 'bloch.npz').exists()
    assert report['wavecar_interface_energy_tolerance_ev'] == (1e-8 if energy_tol is None else energy_tol)
    assert report['residuals']['wavecar_interface_energy_max_ev'] == delta
    assert Path(f'{seed}.eig').read_bytes() == eig_before
    worst = report['energy_comparison']
    for key in ('interface_kpoint_1based', 'wavecar_kpoint_1based', 'compact_band_1based', 'vasp_band_1based'):
        assert worst[key] == 1
    assert worst['wavecar_energy_ev'] == delta
    assert worst['interface_energy_ev'] == 0.
    assert worst['difference_ev'] == delta


def test_larger_energy_tolerance_does_not_relax_mmn_gate(tmp_path, monkeypatch):
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
                                  workers=1, memory_gb=4, energy_tol=1e-3)
    report = json.loads((output / 'report.json').read_text())
    assert report['numerical_gate'] == 1e-6
    assert report['residuals']['mmn_covariance_max'] == 2e-6
    assert not report['sawf_ready'] and not (output / 'bloch.npz').exists()


def test_coarse_fortran_eig_format_is_not_an_extra_energy_acceptance_gate(tmp_path):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, 1, 1)
    Path(f'{seed}.eig').write_text('1 1 0.0D+00\n')
    report = export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', tmp_path / 'symmetry',
                             workers=1, memory_gb=4)
    assert report['sawf_ready']
    assert report['wavecar_interface_energy_tolerance_ev'] == 1e-8
    assert report['residuals']['wavecar_interface_energy_max_ev'] == 0.
