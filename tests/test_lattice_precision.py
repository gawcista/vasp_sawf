"""Lattice serialization checks distinguish native WIN rounding from a different cell."""

import json
import os
from pathlib import Path

import numpy as np
import pytest


def _win(lattice, *, digits=7, unit='ang'):
    rows = '\n'.join(' '.join(f'{value:.{digits}f}' for value in row) for row in lattice)
    return ('num_bands=1\nnum_wann=1\nmp_grid=1 1 1\n'
            f'begin unit_cell_cart\n{unit}\n{rows}\nend unit_cell_cart\n'
            'begin kpoints\n0 0 0\nend kpoints\n')


def test_native_vasp_seven_decimal_lattice_rounding_passes_without_transposing():
    from vasp_sawf.inputs import _read_win
    from vasp_sawf.wavecar import check_lattice_match

    exact = np.array([[4.123456749, .123456749, 0], [-.234567849, 3.234567849, 0], [0, 0, 5.456789149]])
    text = _win(exact)
    printed = _read_win(text)[0]
    assert not np.allclose(exact, printed, atol=1e-8, rtol=0)
    result = check_lattice_match(exact, printed)
    assert result['matches']
    assert result['max_abs_difference_angstrom'] == pytest.approx(4.9e-8, abs=1e-15)
    np.testing.assert_array_equal(result['wavecar_lattice_angstrom'], exact)
    np.testing.assert_array_equal(result['win_lattice_angstrom'], printed)
    np.testing.assert_array_equal(_read_win(text)[0], printed)
    with pytest.raises(ValueError, match='WAVECAR lattice disagrees'):
        check_lattice_match(exact.T, printed)


@pytest.mark.parametrize('digits,delta', [(7, 1.1e-5), (12, 1.1e-5), (0, .1)])
def test_real_cell_changes_cannot_be_hidden_by_rounding_or_coarse_text(digits, delta):
    from vasp_sawf.inputs import _read_win
    from vasp_sawf.wavecar import check_lattice_match

    text = _win(np.diag([2., 3., 4.]), digits=digits)
    printed = _read_win(text)[0]
    different = printed.copy()
    different[0, 0] += delta
    with pytest.raises(ValueError, match='WAVECAR lattice disagrees') as caught:
        check_lattice_match(different, printed)
    message = str(caught.value)
    assert 'WAVECAR' in message and 'WIN' in message and 'difference' in message
    assert caught.value.lattice_comparison['matches'] is False
    assert caught.value.lattice_comparison['worst_component_1based'] == [1, 1]


def test_bohr_precision_and_fortran_exponents_are_converted_with_the_lattice():
    from vasp_sawf.inputs import _read_win, _BOHR_ANGSTROM
    from vasp_sawf.wavecar import check_lattice_match

    exact_bohr = np.diag([4.123456749, 6.234567849, 8.456789149])
    text = _win(exact_bohr, unit='bohr').replace('4.1234567', '4.1234567D+00')
    result = check_lattice_match(exact_bohr * _BOHR_ANGSTROM, _read_win(text)[0])
    assert result['matches']
    assert result['max_abs_difference_angstrom'] < 2.65e-8


def test_explicit_lattice_tolerance_controls_the_comparison_without_a_precision_override():
    from vasp_sawf.wavecar import check_lattice_match

    reference = np.diag([2., 3., 4.])
    shifted = reference.copy()
    shifted[0, 0] += 5e-6
    assert check_lattice_match(shifted, reference)['absolute_tolerance_angstrom'] == 1e-5
    with pytest.raises(ValueError, match='WAVECAR lattice disagrees'):
        check_lattice_match(shifted, reference, tol=1e-6)


def test_reader_error_includes_both_cells_before_any_coefficient_read(tmp_path):
    from test_wavecar import _write_wavecar
    from vasp_sawf.wavecar import inspect_selected_wavecar, read_selected_wavecar

    path, lattice, _ = _write_wavecar(tmp_path)
    wrong = lattice.copy()
    wrong[0, 0] += .1
    for reader, kwargs in [(inspect_selected_wavecar, {}), (read_selected_wavecar, {'kpoints_1based': [1]})]:
        with pytest.raises(ValueError, match='difference') as caught:
            reader(path, bands_1based=[1], lattice=wrong, **kwargs)
        assert caught.value.lattice_comparison['wavecar_lattice_angstrom'] == lattice.tolist()
        assert caught.value.lattice_comparison['win_lattice_angstrom'] == wrong.tolist()


def test_full_extraction_uses_validated_binary_cell_for_coefficient_reads(tmp_path, monkeypatch):
    from test_scalar_extraction import _atomic_inputs
    import vasp_sawf.wavecar as reader
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, 1, 1)
    win_path = Path(f'{seed}.win')
    win_path.write_text(win_path.read_text().replace('2 0 0\n0 3 0\n0 0 4',
                        '2.0000000 0.0000000 0.0000000\n0.0000000 3.0000000 0.0000000\n0.0000000 0.0000000 4.0000000'))
    original_win = win_path.read_bytes()
    with (source / 'WAVECAR').open('r+b') as handle:
        handle.seek(512 + 3*8)
        np.array([2.000000049], dtype='<f8').tofile(handle)
    real_read = reader.read_selected_wavecar
    observed = []
    def checked_read(*args, **kwargs):
        observed.append(np.asarray(kwargs['lattice']).copy())
        return real_read(*args, **kwargs)
    monkeypatch.setattr(reader, 'read_selected_wavecar', checked_read)
    report = export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', tmp_path / 'symmetry',
                             workers=1, memory_gb=4)
    assert report['sawf_ready'] and report['lattice_comparison']['matches']
    assert observed and observed[0][0, 0] == 2.000000049
    assert win_path.read_bytes() == original_win


def test_failed_extraction_saves_lattice_diagnostics_and_never_reads_coefficients(tmp_path, monkeypatch):
    from test_scalar_extraction import _atomic_inputs
    import vasp_sawf.wavecar as reader
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, 1, 1)
    with (source / 'WAVECAR').open('r+b') as handle:
        handle.seek(512 + 3*8)
        np.array([2.1], dtype='<f8').tofile(handle)
    def forbidden(*args, **kwargs):
        raise AssertionError('A different cell reached coefficient I/O')
    monkeypatch.setattr(reader, 'read_selected_wavecar', forbidden)
    output = tmp_path / 'symmetry'
    with pytest.raises(ValueError, match='WAVECAR lattice disagrees'):
        export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output, workers=1)
    report = json.loads((output / 'report.json').read_text())
    assert not report['sawf_ready'] and not report['lattice_comparison']['matches']
    assert report['lattice_comparison']['wavecar_lattice_angstrom'][0][0] == 2.1
    assert not (output / 'bloch.npz').exists()


@pytest.mark.real_data
def test_real_tsns_poscar_and_win_differ_only_by_native_lattice_serialization():
    from pymatgen.io.vasp.inputs import Poscar
    from vasp_sawf.inputs import _read_win
    from vasp_sawf.wavecar import check_lattice_match

    fixture = os.environ.get('SAWF_TSNS_LATTICE_FIXTURE')
    if not fixture:
        pytest.skip('Small read-only tSnS POSCAR/WIN fixture was not specified')
    source = Path(fixture)
    exact = Poscar.from_file(source / 'POSCAR', check_for_potcar=False).structure.lattice.matrix
    text = (source / 'wannier90.win').read_text()
    printed = _read_win(text)[0]
    assert not np.allclose(exact, printed, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(np.round(exact, 7), printed)
    result = check_lattice_match(exact, printed)
    assert result['matches']
    assert result['max_abs_difference_angstrom'] == pytest.approx(4.9834420146055436e-8)
