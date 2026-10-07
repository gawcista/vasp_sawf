"""Analytic scalar-channel checks, independent of the WAVECAR reader writer tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_scalar_transform_keeps_translation_and_conjugation_without_spin_rotation():
    from vasp_sawf.symmetry import _independent_transform

    point = SimpleNamespace(spinor=False, k=np.zeros(3),
                            ig=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
                            WF=np.array([[[1+2j], [3+4j], [5+6j]]]))
    operation = SimpleNamespace(rotation=np.eye(3), translation=np.array([.25, 0, 0]),
                                time_reversal=True)
    actual = _independent_transform(point, operation)
    expected = point.WF[:, [0, 2, 1]].conj() * np.array([1, -1j, 1j])[None, :, None]
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_scalar_dimension_and_channel_binding():
    from vasp_sawf.symmetry import _closed_spinor_dimension, _spin_spec

    assert _closed_spinor_dimension(np.ones((1, 3, 3)), spinor=False) == 3
    scalar = {'spinor': False, 'source_ispin': 2}
    spec = _spin_spec(scalar, Path('wannier90.2'), None, 2)
    assert spec == dict(spinor=False, source_ispin=2, spin_channel=2,
                        antiunitary_kind='channel_complex_conjugation', time_reversal_square=1)
    with pytest.raises(ValueError, match='channel'):
        _spin_spec(scalar, Path('wannier90'), None, 2)
    with pytest.raises(ValueError, match='channel'):
        _spin_spec(scalar, Path('wannier90.1'), 2, 2)
    with pytest.raises(ValueError, match='WAVECAR'):
        _spin_spec(scalar, Path('wannier90.2'), None, 1)
    with pytest.raises(ValueError, match='channel'):
        _spin_spec(scalar, Path('wannier90'), 1.0, 2)
    with pytest.raises(ValueError, match='channel'):
        _spin_spec(scalar, Path('renamed'), 1, 2, win_spin='down')
    with pytest.raises(ValueError, match='channel'):
        _spin_spec(scalar, Path('wannier90.1'), None, 2, win_spin='down')
    assert _spin_spec(scalar, Path('renamed'), None, 2, win_spin='down')['spin_channel'] == 2


def _atomic_inputs(folder, ispin, channel):
    folder.mkdir()
    seed = folder / ('wannier90' if ispin == 1 else f'wannier90.{channel}')
    Path(f'{seed}.win').write_text(
        'num_bands=1\nnum_wann=1\nspinors=false\nmp_grid=1 1 1\n'
        'begin unit_cell_cart\n2 0 0\n0 3 0\n0 0 4\nend unit_cell_cart\n'
        'begin atoms_frac\nH 0 0 0\nend atoms_frac\n'
        'begin kpoints\n0 0 0\nend kpoints\n')
    if ispin == 2:
        with Path(f'{seed}.win').open('a') as handle:
            handle.write('spin = ' + ('up' if channel == 1 else 'down') + '\n')
    Path(f'{seed}.amn').write_text('analytic G=0 orbital\n1 1 1\n1 1 1 1 0\n')
    Path(f'{seed}.eig').write_text(f'1 1 {channel-1:.8f}\n')
    Path(f'{seed}.mmn').write_text('analytic atomic orbital\n1 1 6\n' + ''.join(
        f'1 1 {shift}\n1 0\n' for shift in ('1 0 0', '-1 0 0', '0 1 0', '0 -1 0', '0 0 1', '0 0 -1')))
    with (folder / 'WAVECAR').open('wb') as handle:
        handle.truncate(512 * (2 + ispin * 2))
        np.array([512, ispin, 45200], dtype='<f8').tofile(handle)
        handle.seek(512)
        np.array([1, 1, 100, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0], dtype='<f8').tofile(handle)
        g = np.array([(x, y, z) for x in range(-5, 6) for y in range(-5, 6) for z in range(-5, 6)])
        ng = np.count_nonzero(3.80998212 * np.sum((2*np.pi*g / [2, 3, 4])**2, axis=1) < 100)
        for spin in range(ispin):
            handle.seek(512 * (2 + 2 * spin))
            np.array([ng, 0, 0, 0, spin, 0, 1], dtype='<f8').tofile(handle)
            handle.seek(512 * (3 + 2 * spin))
            coefficients = np.zeros(ng, dtype='<c8')
            coefficients[0] = 1
            coefficients.tofile(handle)
    (folder / 'OUTCAR').write_text(
        f'Startparameter for this run:\n ISPIN = {ispin}\n LNONCOLLINEAR = F\n LSORBIT = F\n'
        ' LWAVE = F\n number of electron 1.000 magnetization 1.0000000\n'
        ' magnetization (x)\n\n # of ion       s       p       d       tot\n'
        ' -------------------------------------------------------\n'
        ' 1 1.000 0.000 0.000 1.000\n'
        ' -------------------------------------------------------\n tot 1.000 0.000 0.000 1.000\n'
        ' POSITION                                       TOTAL-FORCE (eV/Angst)\n'
        ' -------------------------------------------------------------------\n'
        ' 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000\n'
        ' -------------------------------------------------------------------\n'
        'Subroutine IBZKPT returns following result:\nFound 1 irreducible k-points:\n'
        'Following reciprocal coordinates:\n0 0 0 1\nFollowing cartesian coordinates:\n'
        'Subroutine IBZKPT_HF returns following result:\nFound 1 k-points in 1st BZ\n'
        'Following reciprocal coordinates:   # in IRBZ\n0 0 0 1 1 t-inv F\n')
    return seed


@pytest.mark.parametrize('ispin,channel', [(1, 1), (2, 1), (2, 2)])
def test_scalar_atomic_extraction_to_sawf(tmp_path, ispin, channel):
    from vasp_sawf.symmetry import export_symmetry
    from vasp_sawf.localize import run_sawf

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, ispin, channel)
    symmetry = tmp_path / 'symmetry'
    report = export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', symmetry,
                             workers=1, memory_gb=4)
    assert report['sawf_ready']
    assert report['spin']['spinor'] is False
    assert report['spin']['spin_channel'] == channel
    assert report['spin']['time_reversal_square'] == 1
    assert report['residuals']['conjugation_squared_minus_identity_max'] < 1e-14
    with np.load(symmetry / 'bloch.npz', allow_pickle=False) as arrays:
        assert arrays['d'].shape[-2:] == (1, 1)
        np.testing.assert_allclose(arrays['d'], 1., atol=1e-14)
        assert arrays['spin_channel'].item() == channel
    model_dir = tmp_path / 'model'
    model = run_sawf(seed, symmetry, model_dir, center=[[0, 0, 0]], orbital=['s'], num_iter=8)
    assert model['sawf_ready'] and model['converged']
    assert model['spin'] == report['spin']
    assert 'TRIM_Kramers_max_split_eV' not in model
    if ispin == 2:
        assert not (model_dir / 'WANPROJ').exists()
        assert model['spin']['antiunitary_kind'] == 'channel_complex_conjugation'
    else:
        assert (model_dir / 'WANPROJ').is_file()
    assert json.loads((model_dir / 'summary.json').read_text())['sawf_ready']
