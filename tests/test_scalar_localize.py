"""Scalar SAWF uses orbital conjugation, without spin doubling or Kramers assumptions."""

from types import SimpleNamespace

import numpy as np
import pytest


def _scalar_symmetrizer(orbital):
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.localize import _target_representation
    from vasp_sawf.symmetry import make_symmetrizer

    nb = 1 if orbital == 's' else 3
    group = SpaceGroup(Lattice=np.eye(3), spinor=False,
        rotations=[np.eye(3, dtype=int)] * 2, translations=[np.zeros(3)] * 2,
        time_reversals=[False, True], number=1, name='P1+K')
    bundle = SimpleNamespace(kpoints=np.zeros((1, 3)), mesh=np.ones(3, dtype=int),
                             eig=np.arange(nb)[None, :])
    sym = make_symmetrizer(bundle, group, np.tile(np.eye(nb), (2, 1, 1, 1)))
    blocks, positions, target = _target_representation(sym, [0, 0, 0], orbital)
    return sym, blocks, positions, target


@pytest.mark.parametrize('orbital,nb', [('s', 1), ('p', 3)])
def test_scalar_target_and_strict_sawf_preserve_odd_dimension(orbital, nb):
    from wannierberri.w90files.chk import CheckPoint
    from vasp_sawf.localize import wannierise_strict, StrictSymmetrizerUirr

    sym, blocks, positions, target = _scalar_symmetrizer(orbital)
    assert sym.num_wann == nb
    assert all(not block.spinor for block in blocks)
    assert positions.shape == (nb, 3)
    rng = np.random.default_rng(56)
    trial = rng.normal(size=(nb, nb)) + 1j*rng.normal(size=(nb, nb))
    gauge = StrictSymmetrizerUirr(sym, 0)(trial)
    np.testing.assert_allclose(gauge, gauge.conj(), atol=1e-12)
    np.testing.assert_allclose(gauge.conj().T @ gauge, np.eye(nb), atol=1e-12)
    identity = np.eye(nb, dtype=complex)
    matrices = {
        'amn': SimpleNamespace(data={0: identity.copy()}, NK=1, NB=nb, NW=nb),
        'mmn': SimpleNamespace(data={0: np.repeat(identity[None], 6, axis=0)}, NK=1, NB=nb),
        'eig': SimpleNamespace(data={0: np.arange(nb, dtype=float)}, NK=1, NB=nb),
    }
    data = SimpleNamespace(**matrices, irreducible=False, wannierised=False,
        chk=CheckPoint(real_lattice=np.eye(3), kpt_latt=np.zeros((1, 3)),
                       num_wann=nb, num_bands=nb, num_kpts=1, mp_grid=np.ones(3, dtype=int)),
        mp_grid=np.ones(3, dtype=int),
        bkvec=SimpleNamespace(neighbours={0: np.zeros(6, dtype=int)}, wk=np.ones(6)/2,
                               NNB=6, bk_cart=np.r_[np.eye(3), -np.eye(3)]))
    data.get_file = matrices.__getitem__
    data.set_symmetrizer = lambda value, **kwargs: setattr(data, 'symmetrizer', value)
    before = {key: value.data[0].copy() for key, value in matrices.items()}
    report = wannierise_strict(data, sym, num_iter=8)
    assert report['converged'] and report['excluded_bands'] == []
    assert report['shape_nk_nb_nw'] == [1, nb, nb]
    np.testing.assert_allclose(report['spreads_angstrom2'], 0, atol=1e-13)
    np.testing.assert_allclose(data.chk.v_matrix[0], identity, atol=1e-13)
    for key in before:
        np.testing.assert_array_equal(matrices[key].data[0], before[key])


def _spin_payload(channel=1):
    spin = dict(spinor=False, source_ispin=2, spin_channel=channel,
                antiunitary_kind='channel_complex_conjugation', time_reversal_square=1)
    arrays = {key: np.asarray(value) for key, value in spin.items() if key != 'time_reversal_square'}
    arrays['spacegroup_spinor'] = np.asarray(False)
    return arrays, {'spin': spin}


@pytest.mark.parametrize('mutation', ['channel', 'kind', 'square', 'group', 'win', 'missing', 'partial',
                                     'npz_square', 'missing_group', 'outcar'])
def test_scalar_spin_metadata_rejects_mismatched_semantics(mutation):
    from vasp_sawf.localize import _validate_spin_metadata

    arrays, report = _spin_payload()
    win = 'spinors = false\n'
    if mutation == 'channel':
        arrays['spin_channel'] = np.asarray(2)
    elif mutation == 'kind':
        report['spin']['antiunitary_kind'] = 'physical_time_reversal'
        arrays['antiunitary_kind'] = np.asarray('physical_time_reversal')
    elif mutation == 'square':
        report['spin']['time_reversal_square'] = -1
    elif mutation == 'group':
        arrays['spacegroup_spinor'] = np.asarray(True)
    elif mutation == 'win':
        win = 'spinors = true\n'
    elif mutation == 'missing':
        arrays.pop('spin_channel')
    elif mutation == 'partial':
        report.pop('spin')
    elif mutation == 'npz_square':
        arrays['time_reversal_square'] = np.asarray(-1)
    elif mutation == 'missing_group':
        arrays.pop('spacegroup_spinor')
    else:
        report['outcar'] = {'spin_context': {'ISPIN': 1}}
    with pytest.raises(ValueError):
        _validate_spin_metadata(arrays, report, win_text=win)


def test_scalar_spin_metadata_roundtrip_and_legacy_soc():
    from vasp_sawf.localize import _validate_spin_metadata

    arrays, report = _spin_payload(2)
    assert _validate_spin_metadata(arrays, report, win_text='spinors=.false.') == report['spin']
    legacy = _validate_spin_metadata({'spacegroup_spinor': np.asarray(True)}, {})
    assert legacy['spinor'] is True and legacy['time_reversal_square'] == -1
    with pytest.raises(ValueError):
        _validate_spin_metadata({'spacegroup_spinor': np.asarray(False)}, {})


@pytest.mark.parametrize('field', ['spin=down', 'spin=invalid', 'spin=up\nspin=up', 'spin=', 'spin=up down'])
def test_win_spin_rejects_wrong_or_ambiguous_channel_even_for_equal_energies(field):
    from vasp_sawf.localize import _validate_spin_metadata

    arrays, report = _spin_payload(1)
    arrays['eig'] = np.zeros((1, 3))
    with pytest.raises(ValueError, match='spin'):
        _validate_spin_metadata(arrays, report, win_text='spinors=false\n' + field)


@pytest.mark.parametrize('channel,field', [(1, 'spin=UP'), (2, 'spin: down # selected source channel')])
def test_win_spin_channel_is_bound_independently_of_filename(channel, field):
    from vasp_sawf.localize import _validate_spin_metadata

    arrays, report = _spin_payload(channel)
    assert _validate_spin_metadata(arrays, report, win_text=field)['spin_channel'] == channel


@pytest.mark.parametrize('channel,expected', [(1, [1., 2.]), (2, [4., 7.])])
def test_dft_two_channel_selection_is_explicit_and_independently_known(tmp_path, channel, expected):
    from vasp_sawf.bands import read_dft_eigenval

    path = tmp_path / 'EIGENVAL'
    path.write_text('1 1 1 2\nheader\nheader\nheader\nheader\n2 1 2\n\n'
                    ' 0.0 0.0 0.0 1.0\n1 1.0 4.0 1.0 0.0\n2 2.0 7.0 0.0 0.0\n')
    result = read_dft_eigenval(path, [1, 2], spin_channel=channel, source_ispin=2)
    np.testing.assert_array_equal(result['eigenvalues_eV'], [expected])
    assert result['spin_channel'] == channel and result['source_ispin'] == 2
    with pytest.raises(ValueError):
        read_dft_eigenval(path, [1, 2])


@pytest.mark.parametrize('nb', [1, 3])
def test_scalar_single_channel_wanproj_accepts_odd_orbital_count(tmp_path, nb):
    from vasp_sawf.wanproj import write_wanproj, read_wanproj

    payload = dict(U=np.eye(nb)[None].astype(complex), kpoints=np.zeros((1, 3)),
                   bands_vasp_1based=np.arange(1, nb+1), source_nb=nb, mesh=np.ones(3, dtype=int))
    write_wanproj(tmp_path / 'WANPROJ', **payload)
    assert read_wanproj(tmp_path / 'WANPROJ')['U'].shape == (1, nb, nb)


def test_full_three_band_scalar_model_has_no_spurious_kramers_condition(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.localize import run_sawf
    from vasp_sawf.symmetry import _cell_from_win, _spacegroup_arrays, _spacegroup_from_context, build_symmetry_maps

    source = tmp_path / 'inputs'
    source.mkdir()
    seed = source / 'wannier90'
    win = ('num_bands=3\nnum_wann=3\nspinors=false\nmp_grid=1 1 1\n'
        'begin unit_cell_cart\n2 0 0\n0 3 0\n0 0 4\nend unit_cell_cart\n'
        'begin atoms_frac\nH .031 .072 .119\nHe .237 .323 .421\nLi .613 .577 .891\nend atoms_frac\n'
        'begin kpoints\n0 0 0\nend kpoints\n'
        'begin kpoint_path\nG 0 0 0 X .5 0 0\nend kpoint_path\n')
    Path(f'{seed}.win').write_text(win)
    Path(f'{seed}.amn').write_text('analytic point orbitals\n3 1 3\n' + ''.join(
        f'{i+1} {j+1} 1 {int(i == j)} 0\n' for j in range(3) for i in range(3)))
    Path(f'{seed}.eig').write_text('1 1 0\n2 1 1\n3 1 3\n')
    matrix = ''.join(f'{int(i == j)} 0\n' for j in range(3) for i in range(3))
    Path(f'{seed}.mmn').write_text('analytic point orbitals\n3 1 6\n' + ''.join(
        f'1 1 {shift}\n{matrix}' for shift in ('1 0 0', '-1 0 0', '0 1 0', '0 -1 0', '0 0 1', '0 0 -1')))
    bundle = read_inputs(seed, source_nb=3)
    context = dict(spinor=False, source_ispin=1, ISPIN=1, LNONCOLLINEAR=False, LSORBIT=False)
    positions, types, _ = _cell_from_win(win, bundle.lattice)
    group = _spacegroup_from_context(bundle, positions, types, context)
    assert group.size == 2
    arrays = _spacegroup_arrays(group)
    arrays.update(d=np.tile(np.eye(3), (2, 1, 1, 1)), kpoints=bundle.kpoints, eig=bundle.eig,
                  bands_vasp_1based=bundle.bands_vasp_1based, kmap=build_symmetry_maps(bundle, group)[0])
    spin = dict(spinor=False, source_ispin=1, spin_channel=1,
                antiunitary_kind='orbital_complex_conjugation', time_reversal_square=1)
    arrays.update({key: value for key, value in spin.items() if key != 'time_reversal_square'})
    symmetry = tmp_path / 'symmetry'
    symmetry.mkdir()
    np.savez(symmetry / 'bloch.npz', **arrays)
    content = (symmetry / 'bloch.npz').read_bytes()
    report = dict(schema='sawf-bridge-bloch-v1', status='ready', sawf_ready=True,
        numerical_checks_passed=True, source_hashes=bundle.hashes, source_num_bands=3,
        spin=spin, outcar={'spin_context': context}, independent_ibz_little_checks=2,
        residuals={'ibz_coefficient_closure_relative_max': 0.},
        bloch_sha256=hashlib.sha256(content).hexdigest(), bloch_bytes=len(content))
    (symmetry / 'report.json').write_text(json.dumps(report))
    eigenval = source / 'EIGENVAL'
    eigenval.write_text('1 1 1 1\nheader\nheader\nheader\nheader\n2 2 3\n\n'
        ' 0.0 0.0 0.0 1.0\n1 0.0 1.0\n2 1.0 0.0\n3 3.0 0.0\n\n'
        ' 0.5 0.0 0.0 1.0\n1 0.0 1.0\n2 1.0 0.0\n3 3.0 0.0\n')
    model = tmp_path / 'model'
    result = run_sawf(seed, symmetry, model, center=[0, 0, 0], orbital='p', num_iter=8,
                      dft_eigenval=eigenval)
    assert result['converged'] and result['sawf_ready']
    assert result['spin'] == spin
    assert 'TRIM_Kramers_max_split_eV' not in result
    assert result['bloch_K_squared_minus_identity_max'] == 0.
    np.testing.assert_allclose(result['spreads_angstrom2'], 0., atol=1e-13)
    with np.load(model / 'model.npz') as final:
        np.testing.assert_array_equal(final['eigenvalues_eV'], [[0., 1., 3.]])
        np.testing.assert_allclose(final['U'], np.eye(3)[None], atol=1e-13)
    with np.load(model / 'bands.npz', allow_pickle=False) as bands:
        assert bands['source_ispin'].item() == 1
        assert bands['spin_channel'].item() == 1
        assert bands['spinor'].item() is False
        np.testing.assert_allclose(bands['sawf'], [[0., 1., 3.], [0., 1., 3.]], atol=1e-13)
