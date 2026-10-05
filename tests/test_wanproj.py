"""Protect source band indices and complex phases at the VASP WANPROJ boundary."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


def _payload():
    return dict(
        U=np.array([[[.6, .8j], [.8, -.6j]],
                    [[.6j, -.8], [.8j, .6]]]),
        kpoints=np.array([[.5, 0., 0.], [0., 0., 0.]]),
        bands_vasp_1based=np.array([33, 38]), source_nb=72, mesh=np.array([2, 1, 1]))


def _bound_model(tmp_path):
    model, symmetry, inputs = [tmp_path / name for name in ('model', 'symmetry', 'inputs')]
    for folder in (model, symmetry, inputs):
        folder.mkdir()
    payload = _payload()
    eig = np.array([[4., 5.], [4.1, 5.1]])
    source_hashes = {name: hashlib.sha256(name.encode()).hexdigest()
                     for name in ('win', 'amn', 'mmn', 'eig')}
    np.savez(symmetry / 'bloch.npz', kpoints=payload['kpoints'], eig=eig,
             bands_vasp_1based=payload['bands_vasp_1based'], d=np.eye(2)[None, None])
    content = (symmetry / 'bloch.npz').read_bytes()
    symreport = dict(schema='sawf-bridge-bloch-v1', status='ready', sawf_ready=True,
                     numerical_checks_passed=True, source_hashes=source_hashes,
                     source_num_bands=72, mesh=[2, 1, 1],
                     bands_vasp_1based=[33, 38],
                     residuals={'ibz_coefficient_closure_relative_max': 1e-7},
                     bloch_sha256=hashlib.sha256(content).hexdigest(), bloch_bytes=len(content))
    (symmetry / 'report.json').write_text(json.dumps(symreport))
    np.savez(model / 'model.npz', U=payload['U'], kpoints=payload['kpoints'],
             eigenvalues_eV=eig, lattice=np.eye(3), centers_angstrom=np.zeros((2, 3)),
             spreads_angstrom2=np.ones(2), R=np.zeros((1, 3), dtype=int),
             H_R=np.zeros((1, 2, 2), dtype=complex))
    summary = dict(schema='sawf-bridge-model-v2', status='ready', sawf_ready=True,
                   converged=True, numerical_checks_passed=True, model_file='model.npz',
                   model_sha256=hashlib.sha256((model / 'model.npz').read_bytes()).hexdigest(),
                   symmetry_package_sha256=symreport['bloch_sha256'],
                   source_seed=str(inputs / 'wannier90'), source_hashes=dict(source_hashes),
                   bands_vasp_1based=[33, 38])
    (model / 'summary.json').write_text(json.dumps(summary))
    return model, symmetry, inputs, summary, symreport


@pytest.mark.parametrize('source_nb', [72, 80])
def test_writer_preserves_complex_U_order_and_original_band_indices(tmp_path, source_nb):
    from vasp_sawf.wanproj import read_wanproj, write_wanproj
    payload = _payload()
    payload['source_nb'] = source_nb
    before = payload['U'].copy()
    path = tmp_path / 'WANPROJ'
    report = write_wanproj(path, **payload)
    lines = path.read_text().splitlines()
    assert list(map(int, lines[1].split())) == [1, 2, source_nb, 2]
    # Independent literal checks detect conjugation, transposition and compact indices.
    rows = [line.split() for line in lines[5:9]]
    assert [tuple(map(int, row[:2])) for row in rows] == [(33, 1), (33, 2), (38, 1), (38, 2)]
    np.testing.assert_array_equal([[float(x) for x in row[2:]] for row in rows],
                                  [[.6, 0.], [0., .8], [.8, 0.], [0., -.6]])
    restored = read_wanproj(path)
    np.testing.assert_array_equal(restored['U'], before)
    np.testing.assert_array_equal(restored['kpoints'], payload['kpoints'])
    np.testing.assert_array_equal(restored['bands_vasp_1based'], [33, 38])
    np.testing.assert_array_equal(payload['U'], before)
    assert report['readback_exact'] is True
    assert report['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['nan', 'nonunitary', 'rectangular', 'duplicate_k',
                                  'shifted_mesh', 'incomplete_mesh', 'mesh_overflow', 'duplicate_band',
                                  'out_of_bounds', 'noninteger_band', 'missing_source_nb'])
def test_invalid_payload_is_rejected_before_file_creation(tmp_path, change):
    from vasp_sawf.wanproj import write_wanproj
    payload = _payload()
    if change == 'nan':
        payload['U'][0, 0, 0] = np.nan
    elif change == 'nonunitary':
        payload['U'][0] *= 2
    elif change == 'rectangular':
        payload['U'] = payload['U'][:, :, :1]
    elif change == 'duplicate_k':
        payload['kpoints'][1] = payload['kpoints'][0] + [1, 0, 0]
    elif change == 'shifted_mesh':
        payload['kpoints'] += [.25, 0, 0]
    elif change == 'incomplete_mesh':
        payload['mesh'] = [3, 1, 1]
    elif change == 'mesh_overflow':
        payload['U'] = np.repeat(payload['U'][:1], 8, axis=0)
        payload['kpoints'] = np.array([[a, b, c] for a in (0., .5)
                                      for b in (0., .5) for c in (0., .5)])
        payload['mesh'] = [2**62 + 1, 4, 2]
    elif change == 'duplicate_band':
        payload['bands_vasp_1based'] = [33, 33]
    elif change == 'out_of_bounds':
        payload['bands_vasp_1based'] = [33, 73]
    elif change == 'noninteger_band':
        payload['bands_vasp_1based'] = [33., 38.5]
    else:
        payload['source_nb'] = None
    output = tmp_path / 'WANPROJ'
    with pytest.raises(ValueError):
        write_wanproj(output, **payload)
    assert not output.exists()


def test_writer_refuses_existing_file_and_dangling_symlink(tmp_path):
    from vasp_sawf.wanproj import write_wanproj
    original = tmp_path / 'WANPROJ'
    original.write_text('preserve me')
    link = tmp_path / 'link'
    target = tmp_path / 'missing'
    link.symlink_to(target)
    for path in (original, link):
        with pytest.raises((ValueError, FileExistsError)):
            write_wanproj(path, **_payload())
    assert original.read_text() == 'preserve me'
    assert link.is_symlink() and not target.exists()


def test_gamma_coordinates_may_be_integers_and_cell_phases_are_retained(tmp_path):
    from vasp_sawf.wanproj import read_wanproj, write_wanproj
    payload = _payload()
    payload.update(U=payload['U'][:1], kpoints=[[0, 0, 0]], mesh=[1, 1, 1])
    write_wanproj(tmp_path / 'integer-gamma', **payload)
    kpoints = np.array([[2/3, 0., 0.], [0., 0., 0.], [1/3, 0., 0.]])
    u = np.repeat(payload['U'], 3, axis=0)
    u[:, :, 1] *= np.exp(-2j*np.pi*kpoints[:, 0])[:, None]
    payload.update(U=u, kpoints=kpoints, mesh=[3, 1, 1])
    write_wanproj(tmp_path / 'cell-phases', **payload)
    np.testing.assert_array_equal(read_wanproj(tmp_path / 'cell-phases')['U'], u)


def test_export_binds_saved_model_to_symmetry_and_preserves_all_model_arrays(tmp_path):
    from vasp_sawf.wanproj import export_wanproj, read_wanproj
    model, symmetry, _, _, _ = _bound_model(tmp_path)
    originals = {p: p.read_bytes() for folder in (model, symmetry) for p in folder.iterdir()}
    output = tmp_path / 'export'
    report = export_wanproj(model, symmetry, output)
    restored = read_wanproj(output / 'WANPROJ')
    np.testing.assert_array_equal(restored['U'], _payload()['U'])
    assert restored['source_num_bands'] == 72
    assert report['status'] == 'ready' and report['wavefunction_bytes_read'] == 0
    assert json.loads((output / 'report.json').read_text())['wanproj']['sha256'] == report['wanproj']['sha256']
    assert all(p.read_bytes() == contents for p, contents in originals.items())


@pytest.mark.parametrize('change', ['status', 'converged', 'sawf_ready', 'numerical_checks_passed',
                                  'model_hash', 'symmetry_hash', 'source_hashes', 'bands',
                                  'kpoints', 'eig', 'source_nb', 'mesh'])
def test_export_rejects_unready_or_mismatched_bound_model(tmp_path, change):
    from vasp_sawf.wanproj import export_wanproj
    model, symmetry, _, summary, symreport = _bound_model(tmp_path)
    if change == 'status':
        summary['status'] = 'not_ready'
    elif change in ('converged', 'sawf_ready', 'numerical_checks_passed'):
        summary[change] = False
    elif change == 'model_hash':
        summary['model_sha256'] = '0' * 64
    elif change == 'symmetry_hash':
        summary['symmetry_package_sha256'] = '0' * 64
    elif change == 'source_hashes':
        summary['source_hashes']['win'] = '0' * 64
    elif change == 'bands':
        summary['bands_vasp_1based'] = [34, 38]
    elif change == 'source_nb':
        symreport['source_num_bands'] = None
    elif change == 'mesh':
        symreport['mesh'] = [3, 1, 1]
    else:
        with np.load(model / 'model.npz') as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays['kpoints' if change == 'kpoints' else 'eigenvalues_eV'][0, 0] += .01
        np.savez(model / 'model.npz', **arrays)
        summary['model_sha256'] = hashlib.sha256((model / 'model.npz').read_bytes()).hexdigest()
    (model / 'summary.json').write_text(json.dumps(summary))
    (symmetry / 'report.json').write_text(json.dumps(symreport))
    output = tmp_path / 'export'
    with pytest.raises(ValueError):
        export_wanproj(model, symmetry, output)
    assert not output.exists()


def test_export_refuses_overwrite_and_all_protected_source_trees(tmp_path):
    from vasp_sawf.wanproj import export_wanproj
    model, symmetry, inputs, _, _ = _bound_model(tmp_path)
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'keep').write_text('unchanged')
    link = tmp_path / 'link'
    link.symlink_to(tmp_path / 'missing')
    for target in (existing, link, model / 'export', symmetry / 'export', inputs / 'export'):
        with pytest.raises(ValueError):
            export_wanproj(model, symmetry, target)
    assert (existing / 'keep').read_text() == 'unchanged'
    assert link.is_symlink()


@pytest.mark.real_data
def test_native_vasp_wanproj_has_same_complex_orientation_as_its_checkpoint():
    from vasp_sawf.wanproj import read_wanproj
    from wannierberri.w90files import CheckPoint
    native, seed = os.environ.get('SAWF_NATIVE_WANPROJ'), os.environ.get('SAWF_NATIVE_SEED')
    if not native or not seed:
        pytest.skip('Native VASP WANPROJ and its matching checkpoint were not specified')
    parsed = read_wanproj(native)
    chk = CheckPoint()
    chk.from_w90_file(seed)
    assert chk.num_bands == chk.num_wann
    expected = np.array([chk.v_matrix[k] for k in range(chk.num_kpts)])
    np.testing.assert_allclose(parsed['kpoints'], chk.kpt_latt, atol=1e-15, rtol=0)
    np.testing.assert_allclose(parsed['U'], expected, atol=1e-14, rtol=0)


@pytest.mark.real_data
def test_ready_localization_exports_final_U_without_changing_model_array_schema(tmp_path):
    from vasp_sawf.localize import run_sawf
    from vasp_sawf.wanproj import read_wanproj
    seed, symmetry = os.environ.get('SAWF_SRVO3_SEED'), os.environ.get('SAWF_SRVO3_SYMMETRY')
    if not seed or not symmetry:
        pytest.skip('The accepted SrVO3 interface and symmetry package were not specified')
    output = tmp_path / 'model'
    report = run_sawf(seed, symmetry, output, center=[.5, .5, .5], orbital='t2g')
    parsed = read_wanproj(output / 'WANPROJ')
    assert report['status'] == 'ready' and report['wanproj']['readback_exact'] is True
    assert report['source_num_bands'] == parsed['source_num_bands'] == 72
    assert report['mesh'] == [6, 6, 6]
    np.testing.assert_array_equal(parsed['bands_vasp_1based'], [33, 34, 35, 36, 37, 38])
    with np.load(output / 'model.npz') as model:
        assert set(model.files) == {'U', 'kpoints', 'eigenvalues_eV', 'centers_angstrom',
                                    'spreads_angstrom2', 'lattice', 'R', 'H_R'}
        np.testing.assert_array_equal(parsed['U'], model['U'])
        np.testing.assert_array_equal(parsed['kpoints'], model['kpoints'])
