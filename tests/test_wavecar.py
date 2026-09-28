import os
from pathlib import Path

import numpy as np
import pytest


def _write_wavecar(tmp_path, *, rtag=45200, nb=4, recl=2048):
    """Generate known record contents; in-sphere G vectors come from independent Cartesian enumeration."""
    path = tmp_path / 'WAVECAR'
    lattice = np.eye(3) * 4
    reciprocal = np.linalg.inv(lattice).T * 2 * np.pi
    kpoints = np.array([[0., 0, 0], [.25, 0, 0]])
    cutoff = 40.
    refs = []
    with path.open('wb') as handle:
        handle.truncate(recl * (2 + len(kpoints) * (nb + 1)))
        handle.seek(0)
        np.array([recl, 1, rtag], dtype='<f8').tofile(handle)
        handle.seek(recl)
        np.r_[len(kpoints), nb, cutoff, lattice.ravel(), 0.].astype('<f8').tofile(handle)
        if (4 + 3 * nb) * 8 > recl:
            return path, lattice, refs
        for ik, k in enumerate(kpoints):
            gs = []
            for gz in [0, 1, 2, 3, -3, -2, -1]:
                for gy in [0, 1, 2, 3, -3, -2, -1]:
                    for gx in [0, 1, 2, 3, -3, -2, -1]:
                        g = np.array([gx, gy, gz])
                        if np.linalg.norm((g + k) @ reciprocal) ** 2 * 3.80998212 < cutoff:
                            gs.append(g)
            gs = np.array(gs)
            ng = len(gs)
            coefficients = (np.arange(nb * ng * 2).reshape(nb, ng, 2)
                            + 1j * (1000 + ik)).astype('complex64')
            energies = np.arange(nb, dtype=float) + ik / 10
            handle.seek(recl * (2 + ik * (nb + 1)))
            rows = np.column_stack((energies, np.zeros(nb), np.ones(nb)))
            np.r_[2 * ng, k, rows.ravel()].astype('<f8').tofile(handle)
            for ib in range(nb):
                handle.seek(recl * (3 + ik * (nb + 1) + ib))
                coefficients[ib].flatten(order='F').tofile(handle)
            refs.append((gs, coefficients, energies, k))
    return path, lattice, refs


def test_selected_records_keep_exact_coefficients_order_and_bounded_read_ledger(tmp_path):
    from core.wavecar import read_selected_wavecar

    path, lattice, refs = _write_wavecar(tmp_path)
    result = read_selected_wavecar(path, bands_1based=[4, 2], kpoints_1based=[2], lattice=lattice)
    assert result.header.num_kpoints == 2
    assert result.header.num_bands == 4
    assert len(result.kpoints) == 1
    kp = result.kpoints[0]
    g, raw, energies, k = refs[1]
    lookup = {tuple(v): i for i, v in enumerate(g)}
    permutation = [lookup[tuple(v)] for v in kp.ig[:, :3]]
    np.testing.assert_array_equal(kp.WF, raw[[3, 1]][:, permutation])
    np.testing.assert_array_equal(kp.Energy_raw, energies[[3, 1]])
    np.testing.assert_array_equal(kp.k, k)
    assert kp.WF.dtype == np.dtype('complex64')
    assert kp.ik0 == 2
    coefficients = [r for r in result.read_ledger if r['kind'] == 'coefficients']
    assert [r['band_1based'] for r in coefficients] == [4, 2]
    assert all(r['kpoint_1based'] == 2 for r in coefficients)
    assert sum(r['bytes_returned'] for r in result.read_ledger) == 24 + 104 + (4 + 12) * 8 + 2 * len(g) * 2 * 8
    assert result.source_status == result.gauge_status == 'unproven'


def test_header_inspection_reads_only_128_bytes_and_explicit_none_selects_all_k(tmp_path, monkeypatch):
    from core.wavecar import inspect_wavecar, read_selected_wavecar

    path, lattice, refs = _write_wavecar(tmp_path)
    original = np.fromfile
    observations = []

    def observed(handle, *args, **kwargs):
        position = handle.tell()
        result = original(handle, *args, **kwargs)
        observations.append((position, result.nbytes))
        return result

    with monkeypatch.context() as context:
        context.setattr(np, 'fromfile', observed)
        header = inspect_wavecar(path)
    assert observations == [(0, 24), (2048, 104)]
    assert header.num_bands == 4
    assert header.num_kpoints == 2
    result = read_selected_wavecar(path, bands_1based=[2], kpoints_1based=None, lattice=lattice)
    assert result.kpoints_1based == (1, 2)
    assert [kp.ik0 for kp in result.kpoints] == [1, 2]


@pytest.mark.parametrize('problem', ['rtag', 'multi_energy_record', 'lattice', 'k_range',
                                      'band_range', 'duplicate_bands', 'truncated'])
def test_unsupported_or_inconsistent_wavecar_is_rejected(tmp_path, problem):
    from core.wavecar import WavecarReadError, read_selected_wavecar

    path, lattice, _ = _write_wavecar(tmp_path, rtag=45210 if problem == 'rtag' else 45200,
                                     nb=90 if problem == 'multi_energy_record' else 4)
    bands, kpoints = [2], [1]
    if problem == 'lattice':
        lattice[0, 0] += .1
    elif problem == 'k_range':
        kpoints = [3]
    elif problem == 'band_range':
        bands = [0]
    elif problem == 'duplicate_bands':
        bands = [2, 2]
    elif problem == 'truncated':
        with path.open('r+b') as handle:
            handle.truncate(path.stat().st_size - 10)
    with pytest.raises(WavecarReadError):
        read_selected_wavecar(path, bands_1based=bands, kpoints_1based=kpoints, lattice=lattice)


@pytest.mark.real_data
def test_real_srvo3_selected_bands_exactly_match_pymatgen_and_direct_records():
    fixture = os.environ.get('SAWF_SRVO3_WANNIER')
    if fixture is None:
        pytest.skip('SAWF_SRVO3_WANNIER was not explicitly set for the small-system fixture')
    from pymatgen.io.vasp.outputs import Wavecar
    from core.inputs import read_inputs
    from core.wavecar import read_selected_wavecar

    fixture = Path(fixture)
    path = fixture / 'WAVECAR'
    assert path.stat().st_size == 37333632
    bundle = read_inputs(fixture / 'wannier90', source_nb=72)
    bands = list(range(33, 39))
    result = read_selected_wavecar(path, bands_1based=bands, kpoints_1based=range(1, 21), lattice=bundle.lattice)
    reference = Wavecar(str(path), vasp_type='ncl')
    assert len(result.kpoints) == 20
    with path.open('rb') as handle:
        recl, nspin, rtag = np.fromfile(handle, dtype='<f8', count=3).astype(int)
        for ik, kp in enumerate(result.kpoints):
            g = np.rint(reference.Gpoints[ik]).astype(int)
            lookup = {tuple(v): i for i, v in enumerate(g)}
            permutation = [lookup[tuple(v)] for v in kp.ig[:, :3]]
            expected = np.array(reference.coeffs[ik][32:38]).transpose(0, 2, 1)[:, permutation]
            np.testing.assert_array_equal(kp.WF, expected)
            np.testing.assert_array_equal(kp.k, reference.kpoints[ik])
            np.testing.assert_array_equal(kp.Energy_raw, reference.band_energy[ik][32:38, 0])
            for out_band, original_band in enumerate(bands):
                handle.seek(int(recl * (3 + ik * 73 + original_band - 1)))
                raw = np.fromfile(handle, dtype='<c8', count=2 * len(g)).reshape(2, len(g)).T
                np.testing.assert_array_equal(kp.WF[out_band], raw[permutation])
    coefficient_reads = [r for r in result.read_ledger if r['kind'] == 'coefficients']
    assert len(coefficient_reads) == 120
    assert set(r['band_1based'] for r in coefficient_reads) == set(bands)
    assert set(r['kpoint_1based'] for r in coefficient_reads) == set(range(1, 21))
    assert sum(r['bytes_returned'] for r in result.read_ledger) < path.stat().st_size // 10
