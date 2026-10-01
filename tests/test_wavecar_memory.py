import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_wavecar import _write_wavecar


def test_metadata_reads_headers_only_and_preserves_original_indices(tmp_path, monkeypatch):
    from vasp_sawf.wavecar import inspect_selected_wavecar

    path, lattice, refs = _write_wavecar(tmp_path)
    observed = []
    original = np.fromfile

    def trace(handle, *args, **kwargs):
        offset = handle.tell()
        data = original(handle, *args, **kwargs)
        observed.append((offset, data.nbytes))
        return data

    with monkeypatch.context() as patch:
        patch.setattr(np, 'fromfile', trace)
        metadata = inspect_selected_wavecar(path, bands_1based=[4, 2], lattice=lattice)
    assert observed == [(0, 24), (2048, 104), (4096, 128), (14336, 128)]
    assert metadata.bands_1based == (4, 2)
    assert metadata.kpoints_1based == (1, 2)
    assert metadata.coefficient_counts == tuple(2 * len(ref[0]) for ref in refs)
    np.testing.assert_array_equal(metadata.kpoints, [ref[3] for ref in refs])
    np.testing.assert_array_equal(metadata.energies, [ref[2][[3, 1]] for ref in refs])
    assert {record['kind'] for record in metadata.read_ledger} == {
        'file_header', 'dimension_header', 'kpoint_header'}
    assert sum(record['bytes_returned'] for record in metadata.read_ledger) == 384
    assert metadata.source_identity['resolved_path'] == str(path.resolve())
    assert metadata.source_identity['st_size'] == path.stat().st_size
    json.dumps(metadata.source_identity)


def test_metadata_one_k_selection_matches_actual_reader_without_changing_gauge(tmp_path):
    from vasp_sawf.wavecar import inspect_selected_wavecar, read_selected_wavecar

    path, lattice, refs = _write_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[2], kpoints_1based=[2], lattice=lattice)
    selected = read_selected_wavecar(path, bands_1based=[2], kpoints_1based=[2], lattice=lattice,
                                     expected_source_identity=metadata.source_identity)
    assert metadata.kpoints_1based == (2,)
    np.testing.assert_array_equal(metadata.kpoints[0], selected.kpoints[0].k)
    np.testing.assert_array_equal(metadata.energies[0], selected.kpoints[0].Energy_raw)
    assert metadata.coefficient_counts == (selected.kpoints[0].WF.shape[1] * 2,)


def test_metadata_source_change_stops_worker_before_coefficients(tmp_path, monkeypatch):
    from vasp_sawf.wavecar import (WavecarReadError, inspect_selected_wavecar,
                                  read_selected_wavecar, verify_wavecar_source)

    path, lattice, _ = _write_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[1], lattice=lattice)
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    original = np.fromfile
    read_offsets = []

    def trace(handle, *args, **kwargs):
        read_offsets.append(handle.tell())
        return original(handle, *args, **kwargs)

    with pytest.raises(WavecarReadError, match='source changed'):
        verify_wavecar_source(path, metadata.source_identity)
    with monkeypatch.context() as patch:
        patch.setattr(np, 'fromfile', trace)
        with pytest.raises(WavecarReadError, match='source changed'):
            read_selected_wavecar(path, bands_1based=[1], kpoints_1based=[1], lattice=lattice,
                                   expected_source_identity=metadata.source_identity)
    assert set(read_offsets) <= {0, 2048}


def test_metadata_rejects_invalid_coefficient_count(tmp_path):
    from vasp_sawf.wavecar import WavecarReadError, inspect_selected_wavecar

    path, lattice, _ = _write_wavecar(tmp_path)
    with path.open('r+b') as handle:
        handle.seek(4096)
        np.array([3.], dtype='<f8').tofile(handle)
    with pytest.raises(WavecarReadError, match='two-component coefficient count'):
        inspect_selected_wavecar(path, bands_1based=[1], lattice=lattice)


def test_memory_estimate_uses_actual_spinor_counts_and_bounds_workers(tmp_path):
    from vasp_sawf.wavecar import estimate_wavecar_memory, inspect_selected_wavecar

    path, lattice, refs = _write_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[4, 2], lattice=lattice)
    estimate = estimate_wavecar_memory(metadata, workers=8)
    counts = [2 * len(ref[0]) for ref in refs]
    assert estimate['is_estimate'] is True
    assert estimate['workers'] == 2
    assert estimate['selected_coefficient_read_bytes'] == sum(counts) * 2 * 8
    assert estimate['max_gvectors'] == max(counts) // 2
    assert estimate['per_worker']['raw_complex64_bytes'] == max(counts) * 2 * 8
    assert estimate['per_worker']['one_complex128_copy_bytes'] == max(counts) * 2 * 16
    assert estimate['per_worker_estimated_bytes'] >= max(counts) * 2 * 24
    assert estimate['all_workers_estimated_bytes'] == estimate['per_worker_estimated_bytes'] * 2
    assert 'not a guaranteed peak' in estimate['limitations']
    json.dumps(estimate)


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_memory_estimate_rejects_invalid_worker_count(tmp_path, workers):
    from vasp_sawf.wavecar import WavecarReadError, estimate_wavecar_memory, inspect_selected_wavecar

    path, lattice, _ = _write_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[1], lattice=lattice)
    with pytest.raises(WavecarReadError, match='workers'):
        estimate_wavecar_memory(metadata, workers=workers)


@pytest.mark.real_data
def test_real_srvo3_metadata_matches_independent_energy_record_reads():
    fixture = os.environ.get('SAWF_SRVO3_WANNIER')
    if fixture is None:
        pytest.skip('SAWF_SRVO3_WANNIER was not explicitly set for the small-system fixture')
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.wavecar import inspect_selected_wavecar

    source = Path(fixture)
    path = source / 'WAVECAR'
    assert path.stat().st_size == 37333632
    bundle = read_inputs(source / 'wannier90', source_nb=72)
    metadata = inspect_selected_wavecar(path, bands_1based=range(33, 39), lattice=bundle.lattice)
    with path.open('rb') as handle:
        recl, nspin, rtag = np.fromfile(handle, dtype='<f8', count=3).astype(int)
        handle.seek(int(recl))
        nk, nb = np.fromfile(handle, dtype='<f8', count=2).astype(int)
        assert (nk, nb, nspin, rtag) == (20, 72, 1, 45200)
        for ik in range(nk):
            handle.seek(int(recl * (2 + ik * (nb + 1))))
            record = np.fromfile(handle, dtype='<f8', count=4 + 3 * nb)
            assert metadata.coefficient_counts[ik] == int(record[0])
            np.testing.assert_array_equal(metadata.kpoints[ik], record[1:4])
            np.testing.assert_array_equal(metadata.energies[ik], record[4:].reshape(nb, 3)[32:38, 0])
    assert sum(row['bytes_returned'] for row in metadata.read_ledger) == 128 + 20 * (4 + 3 * 72) * 8
    assert all(row['kind'] != 'coefficients' for row in metadata.read_ledger)
