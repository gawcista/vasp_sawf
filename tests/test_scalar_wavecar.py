"""Scalar records checked against independent layouts and external readers."""

import itertools
import os
from pathlib import Path

import numpy as np
import pytest


def _scalar_file(tmp_path, channels=2):
    path = tmp_path / 'WAVECAR'
    lattice = np.eye(3) * 4
    reciprocal = np.eye(3) * (np.pi / 2)
    kpoints = np.array([[0., 0, 0], [.25, 0, 0]])
    references = []
    records = [np.zeros(256, dtype='<f8') for _ in range(2 + channels * 2 * 4)]
    records[0][:3] = [2048, channels, 45200]
    records[1][:13] = np.r_[2, 3, 40., lattice.ravel(), 0.]
    for channel in range(channels):
        channel_refs = []
        for ik, k in enumerate(kpoints):
            axis = [0, 1, 2, 3, -3, -2, -1]
            g = np.array([(x, y, z) for z, y, x in itertools.product(axis, repeat=3)
                          if np.linalg.norm((np.array([x, y, z]) + k) @ reciprocal)**2 * 3.80998212 < 40.])
            energy = np.arange(3.) + 20 * channel + ik / 10
            coefficients = (1000 * channel + 100 * ik + np.arange(3 * len(g)).reshape(3, len(g))
                            + 1j * np.arange(len(g))).astype('complex64')
            record = 2 + channel * 8 + ik * 4
            records[record][:13] = np.r_[len(g), k, np.column_stack((energy, np.zeros(3), np.ones(3))).ravel()]
            for band in range(3):
                records[record + 1 + band].view('<c8')[:len(g)] = coefficients[band]
            channel_refs.append((g, coefficients, energy, k))
        references.append(channel_refs)
    path.write_bytes(b''.join(record.tobytes() for record in records))
    return path, lattice, references


@pytest.mark.parametrize('channel', [1, 2])
def test_scalar_channel_exact_coefficients_energy_and_bounded_offsets(tmp_path, channel):
    from vasp_sawf.wavecar import inspect_selected_wavecar, read_selected_wavecar, estimate_wavecar_memory

    path, lattice, refs = _scalar_file(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[3, 1], lattice=lattice,
                                        spinor=False, spin_channel=channel)
    selected = read_selected_wavecar(path, bands_1based=[3, 1], kpoints_1based=[2], lattice=lattice,
                                     spinor=False, spin_channel=channel)
    assert metadata.spinor is selected.spinor is False
    assert metadata.spin_channel == selected.spin_channel == channel
    assert selected.header.spin_channels == 2
    point = selected.kpoints[0]
    g, coefficients, energy, k = refs[channel - 1][1]
    lookup = {tuple(v): i for i, v in enumerate(g)}
    permutation = [lookup[tuple(v)] for v in point.ig[:, :3]]
    np.testing.assert_array_equal(point.WF, coefficients[[2, 0]][:, permutation, None])
    np.testing.assert_array_equal(point.Energy_raw, energy[[2, 0]])
    np.testing.assert_array_equal(point.k, k)
    assert point.spinor is False
    assert point.WF.dtype == np.dtype('complex64')
    coefficient_reads = [row for row in selected.read_ledger if row['kind'] == 'coefficients']
    assert [row['offset_bytes'] for row in coefficient_reads] == [2048 * (9 + (channel - 1) * 8),
                                                                 2048 * (7 + (channel - 1) * 8)]
    assert all(row['spin_channel_1based'] == channel for row in coefficient_reads)
    assert all(row['bytes_returned'] == len(g) * 8 for row in coefficient_reads)
    estimate = estimate_wavecar_memory(metadata)
    assert estimate['spinor_components'] == 1
    assert estimate['max_gvectors'] == max(metadata.coefficient_counts)


@pytest.mark.parametrize('channels,spinor,channel', [(1, False, 2), (2, True, 1),
                                                   (2, False, 0), (2, False, True),
                                                   (2, False, 3), (2, 'false', 1)])
def test_spin_channel_and_component_misuse_is_rejected(tmp_path, channels, spinor, channel):
    from vasp_sawf.wavecar import WavecarReadError, inspect_selected_wavecar
    path, lattice, _ = _scalar_file(tmp_path, channels=channels)
    with pytest.raises(WavecarReadError):
        inspect_selected_wavecar(path, bands_1based=[1], lattice=lattice, spinor=spinor, spin_channel=channel)


def test_scalar_single_channel_preserves_odd_g_count(tmp_path):
    from vasp_sawf.wavecar import read_selected_wavecar
    path, lattice, refs = _scalar_file(tmp_path, channels=1)
    point = read_selected_wavecar(path, bands_1based=[1], kpoints_1based=[1], lattice=lattice,
                                  spinor=False).kpoints[0]
    assert len(point.ig) == len(refs[0][0][0])
    assert len(point.ig) % 2 == 1


def test_two_channel_selection_matches_independent_pymatgen_reader(tmp_path):
    from pymatgen.io.vasp.outputs import Wavecar
    from vasp_sawf.wavecar import read_selected_wavecar

    path, lattice, _ = _scalar_file(tmp_path)
    reference = Wavecar(str(path), vasp_type='std')
    for channel in (1, 2):
        selected = read_selected_wavecar(path, bands_1based=[2, 3], kpoints_1based=[1, 2],
                                         lattice=lattice, spinor=False, spin_channel=channel)
        for ik, point in enumerate(selected.kpoints):
            lookup = {tuple(v): i for i, v in enumerate(np.rint(reference.Gpoints[ik]).astype(int))}
            order = [lookup[tuple(v)] for v in point.ig[:, :3]]
            expected = np.array(reference.coeffs[channel - 1][ik])[[1, 2]][:, order, None]
            np.testing.assert_array_equal(point.WF, expected)
            np.testing.assert_array_equal(point.Energy_raw, reference.band_energy[channel - 1][ik][[1, 2], 0])


def test_scalar_kpoint_adaptation_preserves_original_and_reversible_association():
    from irrep.kpoint import Kpoint
    from vasp_sawf.wavecar import adapt_irrep_kpoint

    g = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    ig = np.column_stack((g, [0, 1, 2], np.zeros(3, int), np.full(3, 3)))
    raw = np.array([[[2 + 1j], [3 + 5j], [7 + 11j]]], dtype='complex64')
    point = Kpoint(ik=2, num_bands=1, RecLattice=np.eye(3), spinor=False,
                   kpt=np.zeros(3), WF=raw.copy(), Energy=np.array([1.]), ig=ig.copy(),
                   upper=None, normalize=False)
    point.weight = .125
    adapted, association = adapt_irrep_kpoint(point, coefficient_count=3, rtag=45200)
    assert adapted.spinor is False
    assert adapted.weight == .125
    assert adapted.ik0 == 3
    np.testing.assert_array_equal(point.WF, raw)
    np.testing.assert_array_equal(point.ig, ig)
    np.testing.assert_array_equal(adapted.WF, raw[:, [2, 0, 1]])
    np.testing.assert_array_equal(adapted.WF[:, np.argsort(association.corrected_to_native)], raw)


@pytest.mark.real_data
def test_real_scalar_reader_matches_pymatgen_and_irrep_records():
    fixture = os.environ.get('SAWF_SCALAR_FIXTURE')
    if fixture is None:
        pytest.skip('SAWF_SCALAR_FIXTURE must explicitly select a small scalar WAVECAR')
    from irrep.readfiles import ParserVasp
    from pymatgen.io.vasp.outputs import Wavecar
    from vasp_sawf.wavecar import read_selected_wavecar, inspect_wavecar

    path = Path(fixture) / 'WAVECAR'
    assert path.stat().st_size < 10 * 1024**2
    header = inspect_wavecar(path)
    selected = read_selected_wavecar(path, bands_1based=[4, 1], kpoints_1based=[1, 2],
                                     lattice=header.lattice, spinor=False)
    reference = Wavecar(str(path), vasp_type='std')
    native = ParserVasp(str(Path(fixture) / 'POSCAR'), str(path), verbosity=0)
    native.parse_header()
    try:
        for ik, point in enumerate(selected.kpoints):
            g = np.rint(reference.Gpoints[ik]).astype(int)
            lookup = {tuple(v): i for i, v in enumerate(g)}
            permutation = [lookup[tuple(v)] for v in point.ig[:, :3]]
            expected = np.array(reference.coeffs[ik])[[3, 0]][:, permutation, None]
            np.testing.assert_array_equal(point.WF, expected)
            raw, energies, k, ng = native.parse_kpoint(ik, header.num_bands, False)
            np.testing.assert_array_equal(point.WF, raw[[3, 0]][:, permutation])
            np.testing.assert_array_equal(point.Energy_raw, energies[[3, 0]])
            np.testing.assert_array_equal(point.k, k)
            assert len(point.ig) == ng
    finally:
        native.fWAV.f.close()
