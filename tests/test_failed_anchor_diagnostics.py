"""Failed extraction must preserve small Gamma data sufficient for an offline replay."""

import json

import numpy as np
import pytest


def _failed_atomic_extraction(tmp_path, monkeypatch):
    from test_scalar_extraction import _atomic_inputs
    import vasp_sawf.extraction as extraction
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'input'
    seed = _atomic_inputs(source, 1, 1)
    original = extraction.evaluate_kpoint
    changed = []

    def inject_bad_anchor(*args, **kwargs):
        result = original(*args, **kwargs)
        assert kwargs.get('gamma') is True
        anti_index = int(np.flatnonzero(args[4]['time_reversals'])[0])
        local_index = result['little_indices'].index(anti_index)
        result['matrices'][local_index, 0, 0] = 2j
        result['closure'] = result['gamma_lstsq'] = float(np.sqrt(5.))
        changed.append(anti_index)
        return result

    monkeypatch.setattr(extraction, 'evaluate_kpoint', inject_bad_anchor)
    output = tmp_path / 'symmetry'
    with pytest.raises(ValueError, match='mmn_covariance_max failed'):
        export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                        workers=1, memory_gb=4)
    report = json.loads((output / 'report.json').read_text())
    assert not (output / 'bloch.npz').exists()
    assert report['status'] == 'not_ready' and not report['sawf_ready']
    assert report['failed_stage'] == 'mmn_transport'
    assert changed and len(changed) == 1
    return seed, report, changed[0]


def test_actual_transport_failure_preserves_complex_gamma_and_measured_scores(tmp_path, monkeypatch):
    from vasp_sawf.inputs import read_inputs

    seed, report, changed = _failed_atomic_extraction(tmp_path, monkeypatch)
    anchor = report['gamma_anchor']
    operations = report['symmetry_operations']['operations']
    ns = report['symmetry_operations']['total']
    assert anchor['wavecar_kpoint_1based'] == anchor['interface_kpoint_1based'] == 1
    assert anchor['operation_indices_0based'] == list(range(ns))
    assert [op['index_0based'] for op in operations] == list(range(ns))
    assert operations[changed]['time_reversal'] is True
    matrices = np.asarray(anchor['matrices_real']) + 1j*np.asarray(anchor['matrices_imag'])
    expected = np.ones((ns, 1, 1), complex)
    expected[changed, 0, 0] = 2j
    np.testing.assert_array_equal(matrices, expected)
    assert anchor['coefficient_closure_relative_max'] == float(np.sqrt(5.))
    assert anchor['independent_lstsq_difference_max'] == float(np.sqrt(5.))
    assert anchor['unitarity_max'] == pytest.approx(3.)
    assert report['transport']['mmn_covariance_max'] == pytest.approx(3.)
    assert report['source_hashes'] == read_inputs(seed, source_nb=1).hashes
    assert len([entry for entry in report['read_ledger'] if entry['kind'] == 'coefficients']) == 1


def test_failed_report_and_native_mmn_replay_without_wavecar(tmp_path, monkeypatch):
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.inputs import read_inputs
    from vasp_sawf.symmetry import build_symmetry_maps, mmn_translation_phases, transport_sewing
    import vasp_sawf.wavecar as wavecar

    seed, report, _ = _failed_atomic_extraction(tmp_path, monkeypatch)

    def forbid_wavecar(*args, **kwargs):
        raise AssertionError('Offline replay must not read WAVECAR')

    for name in ('inspect_wavecar', 'inspect_selected_wavecar', 'read_selected_wavecar'):
        monkeypatch.setattr(wavecar, name, forbid_wavecar)
    bundle = read_inputs(seed, source_nb=report['source_num_bands'])
    operations = report['symmetry_operations']['operations']
    group = SpaceGroup(Lattice=bundle.lattice, spinor=report['spin']['spinor'],
        rotations=np.array([op['rotation'] for op in operations]),
        translations=np.array([op['translation'] for op in operations]),
        time_reversals=[op['time_reversal'] for op in operations])
    anchor = report['gamma_anchor']
    matrices = np.asarray(anchor['matrices_real']) + 1j*np.asarray(anchor['matrices_imag'])
    ordered = np.empty_like(matrices)
    ordered[anchor['operation_indices_0based']] = matrices
    kmap, edge_map = build_symmetry_maps(bundle, group)
    _, replay = transport_sewing(bundle.mmn, bundle.neighbor_indices, kmap, edge_map,
        [op['time_reversal'] for op in operations], ordered,
        anchor_k=anchor['interface_kpoint_1based']-1, edge_phases=mmn_translation_phases(bundle, group))
    for key in ('mmn_covariance_max', 'mmn_covariance_per_operation', 'unitarity_max',
                'min_edge_singular', 'max_edge_condition'):
        np.testing.assert_allclose(replay[key], report['transport'][key], atol=1e-14, rtol=0)
