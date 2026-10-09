"""Focused coefficient checks remain usable when native MMN transport fails."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


def test_cli_forwards_explicit_stored_kpoints(tmp_path, monkeypatch, capsys):
    from vasp_sawf.extract_symmetry import main
    import vasp_sawf.symmetry as symmetry

    calls = []
    monkeypatch.setattr(symmetry, 'export_symmetry',
        lambda *args, **kwargs: calls.append(kwargs) or {'status': 'diagnostic_complete'})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sawf-extract', '--diagnose-kpoints', '1', '20', '23', '24'])
    assert main() == 0
    assert calls[0]['diagnose_kpoints'] == [1, 20, 23, 24]
    assert 'diagnostic.json' in capsys.readouterr().out


def test_diagnostic_measures_bad_mmn_without_replacing_production_outputs(tmp_path):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    seed = _atomic_inputs(source, 1, 1)
    mmn = Path(f'{seed}.mmn')
    mmn.write_text(mmn.read_text().replace('1 1 1 0 0\n1 0', '1 1 1 0 0\n0 1')
                   .replace('1 1 -1 0 0\n1 0', '1 1 -1 0 0\n0 -1'))
    output = tmp_path / 'symmetry'
    with pytest.raises(ValueError, match='mmn_covariance_max'):
        export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output)
    old_report = (output / 'report.json').read_bytes()
    (output / 'bloch.npz').write_bytes(b'preserve unrelated production package')
    original_inputs = {p: p.read_bytes() for p in source.iterdir()}
    report = export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                             diagnose_kpoints=[1], workers=1)
    assert report['status'] == 'diagnostic_complete'
    assert report['diagnostic_only'] and not report['sawf_ready']
    assert not report['numerical_checks_passed']
    assert report['transport']['mmn_covariance_max'] > 1
    assert report['diagnostic_kpoints'][0]['coefficient_closure_relative_max'] < 1e-12
    assert report['diagnostic_kpoints'][0]['group']['group_composition_max'] < 1e-12
    assert report['diagnostic_kpoints'][0]['operations']
    assert report['read_accounting_complete']
    assert (output / 'report.json').read_bytes() == old_report
    assert (output / 'bloch.npz').read_bytes() == b'preserve unrelated production package'
    assert all(p.read_bytes() == original for p, original in original_inputs.items())
    saved = json.loads((output / 'diagnostic.json').read_text())
    assert saved['status'] == 'diagnostic_complete'
    assert saved['evaluated_wavecar_kpoints_1based'] == [1]
    assert {r['kpoint_1based'] for r in saved['read_ledger'] if r['kind'] == 'coefficients'} == {1}


@pytest.mark.parametrize('selected', [[], [0], [-1], [True], [1.5], [2], [1, 1]])
def test_invalid_diagnostic_selection_never_reads_coefficients(tmp_path, monkeypatch, selected):
    from test_scalar_extraction import _atomic_inputs
    import vasp_sawf.wavecar as wavecar
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    seed = _atomic_inputs(source, 1, 1)
    def forbid(*args, **kwargs):
        raise AssertionError('Invalid selected k points must fail before coefficient reads')
    monkeypatch.setattr(wavecar, 'read_selected_wavecar', forbid)
    with pytest.raises(ValueError, match='kpoints|k points|k-point'):
        export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', tmp_path / 'symmetry',
                         diagnose_kpoints=selected)


@pytest.mark.parametrize('target', ['OUTCAR', 'report.json', 'bloch.npz'])
def test_diagnostic_cannot_overwrite_input_or_production_inode_alias(tmp_path, target):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    seed = _atomic_inputs(source, 1, 1)
    output = tmp_path / 'symmetry'
    output.mkdir()
    backing = source / target if target == 'OUTCAR' else output / target
    if target != 'OUTCAR':
        backing.write_bytes(b'protected production result')
    original = backing.read_bytes()
    os.link(backing, output / 'diagnostic.json')
    with pytest.raises(ValueError, match='input|Input|protected|Output'):
        export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                         diagnose_kpoints=[1])
    assert backing.read_bytes() == original


@pytest.mark.parametrize('failure', [None, 'transport', 'worker'])
def test_selected_diagnostic_keeps_stored_indices_distinct_from_permuted_interface(tmp_path, monkeypatch, failure):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from irrep.spacegroup import SpaceGroup
    from test_nonsymmorphic import _two_site_model
    import vasp_sawf.symmetry as symmetry
    import vasp_sawf.extraction as extraction

    original, old_group, mmn, exact = _two_site_model()
    order = np.array([0, 2, 1, 3])
    values = old_group.as_dict()
    for key in ('rotations', 'translations', 'time_reversals', 'spinor_rotations'):
        values[key] = np.asarray(values[key])[order]
    group = SpaceGroup(**values)
    exact = exact[order]
    permutation = np.array([2, 0, 3, 1])
    raw_to_interface = np.argsort(permutation)
    bundle = SimpleNamespace(kpoints=original.kpoints[permutation], mesh=original.mesh,
        neighbor_indices=raw_to_interface[original.neighbor_indices[permutation]],
        neighbor_shifts=original.neighbor_shifts[permutation], mmn=mmn[permutation], eig=np.zeros((4, 4)))
    kmap, edges = symmetry.build_symmetry_maps(bundle, group)
    phases = symmetry.mmn_translation_phases(bundle, group)
    anti = np.array([op.time_reversal for op in group.symmetries])
    little = [np.flatnonzero(kmap[:, k] == k).tolist() for k in raw_to_interface]
    assert little[3] == [0, 2]
    metadata = SimpleNamespace(kpoints=original.kpoints, bands_1based=(1, 2, 3, 4),
        header=SimpleNamespace(lattice=np.eye(3), num_kpoints=4), source_identity={})

    def result(raw):
        return dict(kpoint_1based=raw+1, little_indices=little[raw], matrices=exact[little[raw], raw],
            closure=0., kpoint_energies_ev=[0.]*4,
            operation_diagnostics=[{'operation_index_0based': op} for op in little[raw]],
            read_ledger=[{'bytes_returned': 1, 'kind': 'coefficients', 'kpoint_1based': raw+1}],
            read_seconds=0., compute_seconds=0., peak_rss_kib=1)

    def evaluate(tasks, *, workers, diagnostics):
        assert diagnostics and workers == 2
        assert [task[3] for task in tasks] == [3, 4]
        for task in tasks:
            if failure == 'worker' and task[3] == 4:
                raise OSError('Interrupted coefficient read')
            yield result(task[3]-1)

    monkeypatch.setattr(extraction, 'evaluate_kpoints', evaluate)
    if failure == 'transport':
        def fail(*args, **kwargs):
            raise ValueError('Singular MMN transport')
        monkeypatch.setattr(symmetry, 'transport_sewing', fail)
    report = {'stage_seconds': {}, 'metadata_read_bytes': 0, 'header_preflight_read_bytes': 0,
              'planned_wavecar_kpoints_1based': [1, 3, 4]}
    context = pytest.raises(OSError, match='Interrupted') if failure == 'worker' else nullcontext()
    with context:
        symmetry._diagnose_selected(report, tmp_path / 'diagnostic.json', bundle, group, metadata,
            tmp_path / 'unused-WAVECAR', {'spin_channel': 1}, [0, 2, 3], raw_to_interface, little,
            result(0), kmap, edges, anti, phases, workers=2)
    points = report['diagnostic_kpoints']
    if failure == 'worker':
        saved = json.loads((tmp_path / 'diagnostic.json').read_text())
        assert saved['planned_wavecar_kpoints_1based'] == [1, 3, 4]
        assert saved['evaluated_wavecar_kpoints_1based'] == [1, 3]
        assert not saved.get('read_accounting_complete', False)
        return
    assert report['evaluated_wavecar_kpoints_1based'] == [1, 3, 4]
    assert [point['interface_kpoint_1based'] for point in points] == [2, 1, 3]
    for point in points:
        assert point['group']['group_composition_max'] < 1e-14
        for operation in point['operations']:
            difference = operation['transport_difference_max_abs']
            assert difference is None if failure == 'transport' else difference < 1e-14
    assert report['direct_mmn_covariance']['mmn_covariance_max'] < 1e-14
    assert report['direct_mmn_covariance']['per_operation'][1]['mmn_covariance_max'] is None
    if failure == 'transport':
        assert report['transport_error'] == 'Singular MMN transport'
    assert report['read_accounting_complete']


@pytest.mark.real_data
def test_real_srvo3_focused_diagnostics_match_existing_source_report(tmp_path):
    from vasp_sawf.symmetry import export_symmetry

    source = os.environ.get('SAWF_SRVO3_WANNIER')
    if not source:
        pytest.skip('SAWF_SRVO3_WANNIER was not explicitly set')
    source = Path(source)
    report = export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR',
                             tmp_path / 'diagnostics', diagnose_kpoints=[2, 8], workers=1)
    assert report['evaluated_wavecar_kpoints_1based'] == [1, 2, 8]
    assert len(report['diagnostic_kpoints']) == 3
    assert {r['kpoint_1based'] for r in report['read_ledger'] if r['kind'] == 'coefficients'} == {1, 2, 8}
    assert report['transport']['mmn_covariance_max'] < 1e-6
    assert report['direct_mmn_covariance']['edges_checked'] > 0
    assert report['direct_mmn_covariance']['mmn_covariance_max'] < 1e-6
    for point in report['diagnostic_kpoints']:
        assert point['group']['group_composition_max'] < 1e-6
        for op in point['operations']:
            assert op['independent_lstsq_difference_max'] < 1e-10
            assert op['transport_difference_max_abs'] < 1e-6
    assert not report['sawf_ready']
    assert not (tmp_path / 'diagnostics' / 'bloch.npz').exists()
