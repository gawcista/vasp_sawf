"""Extraction scheduling and CLI behavior, independent of physical acceptance."""

from pathlib import Path
import os
import sys

import numpy as np
import pytest


def _worker_wavecar(tmp_path):
    from test_wavecar import _write_wavecar

    path, lattice, refs = _write_wavecar(tmp_path, nb=2)
    with path.open('r+b') as handle:
        for ik, (g, _, _, _) in enumerate(refs):
            g0 = int(np.flatnonzero(np.all(g == 0, axis=1))[0])
            for band in range(2):
                coefficients = np.zeros((len(g), 2), dtype='complex64')
                coefficients[g0, band] = 1
                handle.seek(2048 * (3 + ik * 3 + band))
                coefficients.flatten(order='F').tofile(handle)
    return path, lattice


def test_cli_uses_standard_vasp_names_in_current_directory(tmp_path, monkeypatch):
    from vasp_sawf.extract_symmetry import main
    import vasp_sawf.symmetry as symmetry

    calls = []
    def capture(seed, wavecar, outcar, output, **kwargs):
        calls.append((seed, wavecar, outcar, output, kwargs))
        return {'status': 'ready'}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(symmetry, 'export_symmetry', capture)
    monkeypatch.setattr(sys, 'argv', ['sawf-extract', '--output', '../symmetry'])
    assert main() == 0
    assert calls[0][:4] == (Path('wannier90'), Path('WAVECAR'), Path('OUTCAR'), Path('../symmetry'))


def test_worker_budget_limits_cpu_count_and_memory_without_changing_kpoints():
    from vasp_sawf.extraction import worker_plan

    plan = worker_plan(requested=72, allocated_cpus=72, jobs=24,
                       memory_bytes=16 * 1024**3, per_worker_bytes=2 * 1024**3)
    assert plan['workers'] == 6
    assert plan['jobs'] == 24
    assert plan['memory_reserve_fraction'] == .25
    assert worker_plan(requested=72, allocated_cpus=4, jobs=24,
                       memory_bytes=128 * 1024**3, per_worker_bytes=1024**3)['workers'] == 4
    with pytest.raises(ValueError, match='memory'):
        worker_plan(requested=1, allocated_cpus=1, jobs=24,
                    memory_bytes=1024**3, per_worker_bytes=2 * 1024**3)


def test_one_kpoint_worker_returns_small_matrices_without_retaining_wavefunctions(tmp_path):
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.extraction import evaluate_kpoint
    from vasp_sawf.wavecar import inspect_selected_wavecar

    path, lattice = _worker_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[1, 2], lattice=lattice)
    sg = SpaceGroup(Lattice=lattice, spinor=True, rotations=[np.eye(3, dtype=int)],
                    translations=np.zeros((1, 3)), time_reversals=[False],
                    spinor_rotations=[np.eye(2, dtype=complex)])
    result = evaluate_kpoint(path, [1, 2], lattice, 2, sg.as_dict(), [0],
                             metadata.source_identity, gamma=False)
    assert result['matrices'].shape == (1, 2, 2)
    np.testing.assert_allclose(result['matrices'][0], np.eye(2), atol=1e-11)
    assert result['closure'] < 1e-11
    coefficients = [row for row in result['read_ledger'] if row['kind'] == 'coefficients']
    assert [(r['kpoint_1based'], r['band_1based']) for r in coefficients] == [(2, 1), (2, 2)]
    assert not any(hasattr(value, 'WF') for value in result.values())
    assert sum(value.nbytes for value in result.values() if isinstance(value, np.ndarray)) < 1024


def test_process_workers_match_serial_results_and_each_read_only_one_kpoint(tmp_path):
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.extraction import evaluate_kpoints
    from vasp_sawf.wavecar import inspect_selected_wavecar

    path, lattice = _worker_wavecar(tmp_path)
    metadata = inspect_selected_wavecar(path, bands_1based=[1, 2], lattice=lattice)
    sg = SpaceGroup(Lattice=lattice, spinor=True, rotations=[np.eye(3, dtype=int)],
                    translations=np.zeros((1, 3)), time_reversals=[False],
                    spinor_rotations=[np.eye(2, dtype=complex)])
    tasks = [(path, [1, 2], lattice, ik, sg.as_dict(), [0], metadata.source_identity) for ik in (1, 2)]
    serial = list(evaluate_kpoints(tasks, workers=1))
    parallel = list(evaluate_kpoints(tasks, workers=2))
    for expected, actual in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(actual['matrices'], expected['matrices'])
        assert actual['closure'] == expected['closure']
        assert actual['read_ledger'] == expected['read_ledger']


@pytest.mark.real_data
def test_real_srvo3_streaming_matches_accepted_bundle(tmp_path):
    import json
    from vasp_sawf.symmetry import export_symmetry

    source = os.environ.get('SAWF_SRVO3_WANNIER')
    reference = os.environ.get('SAWF_SRVO3_SYMMETRY')
    if not source or not reference:
        pytest.skip('Small SrVO3 WAVECAR and accepted symmetry bundle were not supplied')
    source = Path(source)
    reference_arrays = np.load(Path(reference) / 'bloch.npz', allow_pickle=False)
    reference_report = json.loads((Path(reference) / 'report.json').read_text())
    for workers in (1, 2):
        output = tmp_path / f'workers-{workers}'
        report = export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR',
                                 output, workers=workers)
        assert report['sawf_ready']
        assert report['physical_acceptance_status'] == 'accepted_for_single_particle_model'
        with np.load(output / 'bloch.npz', allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays['d'], reference_arrays['d'], atol=1e-12, rtol=0)
        assert report['residuals']['ibz_coefficient_closure_relative_max'] == pytest.approx(
            reference_report['residuals']['ibz_coefficient_closure_relative_max'], rel=1e-8)
        reads = [row for row in report['read_ledger'] if row['kind'] == 'coefficients']
        assert len(reads) == 20 * 6
        assert len({(row['kpoint_1based'], row['band_1based']) for row in reads}) == 20 * 6


@pytest.mark.real_data
def test_failed_transport_keeps_completed_gamma_read_and_resource_evidence(tmp_path, monkeypatch):
    import json
    import vasp_sawf.symmetry as symmetry

    source = os.environ.get('SAWF_SRVO3_WANNIER')
    if not source:
        pytest.skip('Small SrVO3 WAVECAR was not supplied')
    source = Path(source)
    def fail_transport(*args, **kwargs):
        raise RuntimeError('Injected transport failure after Gamma evaluation')
    monkeypatch.setattr(symmetry, 'transport_sewing', fail_transport)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='Injected transport failure'):
        symmetry.export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR', output, workers=1)
    report = json.loads((output / 'report.json').read_text())
    assert report['failed_stage'] == 'mmn_transport'
    assert not report['sawf_ready'] and not report['read_accounting_complete']
    assert len([row for row in report['read_ledger'] if row['kind'] == 'coefficients']) == 6
    assert len(report['kpoint_resources']) == 1
    assert report['kpoint_resources'][0]['peak_rss_kib'] > 0
    assert not (output / 'bloch.npz').exists()
