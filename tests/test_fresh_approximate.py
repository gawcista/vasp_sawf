"""Fresh numerical-budget extraction does not depend on a material registry."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('flag', ['--matrix-tol', '--mmn-tol'])
def test_fresh_cli_dispatches_without_prior_diagnostic(monkeypatch, tmp_path, flag):
    from vasp_sawf import approximate, extract_symmetry
    calls = []
    def extract(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'accepted_approximate'}
    monkeypatch.setattr(approximate, 'extract_fresh_approximate', extract, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sawf-extract', '--accept-approximate',
                                    flag, '1e-3', '--hamiltonian-tol', '1e-4'])
    assert extract_symmetry.main() == 0
    assert calls[0][0] == (Path('wannier90'), Path('WAVECAR'), Path('OUTCAR'), Path('symmetry'))
    assert calls[0][1]['mmn_tol'] == 1e-3
    assert calls[0][1]['hamiltonian_tol'] == 1e-4


@pytest.mark.parametrize('options', [[], ['--mmn-tol', '1e-3'], ['--hamiltonian-tol', '1e-4']])
def test_fresh_acceptance_requires_explicit_paired_budgets(monkeypatch, options):
    from vasp_sawf.extract_symmetry import main
    monkeypatch.setattr(sys, 'argv', ['sawf-extract', '--accept-approximate', *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize('budget', [0., -1., True, float('nan'), float('inf')])
def test_explicit_budget_validation_precedes_output_mutation(tmp_path, budget):
    from vasp_sawf.approximate import extract_fresh_approximate
    output = tmp_path/'symmetry'
    output.mkdir()
    (output/'report.json').write_text('original')
    with pytest.raises(ValueError, match='tolerance|budget'):
        extract_fresh_approximate(tmp_path/'wannier90', tmp_path/'WAVECAR', tmp_path/'OUTCAR',
                                 output, mmn_tol=budget, hamiltonian_tol=1e-4)
    assert (output/'report.json').read_text() == 'original'


def test_fresh_failure_invalidates_ready_and_preserves_other_files(monkeypatch, tmp_path):
    from vasp_sawf.approximate import extract_fresh_approximate
    from vasp_sawf import symmetry
    output = tmp_path/'symmetry'
    output.mkdir()
    (output/'report.json').write_text('{"sawf_ready": true}')
    (output/'bloch.npz').write_bytes(b'stale')
    (output/'keep').write_bytes(b'keep')
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError('source mismatch')
    monkeypatch.setattr(symmetry, 'export_symmetry', fail)
    with pytest.raises(ValueError, match='source mismatch'):
        extract_fresh_approximate(tmp_path/'wannier90', tmp_path/'WAVECAR', tmp_path/'OUTCAR',
                                 output, mmn_tol=1e-3, hamiltonian_tol=1e-4)
    assert calls[0][1]['diagnose_all'] is True
    assert not (output/'bloch.npz').exists()
    assert json.loads((output/'report.json').read_text())['sawf_ready'] is False
    assert (output/'keep').read_bytes() == b'keep'


def test_all_stored_diagnostic_selection_comes_from_actual_header(monkeypatch, tmp_path):
    from vasp_sawf import symmetry
    monkeypatch.setattr(symmetry, 'inspect_wavecar', lambda path: SimpleNamespace(num_kpoints=7, num_bands=12))
    def stop(*args, **kwargs):
        raise ValueError('stop after header')
    monkeypatch.setattr(symmetry, 'read_inputs', stop)
    output = tmp_path/'symmetry'
    with pytest.raises(ValueError, match='stop after header'):
        symmetry.export_symmetry(tmp_path/'wannier90', tmp_path/'WAVECAR', tmp_path/'OUTCAR',
                                 output, diagnose_all=True)
    report = json.loads((output/'diagnostic.json').read_text())
    assert report['requested_wavecar_kpoints_1based'] == list(range(1, 8))
    assert report['status'] == 'diagnostic_failed'


@pytest.mark.parametrize('name', ['bloch.npz', 'report.json', 'diagnostic.json'])
def test_fresh_extraction_cannot_overwrite_a_wavecar_alias(tmp_path, name):
    from vasp_sawf.approximate import extract_fresh_approximate
    wavecar = tmp_path/'WAVECAR'
    wavecar.write_bytes(b'protected original data')
    output = tmp_path/'symmetry'
    output.mkdir()
    os.link(wavecar, output/name)
    with pytest.raises(ValueError, match='overlaps a protected input'):
        extract_fresh_approximate(tmp_path/'wannier90', wavecar, tmp_path/'OUTCAR', output,
                                 mmn_tol=1e-3, hamiltonian_tol=1e-4)
    assert wavecar.read_bytes() == b'protected original data'


def test_large_action_change_or_independent_disagreement_cannot_hide_behind_small_mmn():
    from vasp_sawf.approximate import _check_explicit_metrics
    from copy import deepcopy
    action = dict(unitarity_entry_max=0., per_operation=[dict(mmn_covariance_entry_max=0.,
                  energy_covariance_spectral_max_ev=0.)])
    report = dict(raw_action=action, reconstructed_action=action, action_change_entry_max=0., source_checks={
        'raw_vs_measured_entry_max': 0., 'corrected_vs_measured_entry_max': 0.,
        'source_unitarity_entry_max': 0., 'source_energy_covariance_spectral_max_ev': 0.})
    policy = dict(matrix_tolerance=1e-3, hamiltonian_budget_ev=1e-4)
    for field in ('action_change_entry_max', 'corrected_vs_measured_entry_max'):
        damaged = deepcopy(report)
        (damaged if field == 'action_change_entry_max' else damaged['source_checks'])[field] = .1
        with pytest.raises(ValueError, match=field):
            _check_explicit_metrics(damaged, policy)


@pytest.mark.skipif(not os.environ.get('SAWF_APPROXIMATE_SEED'), reason='Real extracted evidence is optional')
def test_real_fresh_policy_recomputes_without_registry_and_rejects_larger_errors(monkeypatch, tmp_path):
    from vasp_sawf import approximate
    seed = Path(os.environ['SAWF_APPROXIMATE_SEED'])
    diagnostic = Path(os.environ['SAWF_APPROXIMATE_DIAGNOSTIC'])
    def no_registry(*args):
        raise AssertionError('A fresh numerical-budget policy must not consult a material registry')
    monkeypatch.setattr(approximate, '_decision', no_registry)
    output = tmp_path/'symmetry'
    report = approximate.export_approximate(seed, diagnostic, output, mmn_tol=1e-3, hamiltonian_tol=1e-4)
    assert report['acceptance']['policy'] == 'explicit_numerical_budgets_v1'
    assert 'target' not in report['acceptance']
    assert report['source_full_closure_certified'] is False
    assert report['source_checks']['coefficient_closure_relative_max'] > 1e-3
    assert report['source_checks']['corrected_vs_measured_entry_max'] < 1e-3
    approximate._read_package(seed, output)
    shared = tmp_path/'shared'
    shared.mkdir()
    (shared/'diagnostic.json').write_bytes((diagnostic/'diagnostic.json').read_bytes())
    approximate.export_approximate(seed, shared, shared, mmn_tol=1e-3, hamiltonian_tol=1e-4)
    assert (shared/'diagnostic.json').read_bytes() == (diagnostic/'diagnostic.json').read_bytes()
    with pytest.raises(ValueError, match='mmn|matrix'):
        approximate.export_approximate(seed, diagnostic, tmp_path/'too_strict',
                                      mmn_tol=1e-4, hamiltonian_tol=1e-4)
    with pytest.raises(ValueError, match='energy'):
        approximate.export_approximate(seed, diagnostic, tmp_path/'energy_too_strict',
                                      mmn_tol=1e-3, hamiltonian_tol=1e-6)
    changed = dict(report)
    changed['acceptance'] = dict(report['acceptance'], active_axes=[0, 1, 2])
    (output/'report.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='policy|acceptance'):
        approximate._read_package(seed, output)
    damaged = json.loads((diagnostic/'diagnostic.json').read_text())
    point = damaged['diagnostic_kpoints'][1]
    point['matrices_real'][-1] = (-np.asarray(point['matrices_real'][-1])).tolist()
    point['matrices_imag'][-1] = (-np.asarray(point['matrices_imag'][-1])).tolist()
    foreign = tmp_path/'foreign'
    foreign.mkdir()
    (foreign/'diagnostic.json').write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match='raw_vs_measured|corrected_vs_measured'):
        approximate.export_approximate(seed, foreign, tmp_path/'foreign_output',
                                      mmn_tol=1e-3, hamiltonian_tol=1e-4)


@pytest.mark.real_data
@pytest.mark.skipif(not os.environ.get('SAWF_SRVO3_WANNIER'), reason='A real small WAVECAR is optional')
def test_real_fresh_wavecar_extraction_without_registry(monkeypatch, tmp_path):
    from vasp_sawf import approximate
    from vasp_sawf.wavecar import inspect_wavecar
    source = Path(os.environ['SAWF_SRVO3_WANNIER'])
    header = inspect_wavecar(source/'WAVECAR')
    def no_registry(*args):
        raise AssertionError('Fresh extraction must not consult a material registry')
    monkeypatch.setattr(approximate, '_decision', no_registry)
    output = tmp_path/'symmetry'
    report = approximate.extract_fresh_approximate(source/'wannier90', source/'WAVECAR',
        source/'OUTCAR', output, workers=1, memory_gb=2., mmn_tol=1e-3, hamiltonian_tol=1e-4)
    diagnostic = json.loads((output/'diagnostic.json').read_text())
    assert diagnostic['evaluated_wavecar_kpoints_1based'] == list(range(1, header.num_kpoints + 1))
    reads = [row for row in diagnostic['read_ledger'] if row['kind'] == 'coefficients']
    assert len(reads) == header.num_kpoints * report['shape_nk_nb_nw'][1]
    assert len({(row['kpoint_1based'], row['band_1based']) for row in reads}) == len(reads)
    assert report['wavefunction_bytes_read'] == sum(row['bytes_returned'] for row in reads)
    assert report['accepted'] and not report['source_full_closure_certified']
    approximate._read_package(source/'wannier90', output)
