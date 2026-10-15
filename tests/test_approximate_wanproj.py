"""Accepted approximate gauges keep their evidence and limitations at export."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_model_analysis import saved_model


@pytest.mark.parametrize('inside_inputs', [False, True])
def test_approximate_export_reuses_saved_gauge_and_relocated_inputs(saved_model, tmp_path, monkeypatch, inside_inputs):
    from vasp_sawf import localize
    from vasp_sawf.wanproj import export_wanproj, read_wanproj

    seed, model = saved_model
    relocated = tmp_path/'relocated'
    seed.parent.rename(relocated)
    seed = relocated/seed.name
    package = tmp_path/'package'
    protected = {path: path.read_bytes() for folder in (model, package, relocated) for path in folder.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved WANPROJ export must not repeat localization')
    monkeypatch.setattr(localize, '_ordinary_initial_gauge', forbidden)
    monkeypatch.setattr(localize, 'wannierise_full_mesh', forbidden)
    output = (relocated if inside_inputs else tmp_path)/'export'
    report = export_wanproj(model, package, output, seed=seed)
    restored = read_wanproj(output/'WANPROJ')
    with np.load(model/'model.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(restored['U'], arrays['U'])
        np.testing.assert_array_equal(restored['kpoints'], arrays['kpoints'])
    source = json.loads((model/'summary.json').read_text())
    assert report['status'] == report['source_model_status'] == 'accepted_approximate'
    assert report['acceptance'] == source['acceptance']
    assert report['limitation'] == source['limitation']
    assert report['strict_ready'] is False and report['source_full_closure_certified'] is False
    assert report['hamiltonian_uniform_upper_ev'] <= report['hamiltonian_budget_ev']
    assert report['wavefunction_bytes_read'] == 0 and report['localization_repeated'] is False
    assert report['wanproj']['readback_exact']
    assert restored['source_num_bands'] == 1 and restored['bands_vasp_1based'].tolist() == [1]
    assert all(path.read_bytes() == contents for path, contents in protected.items())
    assert set(path.name for path in output.iterdir()) == {'WANPROJ', 'report.json'}


@pytest.mark.parametrize('change', ['unaccepted', 'unconverged', 'model_hash', 'gauge', 'bundle', 'standalone_channel'])
def test_approximate_export_rejects_inconsistent_models(saved_model, tmp_path, change):
    from vasp_sawf.wanproj import export_wanproj

    seed, model = saved_model
    package = tmp_path/'package'
    summary = json.loads((model/'summary.json').read_text())
    if change == 'unaccepted':
        summary['accepted'] = False
    elif change == 'unconverged':
        summary['converged'] = False
    elif change == 'model_hash':
        summary['output_sha256']['model.npz'] = '0'*64
    elif change == 'gauge':
        with np.load(model/'model.npz', allow_pickle=False) as saved:
            arrays = dict(saved)
        arrays['U'] *= 2.
        np.savez_compressed(model/'model.npz', **arrays)
        summary['output_sha256']['model.npz'] = hashlib.sha256((model/'model.npz').read_bytes()).hexdigest()
    elif change == 'bundle':
        report = json.loads((package/'report.json').read_text())
        report['source_diagnostic_sha256'] = '0'*64
        (package/'report.json').write_text(json.dumps(report))
    else:
        summary['spin']['source_ispin'] = 2
    (model/'summary.json').write_text(json.dumps(summary))
    output = tmp_path/'export'
    with pytest.raises(ValueError):
        export_wanproj(model, package, output, seed=seed)
    assert not output.exists()


def test_approximate_export_checks_inputs_again_before_writing(saved_model, tmp_path, monkeypatch):
    from vasp_sawf import model_analysis
    from vasp_sawf.wanproj import export_wanproj

    seed, model = saved_model
    original = model_analysis.analyze_model
    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        with Path(f'{seed}.eig').open('a') as handle:
            handle.write('\n')
        return report
    monkeypatch.setattr(model_analysis, 'analyze_model', changed)
    output = tmp_path/'export'
    with pytest.raises(ValueError, match='changed'):
        export_wanproj(model, tmp_path/'package', output, seed=seed)
    assert not output.exists()


def test_approximate_export_cli_accepts_relocated_seed(saved_model, tmp_path, monkeypatch, capsys):
    from vasp_sawf.wanproj import main

    seed, model = saved_model
    monkeypatch.setattr(sys, 'argv', ['sawf-export-wanproj', '--model', str(model),
        '--symmetry', str(tmp_path/'package'), '--output', str(tmp_path/'export'), '--seed', str(seed)])
    assert main() == 0
    assert 'accepted_approximate' in capsys.readouterr().out
