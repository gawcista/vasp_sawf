"""Explicit approximate acceptance never silently changes strict extraction."""

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


def test_extract_approved_diagnostic_does_not_enter_wavecar_route(monkeypatch, tmp_path):
    from vasp_sawf.extract_symmetry import main

    module = ModuleType('vasp_sawf.approximate')
    calls = []
    def capture(*args):
        calls.append(args)
        return {'status': 'accepted_approximate'}
    module.export_approximate = capture
    monkeypatch.setitem(sys.modules, 'vasp_sawf.approximate', module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sawf-extract', '--from-diagnostic', 'symmetry_full',
                                    '--accept-approximate'])
    assert main() == 0
    assert calls == [(Path('wannier90'), Path('symmetry_full'), Path('symmetry'))]


@pytest.mark.parametrize('options', [
    ['--from-diagnostic', 'symmetry_full'], ['--accept-approximate'],
    ['--from-diagnostic', 'symmetry_full', '--accept-approximate', '--diagnose-kpoints', '1'],
])
def test_extract_requires_explicit_paired_acceptance_options(monkeypatch, options):
    from vasp_sawf.extract_symmetry import main

    monkeypatch.setattr(sys, 'argv', ['sawf-extract', *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_run_dispatches_accepted_package_without_trial_flag(monkeypatch, tmp_path):
    from vasp_sawf.run_sawf import main

    folder = tmp_path/'symmetry'
    folder.mkdir()
    (folder/'report.json').write_text(json.dumps({'schema': 'sawf-bridge-approximate-bloch-v1'}))
    module = ModuleType('vasp_sawf.approximate')
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'accepted_approximate', 'converged': True, 'accepted': True}
    module.run_approximate = capture
    monkeypatch.setitem(sys.modules, 'vasp_sawf.approximate', module)
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--seed', 'wannier90', '--symmetry', str(folder),
        '--center', '.25', '.75', '.16', '--orbital', 's', '--output', str(tmp_path/'model')])
    assert main() == 0
    assert calls[0][0] == (Path('wannier90'), folder, tmp_path/'model')
    assert calls[0][1]['center'] == [[.25, .75, .16]]
    assert calls[0][1]['orbital'] == ['s']
    assert calls[0][1]['num_iter'] == 1000
