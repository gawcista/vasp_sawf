"""Automatic targets use current interface results rather than material presets."""

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _centers():
    from test_multicenter import _group
    group = _group('screw')
    centers = np.repeat([[.2, .17, .3], [.8, .67, .7]], 2, axis=0) @ group.lattice
    return group, centers


def _write_wout(path, centers):
    text = 'Final State\n'
    for i, center in enumerate(centers, 1):
        text += f' WF centre and spread {i} ( {center[0]:.9f}, {center[1]:.9f}, {center[2]:.9f} ) 1.0\n'
    path.write_text(text)


def test_auto_target_reads_matching_wout_and_records_its_hash(tmp_path):
    from vasp_sawf.localize import _resolve_target
    group, centers = _centers()
    path = tmp_path/'wannier90.wout'
    _write_wout(path, centers[[2, 0, 3, 1]])
    before = path.read_bytes()
    positions, orbitals, report = _resolve_target(
        tmp_path/'wannier90', centers, group.lattice, group, None, None, center_tol=.01)
    assert len(positions) == 1 and orbitals == ['s']
    assert report['source'] == 'wannier90.wout'
    assert report['source_sha256'] == hashlib.sha256(before).hexdigest()
    assert report['ordinary_center_match_max_angstrom'] < 1e-8
    assert path.read_bytes() == before


def test_auto_target_without_wout_uses_fresh_original_amn_centers(tmp_path):
    from vasp_sawf.localize import _resolve_target
    group, centers = _centers()
    _, _, report = _resolve_target(tmp_path/'wannier90', centers, group.lattice, group,
                                  None, None, center_tol=.01)
    assert report['source'] == 'current_original_amn_localization'


def test_auto_target_rejects_wout_inconsistent_with_current_amn(tmp_path):
    from vasp_sawf.localize import _resolve_target
    group, centers = _centers()
    _write_wout(tmp_path/'wannier90.wout', centers + [.2, 0., 0.])
    with pytest.raises(ValueError, match='WOUT.*current|current.*WOUT'):
        _resolve_target(tmp_path/'wannier90', centers, group.lattice, group, None, None, center_tol=.01)


def test_manual_target_does_not_require_or_read_wout(tmp_path):
    from vasp_sawf.localize import _resolve_target
    group, centers = _centers()
    (tmp_path/'wannier90.wout').write_text('Not a Wannier result')
    p, orbital, report = _resolve_target(tmp_path/'wannier90', centers, group.lattice, group,
                                       [.2, .17, .3], 's', center_tol=.01)
    np.testing.assert_array_equal(p, [[.2, .17, .3]])
    assert orbital == ['s'] and report['source'] == 'explicit_target'


def test_sawf_cli_defaults_to_auto_target_and_current_inputs(monkeypatch, tmp_path):
    from vasp_sawf.run_sawf import main
    module = ModuleType('vasp_sawf.localize')
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'sawf_ready', 'converged': True}
    module.run_sawf = capture
    monkeypatch.setitem(sys.modules, 'vasp_sawf.localize', module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--output', 'model'])
    assert main() == 0
    args, kwargs = calls[0]
    assert args == (Path('wannier90'), Path('symmetry'), Path('model'))
    assert kwargs['center'] is None and kwargs['orbital'] is None
    assert kwargs['center_tol'] == .01


@pytest.mark.parametrize('extra', [['--center', '0', '0', '0'], ['--orbital', 's']])
def test_sawf_cli_rejects_partial_manual_target(monkeypatch, extra):
    from vasp_sawf.run_sawf import main
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--output', 'model', *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_model_output_can_be_new_child_of_calculation_directory(tmp_path):
    from vasp_sawf.localize import _new_output
    source = tmp_path/'wannier90.win'
    source.write_bytes(b'protected')
    output = _new_output(tmp_path/'sawf', [source], allow_input_directory_child=True)
    assert output.is_dir() and source.read_bytes() == b'protected'
    with pytest.raises(ValueError):
        _new_output(source/'child', [source], allow_input_directory_child=True)
