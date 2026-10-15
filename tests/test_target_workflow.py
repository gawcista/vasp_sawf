"""Automatic targets use current interface results rather than material presets."""

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


@pytest.mark.parametrize('wout_kind', ['absent', 'setup', 'partial', 'matching', 'shifted', 'unrelated', 'directory'])
def test_auto_target_uses_current_amn_centers_regardless_of_wout(tmp_path, monkeypatch, wout_kind):
    from vasp_sawf.localize import _resolve_target
    group, centers = _centers()
    path = tmp_path/'wannier90.wout'
    if wout_kind == 'setup':
        path.write_text('Wannier90 is running in LIBRARY MODE\n'
                        'Finished setting up k-point neighbours.\n'
                        'Exiting wannier_setup in wannier90\n')
    elif wout_kind == 'partial':
        _write_wout(path, centers[:1])
    elif wout_kind == 'matching':
        _write_wout(path, centers[[2, 0, 3, 1]])
    elif wout_kind == 'shifted':
        _write_wout(path, centers + [0., .001, 0.])
    elif wout_kind == 'unrelated':
        _write_wout(path, centers + [.2, 0., 0.])
    elif wout_kind == 'directory':
        path.mkdir()
    before = path.read_bytes() if path.is_file() else None
    monkeypatch.chdir(tmp_path)
    positions, orbitals, report = _resolve_target(
        centers, group.lattice, group, None, None, center_tol=.01)
    np.testing.assert_allclose(positions, [[.2, .17, .3]], atol=1e-12)
    assert orbitals == ['s']
    assert report['source'] == 'current_original_amn_localization'
    if before is not None:
        assert path.read_bytes() == before
    else:
        assert path.is_dir() if wout_kind == 'directory' else not path.exists()


def test_manual_target_does_not_require_or_read_wout(tmp_path):
    from vasp_sawf.localize import _resolve_target
    group, centers = _centers()
    (tmp_path/'wannier90.wout').write_text('Not a Wannier result')
    p, orbital, report = _resolve_target(centers, group.lattice, group,
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
