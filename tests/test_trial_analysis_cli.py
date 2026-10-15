"""The analysis route reuses saved trials without entering localization."""

import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_analysis_dispatch_needs_no_new_target_or_symmetry(monkeypatch, tmp_path):
    from vasp_sawf.run_sawf import main
    module = ModuleType('vasp_sawf.analysis')
    calls = []
    def analyze(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'analysis_complete', 'accepted': False, 'sawf_ready': False}
    module.analyze_trial = analyze
    monkeypatch.setitem(sys.modules, 'vasp_sawf.analysis', module)
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--analyze-trial', 'trial',
        '--output', str(tmp_path/'analysis'), '--points-per-segment', '51',
        '--dft-eigenval', 'EIGENVAL.0', 'EIGENVAL.1', '--energy-reference-ev', '-.9'])
    assert main() == 0
    assert calls[0][0] == (Path('trial'), tmp_path/'analysis')
    assert calls[0][1]['seed'] is None
    assert calls[0][1]['points_per_segment'] == 51
    assert calls[0][1]['dft_eigenval'] == [Path('EIGENVAL.0'), Path('EIGENVAL.1')]


@pytest.mark.parametrize('extra', [['--trial'], ['--center', '0', '0', '0'],
    ['--orbital', 's'], ['--symmetry', 'symmetry'], ['--num-iter', '500']])
def test_analysis_rejects_options_that_would_be_silently_ignored(monkeypatch, extra):
    from vasp_sawf.run_sawf import main
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--analyze-trial', 'trial',
                                    '--output', 'analysis', *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_localization_rejects_incomplete_explicit_target(monkeypatch):
    from vasp_sawf.run_sawf import main
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--output', 'model', '--orbital', 's'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_existing_single_dft_argument_keeps_path_semantics(monkeypatch, tmp_path):
    from vasp_sawf.run_sawf import main
    import vasp_sawf.localize as localize
    calls = []
    def localize_model(*args, **kwargs):
        calls.append(kwargs)
        return {'status': 'ready', 'converged': True}
    monkeypatch.setattr(localize, 'run_sawf', localize_model)
    monkeypatch.setattr(sys, 'argv', ['sawf-run', '--seed', 'w', '--symmetry', 'symmetry',
        '--center', '0', '0', '0', '--orbital', 's', '--output', str(tmp_path/'model'),
        '--dft-eigenval', 'EIGENVAL'])
    assert main() == 0
    assert calls[0]['dft_eigenval'] == Path('EIGENVAL')
    assert calls[0]['num_iter'] == 1000
