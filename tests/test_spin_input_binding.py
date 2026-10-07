"""WIN spin declarations must reach extraction before any coefficient reads."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('contradiction', ['spin_channel', 'spinor_mode'])
def test_extraction_binds_raw_win_spin_before_wavefunction_reads(tmp_path, monkeypatch, contradiction):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.symmetry import export_symmetry
    import vasp_sawf.wavecar as wavecar

    source = tmp_path / 'inputs'
    original = _atomic_inputs(source, 2, 2)
    seed = source / 'renamed'
    for suffix in ('win', 'amn', 'mmn', 'eig'):
        Path(f'{original}.{suffix}').rename(Path(f'{seed}.{suffix}'))
    selected_channel = 1
    if contradiction == 'spinor_mode':
        win = Path(f'{seed}.win')
        win.write_text(win.read_text().replace('spinors=false', 'spinors=true'))
        selected_channel = 2

    def forbid_wavefunction_stage(*args, **kwargs):
        raise AssertionError('Spin binding must fail before selected WAVECAR records are read')

    monkeypatch.setattr(wavecar, 'inspect_selected_wavecar', forbid_wavefunction_stage)
    monkeypatch.setattr(wavecar, 'read_selected_wavecar', forbid_wavefunction_stage)
    output = tmp_path / 'symmetry'
    with pytest.raises(ValueError, match='channel|spinors'):
        export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output,
                        workers=1, memory_gb=4, spin_channel=selected_channel)
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'not_ready'
    assert not (output / 'bloch.npz').exists()
    assert 'metadata_read_bytes' not in report


def test_win_sections_retains_spin_declarations_and_rejects_duplicates(tmp_path):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.inputs import _win_sections

    seed = _atomic_inputs(tmp_path / 'inputs', 2, 2)
    text = Path(f'{seed}.win').read_text()
    scalars, _ = _win_sections(text)
    assert scalars['spin'] == 'down'
    assert scalars['spinors'] == 'false'
    for key, value in [('spin', 'down'), ('spinors', 'false')]:
        with pytest.raises(ValueError, match='duplicate'):
            _win_sections(text + f'{key}={value}\n')
