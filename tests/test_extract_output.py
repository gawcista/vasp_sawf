import hashlib
import json
import os

import numpy as np
import pytest


def _inputs(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    return source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR'


@pytest.mark.parametrize('unrelated', [False, True])
def test_existing_output_without_generated_files_is_reused_silently(tmp_path, capsys, unrelated):
    from vasp_sawf.symmetry import export_symmetry

    inputs = _inputs(tmp_path)
    output = tmp_path / 'symmetry'
    output.mkdir()
    if unrelated:
        (output / 'notes.txt').write_text('Keep this user file')
    with pytest.raises(ValueError, match='workers'):
        export_symmetry(*inputs, output, workers=0)
    assert not capsys.readouterr().err
    assert json.loads((output / 'report.json').read_text())['status'] == 'not_ready'
    if unrelated:
        assert (output / 'notes.txt').read_text() == 'Keep this user file'


@pytest.mark.parametrize('existing', [('report.json',), ('bloch.npz',), ('bloch.npz', 'report.json')])
def test_failed_rerun_warns_only_for_existing_generated_files_and_invalidates_old_pair(tmp_path, capsys, existing):
    from vasp_sawf.symmetry import export_symmetry

    inputs = _inputs(tmp_path)
    output = tmp_path / 'symmetry'
    output.mkdir()
    for name in existing:
        (output / name).write_text('old ready output')
    (output / 'keep.txt').write_text('untouched')
    with pytest.raises(ValueError, match='workers'):
        export_symmetry(*inputs, output, workers=0)
    warning = capsys.readouterr().err
    assert 'Warning' in warning
    assert 'keep.txt' not in warning
    for name in ('bloch.npz', 'report.json'):
        assert (name in warning) == (name in existing)
    assert not (output / 'bloch.npz').exists()
    report = json.loads((output / 'report.json').read_text())
    assert not report['sawf_ready'] and not report['numerical_checks_passed']
    assert report['status'] == 'not_ready'
    assert (output / 'keep.txt').read_text() == 'untouched'


def test_previous_ready_state_is_cleared_before_any_wavecar_read(tmp_path, monkeypatch):
    import vasp_sawf.symmetry as symmetry

    inputs = _inputs(tmp_path)
    output = tmp_path / 'symmetry'
    output.mkdir()
    (output / 'bloch.npz').write_bytes(b'old package')
    (output / 'report.json').write_text(json.dumps({'status': 'ready', 'sawf_ready': True}))

    def interrupted_read(*args, **kwargs):
        assert not (output / 'bloch.npz').exists()
        assert json.loads((output / 'report.json').read_text())['status'] == 'not_ready'
        raise RuntimeError('Simulated read failure')

    monkeypatch.setattr(symmetry, 'inspect_wavecar', interrupted_read)
    with pytest.raises(RuntimeError, match='Simulated read failure'):
        symmetry.export_symmetry(*inputs, output)


@pytest.mark.parametrize('name', ['bloch.npz', 'report.json'])
@pytest.mark.parametrize('kind', ['directory', 'symlink', 'dangling_symlink', 'fifo'])
def test_nonregular_generated_paths_are_rejected_before_any_output_change(tmp_path, capsys, name, kind):
    from vasp_sawf.symmetry import export_symmetry

    inputs = _inputs(tmp_path)
    output = tmp_path / 'symmetry'
    output.mkdir()
    target = output / name
    other = output / ('report.json' if name == 'bloch.npz' else 'bloch.npz')
    other.write_text('keep until validation succeeds')
    backing = tmp_path / 'backing'
    if kind == 'directory':
        target.mkdir()
    elif kind == 'fifo':
        os.mkfifo(target)
    else:
        if kind == 'symlink':
            backing.write_text('protected backing')
        target.symlink_to(backing)
    with pytest.raises(ValueError, match='Output|output'):
        export_symmetry(*inputs, output)
    assert not capsys.readouterr().err
    assert other.read_text() == 'keep until validation succeeds'
    if kind == 'symlink':
        assert backing.read_text() == 'protected backing'
    elif kind == 'dangling_symlink':
        assert not backing.exists()


@pytest.mark.parametrize('name', ['bloch.npz', 'report.json'])
def test_generated_file_hardlink_to_protected_input_is_rejected(tmp_path, capsys, name):
    from vasp_sawf.symmetry import export_symmetry

    inputs = _inputs(tmp_path)
    inputs[2].write_text('original OUTCAR remains protected')
    output = tmp_path / 'symmetry'
    output.mkdir()
    os.link(inputs[2], output / name)
    with pytest.raises(ValueError, match='input|Input'):
        export_symmetry(*inputs, output)
    assert not capsys.readouterr().err
    assert inputs[2].read_text() == 'original OUTCAR remains protected'
    assert (output / name).read_text() == inputs[2].read_text()


def test_successful_rerun_replaces_only_generated_bound_pair(tmp_path, capsys):
    from test_scalar_extraction import _atomic_inputs
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    seed = _atomic_inputs(source, 1, 1)
    output = tmp_path / 'symmetry'
    output.mkdir()
    (output / 'bloch.npz').write_bytes(b'old package')
    (output / 'report.json').write_text('old report')
    (output / 'notes.txt').write_text('Keep this user file')
    result = export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output, workers=1, memory_gb=4)
    assert result['sawf_ready']
    assert result['bloch_sha256'] == hashlib.sha256((output / 'bloch.npz').read_bytes()).hexdigest()
    assert json.loads((output / 'report.json').read_text())['bloch_sha256'] == result['bloch_sha256']
    with np.load(output / 'bloch.npz', allow_pickle=False) as package:
        assert package['d'].shape[-2:] == (1, 1)
    assert (output / 'notes.txt').read_text() == 'Keep this user file'
    assert 'notes.txt' not in capsys.readouterr().err
