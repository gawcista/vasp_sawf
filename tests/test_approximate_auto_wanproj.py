"""Accepted approximate runs export their saved gauge and clean up failed exports."""

import hashlib
import json

import numpy as np
import pytest

from test_model_analysis import saved_model


def test_approximate_run_writes_final_wanproj(saved_model):
    from vasp_sawf.wanproj import read_wanproj

    _, model = saved_model
    summary = json.loads((model/'summary.json').read_text())
    assert (model/'WANPROJ').is_file()
    restored = read_wanproj(model/'WANPROJ')
    with np.load(model/'model.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(restored['U'], arrays['U'])
        np.testing.assert_array_equal(restored['kpoints'], arrays['kpoints'])
        np.testing.assert_array_equal(restored['bands_vasp_1based'], arrays['bands_vasp_1based'])
    assert restored['source_num_bands'] == summary['source_num_bands']
    assert summary['wanproj']['readback_exact'] is True
    assert summary['output_sha256']['WANPROJ'] == hashlib.sha256((model/'WANPROJ').read_bytes()).hexdigest()
    assert summary['status'] == 'accepted_approximate'
    assert summary['strict_ready'] is False


def test_failed_approximate_wanproj_leaves_no_model(saved_model, tmp_path, monkeypatch):
    from vasp_sawf.approximate import run_approximate
    import vasp_sawf.wanproj as wanproj

    seed, _ = saved_model
    output = tmp_path/'failed_model'
    def fail(path, **kwargs):
        path.write_text('partial export')
        raise ValueError('Simulated WANPROJ write failure')
    monkeypatch.setattr(wanproj, 'write_wanproj', fail)
    with pytest.raises(ValueError, match='Simulated WANPROJ'):
        run_approximate(seed, tmp_path/'package', output,
                        center=[0, 0, 0], orbital='s', num_iter=8, points_per_segment=3)
    report = json.loads((output/'summary.json').read_text())
    assert report['status'] == 'not_ready' and report['accepted'] is False
    assert not (output/'WANPROJ').exists()
    assert not (output/'model.npz').exists()
