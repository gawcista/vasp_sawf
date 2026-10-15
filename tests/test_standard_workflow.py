"""Real compact evidence exercises a fresh target without a material registry."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.real_data
@pytest.mark.skipif(not os.environ.get('SAWF_STANDARD_FULL'), reason='Full compact-data regression is opt-in')
@pytest.mark.parametrize('with_wout', [True, False])
def test_new_input_prefix_and_hashes_auto_target_end_to_end(tmp_path, monkeypatch, with_wout):
    import vasp_sawf.approximate as approximate

    source = Path(os.environ['SAWF_APPROXIMATE_SEED'])
    diagnostic = Path(os.environ['SAWF_APPROXIMATE_DIAGNOSTIC'])
    seed = tmp_path/'fresh_input'
    extensions = ('win', 'mmn', 'amn', 'eig', 'wout') if with_wout else ('win', 'mmn', 'amn', 'eig')
    for extension in extensions:
        Path(f'{seed}.{extension}').write_bytes(Path(f'{source}.{extension}').read_bytes())
    with Path(f'{seed}.win').open('a') as handle:
        handle.write('\n! Independent input prefix for automatic workflow regression\n')
    hashes = {extension: hashlib.sha256(Path(f'{seed}.{extension}').read_bytes()).hexdigest()
              for extension in ('win', 'mmn', 'amn', 'eig')}
    record = json.loads((diagnostic/'diagnostic.json').read_bytes())
    record.update(source_hashes=hashes, source_seed=str(seed))
    fixture = tmp_path/'measured_fixture'
    fixture.mkdir()
    (fixture/'diagnostic.json').write_text(json.dumps(record))
    def forbidden_registry(*args, **kwargs):
        raise AssertionError('A new input must never consult a material acceptance registry')
    monkeypatch.setattr(approximate, '_decision', forbidden_registry)
    package = approximate.export_approximate(seed, fixture, tmp_path/'symmetry',
        mmn_tol=1e-3, hamiltonian_tol=1e-4)
    model = approximate.run_approximate(seed, tmp_path/'symmetry', tmp_path/'sawf')
    assert package['accepted'] and model['accepted'] and model['converged']
    assert model['source_hashes'] == hashes
    selection = model['target']['selection']
    assert selection['source'] == ('wannier90.wout' if with_wout else 'current_original_amn_localization')
    assert selection['inference']['local_representation'] == 'local_kramers_doublet'
    assert selection['inference']['atomic_orbital_assignment'] is False
    assert selection['inference']['orbit_site_counts'] == [2, 2]
    assert model['hamiltonian_uniform_upper_ev'] <= 1e-4
    with np.load(tmp_path/'sawf/model.npz', allow_pickle=False) as arrays:
        assert arrays['U'].shape == (36, 8, 8)
        np.testing.assert_allclose(arrays['U'].conj().swapaxes(-1, -2) @ arrays['U'],
                                   np.broadcast_to(np.eye(8), (36, 8, 8)), atol=1e-12)
        trial = json.loads(arrays['trial_summary_utf8'].tobytes())
        assert trial['all_k_gauge_covariance_max'] < 1e-6
    assert all(hashlib.sha256(Path(f'{seed}.{ext}').read_bytes()).hexdigest() == digest
               for ext, digest in hashes.items())
