"""Plots consume measured model comparisons without material-specific operations."""

import json

import numpy as np


def test_symmetry_plot_uses_actual_operations_and_mev(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from vasp_sawf import plot_symmetry as plotting

    source = tmp_path/'comparison.json'
    data = dict(schema='sawf-model-comparison-v1', nk=36,
        operations=[dict(index=1, antiunitary=True, rotation=np.eye(3, dtype=int).tolist(),
                         translation=[0, 0, 0]),
                    dict(index=2, antiunitary=False, rotation=np.diag([-1, 1, -1]).tolist(),
                         translation=[0, .5, .337])],
        metrics=dict(ordinary_aligned=dict(gauge_entry_max=[.002, .003],
                                           hamiltonian_spectral_max_ev=[.00002, .00003]),
                     sawf=dict(gauge_entry_max=[1e-14, 2e-14],
                               hamiltonian_spectral_max_ev=[.00001, .000015])))
    source.write_text(json.dumps(data))
    before = source.read_bytes()
    captured = []
    save = Figure.savefig
    def capture(figure, *args, **kwargs):
        captured.append([[bar.get_height() for bar in axis.patches] for axis in figure.axes])
        assert '36' in figure.axes[0].get_title()
        assert 'meV' in figure.axes[1].get_ylabel()
        assert all('SrVO' not in text.get_text() for text in figure.texts)
        return save(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', capture)
    monkeypatch.setattr(plotting, 'DPI', 40)
    plotting.plot_symmetry(source, tmp_path/'figures')
    np.testing.assert_allclose(captured[0][0], [.002, .003, 1e-14, 2e-14])
    np.testing.assert_allclose(captured[0][1], [.02, .03, .01, .015])
    assert (tmp_path/'figures/symmetry_comparison.png').stat().st_size > 0
    assert (tmp_path/'figures/symmetry_comparison.pdf').stat().st_size > 0
    assert source.read_bytes() == before
