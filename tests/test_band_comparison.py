import os
from decimal import Decimal
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from core.bands import path_geometry, read_dft_eigenval, read_win_path


def test_path_does_not_add_distance_or_join_disconnected_segments():
    segments = [('G', [0, 0, 0], 'X', [.5, 0, 0]),
                ('R', [.5, .5, .5], 'M', [.5, .5, 0])]
    k = np.array([[0, 0, 0], [.25, 0, 0], [.5, 0, 0],
                  [.5, .5, .5], [.5, .5, .25], [.5, .5, 0]])
    result = path_geometry(k, segments, np.eye(3) * 2)
    np.testing.assert_allclose(result['distance_inv_angstrom'], [0, np.pi/4, np.pi/2, np.pi/2, 3*np.pi/4, np.pi])
    assert result['segment_slices'] == [(0, 3), (3, 6)]
    assert result['tick_labels'] == ['G', 'X|R', 'M']


def test_shared_endpoint_is_accepted_without_duplication():
    segments = [('G', [0, 0, 0], 'X', [.5, 0, 0]),
                ('X', [.5, 0, 0], 'M', [.5, .5, 0])]
    k = np.array([[0, 0, 0], [.5, 0, 0], [.5, .5, 0]])
    result = path_geometry(k, segments, np.eye(3) * 2)
    assert result['segment_slices'] == [(0, 2), (1, 3)]
    assert result['tick_labels'] == ['G', 'X', 'M']
    np.testing.assert_allclose(result['tick_positions_inv_angstrom'], [0, np.pi/2, np.pi])


@pytest.mark.parametrize('k', [
    [[0, 0, 0], [.25, .1, 0], [.5, 0, 0]],
    [[0, 0, 0], [.4, 0, 0], [.2, 0, 0], [.5, 0, 0]],
    [[0, 0, 0], [.25, 0, 0]],
    [[0, 0, 0], [.5, 0, 0], [.2, 0, 0]],
])
def test_invalid_or_unconsumed_path_points_are_rejected(k):
    with pytest.raises(ValueError):
        path_geometry(k, [('G', [0, 0, 0], 'X', [.5, 0, 0])], np.eye(3))


@pytest.mark.parametrize('bands', [[0], [2, 1], [1, 1], [1.5], []])
def test_invalid_band_selection_is_rejected_before_reading(bands):
    with pytest.raises(ValueError):
        read_dft_eigenval('/nonexistent/EIGENVAL', bands)


def test_win_path_preserves_non_ascii_labels(tmp_path):
    win = tmp_path / 'test.win'
    win.write_text('begin kpoint_path\nΓ 0 0 0 X .5 0 0 ! edge\nend kpoint_path\n')
    assert read_win_path(win) == [('Γ', [0., 0., 0.], 'X', [.5, 0., 0.])]


@pytest.mark.real_data
def test_real_eigenval_internal_blank_cannot_silently_zero_a_band(tmp_path):
    root_text = os.environ.get('SAWF_SRVO3_ROOT')
    if not root_text:
        pytest.skip('The read-only real SrVO3 directory must be explicitly specified')
    lines = (Path(root_text) / 'bandsoc/EIGENVAL').read_text().splitlines()
    first_band33 = next(i for i, row in enumerate(lines) if row.split()[:1] == ['33'] and len(row.split()) == 3 and i > 6)
    lines.insert(first_band33, '')
    changed = tmp_path / 'EIGENVAL'
    changed.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):
        read_dft_eigenval(changed, range(33, 39))


@pytest.mark.real_data
def test_real_srvo3_path_and_eigenval_match_independent_xml():
    root_text = os.environ.get('SAWF_SRVO3_ROOT')
    if not root_text:
        pytest.skip('The read-only real SrVO3 directory must be explicitly specified')
    root = Path(root_text)
    data = read_dft_eigenval(root / 'bandsoc/EIGENVAL', range(33, 39))
    tree = ET.parse(root / 'bandsoc/vasprun.xml')
    xml_k = np.array([[float(x) for x in row.text.split()] for row in
                      tree.findall('./kpoints/varray[@name="kpointlist"]/v')])
    final_eig = tree.findall('./calculation/eigenvalues')[-1]
    xml_eig = np.array([[[float(x) for x in row.text.split()] for row in k.findall('r')]
                        for k in final_eig.findall('./array/set/set/set')])[:, 32:38, 0]
    eig_body = [line.split() for line in (root / 'bandsoc/EIGENVAL').read_text().splitlines()[6:] if line.strip()]
    eig_tokens = np.array([eig_body[i*73][:3] for i in range(120)])
    xml_tokens = np.array([row.text.split() for row in tree.findall('./kpoints/varray[@name="kpointlist"]/v')])
    quantum = np.vectorize(lambda s: float(Decimal(10) ** Decimal(s).as_tuple().exponent))
    printed_bound = (quantum(eig_tokens) + quantum(xml_tokens)) / 2
    assert np.all(abs(data['kpoints'] - xml_k) <= printed_bound + np.finfo(float).eps)
    np.testing.assert_allclose(data['eigenvalues_eV'], xml_eig, atol=5.1e-5, rtol=0)
    assert data['eigenvalues_eV'].shape == (120, 6)
    assert data['source_num_bands'] == 72
    geometry = path_geometry(data['kpoints'], read_win_path(root / 'wannier/wannier90.win'),
                             np.eye(3) * 3.8610845)
    assert geometry['segment_slices'] == [(i, i+20) for i in range(0, 120, 20)]
    assert geometry['tick_labels'] == ['Γ', 'X', 'M', 'Γ', 'R', 'X|R', 'M']
