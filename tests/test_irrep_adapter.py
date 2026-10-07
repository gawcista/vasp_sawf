import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _case():
    g = np.array([[-7, 0, 0], [7, 0, 0], [0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    ig = np.column_stack((g, [3, 2, 0, 4, 1], np.zeros(5, int), np.full(5, 5)))
    raw = (np.arange(20).reshape(2, 5, 2) + 1j * np.arange(20, 40).reshape(2, 5, 2)).astype('complex64')
    return ig, raw, raw[:, ig[:, 3], :].copy()


def test_exact_record_association_is_reversible_without_changing_g_labels():
    from vasp_sawf.wavecar import repair_vasp_g_association

    ig, raw, wf = _case()
    ig[:, 3] = [2, 3, 0, 4, 1]
    wf = raw[:, ig[:, 3], :].copy()
    before_ig, before_wf = ig.copy(), wf.copy()
    fixed = repair_vasp_g_association(ig, wf, coefficient_count=10, rtag=45200)
    np.testing.assert_array_equal(fixed.ig[:, 3], [3, 2, 0, 4, 1])
    np.testing.assert_array_equal(fixed.ig[:, [0, 1, 2, 4, 5]], ig[:, [0, 1, 2, 4, 5]])
    np.testing.assert_array_equal(fixed.coefficients, raw[:, [3, 2, 0, 4, 1], :])
    np.testing.assert_array_equal(fixed.coefficients[:, np.argsort(fixed.corrected_to_native), :], wf)
    np.testing.assert_array_equal(fixed.native_to_record, before_ig[:, 3])
    np.testing.assert_array_equal(fixed.corrected_to_record, fixed.ig[:, 3])
    np.testing.assert_array_equal(ig, before_ig)
    np.testing.assert_array_equal(wf, before_wf)
    assert fixed.coefficients.dtype == np.dtype('complex64')


def test_fft_order_handles_zero_axes_and_asymmetric_extrema():
    from vasp_sawf.wavecar import repair_vasp_g_association

    record_g = np.array([[0, 0, 0], [2, 0, 0], [-3, 0, 0], [-1, 0, 0],
                         [0, 1, 0], [-3, 1, 0], [2, -2, 0], [-1, -2, 0],
                         [0, 0, 2], [2, -2, 2], [0, 1, -1]])
    order = np.array([5, 1, 6, 3, 8, 0, 10, 9, 4, 2, 7])
    ig = np.column_stack((record_g[order], np.arange(11), np.zeros(11, int), np.full(11, 11)))
    wf = np.arange(22, dtype='float32').reshape(1, 11, 2).astype('complex64')
    fixed = repair_vasp_g_association(ig, wf, coefficient_count=22, rtag=45200)
    np.testing.assert_array_equal(fixed.corrected_to_record, order)
    np.testing.assert_array_equal(fixed.coefficients, wf[:, order, :])


def test_skew_reciprocal_non_gamma_enumeration_has_fft_record_order():
    from vasp_sawf.wavecar import repair_vasp_g_association

    reciprocal = np.array([[1.0, 0.2, 0.1], [0.0, 1.3, 0.3], [0.1, 0.0, 0.9]])
    k = np.array([0.25, -0.2, 0.1])
    axis = [0, 1, 2, 3, -3, -2, -1]
    record_g = []
    for gz in axis:
        for gy in axis:
            for gx in axis:
                g = np.array([gx, gy, gz])
                if np.linalg.norm((g + k) @ reciprocal) ** 2 < 2.5:
                    record_g.append(g)
    record_g = np.array(record_g)
    ng = len(record_g)
    order = np.arange(ng)[::-1]
    ig = np.column_stack((record_g[order], np.arange(ng), np.zeros(ng, int), np.full(ng, ng)))
    raw = np.arange(2 * ng, dtype='float32').reshape(1, ng, 2).astype('complex64')
    fixed = repair_vasp_g_association(ig, raw, coefficient_count=2 * ng, rtag=45200)
    np.testing.assert_array_equal(fixed.corrected_to_record, order)
    np.testing.assert_array_equal(fixed.coefficients, raw[:, order, :])


def test_new_kpoint_preserves_weight_and_does_not_copy_stale_characters():
    from irrep.kpoint import Kpoint
    from vasp_sawf.wavecar import adapt_irrep_kpoint

    ig, raw, wf = _case()
    kp = Kpoint(ik=3, num_bands=2, RecLattice=np.eye(3), spinor=True,
                kpt=np.zeros(3), WF=wf, Energy=np.array([1., 2.]), ig=ig,
                upper=3., normalize=False)
    kp.weight = 0.125
    kp.char = np.array([999.])
    fixed, _ = adapt_irrep_kpoint(kp, coefficient_count=10, rtag=45200)
    assert fixed.weight == 0.125
    assert fixed.ik0 == 4
    assert not hasattr(fixed, 'char')
    np.testing.assert_array_equal(kp.char, [999.])


@pytest.mark.parametrize('problem', ['duplicate_g', 'float_g', 'duplicate_record',
                                     'truncated', 'wrong_spinor', 'precision', 'rtag', 'nonfinite'])
def test_invalid_or_unsupported_input_is_rejected(problem):
    from vasp_sawf.wavecar import GAssociationError, repair_vasp_g_association

    ig, raw, wf = _case()
    count, rtag = 10, 45200
    if problem == 'duplicate_g':
        ig[0, :3] = ig[1, :3]
    elif problem == 'float_g':
        ig = ig.astype(float)
        ig[0, 0] += 0.5
    elif problem == 'duplicate_record':
        ig[0, 3] = ig[1, 3]
    elif problem == 'truncated':
        count = 12
    elif problem == 'wrong_spinor':
        wf = wf[:, :, :1]
    elif problem == 'precision':
        wf = wf.astype('complex128')
    elif problem == 'rtag':
        rtag = 45210
    elif problem == 'nonfinite':
        wf[0, 0, 0] = np.nan
    with pytest.raises(GAssociationError):
        repair_vasp_g_association(ig, wf, coefficient_count=count, rtag=rtag)


def test_export_cli_refuses_protected_output_before_reading_wavecar(tmp_path):
    output = tmp_path / 'WAVECAR'
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-I', '-m', 'vasp_sawf.extract_symmetry',
                             '--seed', str(tmp_path/'wannier90'), '--wavecar', str(tmp_path/'WAVECAR'),
                             '--outcar', str(tmp_path/'OUTCAR'), '--output', str(output)],
                            cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Output' in result.stderr
    assert not output.exists()


def test_export_cli_default_output_creates_only_result_child(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    original = source / 'wannier90.win'
    original.write_bytes(b'Original input\n')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       XDG_CACHE_HOME=str(tmp_path / 'user-cache'))
    for name in ('PYTHONPATH', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR'):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, '-I', '-m', 'vasp_sawf.extract_symmetry'],
                            cwd=source, env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 1, result.stderr
    assert (source / 'symmetry' / 'report.json').is_file()
    assert original.read_bytes() == b'Original input\n'
    assert set(path.name for path in source.iterdir()) == {'wannier90.win', 'symmetry'}


@pytest.mark.real_data
def test_real_srvo3_twenty_kpoints_match_independent_pymatgen():
    fixture = os.environ.get('SAWF_SRVO3_SOC')
    if fixture is None:
        pytest.skip('SAWF_SRVO3_SOC was not explicitly set for the small-system fixture')
    from irrep.bandstructure import BandStructure
    from pymatgen.io.vasp.outputs import Wavecar
    from vasp_sawf.wavecar import adapt_irrep_kpoint

    fixture = Path(fixture)
    wavecar = fixture / 'WAVECAR'
    assert wavecar.stat().st_size == 37333632
    bs = BandStructure(fWAV=str(wavecar), fPOS=str(fixture / 'POSCAR'),
                       IBstart=32, IBend=38, spinor=True, normalize=False,
                       include_TR=True, save_wf=True, Ecut=None, EF='0.0',
                       calculate_traces=False, irreps=False)
    ref = Wavecar(str(wavecar), vasp_type='ncl')
    assert len(bs.kpoints) == int(ref.nk) == 20
    mismatched_native = 0
    for kp in bs.kpoints:
        ik = int(kp.ik0) - 1
        gvec = np.array(ref.Gpoints[ik], dtype=int)
        lookup = {tuple(g): i for i, g in enumerate(gvec)}
        assert set(lookup) == set(map(tuple, kp.ig[:, :3]))
        order = np.array([lookup[tuple(g)] for g in kp.ig[:, :3]])
        coefficients = np.array(ref.coeffs[ik][32:38]).transpose(0, 2, 1)[:, order, :]
        mismatched_native += int(not np.array_equal(kp.WF, coefficients))
        adapted, mapping = adapt_irrep_kpoint(kp, coefficient_count=2 * len(gvec), rtag=45200)
        assert adapted is not kp
        assert adapted.WF.dtype == np.dtype('complex64')
        np.testing.assert_array_equal(adapted.WF, coefficients)
        np.testing.assert_array_equal(adapted.ig[:, 3], order)
        np.testing.assert_array_equal(adapted.ig[:, :3], kp.ig[:, :3])
        np.testing.assert_array_equal(adapted.k, ref.kpoints[ik])
        np.testing.assert_array_equal(adapted.Energy_raw, ref.band_energy[ik][32:38, 0])
        np.testing.assert_array_equal(adapted.WF[:, np.argsort(mapping.corrected_to_native), :], kp.WF)
    assert mismatched_native == 20
