import json
from pathlib import Path

import numpy as np
import pytest


def _spin_outcar():
    return '''MAGMOM = 6*0
Startparameter for this run:
 ISTART = 0
 ISPIN = 1
 LNONCOLLINEAR = T
 LSORBIT = T
 ISYM = 2
 LWAVE = T
 transformation matrix from SAXIS to cartesian coordinates
 ---------------------------------------------------------
 1.0000000 m_x 0.0000000 m_y 0.0000000 m_z
 0.0000000 m_x 1.0000000 m_y 0.0000000 m_z
 0.0000000 m_x 0.0000000 m_y 1.0000000 m_z
 number of electron 2.00000 magnetization 0.0000000 0.0000000 0.0000000
'''


def test_outcar_uses_effective_parameters_and_validates_spin_axes():
    from vasp_sawf.symmetry import _outcar_spin_context

    report = _outcar_spin_context('LSORBIT = F\n' + _spin_outcar(), 2)
    assert report['LSORBIT'] is True
    assert report['ISYM'] == 2
    assert report['input_magnetic_moments_zero']
    for text in [_spin_outcar().replace('LSORBIT = T', 'LSORBIT = F'),
                 _spin_outcar().replace('6*0', '5*0 1'),
                 _spin_outcar().replace('1.0000000 m_x', '0.0000000 m_x')]:
        with pytest.raises(ValueError):
            _outcar_spin_context(text, 2)


def test_nonfinite_final_magnetization_cannot_pass_zero_test():
    from vasp_sawf.symmetry import _outcar_spin_context

    text = _spin_outcar().replace('magnetization 0.0000000', 'magnetization nan')
    with pytest.raises(ValueError, match='magnetic moment'):
        _outcar_spin_context(text, 2)


def test_nonfinite_local_residual_cannot_disappear_in_maximum():
    from vasp_sawf.symmetry import _finite_max

    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='finite'):
            _finite_max(0.0, value)
    assert _finite_max(2e-7, 3e-7) == 3e-7


def _accepted_sources():
    return {
        'win': '8470fcee1d8bf6b5c1009fe6ae3111192b66c43f551a8803ed39e6e1f9c0f2ba',
        'amn': '15ae2190883bfd963b083c5a31eee8a371e469174b5aad2755d4a8f570c767b6',
        'eig': 'e5369db82649a277a4819c0cdb9e7f5cf85ee9f3875b4bcceeeee567c138837d',
        'mmn': '9c055588d9f0ae702a5188e8c8241d0f024141019a88b24ec27e3f78fc553d62',
    }


def test_reviewed_closure_is_accepted_automatically_without_changing_residual():
    from vasp_sawf.symmetry import _check_coefficient_closure
    report = {'residuals': {}, 'source_hashes': _accepted_sources()}
    _check_coefficient_closure(report, 2.589629272055618e-6)
    assert report['residuals']['ibz_coefficient_closure_relative_max'] == 2.589629272055618e-6
    assert report['coefficient_closure']['reference_tolerance'] == 1e-6
    assert report['coefficient_closure']['reference_status'] == 'above_reference'
    assert report['physical_acceptance_status'] == 'accepted_for_single_particle_model'
    assert report['coefficient_closure']['acceptance_id'] == 'srvo3-closure-20260924'


@pytest.mark.parametrize('changed_source', ['win','amn','mmn','eig'])
def test_acceptance_cannot_transfer_to_another_interface(changed_source):
    from vasp_sawf.symmetry import _check_coefficient_closure
    report = {'residuals': {}, 'source_hashes': _accepted_sources()}
    report['source_hashes'][changed_source] = 'different'
    with pytest.raises(ValueError, match='[Cc]oefficient closure'):
        _check_coefficient_closure(report, 2.589629272055618e-6)


def test_same_material_cannot_hide_a_larger_closure_error():
    from vasp_sawf.symmetry import _check_coefficient_closure
    report = {'residuals': {}, 'source_hashes': _accepted_sources()}
    with pytest.raises(ValueError, match='[Cc]oefficient closure'):
        _check_coefficient_closure(report, 4e-6)


@pytest.mark.parametrize('value', [np.nan, np.inf, -1e-8])
def test_invalid_closure_is_rejected_even_for_accepted_source(value):
    from vasp_sawf.symmetry import _check_coefficient_closure
    with pytest.raises(ValueError):
        _check_coefficient_closure({'residuals': {},'source_hashes': _accepted_sources()}, value)


def test_unreviewed_small_residual_does_not_claim_physical_acceptance():
    from vasp_sawf.symmetry import _check_coefficient_closure
    report = {'residuals': {}, 'source_hashes': {'win':'other'}}
    _check_coefficient_closure(report, 1e-7)
    assert report['physical_acceptance_status'] == 'not_assessed'
    assert 'acceptance_id' not in report['coefficient_closure']


def test_closure_acceptance_never_bypasses_other_matrix_gates():
    from vasp_sawf.symmetry import _check, _check_coefficient_closure
    report = {'residuals': {}, 'source_hashes': _accepted_sources()}
    _check_coefficient_closure(report, 2.589629272055618e-6)
    with pytest.raises(ValueError, match='mmn_covariance_max'):
        _check(report['residuals'], 'mmn_covariance_max', 2e-6)


def test_atoms_are_read_from_win_with_units_and_species():
    from vasp_sawf.symmetry import _cell_from_win

    lattice = np.eye(3) * 2
    template = '''num_bands=6\nnum_wann=6\nmp_grid=6 6 6
begin unit_cell_cart\n2 0 0\n0 2 0\n0 0 2\nend unit_cell_cart
begin kpoints\n0 0 0\nend kpoints
begin atoms_cart\nang\nSr 0 0 0\nV 1 1 1\nO 1 1 0\nend atoms_cart
'''
    positions, types, species = _cell_from_win(template, lattice)
    np.testing.assert_array_equal(positions, [[0, 0, 0], [.5, .5, .5], [.5, .5, 0]])
    assert species == ['Sr', 'V', 'O']
    np.testing.assert_array_equal(types, [1, 2, 3])


def test_failed_export_writes_not_ready_without_package(tmp_path):
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    source.mkdir()
    output = tmp_path / 'new-output'
    with pytest.raises(Exception):
        export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR', output)
    report = json.loads((output / 'report.json').read_text())
    assert report['sawf_ready'] is False
    assert report['status'] == 'not_ready'
    assert not (output / 'bloch.npz').exists()


def test_source_failure_still_writes_not_ready_and_runtime(tmp_path):
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    source.mkdir()
    output = tmp_path / 'trial-output'
    with pytest.raises(Exception):
        export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR',
                        output)
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'not_ready'
    assert not report['sawf_ready']
    assert report['numerical_checks_passed'] is False
    assert report['wall_seconds'] >= 0
    assert report['peak_rss_kib'] > 0
    assert report['runtime_versions']['irrep'] == '2.6.3'
    assert not (output / 'bloch.npz').exists()


@pytest.mark.parametrize('atoms,kpoint,error', [
    ('Si 0 0 0', '.5 0 0', 'Gamma-centered'),
])
def test_export_rejects_unsupported_geometry_before_coefficient_io(tmp_path, monkeypatch, atoms, kpoint, error):
    import vasp_sawf.symmetry as export

    source = tmp_path / 'inputs'
    source.mkdir()
    seed = source / 'wannier90'
    Path(f'{seed}.win').write_text(
        'num_bands=2\nnum_wann=2\nspinors=true\nmp_grid=1 1 1\n'
        'begin unit_cell_cart\n2 0 0\n0 3 0\n0 0 4\nend unit_cell_cart\n'
        f'begin atoms_frac\n{atoms}\nend atoms_frac\n'
        f'begin kpoints\n{kpoint}\nend kpoints\n')
    Path(f'{seed}.amn').write_text('identity\n2 1 2\n1 1 1 1 0\n2 1 1 0 0\n1 2 1 0 0\n2 2 1 1 0\n')
    Path(f'{seed}.eig').write_text('1 1 0.00000000\n2 1 0.00000000\n')
    Path(f'{seed}.mmn').write_text('identity\n2 1 6\n' + ''.join(
        f'1 1 {shift}\n1 0\n0 0\n0 0\n1 0\n'
        for shift in ('1 0 0', '-1 0 0', '0 1 0', '0 -1 0', '0 0 1', '0 0 -1')))
    (source / 'OUTCAR').write_text(
        _spin_outcar().replace('6*0', f'{3 * len(atoms.splitlines())}*0') +
        'Subroutine IBZKPT returns following result:\nFound 1 irreducible k-points:\n'
        f'Following reciprocal coordinates:\n{kpoint} 1\nFollowing cartesian coordinates:\n'
        'Subroutine IBZKPT_HF returns following result:\nFound 1 k-points in 1st BZ\n'
        f'Following reciprocal coordinates:   # in IRBZ\n{kpoint} 1 1 t-inv F\n')
    # Headers only; no coefficient data are needed for structural rejection.
    with (source / 'WAVECAR').open('wb') as handle:
        handle.truncate(128 * 5)
        np.array([128, 1, 45200], dtype='<f8').tofile(handle)
        handle.seek(128)
        np.array([1, 2, 20, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0], dtype='<f8').tofile(handle)

    def forbid_coefficients(*args, **kwargs):
        raise AssertionError('Unsupported geometry reached coefficient I/O')

    import vasp_sawf.wavecar as wavecar_reader
    monkeypatch.setattr(wavecar_reader, 'read_selected_wavecar', forbid_coefficients)
    output = tmp_path / 'symmetry'
    with pytest.raises(ValueError, match=error):
        export.export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', output)
    report = json.loads((output / 'report.json').read_text())
    assert not report['sawf_ready']
    assert report['header_preflight_read_bytes'] == 128
    assert not (output / 'bloch.npz').exists()


def test_export_never_overwrites_existing_outputs_or_input_directory(tmp_path):
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    source.mkdir()
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'report.json'
    marker.write_text('original')
    for target in (output, marker, source):
        with pytest.raises(ValueError):
            export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR', target)
    assert marker.read_text() == 'original'


def test_export_creates_new_child_output_without_changing_inputs(tmp_path):
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'inputs'
    source.mkdir()
    original = source / 'wannier90.win'
    original.write_bytes(b'Original input remains untouched\n')
    output = source / 'symmetry'
    with pytest.raises(FileNotFoundError):
        export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR', output)
    assert original.read_bytes() == b'Original input remains untouched\n'
    assert set(path.name for path in source.iterdir()) == {'wannier90.win', 'symmetry'}
    assert json.loads((output / 'report.json').read_text())['status'] == 'not_ready'


@pytest.mark.parametrize('name', ['WAVECAR', 'OUTCAR', 'wannier90', 'wannier90.win',
                                 'wannier90.amn', 'wannier90.mmn', 'wannier90.eig'])
@pytest.mark.parametrize('child', ['', 'child'])
def test_export_never_creates_directories_at_missing_input_paths(tmp_path, name, child):
    from vasp_sawf.symmetry import export_symmetry

    output = tmp_path / name / child
    with pytest.raises(ValueError, match='Output'):
        export_symmetry(tmp_path / 'wannier90', tmp_path / 'WAVECAR', tmp_path / 'OUTCAR', output)
    assert list(tmp_path.iterdir()) == []


def test_export_does_not_create_missing_input_tree_as_output(tmp_path):
    from vasp_sawf.symmetry import export_symmetry

    source = tmp_path / 'missing'
    with pytest.raises(ValueError, match='Output'):
        export_symmetry(source / 'wannier90', source / 'WAVECAR', source / 'OUTCAR', source)
    assert not source.exists()


@pytest.mark.parametrize('existing_target', [False, True])
def test_export_refuses_output_symlinks_without_changing_target(tmp_path, existing_target):
    from vasp_sawf.symmetry import export_symmetry

    actual = tmp_path / 'target'
    if existing_target:
        actual.mkdir()
        (actual / 'keep').write_text('original')
    link = tmp_path / 'symmetry'
    link.symlink_to(actual)
    with pytest.raises(ValueError, match='Output'):
        export_symmetry(tmp_path / 'wannier90', tmp_path / 'WAVECAR', tmp_path / 'OUTCAR', link)
    assert link.is_symlink()
    assert actual.exists() == existing_target
    if existing_target:
        assert (actual / 'keep').read_text() == 'original'


@pytest.mark.parametrize('suffix', ['win', 'amn', 'eig', 'mmn'])
def test_export_protects_actual_file_of_symlinked_interfaces(tmp_path, suffix):
    from vasp_sawf.symmetry import export_symmetry

    inputs, actual = tmp_path / 'inputs', tmp_path / 'actual'
    inputs.mkdir()
    actual.mkdir()
    original = actual / f'source.{suffix}'
    original.write_text('protected input')
    (inputs / f'wannier90.{suffix}').symlink_to(original)
    target = original / 'new-output'
    with pytest.raises(ValueError, match='Output'):
        export_symmetry(inputs / 'wannier90', inputs / 'WAVECAR', inputs / 'OUTCAR', target)
    assert not target.exists()
    assert original.read_text() == 'protected input'


def test_spacegroup_arrays_roundtrip_without_pickle_and_reject_invalid_spin_lift():
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.symmetry import _spacegroup_arrays

    sg = SpaceGroup(Lattice=np.eye(3), spinor=True,
                    rotations=np.array([np.eye(3), np.eye(3)], dtype=int),
                    translations=np.zeros((2, 3)), time_reversals=[False, True],
                    spinor_rotations=np.array([np.eye(2), np.eye(2)], dtype=complex))
    arrays = _spacegroup_arrays(sg)
    assert not any(a.dtype.hasobject for a in arrays.values())
    np.testing.assert_array_equal(arrays['time_reversals'], [False, True])
    sg.symmetries[1].spinor_rotation[0, 0] = 2
    with pytest.raises(ValueError):
        _spacegroup_arrays(sg)


def test_export_derives_dimension_and_species_from_inputs_before_spin_check(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import vasp_sawf.symmetry as export

    source = tmp_path / 'input'
    source.mkdir()
    seed = source / 'wannier90'
    Path(f'{seed}.win').write_text(
        'num_bands=2\nnum_wann=2\nmp_grid=2 2 1\n'
        'begin unit_cell_cart\n1 0 0\n0 1 0\n0 0 1\nend unit_cell_cart\n'
        'begin kpoints\n0 0 0\n.5 0 0\n0 .5 0\n.5 .5 0\nend kpoints\n'
        'begin atoms_frac\nXe 0 0 0\nend atoms_frac\n')
    (source / 'OUTCAR').write_text('missing effective parameters')
    header = SimpleNamespace(num_bands=10)
    bundle = SimpleNamespace(amn=np.zeros((4, 2, 2)), mesh=np.array([2, 2, 1]),
                             lattice=np.eye(3), hashes={})
    monkeypatch.setattr(export, 'inspect_wavecar', lambda path: header)

    def read(seed, *, source_nb):
        assert source_nb == 10
        return bundle

    monkeypatch.setattr(export, 'read_inputs', read)
    with pytest.raises(ValueError, match='OUTCAR effective parameter'):
        export.export_symmetry(seed, source / 'WAVECAR', source / 'OUTCAR', tmp_path / 'result')
    report = json.loads((tmp_path / 'result' / 'report.json').read_text())
    assert report['source_num_bands'] == 10
    assert report['shape_nk_nb_nw'] == [4, 2, 2]


@pytest.mark.parametrize('shape', [(4, 2, 1), (4, 3, 3), (4, 0, 0)])
def test_closed_spinor_subspace_rejects_rectangular_odd_and_empty_targets(shape):
    from vasp_sawf.symmetry import _closed_spinor_dimension

    with pytest.raises(ValueError, match='positive even|equal'):
        _closed_spinor_dimension(np.zeros(shape))


def test_grey_group_requires_partner_for_each_rotation_not_only_equal_counts():
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.symmetry import _validate_grey_group

    identity = np.eye(3, dtype=int)
    inversion = -identity
    rotations = np.array([identity, inversion, identity, inversion])
    group = SpaceGroup(Lattice=np.eye(3), spinor=True, rotations=rotations,
                       translations=np.zeros((4, 3)), time_reversals=[False, False, True, True],
                       spinor_rotations=np.tile(np.eye(2, dtype=complex), (4, 1, 1)))
    assert _validate_grey_group(group) == 2
    group.symmetries[-1].rotation[:] = identity
    with pytest.raises(ValueError, match='grey group|unique|one-to-one'):
        _validate_grey_group(group)
