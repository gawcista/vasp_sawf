from types import SimpleNamespace

import numpy as np
import pytest


def _scalar(ispin=1, moments=None):
    text = f'''Startparameter for this run:
 ISPIN = {ispin}
 LSORBIT = F
 LNONCOLLINEAR = F
 NSW = 0
'''
    if moments is not None:
        text += ''' position of ions in fractional coordinates (direct lattice)
 0.00000000 0.00000000 0.00000000
 0.50000000 0.00000000 0.00000000
'''
        text += ''' magnetization (x)

 # of ion       s       p       d       tot
 --------------------------------------------------
'''
        text += ''.join(f' {i} 0.000 0.000 {value} {value}\n'
                        for i, value in enumerate(moments, 1))
        text += ' --------------------------------------------------\n tot 0.000 0.000 0.000 0.000\n'
    return text


def _soc():
    return '''MAGMOM = 6*0
Startparameter for this run:
 ISTART = 1
 ISPIN = 1
 LSORBIT = T
 LNONCOLLINEAR = T
 ISYM = 2
 LWAVE = F
 transformation matrix from SAXIS to cartesian coordinates
 ---------------------------------------------------------
 1.0000000 m_x 0.0000000 m_y 0.0000000 m_z
 0.0000000 m_x 1.0000000 m_y 0.0000000 m_z
 0.0000000 m_x 0.0000000 m_y 1.0000000 m_z
 number of electron 2.00000 magnetization 0.0000000 0.0000000 0.0000000
'''


def test_scalar_context_needs_neither_saxis_nor_optional_output_metadata():
    from vasp_sawf.spin import _outcar_spin_context

    context = _outcar_spin_context(_scalar(), 2)
    assert not context['spinor']
    assert context['source_ispin'] == 1
    for key in ('LWAVE', 'ISTART', 'ISYM', 'input_magnetic_moments_zero'):
        assert context[key] is None
    assert 'source_status' not in context and 'gauge_status' not in context


@pytest.mark.parametrize('replace,by', [('LWAVE = F', 'LWAVE = nope'),
                                      ('ISPIN = 1', 'ISPIN = 3'),
                                      ('ISYM = 2', 'ISYM = 2.5'),
                                      ('LSORBIT = T', 'LSORBIT = yes')])
def test_present_parameters_are_validated(replace, by):
    from vasp_sawf.spin import _outcar_spin_context

    with pytest.raises(ValueError, match='OUTCAR|ISPIN'):
        _outcar_spin_context(_soc().replace(replace, by), 2)


@pytest.mark.parametrize('tag', ['LWAVE', 'ISTART', 'ISYM', 'LSORBIT', 'LNONCOLLINEAR', 'ISPIN'])
def test_duplicate_parameters_are_rejected(tag):
    from vasp_sawf.spin import _outcar_spin_context

    with pytest.raises(ValueError, match='not unique'):
        _outcar_spin_context(_soc() + f'{tag} = 1\n', 2)


def test_soc_initial_moments_are_diagnostic_and_ispin_does_not_change_spinor_mode():
    from vasp_sawf.spin import _outcar_spin_context

    context = _outcar_spin_context(_soc().replace('6*0', '0 0 1 0 0 -1')
                                  .replace('ISPIN = 1', 'ISPIN = 2'), 2)
    assert context['spinor']
    assert context['source_ispin'] == 1
    assert context['ISPIN'] == 2
    assert context['LWAVE'] is False
    assert context['input_magnetic_moments_zero'] is False
    assert 'candidate' in context['time_reversal_basis'].lower()


@pytest.mark.parametrize('replacement', ['0.1000000', 'nan'])
def test_soc_final_magnetization_is_still_checked(replacement):
    from vasp_sawf.spin import _outcar_spin_context

    with pytest.raises(ValueError, match='magnetic moment'):
        _outcar_spin_context(_soc().replace('magnetization 0.0000000',
                                           f'magnetization {replacement}'), 2)


def test_soc_noncartesian_spin_basis_is_not_silently_accepted():
    from vasp_sawf.spin import _outcar_spin_context

    with pytest.raises(ValueError, match='Cartesian'):
        _outcar_spin_context(_soc().replace('1.0000000 m_x', '0.0000000 m_x'), 2)


def test_initial_moments_and_zero_global_moment_cannot_replace_final_site_table():
    from vasp_sawf.spin import _outcar_spin_context

    text = 'MAGMOM = 1 -1\n' + _scalar(2) + 'number of electron 2 magnetization 0.0000000\n'
    with pytest.raises(ValueError, match=r'final.*magnetization.*LORBIT'):
        _outcar_spin_context(text, 2)


def test_last_complete_site_table_and_its_print_precision_select_candidate():
    from vasp_sawf.spin import _outcar_spin_context

    text = _scalar(2, ['0.123', '0.123'])
    text += ' magnetization (x)' + _scalar(2, ['1.2345', '-1.2345']).split(' magnetization (x)', 1)[1]
    context = _outcar_spin_context(text, 2)
    assert context['final_site_magnetic_moments'] == [1.2345, -1.2345]
    assert 1e-4 <= context['magnetic_candidate_tolerance'] < 1.000001e-4
    assert context['source_ispin'] == 2
    assert context['antiunitary_kind'] == 'channel_complex_conjugation'
    assert context['physical_time_reversal_enforced'] is False


@pytest.mark.parametrize('moments', [['1.000'], ['1.000', 'nan']])
def test_incomplete_or_nonfinite_final_site_table_is_rejected(moments):
    from vasp_sawf.spin import _outcar_spin_context

    with pytest.raises(ValueError, match='magnetization|magnetic moment'):
        _outcar_spin_context(_scalar(2, moments), 2)


@pytest.mark.parametrize('moments,translation_expected', [(['1.000', '1.000'], True),
                                                        (['1.000', '-1.000'], False)])
def test_collinear_group_retains_only_signed_moment_preserving_spatial_operations(moments, translation_expected):
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.spin import _outcar_spin_context, _spacegroup_from_context

    bundle = SimpleNamespace(lattice=np.diag([2., 3., 4.]))
    positions = np.array([[0., 0., 0.], [.5, 0., 0.]])
    context = _outcar_spin_context(_scalar(2, moments), 2)
    group = _spacegroup_from_context(bundle, positions, [1, 1], context)
    unitary = [op for op in group.symmetries if not op.time_reversal]
    anti = [op for op in group.symmetries if op.time_reversal]
    exchange = any(np.array_equal(op.rotation, np.eye(3))
                   and np.allclose(op.translation, [.5, 0., 0.]) for op in unitary)
    assert exchange == translation_expected
    assert len(anti) == len(unitary)
    assert not group.spinor
    for op in unitary:
        assert any(np.array_equal(other.rotation, op.rotation)
                   and np.array_equal(other.translation, op.translation) for other in anti)
    restored = SpaceGroup(**group.as_dict())
    products, shifts, factors = restored.get_product_table(get_diff=True)
    assert products.shape == (group.size, group.size)
    np.testing.assert_array_equal(factors, np.ones_like(factors))


def test_soc_group_construction_preserves_existing_operation_order():
    from irrep.spacegroup import SpaceGroup
    from vasp_sawf.spin import _outcar_spin_context, _spacegroup_from_context

    bundle = SimpleNamespace(lattice=np.diag([2., 3., 4.]))
    positions = np.array([[0., 0., 0.], [.5, 0., 0.]])
    context = _outcar_spin_context(_soc(), 2)
    actual = _spacegroup_from_context(bundle, positions, [1, 1], context)
    expected = SpaceGroup.from_cell(cell=(bundle.lattice, positions, [1, 1]), spinor=True,
                                    magmom=True, include_TR=True, verbosity=0)
    for left, right in zip(actual.symmetries, expected.symmetries, strict=True):
        np.testing.assert_array_equal(left.rotation, right.rotation)
        np.testing.assert_array_equal(left.translation, right.translation)
        np.testing.assert_array_equal(left.spinor_rotation, right.spinor_rotation)
        assert left.time_reversal == right.time_reversal


def test_magnetic_site_indices_cannot_be_assigned_to_reordered_win_atoms():
    from vasp_sawf.spin import _outcar_spin_context, _spacegroup_from_context

    bundle = SimpleNamespace(lattice=np.diag([2., 3., 4.]))
    context = _outcar_spin_context(_scalar(2, ['1.000', '-1.000']), 2)
    with pytest.raises(ValueError, match='atom order|positions'):
        _spacegroup_from_context(bundle, np.array([[.5, 0, 0], [0, 0, 0]]), [1, 1], context)


def test_latest_cartesian_ionic_positions_override_initial_fractional_positions():
    from vasp_sawf.spin import _outcar_spin_context, _spacegroup_from_context

    frame = ''' POSITION                    TOTAL-FORCE (eV/Angst)
 ----------------------------------------------------------
 0.500000 0.000000 0.000000 0.0 0.0 0.0
 1.500000 0.000000 0.000000 0.0 0.0 0.0
 ----------------------------------------------------------
'''
    text = _scalar(2, ['1.000', '-1.000']).replace('NSW = 0', 'NSW = 5')
    text += frame.replace('0.500000', '0.100000').replace('1.500000', '1.100000') + frame
    context = _outcar_spin_context(text, 2)
    assert context['site_positions']['kind'] == 'cartesian'
    np.testing.assert_array_equal(context['site_positions']['coordinates'], [[.5, 0, 0], [1.5, 0, 0]])
    bundle = SimpleNamespace(lattice=np.diag([2., 3., 4.]))
    # Identical atoms may be represented in neighboring cells without changing their order.
    _spacegroup_from_context(bundle, np.array([[1.25, 0, 0], [.75, 0, 0]]), [1, 1], context)
    with pytest.raises(ValueError, match='atom order|positions'):
        _spacegroup_from_context(bundle, np.array([[0., 0, 0], [.5, 0, 0]]), [1, 1], context)


def test_initial_positions_cannot_bind_moments_after_unreported_ionic_updates():
    from vasp_sawf.spin import _outcar_spin_context

    with pytest.raises(ValueError, match='final.*POSITION|static'):
        _outcar_spin_context(_scalar(2, ['1.000', '-1.000']).replace('NSW = 0', 'NSW = 5'), 2)
