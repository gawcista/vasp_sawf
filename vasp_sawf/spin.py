"""Effective VASP spin modes and candidate symmetries for each retained sector."""

from decimal import Decimal, InvalidOperation
import re

import numpy as np


def _effective_parameter(text, key, *, required):
    values = re.findall(rf'^\s*{key}\s*=\s*([^\n]*)', text, re.M)
    if len(values) > 1:
        raise ValueError(f'OUTCAR effective {key} is not unique')
    if not values:
        if required:
            raise ValueError(f'OUTCAR effective {key} is missing')
        return None
    tokens = values[0].split()
    value = tokens[0].upper() if tokens else ''
    if key.startswith('L'):
        if value not in ('T', 'F', '.TRUE.', '.FALSE.'):
            raise ValueError(f'OUTCAR effective {key} is not a valid logical value')
        return value in ('T', '.TRUE.')
    if not re.fullmatch(r'[+-]?\d+', value):
        raise ValueError(f'OUTCAR effective {key} is not a valid integer')
    return int(value)


def _initial_moments(text):
    records = re.findall(r'^\s*MAGMOM\s*=\s*([^\n]*)', text, re.M)
    if len(records) > 1:
        raise ValueError('OUTCAR input MAGMOM is not unique')
    if not records:
        return None
    moments = []
    try:
        for token in records[0].split('!')[0].split('#')[0].split():
            count, value = token.split('*') if '*' in token else ('1', token)
            if not re.fullmatch(r'\d+', count) or int(count) < 1:
                raise ValueError
            moments.extend([float(value.replace('D', 'E').replace('d', 'e'))] * int(count))
    except ValueError as exc:
        raise ValueError('OUTCAR initial magnetic moments are malformed') from exc
    if not moments or not np.isfinite(moments).all():
        raise ValueError('OUTCAR initial magnetic moments are empty or nonfinite')
    return moments


def _final_site_moments(text, nions):
    blocks = list(re.finditer(r'^\s*magnetization\s*\(x\)\s*$', text, re.M))
    if not blocks:
        raise ValueError('ISPIN=2 requires the final per-site magnetization (x) table in OUTCAR; '
                         'provide the matching output with LORBIT enabled to identify the candidate spatial subgroup')
    tail = text[blocks[-1].end():]
    header = re.search(r'^\s*# of ion\s+.*\btot\s*$', tail, re.M)
    if header is None:
        raise ValueError('The final magnetization (x) table has no valid column header')
    values, quanta = [], []
    started = False
    for line in tail[header.end():].splitlines():
        tokens = line.split()
        if not tokens or set(tokens[0]) == {'-'}:
            if started:
                break
            continue
        if not tokens[0].isdigit():
            break
        started = True
        if int(tokens[0]) != len(values) + 1 or len(tokens) < 3:
            raise ValueError('The final magnetization (x) table has invalid or duplicate site indices')
        try:
            printed = Decimal(tokens[-1].replace('D', 'E').replace('d', 'e'))
            if not printed.is_finite():
                raise InvalidOperation
            value = float(printed)
            quantum = float(Decimal(10) ** printed.as_tuple().exponent)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ValueError('The final site magnetic moment is malformed or nonfinite') from exc
        if not np.isfinite(value) or not np.isfinite(quantum) or quantum <= 0:
            raise ValueError('The final site magnetic moment or its precision is nonfinite')
        values.append(value)
        quanta.append(quantum)
    if len(values) != nions:
        raise ValueError('The final magnetization (x) table does not contain exactly one row per atom')
    # Two rounded values can differ by the sum of their half-quantum errors.
    tolerance = max(quanta) + 8 * np.finfo(float).eps * max(1., max(map(abs, values)))
    return values, quanta, tolerance


def _site_positions(text, effective, nions):
    final = list(re.finditer(r'^\s*POSITION\s+TOTAL-FORCE[^\n]*$', text, re.M))
    if final:
        marker, kind = final[-1], 'cartesian'
    else:
        nsw = _effective_parameter(effective, 'NSW', required=False)
        ibrion = _effective_parameter(effective, 'IBRION', required=False)
        if nsw != 0 and ibrion != -1:
            raise ValueError('ISPIN=2 requires a final POSITION/TOTAL-FORCE table, or effective static-run '
                             'parameters with fractional ion positions, to bind the site moments')
        initial = list(re.finditer(r'^\s*position of ions in fractional coordinates[^\n]*$', text, re.M))
        if not initial:
            raise ValueError('ISPIN=2 OUTCAR lacks ion positions needed to bind its site moments to WIN atom order')
        marker, kind = initial[-1], 'fractional'
    coordinates, quanta = [], []
    for line in text[marker.end():].splitlines():
        tokens = line.split()
        if not tokens or set(tokens[0]) == {'-'}:
            if coordinates:
                break
            continue
        if len(tokens) < (6 if kind == 'cartesian' else 3):
            break
        try:
            row = [Decimal(token.replace('D', 'E').replace('d', 'e')) for token in tokens[:3]]
        except InvalidOperation:
            break
        if not all(v.is_finite() for v in row):
            raise ValueError('OUTCAR ion positions contain nonfinite coordinates')
        coordinates.append([float(v) for v in row])
        quanta.append([float(Decimal(10) ** v.as_tuple().exponent) for v in row])
    if (len(coordinates) != nions or not np.isfinite(coordinates).all()
            or not np.isfinite(quanta).all() or np.min(quanta) <= 0):
        raise ValueError('OUTCAR final ion positions are incomplete or have invalid print precision')
    return dict(coordinates=coordinates, kind=kind, print_quanta=quanta,
                source='final_POSITION_TOTAL_FORCE' if final else 'static_fractional_ion_positions')


def _validate_site_order(bundle, positions, context):
    from pymatgen.core import Lattice

    record = context['site_positions']
    lattice = np.asarray(bundle.lattice)
    coordinates = np.asarray(record['coordinates'])
    quanta = np.asarray(record['print_quanta'])
    if record['kind'] == 'cartesian':
        fractional = coordinates @ np.linalg.inv(lattice)
        tolerance = np.linalg.norm(quanta / 2, axis=1)
    else:
        fractional = coordinates
        tolerance = (quanta / 2) @ np.linalg.norm(lattice, axis=1)
    tolerance += 16 * np.finfo(float).eps * max(1., np.linalg.norm(lattice), np.max(abs(coordinates)))
    metric = Lattice(lattice)
    distances = np.array([metric.get_distance_and_image(source, target)[0]
                          for source, target in zip(fractional, positions, strict=True)])
    if np.any(distances > tolerance):
        atom = int(np.argmax(distances - tolerance)) + 1
        raise ValueError(f'OUTCAR ion positions and WIN atom order disagree at atom {atom}; '
                         'preserve the OUTCAR atom order and final structure in WIN before assigning site moments')
    context['site_position_binding'] = dict(max_distance_angstrom=float(distances.max()),
        max_print_precision_tolerance_angstrom=float(tolerance.max()),
        method='Original atom order with exact periodic Cartesian distance; tolerance from printed coordinate precision')


def _outcar_spin_context(text, nions):
    if text.count('Startparameter for this run:') != 1:
        raise ValueError('OUTCAR effective parameter block is missing or not unique')
    effective = text.split('Startparameter for this run:', 1)[1]
    result = {key: _effective_parameter(effective, key, required=key in
              ('ISPIN', 'LNONCOLLINEAR', 'LSORBIT'))
              for key in ('ISTART', 'ISPIN', 'ISYM', 'LNONCOLLINEAR', 'LSORBIT', 'LWAVE')}
    if result['ISPIN'] not in (1, 2):
        raise ValueError('OUTCAR effective ISPIN must be 1 or 2')
    if result['LSORBIT'] != result['LNONCOLLINEAR']:
        raise ValueError('Supported modes require both LSORBIT and LNONCOLLINEAR true for SOC, '
                         'or both false for scalar ISPIN=1/2; noncollinear non-SOC is not implemented')
    spinor = result['LSORBIT']
    initial = _initial_moments(text)
    result.update(spinor=spinor, source_ispin=1 if spinor else result['ISPIN'],
                  input_magnetic_moments=initial,
                  input_magnetic_moments_zero=None if initial is None else all(v == 0 for v in initial),
                  input_magnetic_moments_role='Initial-state metadata, not evidence of the final magnetic state')
    moments = re.findall(r'number of electron[^\n]*magnetization\s+([^\n]+)', text)
    final = None
    if moments:
        try:
            final = [float(v.replace('D', 'E').replace('d', 'e')) for v in moments[-1].split()]
        except ValueError as exc:
            raise ValueError('Final global magnetic moment is malformed') from exc
        if len(final) != (3 if spinor else 1) or not np.isfinite(final).all():
            raise ValueError('Final global magnetic moment has invalid dimensions or nonfinite values')
    result['final_global_magnetization'] = final
    if spinor:
        marker = 'transformation matrix from SAXIS to cartesian coordinates'
        if text.count(marker) != 1:
            raise ValueError('OUTCAR does not provide a unique SAXIS-to-Cartesian transform')
        try:
            rows = text.split(marker, 1)[1].splitlines()[2:5]
            axes = np.array([[float(row.split()[i]) for i in (0, 2, 4)] for row in rows])
        except (ValueError, IndexError) as exc:
            raise ValueError('OUTCAR SAXIS-to-Cartesian transform is malformed') from exc
        if axes.shape != (3, 3) or not np.array_equal(axes, np.eye(3)):
            raise ValueError('Only spinor coordinates with SAXIS aligned to Cartesian axes have been validated')
        if final is None:
            raise ValueError('OUTCAR is missing the final global magnetization')
        if max(map(abs, final)) > 5.1e-8:
            raise ValueError('Final global magnetic moment is nonzero at OUTCAR print precision')
        result.update(saxis_to_cartesian=axes.tolist(), antiunitary_kind='physical_time_reversal',
                      physical_time_reversal_enforced=True,
                      time_reversal_basis='A zero final moment identifies a candidate only; independent wavefunction and PAW MMN antiunitary checks are required')
    else:
        result.update(saxis_to_cartesian=None,
                      antiunitary_kind='channel_complex_conjugation' if result['ISPIN'] == 2 else 'orbital_complex_conjugation',
                      physical_time_reversal_enforced=False,
                      time_reversal_basis='Orbital complex conjugation within one scalar channel; '
                      'this does not impose physical spin-flipping time reversal between ISPIN=2 channels')
        if result['ISPIN'] == 2:
            site, quanta, tolerance = _final_site_moments(text, nions)
            result.update(final_site_magnetic_moments=site, site_moment_print_quanta=quanta,
                          magnetic_candidate_tolerance=tolerance,
                          site_positions=_site_positions(text, effective, nions),
                          magnetic_candidate_basis='Final projected site moments and their printed precision identify '
                          'candidate channel-preserving operations only; numerical sewing and MMN checks remain required')
    return result


def _spacegroup_from_context(bundle, positions, types, context):
    from irrep.spacegroup import SpaceGroup

    cell = (bundle.lattice, positions, types)
    if context['spinor'] or context['source_ispin'] == 1:
        return SpaceGroup.from_cell(cell=cell, spinor=context['spinor'],
                                    magmom=True, include_TR=True, verbosity=0)
    _validate_site_order(bundle, positions, context)
    magnetic = SpaceGroup.from_cell(cell=cell, spinor=False,
                                    magmom=np.asarray(context['final_site_magnetic_moments']),
                                    mag_symprec=context['magnetic_candidate_tolerance'],
                                    include_TR=True, verbosity=0)
    unitary = [op for op in magnetic.symmetries if not op.time_reversal]
    context['candidate_magnetic_group'] = dict(name=magnetic.name, number_str=magnetic.number_str,
        magnetic_operation_count=magnetic.size, retained_spatial_operation_count=len(unitary),
        discarded_channel_exchanging_operation_count=magnetic.size - len(unitary),
        exported_group='Channel-preserving spatial subgroup doubled by orbital complex conjugation')
    return SpaceGroup(Lattice=np.asarray(bundle.lattice), spinor=False,
                      rotations=np.array([op.rotation for op in unitary] * 2),
                      translations=np.array([op.translation for op in unitary] * 2),
                      time_reversals=[False] * len(unitary) + [True] * len(unitary),
                      positions=np.asarray(positions), typat=np.asarray(types),
                      name='channel-preserving spatial subgroup with orbital K', number_str='0')
