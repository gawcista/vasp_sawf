"""IBZ wavefunction anchors, native MMN transport, and full-group/TR validation."""

from dataclasses import dataclass
import re
import numpy as np
from collections import deque
from importlib.metadata import version
import copy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from .inputs import _win_sections, read_inputs
from .wavecar import inspect_wavecar, read_selected_wavecar


# OUTCAR prints reciprocal coordinates with six decimals; this is not a physics gate.
_PRINT_TOL = 5.1e-7
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?'


class VaspSymmetryError(ValueError):
    pass


def _periodic_delta(values):
    return values - np.rint(values)


@dataclass(frozen=True)
class OutcarKpointMap:
    """Source indices are zero-based; coordinates retain the original OUTCAR folding."""

    ibz_kpoints: np.ndarray
    ibz_weights: np.ndarray
    full_kpoints: np.ndarray
    full_weights: np.ndarray
    source_ibz: np.ndarray
    time_reversal: np.ndarray
    spin_flip: np.ndarray
    operator_table_present: bool
    gauge_status: str = 'unproven'

    def match_interface(self, kpoints):
        """Return the interface-to-OUTCAR permutation and integer reciprocal shifts of interface minus OUTCAR."""
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.shape != self.full_kpoints.shape or not np.isfinite(kpoints).all():
            raise VaspSymmetryError('Invalid interface k-grid shape or values')
        delta = kpoints[:, None, :] - self.full_kpoints[None, :, :]
        matches = np.max(np.abs(_periodic_delta(delta)), axis=2) <= _PRINT_TOL
        if not (np.all(matches.sum(axis=0) == 1) and np.all(matches.sum(axis=1) == 1)):
            raise VaspSymmetryError('Interface and OUTCAR full BZ do not form a unique complete bijection')
        indices = matches.argmax(axis=1)
        shifts = np.rint(kpoints - self.full_kpoints[indices]).astype(int)
        return indices, shifts

    def geometric_candidates(self, reciprocal_rotations):
        """List geometric candidates only; matrices act on column vectors without selecting the actual VASP operation."""
        rotations = np.asarray(reciprocal_rotations)
        if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
                or len(rotations) == 0 or not np.issubdtype(rotations.dtype, np.integer)
                or np.any(np.abs(np.linalg.det(rotations)) != 1)
                or len(np.unique(rotations.reshape(len(rotations), 9), axis=0)) != len(rotations)):
            raise VaspSymmetryError('Candidate operations must be distinct invertible integer lattice matrices')
        rows = []
        for k, source, tr in zip(self.full_kpoints, self.source_ibz, self.time_reversal, strict=True):
            images = np.einsum('rij,j->ri', rotations, self.ibz_kpoints[source])
            if tr:
                images = -images
            # Both source and target were rounded by OUTCAR.
            limits = _PRINT_TOL * (1 + np.abs(rotations).sum(axis=2))
            matches = np.all(np.abs(_periodic_delta(images - k)) <= limits, axis=1)
            indices = tuple(int(i) for i in np.flatnonzero(matches))
            if not indices:
                raise VaspSymmetryError('At least one full-BZ point has no geometric candidate in the supplied operation set')
            rows.append(indices)
        return tuple(rows)


def parse_outcar_kpoint_map(text):
    """Parse a single OUTCAR calculation with positive uniform full-BZ weights; reject ambiguity."""
    ibz_mark = 'Subroutine IBZKPT returns following result:'
    full_mark = 'Subroutine IBZKPT_HF returns following result:'
    if text.count(ibz_mark) != 1 or text.count(full_mark) != 1:
        raise VaspSymmetryError('Exactly one IBZKPT and one IBZKPT_HF source block are required')
    ibz_text, full_text = text.split(ibz_mark, 1)[1].split(full_mark, 1)
    ni = re.search(r'Found\s+(\d+)\s+irreducible k-points:', ibz_text)
    nf = re.search(r'Found\s+(\d+)\s+k-points in 1st BZ', full_text)
    if ni is None or nf is None:
        raise VaspSymmetryError('OUTCAR is missing IBZ/full-BZ point counts')
    ni, nf = int(ni[1]), int(nf[1])
    if not 0 < ni <= nf:
        raise VaspSymmetryError('Invalid OUTCAR k-point count')
    try:
        ibz_body = ibz_text.split('Following reciprocal coordinates:', 1)[1].split('Following cartesian coordinates:', 1)[0]
        full_body = full_text.split('Following reciprocal coordinates:   # in IRBZ', 1)[1]
    except IndexError as exc:
        raise VaspSymmetryError('OUTCAR source coordinate table is missing') from exc
    ibz_rows = re.findall(rf'^\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$', ibz_body, re.M)
    full_rows = re.findall(rf'^\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+(\d+)\s+t-inv ([FT])( spinflp)?\s*$', full_body, re.M)
    if len(ibz_rows) != ni or len(full_rows) != nf:
        raise VaspSymmetryError('OUTCAR coordinate row count or t-inv field disagrees with the declaration')
    ibz = np.array([[float(v.replace('D', 'E').replace('d', 'e')) for v in row] for row in ibz_rows])
    full = np.array([[float(v.replace('D', 'E').replace('d', 'e')) for v in row[:4]] for row in full_rows])
    source = np.array([int(row[4]) - 1 for row in full_rows])
    tr = np.array([row[5] == 'T' for row in full_rows])
    spin_flip = np.array([bool(row[6]) for row in full_rows])
    if (not np.isfinite(ibz).all() or not np.isfinite(full).all()
            or np.any(ibz[:, 3] <= 0) or np.any(full[:, 3] <= 0)
            or np.any(source < 0) or np.any(source >= ni)):
        raise VaspSymmetryError('Invalid OUTCAR source indices, weights, or values')
    multiplicity = np.bincount(source, minlength=ni)
    expected = ibz[:, 3] / ibz[:, 3].sum() * nf
    if not np.allclose(multiplicity, expected, rtol=0, atol=_PRINT_TOL * nf):
        raise VaspSymmetryError('Full-BZ source multiplicities disagree with IBZ weights')
    if not np.allclose(full[:, 3], 1 / nf, rtol=0, atol=5.1e-9):
        raise VaspSymmetryError('The parser currently supports only uniform normalized full-BZ weights')
    if (np.max(np.abs(full[:ni, :3] - ibz[:, :3])) > _PRINT_TOL
            or not np.array_equal(source[:ni], np.arange(ni)) or tr[:ni].any()
            or spin_flip[:ni].any()):
        raise VaspSymmetryError('Full-BZ prefix disagrees with the actual IBZ source')
    delta = _periodic_delta(full[:, None, :3] - full[None, :, :3])
    if np.any((np.max(np.abs(delta), axis=2) <= _PRINT_TOL).sum(axis=1) != 1):
        raise VaspSymmetryError('OUTCAR full BZ contains periodically equivalent duplicates')
    return OutcarKpointMap(ibz[:, :3], ibz[:, 3], full[:, :3], full[:, 3], source,
                          tr, spin_flip, 'Space group operators:' in text)


def build_symmetry_maps(bundle, spacegroup):
    """Check all k-point folding and neighbor G shifts; currently accept only zero-translation operations."""
    operations = spacegroup.symmetries
    if any(np.max(abs(op.translation)) > 1e-12 for op in operations):
        raise ValueError('Nonzero translations have not been validated for MMN anchor adaptation; phases are not silently ignored')
    mesh = bundle.mesh
    address = np.rint(bundle.kpoints*mesh).astype(int) % mesh
    if np.max(abs(bundle.kpoints*mesh-np.rint(bundle.kpoints*mesh))) > 1e-8:
        raise ValueError('The current adapter requires a full Gamma-centered grid')
    lookup = {tuple(a): k for k, a in enumerate(address)}
    if len(lookup) != len(address):
        raise ValueError('Duplicate k-point addresses')
    vectors = bundle.kpoints[bundle.neighbor_indices]+bundle.neighbor_shifts-bundle.kpoints[:, None]
    kmap, emap = [], []
    for op in operations:
        transformed = np.array([op.transform_k(k) for k in bundle.kpoints])
        addresses = np.rint(transformed*mesh).astype(int)
        if np.max(abs(transformed*mesh-addresses)) > 1e-8:
            raise ValueError('Symmetry operation does not preserve the full grid')
        km = np.array([lookup[tuple(a)] for a in addresses % mesh])
        em = np.empty(bundle.neighbor_indices.shape, int)
        for k, row in enumerate(vectors):
            for e, vec in enumerate(row):
                target = op.transform_k(vec)
                matches = np.flatnonzero(np.max(abs(vectors[km[k]]-target), axis=1) < 1e-8)
                if len(matches) != 1:
                    raise ValueError('The symmetry-mapped full MMN neighbor is missing or duplicated')
                em[k,e] = matches[0]
                if bundle.neighbor_indices[km[k],matches[0]] != km[bundle.neighbor_indices[k,e]]:
                    raise ValueError('Neighbor endpoint disagrees with the G-shift mapping')
        kmap.append(km)
        emap.append(em)
    return np.array(kmap), np.array(emap)


def transport_sewing(mmn, neighbors, kmap, edge_map, time_reversals, anchors,
                     *, anchor_k=0, reverse_edges=False):
    """Solve M(gk,gb)d(l)=d(k)M(k,b)^{*a} without unitarizing or averaging inputs.

    The tree only generates candidates; all-edge residuals independently check them, and wavefunction anchors require separate validation.
    """
    mmn = np.asarray(mmn, complex)
    neighbors, kmap, edge_map = map(np.asarray, (neighbors,kmap,edge_map))
    anti = np.asarray(time_reversals, bool)
    anchors = np.asarray(anchors, complex)
    nk, nnb, nb, nb2 = mmn.shape
    ns = len(anti)
    if (nb != nb2 or neighbors.shape != (nk,nnb) or kmap.shape != (ns,nk)
            or edge_map.shape != (ns,nk,nnb) or anchors.shape != (ns,nb,nb)
            or not 0 <= anchor_k < nk):
        raise ValueError('Symmetry transport input dimensions or anchors disagree')
    if not np.all(np.isfinite(mmn)) or not np.all(np.isfinite(anchors)):
        raise ValueError('Input contains nonfinite values')
    for a,upper in ((neighbors,nk),(kmap,nk),(edge_map,nnb)):
        if not np.issubdtype(a.dtype,np.integer) or np.any((a<0)|(a>=upper)):
            raise ValueError('Symmetry mapping indices are out of bounds or noninteger')
    if not np.all(np.sort(kmap,axis=1)==np.arange(nk)):
        raise ValueError('Symmetry k-point mapping is not bijective')
    for s in range(ns):
        if not np.array_equal(neighbors[kmap[s,:,None],edge_map[s]],kmap[s,neighbors]):
            raise ValueError('Incorrect symmetry-mapped neighbor endpoint')
    singular=np.linalg.svd(mmn,compute_uv=False)
    if np.any(singular[...,0] == 0):
        raise ValueError('MMN neighbor matrix is singular; regularization must not hide the failure')
    ratios=singular[...,-1]/singular[...,0]
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= np.finfo(float).eps*nb):
        raise ValueError('MMN neighbor matrix is singular; regularization must not hide the failure')
    queue,seen,tree=deque([anchor_k]),{anchor_k},[]
    edges=list(range(nnb))
    if reverse_edges:
        edges.reverse()
    while queue:
        k=queue.popleft()
        for e in edges:
            target=int(neighbors[k,e])
            if target not in seen:
                seen.add(target)
                queue.append(target)
                tree.append((k,e,target))
    if len(seen) != nk:
        raise ValueError('Native MMN graph is disconnected')
    d=np.empty((ns,nk,nb,nb),complex)
    d[:,anchor_k]=anchors
    for k,e,target in tree:
        original=mmn[k,e]
        rhs=d[:,k] @ np.where(anti[:,None,None],original.conj(),original)
        d[:,target]=np.linalg.solve(mmn[kmap[:,k],edge_map[:,k,e]],rhs)
    errors=[]
    for s in range(ns):
        mg=mmn[kmap[s,:,None],edge_map[s]]
        reconstructed=d[s,:,None].swapaxes(-1,-2).conj() @ mg @ d[s,neighbors]
        errors.append(float(np.max(abs(reconstructed-(mmn.conj() if anti[s] else mmn)))))
    return d, {'mmn_covariance_max':max(errors), 'mmn_covariance_per_operation':errors,
               'unitarity_max':float(np.max(abs(d.swapaxes(-1,-2).conj()@d-np.eye(nb)))),
               'min_edge_singular':float(singular.min()),
               'max_edge_condition':float(1/ratios.min()),'tree_edges':len(tree),
               'all_edges':nk*nnb}


def make_symmetrizer(bundle, spacegroup, d):
    """Use the pinned official NPZ interface for the complete raw representation without recomputing AMN or implicit unitarization."""
    if version('wannierberri') != '1.7.0' or version('irrep') != '2.6.3':
        raise ValueError('Only WannierBerri1.7.0 and IrRep2.6.3 have been validated')
    from irrep.utility import select_irreducible, get_mapping_irr
    from wannierberri.symmetry.sawf import SymmetrizerSAWF
    nk,nb=bundle.eig.shape
    if d.shape != (spacegroup.size,nk,nb,nb):
        raise ValueError('Incorrect full Bloch symmetry matrix dimensions')
    irr=select_irreducible(bundle.kpoints,spacegroup)
    mapping,k2irr,from_sym=get_mapping_irr(bundle.kpoints,irr,spacegroup)
    dic={'D_wann_block_indices':np.zeros((0,2),int),'_NB':nb,'_NK':nk,
         'num_wann':0,'comment':'Same-run WAVECAR anchors and native VASP PAW MMN; complete raw six-dimensional representation',
         'kptirr':irr,'kptirr2kpt':mapping,'kpt2kptirr':k2irr,
         'kpt2kptirr_sym':from_sym,'kpt_from_kptirr_isym':from_sym,
         'NKirr':len(irr),'Nsym':spacegroup.size,
         'time_reversals':np.array([op.time_reversal for op in spacegroup.symmetries]),
         'kpoints_all':bundle.kpoints.copy(),'grid':bundle.mesh.copy(),'eig_irr':bundle.eig[irr].copy()}
    for i,k in enumerate(irr):
        dic[f'd_band_block_indices_{i}']=np.array([[0,nb]])
        dic[f'd_band_blocks_{i}_0']=d[:,k].copy()
    sym=SymmetrizerSAWF().from_dict(dic)
    sym.set_spacegroup(spacegroup)
    sym.sym_product_table,sym.translations_diff,sym.spinor_factors=spacegroup.get_product_table(get_diff=True)
    return sym


def group_residuals(d,kmap,time_reversals,product_table,spinor_factors):
    """Compose zero-translation double-group operations over the full grid, retaining antiunitary conjugation."""
    if not np.isfinite(d).all() or not np.isfinite(spinor_factors).all():
        raise ValueError('Group-composition input contains nonfinite values')
    maximum=0.0
    worst=None
    for g in range(len(d)):
        second=d.conj() if time_reversals[g] else d
        lhs=d[g,kmap] @ second
        rhs=spinor_factors[g,:,None,None,None]*d[product_table[g]]
        difference=abs(lhs-rhs)
        if not np.isfinite(difference).all():
            raise ValueError('Group-composition residual contains nonfinite values')
        err=float(difference.max())
        if err>maximum:
            maximum=err
            worst=[g,*map(int,np.unravel_index(np.argmax(difference),difference.shape))]
    return {'group_composition_max':maximum,'worst_product_h_k_row_col':worst}


_GATE = 1e-6  # WB1.7.0 Symmetrizer_Uirr default; this adapter must not relax it.


def _closed_spinor_dimension(amn):
    shape = np.shape(amn)
    if len(shape) != 3 or shape[0] < 1 or shape[1] != shape[2]:
        raise ValueError('Only complete subspaces with equal Bloch band and Wannier counts are currently supported')
    if shape[1] < 1 or shape[1] % 2:
        raise ValueError('The dimension of an SOC/TR-closed subspace must be a positive even integer')
    return shape[1]


def _validate_grey_group(spacegroup):
    operations = spacegroup.symmetries
    if not operations or any(np.max(abs(op.translation)) > 1e-12 for op in operations):
        raise ValueError('Only complete grey groups containing pure TR and zero spatial translations are currently supported')
    unitary = [tuple(op.rotation.ravel()) for op in operations if not op.time_reversal]
    anti = [tuple(op.rotation.ravel()) for op in operations if op.time_reversal]
    identity = tuple(np.eye(3, dtype=int).ravel())
    if (len(set(unitary)) != len(unitary) or len(set(anti)) != len(anti)
            or set(unitary) != set(anti) or identity not in unitary):
        raise ValueError('A grey group requires unique spatial operations in one-to-one correspondence with their TR partners')
    return next(i for i, op in enumerate(operations)
                if op.time_reversal and tuple(op.rotation.ravel()) == identity)


def _cell_from_win(text, lattice):
    _, blocks = _win_sections(text)
    names = [name for name in ('atoms_cart', 'atoms_frac') if name in blocks]
    if len(names) != 1:
        raise ValueError('WIN must contain exactly one atoms_cart or atoms_frac structure block')
    rows = list(blocks[names[0]])
    unit = 'ang'
    if rows and len(rows[0].split()) == 1:
        unit = rows.pop(0).lower()
    if unit not in ('ang', 'bohr') or (names[0] == 'atoms_frac' and unit != 'ang'):
        raise ValueError('WIN atomic coordinate units have not been validated')
    tokens = [row.split() for row in rows]
    if not tokens or any(len(row) != 4 for row in tokens):
        raise ValueError('Invalid WIN atomic coordinate format')
    species = [row[0] for row in tokens]
    positions = np.array([[float(v) for v in row[1:]] for row in tokens])
    if not np.isfinite(positions).all():
        raise ValueError('WIN atomic coordinates contain nonfinite values')
    if names[0] == 'atoms_cart':
        if unit == 'bohr':
            positions *= 0.529177210544
        positions = positions @ np.linalg.inv(lattice)
    indices = {name: i + 1 for i, name in enumerate(dict.fromkeys(species))}
    return positions, np.array([indices[name] for name in species]), species


def _outcar_spin_context(text, nions):
    if text.count('Startparameter for this run:') != 1:
        raise ValueError('OUTCAR effective parameter block is missing or not unique')
    effective = text.split('Startparameter for this run:', 1)[1]
    result = {}
    for key in ('ISTART', 'ISPIN', 'ISYM', 'LNONCOLLINEAR', 'LSORBIT', 'LWAVE'):
        values = re.findall(rf'^\s*{key}\s*=\s*(\S+)', effective, re.M)
        if len(values) != 1:
            raise ValueError(f'OUTCAR effective {key} is missing or not unique')
        result[key] = values[0] == 'T' if key.startswith('L') else int(values[0])
    if not result['LSORBIT'] or not result['LNONCOLLINEAR'] or not result['LWAVE'] or result['ISPIN'] != 1:
        raise ValueError('Effective SOC parameters must confirm that the final spinor WAVECAR was saved in the same run')
    marker = 'transformation matrix from SAXIS to cartesian coordinates'
    if text.count(marker) != 1:
        raise ValueError('OUTCAR does not provide a unique SAXIS-to-Cartesian transform')
    rows = text.split(marker, 1)[1].splitlines()[2:5]
    axes = np.array([[float(row.split()[i]) for i in (0, 2, 4)] for row in rows])
    if axes.shape != (3, 3) or not np.array_equal(axes, np.eye(3)):
        raise ValueError('Only spinor coordinates with SAXIS aligned to Cartesian axes have been validated')
    moments = re.findall(r'^\s*MAGMOM\s*=\s*(.+)$', text, re.M)
    if len(moments) != 1:
        raise ValueError('OUTCAR input MAGMOM record is missing or not unique')
    expanded = []
    for token in moments[0].split('!')[0].split('#')[0].split():
        count, value = token.split('*') if '*' in token else ('1', token)
        expanded.extend([float(value)] * int(count))
    if len(expanded) != 3 * nions or any(v != 0 for v in expanded):
        raise ValueError('A nonmagnetic candidate with explicitly zero initial magnetic moments is currently required')
    magnetization = re.findall(r'number of electron[^\n]*magnetization\s+([^\n]+)', text)
    if not magnetization:
        raise ValueError('OUTCAR is missing the final global magnetization')
    final_moment = np.array([float(v) for v in magnetization[-1].split()])
    if (final_moment.shape != (3,) or not np.isfinite(final_moment).all()
            or np.max(abs(final_moment)) > 5.1e-8):
        raise ValueError('Final global magnetic moment is nonzero at OUTCAR print precision')
    result.update(saxis_to_cartesian=axes.tolist(), input_magnetic_moments_zero=True,
                  final_global_magnetization=final_moment.tolist(),
                  time_reversal_basis='Zero MAGMOM and output moments only identify a candidate; independent wavefunction and PAW MMN antiunitary checks are required')
    return result


def _independent_transform(point, operation):
    g = point.ig[:, :3]
    inverse = np.rint(np.linalg.inv(operation.rotation)).astype(int)
    sign = -1 if operation.time_reversal else 1
    transformed = (g + point.k) @ inverse * sign - point.k
    target_g = np.rint(transformed).astype(int)
    if np.max(abs(transformed - target_g)) > 1e-8:
        raise ValueError('Little-group operation does not map back to integer G')
    lookup = {tuple(v): i for i, v in enumerate(g)}
    order = np.empty(len(g), int)
    for source, target in enumerate(target_g):
        order[lookup[tuple(target)]] = source
    if len(set(map(tuple, target_g))) != len(g):
        raise ValueError('Little-group G mapping is not a complete bijection')
    coefficients = point.WF.conj() if operation.time_reversal else point.WF
    spin = operation.spinor_rotation
    if operation.time_reversal:
        spin = np.array([[0, 1], [-1, 0]]) @ spin.conj()
    return np.einsum('ts,mgs->mgt', spin, coefficients[:, order])


def _check(residuals, name, value, tolerance=_GATE):
    residuals[name] = float(value)
    if not np.isfinite(value) or value > tolerance:
        raise ValueError(f'{name} failed: {value} > {tolerance}')


def _finite_max(previous, value):
    if not np.isfinite(previous) or not np.isfinite(value):
        raise ValueError('Local residuals contain nonfinite values; max aggregation must not hide them')
    return max(previous, value)


def _check_coefficient_closure(report, value):
    """Apply the recorded dataset-specific decision; other sources retain the original numerical reference."""
    if not np.isfinite(value) or value < 0:
        raise ValueError('Coefficient closure residual must be finite and nonnegative')
    decision = json.loads(Path(__file__).with_name('accepted_closure.json').read_text())
    accepted = (report.get('source_hashes') == decision['source_hashes']
                and value <= decision['accepted_max_relative'])
    report['residuals']['ibz_coefficient_closure_relative_max'] = float(value)
    report['coefficient_closure'] = dict(
        metric='relative_Frobenius_coefficient_reconstruction', raw_value=float(value),
        reference_tolerance=_GATE,
        reference_status='within_reference' if value <= _GATE else 'above_reference',
        reference_basis='The original 1e-6 is only a numerical reference; see ACCEPTANCE.md for dataset-specific physical acceptance')
    report['physical_acceptance_status'] = 'not_assessed'
    if accepted:
        report['physical_acceptance_status'] = 'accepted_for_single_particle_model'
        report['coefficient_closure'].update(acceptance_id=decision['id'],
            acceptance_document=decision['document'], acceptance_scope=decision['scope'],
            accepted_max_relative=decision['accepted_max_relative'])
        return decision
    if value > _GATE:
        raise ValueError(f'Coefficient closure residual {value} is not covered by an existing dataset-specific decision; SrVO3 acceptance cannot be inherited')
    return None


def _runtime_summary(report, started):
    report['wall_seconds'] = time.perf_counter() - started
    report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['runtime_versions'] = {'python': platform.python_version(),
                                 **{name: version(name) for name in ('numpy', 'scipy', 'irrep', 'wannierberri', 'spglib')}}


def _spacegroup_arrays(spacegroup):
    from irrep.spacegroup import SpaceGroup

    values = spacegroup.as_dict()
    arrays = {f'spacegroup_{key}': np.asarray(value) for key, value in values.items()}
    for key, array in arrays.items():
        if array.dtype.hasobject or (np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()):
            raise ValueError(f'Space-group field {key} is nonfinite or cannot be saved without pickle')
    for name in ('rotations', 'translations', 'time_reversals', 'spinor_rotations'):
        arrays[name] = arrays[f'spacegroup_{name}']
    ns = spacegroup.size
    rotations, translations, anti, spin = (arrays[k] for k in
                                           ('rotations', 'translations', 'time_reversals', 'spinor_rotations'))
    if (rotations.shape != (ns, 3, 3) or not np.issubdtype(rotations.dtype, np.integer)
            or translations.shape != (ns, 3) or np.max(abs(translations)) > 1e-12
            or anti.shape != (ns,) or anti.dtype != np.dtype(bool) or not np.any(anti)
            or spin.shape != (ns, 2, 2)
            or np.max(abs(spin.conj().swapaxes(-1, -2) @ spin - np.eye(2))) > 1e-12):
        raise ValueError('Invalid space-group operations, TR flags, or spinor-lift arrays')
    restored = SpaceGroup(**{key: arrays[f'spacegroup_{key}'].item()
                             if arrays[f'spacegroup_{key}'].ndim == 0 else arrays[f'spacegroup_{key}']
                             for key in values})
    for original, again in zip(spacegroup.symmetries, restored.symmetries, strict=True):
        if (original.time_reversal != again.time_reversal
                or any(not np.array_equal(getattr(original, key), getattr(again, key))
                       for key in ('rotation', 'translation', 'spinor_rotation'))):
            raise ValueError('Restoring the space-group dictionary changed operation values, order, or TR flags')
    return arrays


def export_symmetry(seed, wavecar, outcar, output_dir):
    """Export Bloch representations and apply approved coefficient-closure decisions without bypassing other checks."""
    started = time.perf_counter()
    seed, wavecar, outcar = (Path(p).resolve() for p in (seed, wavecar, outcar))
    output = Path(output_dir).resolve()
    protected = {p.parent for p in (seed, wavecar, outcar)}
    protected.update(Path(f'{seed}.{suffix}').resolve().parent for suffix in ('win', 'amn', 'eig', 'mmn'))
    if (output.exists() or Path(output_dir).is_symlink()
            or any(output.is_relative_to(parent) for parent in protected)):
        raise ValueError('Output must be a new directory outside the original input directory; overwriting is forbidden')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='sawf-bridge-bloch-v1', status='not_ready', sawf_ready=False,
                  gauge_status='unproven', residuals={}, numerical_gate=_GATE,
                  numerical_checks_passed=False,
                  gate_source='WannierBerri1.7.0 Symmetrizer_Uirr accuracy_threshold default')
    residuals = report['residuals']
    try:
        from irrep.spacegroup import SpaceGroup
        header = inspect_wavecar(wavecar)
        report['header_preflight_read_bytes'] = 128
        bundle = read_inputs(seed, source_nb=header.num_bands)
        report['source_hashes'] = bundle.hashes
        nb = _closed_spinor_dimension(bundle.amn)
        report.update(source_num_bands=header.num_bands, shape_nk_nb_nw=list(bundle.amn.shape),
                      mesh=bundle.mesh.tolist())
        win = Path(f'{seed}.win').read_text()
        positions, typat, _ = _cell_from_win(win, bundle.lattice)
        outcar_bytes = outcar.read_bytes()
        outcar_text = outcar_bytes.decode()
        report['outcar'] = dict(path=str(outcar), sha256=hashlib.sha256(outcar_bytes).hexdigest(),
                                spin_context=_outcar_spin_context(outcar_text, len(positions)))
        table = parse_outcar_kpoint_map(outcar_text)
        interface_to_outcar, shifts = table.match_interface(bundle.kpoints)
        if np.any(shifts):
            raise ValueError('Wavefunction phases for differing interface and OUTCAR folding have not been validated for this calculation')
        if len(table.ibz_kpoints) != header.num_kpoints:
            raise ValueError('OUTCAR source and WAVECAR disagree on the number of stored IBZ points')
        sg = SpaceGroup.from_cell(cell=(bundle.lattice, positions, typat), spinor=True,
                                  magmom=True, include_TR=True, verbosity=0)
        operations = sg.symmetries
        pure_t = _validate_grey_group(sg)
        report['symmetry_operations'] = dict(total=len(operations),
                                             antiunitary=sum(bool(op.time_reversal) for op in operations))
        kmap, edge_map = build_symmetry_maps(bundle, sg)
        anti = np.array([op.time_reversal for op in operations], dtype=bool)
        stored = read_selected_wavecar(wavecar, bands_1based=bundle.bands_vasp_1based,
                                       kpoints_1based=None, lattice=bundle.lattice)
        report['wavecar'] = asdict(stored.header)
        report['wavecar']['lattice'] = stored.header.lattice.tolist()
        stat = wavecar.stat()
        report['wavecar'].update(path=str(wavecar), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                 whole_file_hash_computed=False)
        report['read_ledger'] = list(stored.read_ledger)
        report['read_bytes'] = sum(row['bytes_returned'] for row in stored.read_ledger)
        report['coefficient_storage_dtype'] = 'complex64'
        report['computation_dtype'] = 'complex128_lossless_copy'
        report['bands_vasp_1based'] = list(stored.bands_1based)
        raw_k = np.array([point.k for point in stored.kpoints])
        if np.max(abs(raw_k - table.ibz_kpoints)) > 5.1e-7:
            raise ValueError('WAVECAR k-point coordinates/order disagree with the actual OUTCAR IBZ source')
        reverse_permutation = np.argsort(interface_to_outcar)
        ibz_interface = reverse_permutation[np.arange(header.num_kpoints)]
        if np.max(abs(raw_k - bundle.kpoints[ibz_interface])) > 1e-11:
            raise ValueError('WAVECAR IBZ coordinates or folding disagree with the actual interface representatives')
        raw_energies = np.array([point.Energy_raw for point in stored.kpoints])
        expected_eig = raw_energies[table.source_ibz[interface_to_outcar]]
        serialized = [line.split()[2] for line in Path(f'{seed}.eig').read_text().splitlines() if line.strip()]
        exponents = {Decimal(token).as_tuple().exponent for token in serialized}
        if len(exponents) != 1 or next(iter(exponents)) > -8:
            raise ValueError('EIG print precision is inconsistent or insufficient; refusing to guess a source energy tolerance')
        eig_tolerance = .5 * 10. ** next(iter(exponents)) + 8 * np.finfo(float).eps * max(1., abs(bundle.eig).max())
        report['eig_serialization_tolerance_ev'] = float(eig_tolerance)
        _check(residuals, 'wavecar_interface_energy_max_ev', np.max(abs(expected_eig-bundle.eig)), eig_tolerance)
        points = []
        for raw in stored.kpoints:
            point = copy.copy(raw)
            point.WF = raw.WF.astype(np.complex128)
            if not np.array_equal(point.WF.astype(np.complex64), raw.WF):
                raise ValueError('Precision conversion of the working copy changed original coefficients')
            points.append(point)
        gamma_rows = np.flatnonzero(np.max(abs(raw_k), axis=1) < 1e-12)
        if len(gamma_rows) != 1:
            raise ValueError('Exactly one stored Gamma anchor is required')
        gamma_ibz = int(gamma_rows[0])
        gamma = int(ibz_interface[gamma_ibz])
        anchors = np.array([points[gamma_ibz].symm_matrix(points[gamma_ibz], op,
                            block_indices=np.array([[0, nb]]), unitary=False)[0] for op in operations])
        d, transport = transport_sewing(bundle.mmn, bundle.neighbor_indices, kmap, edge_map, anti, anchors, anchor_k=gamma)
        report['transport'] = transport
        _check(residuals, 'mmn_covariance_max', transport['mmn_covariance_max'])
        _check(residuals, 'unitarity_max', transport['unitarity_max'])
        backwards, _ = transport_sewing(bundle.mmn, bundle.neighbor_indices, kmap, edge_map, anti, anchors,
                                         anchor_k=gamma, reverse_edges=True)
        _check(residuals, 'reverse_tree_difference_max', np.max(abs(d-backwards)))
        ibz_difference = closure = gamma_lstsq = 0.
        checked = 0
        for point, ik in zip(points, ibz_interface, strict=True):
            wf = point.WF.reshape(nb, -1)
            for isym, operation in enumerate(operations):
                if kmap[isym, ik] != ik:
                    continue
                local = point.symm_matrix(point, operation, block_indices=np.array([[0, nb]]), unitary=False)[0]
                transformed = _independent_transform(point, operation).reshape(nb, -1)
                if not np.isfinite(local).all() or not np.isfinite(transformed).all():
                    raise ValueError('Raw IBZ sewing matrices or independently transformed coefficients contain nonfinite values')
                norm = np.linalg.norm(transformed)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError('Independently transformed IBZ coefficient norms are nonfinite or zero')
                closure = _finite_max(closure, float(np.linalg.norm(local.T @ wf-transformed) / norm))
                ibz_difference = _finite_max(ibz_difference, float(np.max(abs(local-d[isym, ik]))))
                if ik == gamma:
                    direct = np.linalg.lstsq(wf.T, transformed.T, rcond=None)[0]
                    gamma_lstsq = _finite_max(gamma_lstsq, float(np.max(abs(local-direct))))
                checked += 1
        report['independent_ibz_little_checks'] = checked
        _check(residuals, 'ibz_little_difference_max', ibz_difference)
        _check_coefficient_closure(report, closure)
        _check(residuals, 'gamma_independent_lstsq_difference_max', gamma_lstsq)
        energy_covariance = np.max(abs(bundle.eig[kmap, :, None] * d - d * bundle.eig[None, :, None, :]))
        _check(residuals, 'energy_covariance_max_ev', energy_covariance)
        sym = make_symmetrizer(bundle, sg, d)
        product = group_residuals(d, kmap, anti, sym.sym_product_table, sym.spinor_factors)
        report['group'] = product
        _check(residuals, 'group_composition_max', product['group_composition_max'])
        t = pure_t
        _check(residuals, 'time_reversal_squared_plus_identity_max', np.max(abs(d[t, kmap[t]] @ d[t].conj()+np.eye(nb))))
        arrays = dict(d=d, kmap=kmap, kpoints=bundle.kpoints, eig=bundle.eig,
                      bands_vasp_1based=bundle.bands_vasp_1based, **_spacegroup_arrays(sg))
        for suffix, expected in bundle.hashes.items():
            if hashlib.sha256(Path(f'{seed}.{suffix}').read_bytes()).hexdigest() != expected:
                raise ValueError('Original interface files changed during export')
        np.savez_compressed(output / 'bloch.npz', **arrays)
        package = (output / 'bloch.npz').read_bytes()
        report['bloch_sha256'] = hashlib.sha256(package).hexdigest()
        report['bloch_bytes'] = len(package)
        report.update(status='ready', sawf_ready=True,
                      numerical_checks_passed=True,
                      numerical_checks_scope='All provenance/representation/PAW MMN numerical checks; coefficient F-closure reference results are reported separately',
                      gauge_status='verified_mmn_anchor_transport',
                      full_shape=list(d.shape), output='bloch.npz', source_status='same_run_numerically_cross_checked',
                      excluded_bands_after_compaction=[], no_polar_projection=True,
                      limitation='Uses input dimensions; requires a closed square subspace, Gamma-centered grid, zero spatial translations, and Cartesian spinor axes. Real-material regression covers SrVO3 only; the target representation is checked separately')
    except Exception as error:
        report['error'] = str(error)
        if (output / 'bloch.npz').exists():
            (output / 'bloch.npz').unlink()
        _runtime_summary(report, started)
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        raise
    _runtime_summary(report, started)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report
