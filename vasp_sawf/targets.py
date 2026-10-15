"""Infer general-position local targets from ordinary Wannier centers."""

from numbers import Real
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Lattice


def read_wout_centers(path, expected_num_wann):
    """Read the last complete Wannier90 Final State, in unwrapped Angstroms."""
    if (isinstance(expected_num_wann, (bool, np.bool_))
            or not isinstance(expected_num_wann, (int, np.integer)) or expected_num_wann < 1):
        raise ValueError('The expected number of Wannier centers must be a positive integer')
    text = Path(path).read_text()
    sections = list(re.finditer(r'^\s*Final State\s*$', text, re.M))
    if not sections:
        raise ValueError(f'{path}: no Wannier90 Final State centers were found')
    section = text[sections[-1].end():].split('Sum of centres and spreads', 1)[0]
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
    expression = (r'^\s*WF centre and spread\s+(\d+)\s*\(\s*('
                  + number + r')\s*,\s*(' + number + r')\s*,\s*('
                  + number + r')\s*\)')
    records = re.findall(expression, section, re.M)
    indices = [int(record[0]) for record in records]
    if len(indices) != expected_num_wann or sorted(indices) != list(range(1, expected_num_wann + 1)):
        raise ValueError(f'{path}: final center indices must cover 1..{expected_num_wann} exactly once')
    centers = np.empty((expected_num_wann, 3))
    for index, *coordinates in records:
        centers[int(index) - 1] = [float(v.replace('D', 'E').replace('d', 'e')) for v in coordinates]
    if not np.isfinite(centers).all():
        raise ValueError(f'{path}: Final State centers contain nonfinite values')
    return centers


def _site_centers(fractional, metric, spinor, tolerance):
    distances = metric.get_all_distances(fractional, fractional)
    multiplicity = 2 if spinor else 1
    if len(fractional) % multiplicity:
        raise ValueError('Automatic local Kramers pairs require an even number of Wannier centers')
    neighbors = [np.flatnonzero(row <= tolerance) for row in distances]
    for i, candidates in enumerate(neighbors):
        if len(candidates) > multiplicity:
            raise ValueError(f'Ambiguous local center multiplicity at column {i}: {len(candidates)} '
                             f'centers lie within {tolerance:g} Angstrom; provide an explicit target')
        if len(candidates) < multiplicity:
            others = np.delete(distances[i], i)
            closest = float(others.min()) if len(others) else float('inf')
            raise ValueError(f'No local Kramers pair for column {i}; closest other center is '
                             f'{closest:.9g} Angstrom, tolerance {tolerance:g} Angstrom')
    groups, sites, used = [], [], set()
    for i, candidates in enumerate(neighbors):
        if i in used:
            continue
        group = candidates.tolist()
        if any(set(neighbors[j]) != set(group) for j in group):
            raise ValueError('Ambiguous non-disjoint local center pairing; provide an explicit target')
        images = [metric.get_distance_and_image(fractional[i], fractional[j])[1] for j in group]
        sites.append(np.mean(fractional[group] + images, axis=0) % 1)
        groups.append(group)
        used.update(group)
    return np.asarray(sites), groups


def _site_action(sites, operations, metric, tolerance):
    count = len(sites)
    permutations = np.empty((len(operations), count), dtype=int)
    shifts = np.empty((len(operations), count, 3), dtype=int)
    worst = 0.
    for g, operation in enumerate(operations):
        transformed = sites @ operation.rotation.T + operation.translation
        distances = metric.get_all_distances(transformed, sites)
        for i, row in enumerate(distances):
            candidates = np.flatnonzero(row <= tolerance)
            if not len(candidates):
                raise ValueError(f'No symmetry-related center for site {i}, unitary operation {g}; '
                                 f'closest distance is {row.min():.9g} Angstrom, '
                                 f'tolerance {tolerance:g} Angstrom')
            if len(candidates) != 1:
                raise ValueError(f'Ambiguous symmetry-related centers for site {i}, operation {g}; '
                                 'provide an explicit target or resolve the center multiplicity')
            j = int(candidates[0])
            permutations[g, i] = j
            distance, image = metric.get_distance_and_image(transformed[i], sites[j])
            shifts[g, i] = image
            worst = max(worst, float(distance))
        if len(set(permutations[g])) != count:
            raise ValueError('Symmetry-related center matching is not a permutation')
    return permutations, shifts, worst


def _check_site_group(operations, permutations, shifts):
    """Keep the lattice-translation cocycle when validating permutation products."""
    identity = np.eye(3, dtype=int)
    for g, left in enumerate(operations):
        for h, right in enumerate(operations):
            rotation = left.rotation @ right.rotation
            translation = left.rotation @ right.translation + left.translation
            matches = []
            for p, product in enumerate(operations):
                difference = translation - product.translation
                if (np.array_equal(rotation, product.rotation)
                        and np.max(abs(difference - np.rint(difference))) < 1e-10):
                    matches.append((p, np.rint(difference).astype(int)))
            if len(matches) != 1:
                raise ValueError('Unitary Seitz operations do not form a unique closed spatial group')
            p, difference = matches[0]
            if not np.array_equal(permutations[g, permutations[h]], permutations[p]):
                raise ValueError('Center permutations violate the spatial group product')
            product_shifts = shifts[h] @ left.rotation.T + shifts[g, permutations[h]]
            if not np.array_equal(product_shifts, shifts[p] + difference):
                raise ValueError('Center cell shifts violate the Seitz group product')
    identities = [g for g, op in enumerate(operations)
                  if np.array_equal(op.rotation, identity)
                  and np.max(abs(op.translation - np.rint(op.translation))) < 1e-10]
    if len(identities) != 1:
        raise ValueError('The spatial group must contain exactly one identity modulo lattice translations')
    for i in range(permutations.shape[1]):
        stabilizer = np.flatnonzero(permutations[:, i] == i)
        if len(stabilizer) != 1 or stabilizer[0] != identities[0]:
            raise ValueError(f'Nontrivial site symmetry at center {i}: unitary operations '
                             f'{stabilizer.tolist()}; an explicit local representation is required '
                             'because center coordinates alone cannot determine it')


def infer_target(centers_cart, lattice, spacegroup, *, center_tol=1e-2):
    """Return representatives, internal WB labels, and the geometric inference ledger.

    Automatic targets cover general positions with one local scalar state or one
    Kramers doublet. The internal WB ``s`` constructs that representation only;
    it is not an atomic-orbital assignment or a replacement for SCDM AMN.
    ``center_tol`` is a Cartesian matching distance, not a symmetry-error budget.
    """
    if (isinstance(center_tol, (bool, np.bool_)) or not isinstance(center_tol, Real)
            or not np.isfinite(center_tol) or center_tol <= 0):
        raise ValueError('Center matching tolerance must be positive and finite, in Angstrom')
    centers = np.asarray(centers_cart, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if (centers.ndim != 2 or centers.shape[1:] != (3,) or not len(centers)
            or not np.isfinite(centers).all() or lattice.shape != (3, 3)
            or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) <= np.finfo(float).tiny):
        raise ValueError('Automatic targets require finite [NW,3] centers and a nonsingular [3,3] lattice')
    spinor = bool(spacegroup.spinor)
    operations_all = list(spacegroup.symmetries)
    if spinor or any(op.time_reversal for op in operations_all):
        from .symmetry import _validate_grey_group
        _validate_grey_group(spacegroup)
    indices = [g for g, op in enumerate(operations_all) if not op.time_reversal]
    operations = [operations_all[g] for g in indices]
    if not operations:
        raise ValueError('Automatic targets require a nonempty spatial group')
    metric = Lattice(lattice)
    fractional = centers @ np.linalg.inv(lattice)
    sites, groups = _site_centers(fractional, metric, spinor, center_tol)
    permutations, shifts, orbit_error = _site_action(sites, operations, metric, center_tol)
    _check_site_group(operations, permutations, shifts)

    # Reynolds averaging acts on unwrapped affine coordinates, including every cell shift.
    projected = np.mean([((sites[permutations[g]] + shifts[g] - op.translation)
                         @ np.linalg.inv(op.rotation).T)
                        for g, op in enumerate(operations)], axis=0) % 1
    exact_shifts = np.empty_like(shifts)
    exact_error = 0.
    for g, op in enumerate(operations):
        difference = projected @ op.rotation.T + op.translation - projected[permutations[g]]
        exact_shifts[g] = np.rint(difference).astype(int)
        exact_error = max(exact_error, float(np.linalg.norm(
            (difference - exact_shifts[g]) @ lattice, axis=1).max()))
    if exact_error > 1e-9:
        raise ValueError(f'Projected center orbit failed affine symmetry: {exact_error:g} Angstrom')
    _check_site_group(operations, permutations, exact_shifts)

    input_images = np.empty((len(centers), 3), dtype=int)
    adjustment = np.empty(len(centers))
    for site, columns in enumerate(groups):
        for column in columns:
            adjustment[column], input_images[column] = metric.get_distance_and_image(
                projected[site], fractional[column])
    if adjustment.max() > center_tol:
        raise ValueError(f'Symmetry-consistent centers require an adjustment of {adjustment.max():.9g} '
                         f'Angstrom, exceeding center tolerance {center_tol:g} Angstrom')

    key = lambda i: tuple(np.round(projected[i], 12))
    remaining, orbits = set(range(len(sites))), []
    while remaining:
        representative = min(remaining, key=key)
        orbit = sorted({int(i) for i in permutations[:, representative]}, key=key)
        if not set(orbit) <= remaining or len(orbit) != len(operations):
            raise ValueError('Center orbits overlap or have a nontrivial site stabilizer')
        remaining.difference_update(orbit)
        orbits.append((representative, orbit))
    representatives = np.array([projected[representative] for representative, _ in orbits])
    representation = 'local_kramers_doublet' if spinor else 'local_scalar_state'
    metadata = dict(
        method='metric center pairing and complete Seitz orbit projection',
        local_representation=representation, atomic_orbital_assignment=False,
        internal_wannierberri_orbital='s', num_wann=len(centers),
        center_tolerance_angstrom=float(center_tol),
        tolerance_role='Geometric pairing and orbit matching only; no physical residual is accepted here',
        input_centers_angstrom=centers.tolist(),
        site_input_columns_0based=groups,
        unprojected_site_centers_fractional=sites.tolist(),
        projected_site_centers_fractional=projected.tolist(),
        input_center_cell_shifts=input_images.tolist(),
        input_center_adjustments_angstrom=adjustment.tolist(),
        max_center_adjustment_angstrom=float(adjustment.max()),
        max_orbit_mismatch_angstrom=orbit_error, projected_covariance_angstrom=exact_error,
        unitary_operation_indices_0based=indices, site_permutations=permutations.tolist(),
        site_cell_shifts=exact_shifts.tolist(),
        orbit_site_counts=[len(orbit) for _, orbit in orbits],
        orbits=[dict(position_fractional=projected[representative].tolist(),
                     site_indices_0based=orbit, local_representation=representation)
                for representative, orbit in orbits],
        compatibility='Center geometry supplies a candidate; full band/target compatibility is checked by SAWF',
    )
    return representatives, ['s'] * len(orbits), metadata
