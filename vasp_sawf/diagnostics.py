"""Measured small-matrix diagnostics without acceptance decisions or matrix repair."""

import numpy as np
from numbers import Integral


def mmn_invariant_report(bundle, kmap, edge_map):
    """Compare mapped-edge singular values, invariant under unitary endpoint gauges.

    Weyl's inequality and ||R||_2 <= ||R||_F <= NB*max(abs(R)) give
    max(abs(R)) >= max(abs(delta(sigma)))/NB for exactly unitary sewing.
    This necessary bound does not establish the existence of a valid sewing matrix.
    """
    mmn = np.asarray(bundle.mmn)
    neighbors = np.asarray(bundle.neighbor_indices)
    shifts = np.asarray(bundle.neighbor_shifts)
    kmap, edge_map = map(np.asarray, (kmap, edge_map))
    energies = np.asarray(bundle.eig)
    if mmn.ndim != 4 or not np.isfinite(mmn).all():
        raise ValueError('MMN diagnostics require finite four-dimensional matrices')
    nk, nnb, nb, nb2 = mmn.shape
    if (min(nk, nnb, nb) < 1 or nb != nb2 or neighbors.shape != (nk, nnb)
            or shifts.shape != (nk, nnb, 3) or kmap.ndim != 2
            or kmap.shape[1] != nk or len(kmap) == 0
            or edge_map.shape != (len(kmap), nk, nnb)
            or energies.shape != (nk, nb) or not np.isfinite(energies).all()):
        raise ValueError('MMN diagnostic dimensions disagree')
    for array, upper in ((neighbors, nk), (kmap, nk), (edge_map, nnb)):
        if (not np.issubdtype(array.dtype, np.integer)
                or np.any((array < 0) | (array >= upper))):
            raise ValueError('MMN diagnostic mappings must have valid integer indices')
    if not np.issubdtype(shifts.dtype, np.integer):
        raise ValueError('MMN neighbor translations must be integer vectors')
    if not np.all(np.sort(kmap, axis=1) == np.arange(nk)):
        raise ValueError('MMN diagnostic k-point mapping must be bijective')
    singular = np.linalg.svd(mmn, compute_uv=False)
    energies = np.sort(energies, axis=1)
    rows = []
    for operation_index, (km, em) in enumerate(zip(kmap, edge_map, strict=True)):
        if not np.array_equal(neighbors[km[:, None], em], km[neighbors]):
            raise ValueError('MMN symmetry-mapped neighbor endpoint disagrees')
        delta = abs(singular[km[:, None], em] - singular)
        k, edge, band = map(int, np.unravel_index(np.argmax(delta), delta.shape))
        mapped_k, mapped_edge = int(km[k]), int(em[k, edge])
        maximum = float(delta[k, edge, band])
        rows.append({
            'operation_index': operation_index,
            'singular_value_difference_max': maximum,
            'unitary_covariance_entrywise_lower_bound': maximum/nb,
            'energy_spectrum_difference_max_ev': float(abs(energies[km] - energies).max()),
            'per_k_singular_value_difference_max': delta.max(axis=(1, 2)).tolist(),
            'worst_edge': {
                'source_k_1based': k + 1,
                'source_neighbor_k_1based': int(neighbors[k, edge]) + 1,
                'source_neighbor_index_0based': edge,
                'source_G': shifts[k, edge].tolist(),
                'mapped_source_k_1based': mapped_k + 1,
                'mapped_neighbor_k_1based': int(neighbors[mapped_k, mapped_edge]) + 1,
                'mapped_neighbor_index_0based': mapped_edge,
                'mapped_G': shifts[mapped_k, mapped_edge].tolist(),
                'singular_value_index_0based': band,
            },
        })
    maximum = max(row['singular_value_difference_max'] for row in rows)
    return {
        'kpoints_checked': nk,
        'edges_checked_per_operation': nk*nnb,
        'num_bands': nb,
        'singular_value_difference_max': maximum,
        'unitary_covariance_entrywise_lower_bound': maximum/nb,
        'bound_scope': 'Necessary bound for exactly unitary endpoint sewing, not an acceptance test.',
        'per_operation': rows,
    }


def little_group_report(matrices, operation_indices, kpoint, spacegroup):
    """Check a closed set of stabilizers using complete Seitz and antiunitary products."""
    matrices = np.asarray(matrices, dtype=complex)
    indices = np.asarray(list(operation_indices))
    kpoint = np.asarray(kpoint, dtype=float)
    operations = spacegroup.symmetries
    if (indices.ndim != 1 or len(indices) == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any((indices < 0) | (indices >= len(operations)))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Little-group operation indices must be distinct valid integers')
    if (matrices.ndim != 3 or matrices.shape[0] != len(indices)
            or matrices.shape[1] < 1 or matrices.shape[1] != matrices.shape[2]
            or not np.isfinite(matrices).all()
            or kpoint.shape != (3,) or not np.isfinite(kpoint).all()):
        raise ValueError('Little-group matrices and k point have invalid dimensions or values')
    for index in indices:
        delta = operations[index].transform_k(kpoint) - kpoint
        if np.max(abs(delta - np.rint(delta))) > 1e-8:
            raise ValueError('Requested operation does not stabilize this k point modulo reciprocal vectors')
    product, translations, factors = map(np.asarray, spacegroup.get_product_table(get_diff=True))
    ns = len(operations)
    if (product.shape != (ns, ns) or translations.shape != (ns, ns, 3)
            or factors.shape != (ns, ns) or not np.isfinite(translations).all()
            or not np.isfinite(factors).all()
            or np.max(abs(translations - np.rint(translations))) > 1e-12):
        raise ValueError('Invalid Seitz product table, lattice translations, or spin factors')
    local = {int(index): j for j, index in enumerate(indices)}
    if any(int(product[g, h]) not in local for g in indices for h in indices):
        raise ValueError('Selected little-group operations are not closed under products')
    identity = np.eye(matrices.shape[1])
    unitarity = np.max(abs(matrices.conj().swapaxes(1, 2) @ matrices - identity), axis=(1, 2))
    maximum, worst = 0., None
    for j, g in enumerate(indices):
        for l, h in enumerate(indices):
            p = int(product[g, h])
            # The product maps k to k modulo integers; lattice translations make the folding immaterial.
            phase = np.exp(-2j*np.pi*np.dot(kpoint, translations[g, h]))
            lhs = matrices[j] @ (matrices[l].conj() if operations[g].time_reversal else matrices[l])
            rhs = phase*factors[g, h]*matrices[local[p]]
            difference = abs(lhs - rhs)
            error = float(difference.max())
            if error > maximum:
                maximum = error
                row, column = map(int, np.unravel_index(np.argmax(difference), difference.shape))
                worst = {'left_operation_index': int(g), 'right_operation_index': int(h),
                         'product_operation_index': p, 'row': row, 'column': column}
    pure_tr = [j for j, index in enumerate(indices)
               if (operations[index].time_reversal
                   and np.array_equal(operations[index].rotation, np.eye(3, dtype=int))
                   and np.max(abs(operations[index].translation)) < 1e-12)]
    sign = -1 if spacegroup.spinor else 1
    tr_square = max((float(abs(matrices[j] @ matrices[j].conj() - sign*identity).max())
                     for j in pure_tr), default=None)
    return {
        'operation_indices': indices.tolist(),
        'kpoint_fractional': kpoint.tolist(),
        'group_composition_max': maximum,
        'worst_product': worst,
        'unitarity_max': float(unitarity.max()),
        'unitarity_per_operation': unitarity.tolist(),
        'tr_operation_indices': [int(indices[j]) for j in pure_tr],
        'tr_square_max': tr_square,
        'tr_square_expected_sign': sign if pure_tr else None,
    }


def direct_mmn_covariance_report(bundle, kmap, edge_map, time_reversals,
                                 edge_phases, local_matrices):
    """Check native MMN edges with independently measured matrices at both endpoints.

    Only little-group operations with a measured matrix at both endpoints are
    included. No MMN transport, matrix projection, or basis alignment is performed.
    """
    mmn = np.asarray(bundle.mmn)
    neighbors, shifts = map(np.asarray, (bundle.neighbor_indices, bundle.neighbor_shifts))
    kmap, edge_map = map(np.asarray, (kmap, edge_map))
    anti = np.asarray(time_reversals, dtype=bool)
    phases = np.asarray(edge_phases, dtype=complex)
    if mmn.ndim != 4 or not np.isfinite(mmn).all():
        raise ValueError('Direct MMN diagnostics require finite four-dimensional matrices')
    nk, nnb, nb, nb2 = mmn.shape
    ns = len(anti)
    if (min(nk, nnb, nb, ns) < 1 or nb != nb2 or anti.shape != (ns,)
            or neighbors.shape != (nk, nnb) or shifts.shape != (nk, nnb, 3)
            or kmap.shape != (ns, nk) or edge_map.shape != (ns, nk, nnb)
            or phases.shape != (ns, nk, nnb) or not np.isfinite(phases).all()
            or np.max(abs(abs(phases) - 1)) > 1e-12):
        raise ValueError('Direct MMN diagnostic dimensions or Seitz phases disagree')
    for array, upper in ((neighbors, nk), (kmap, nk), (edge_map, nnb)):
        if (not np.issubdtype(array.dtype, np.integer)
                or np.any((array < 0) | (array >= upper))):
            raise ValueError('Direct MMN mappings must have valid integer indices')
    if (not np.issubdtype(shifts.dtype, np.integer)
            or not np.all(np.sort(kmap, axis=1) == np.arange(nk))):
        raise ValueError('Direct MMN shifts or k-point permutation are invalid')
    for g in range(ns):
        if not np.array_equal(neighbors[kmap[g, :, None], edge_map[g]], kmap[g, neighbors]):
            raise ValueError('Direct MMN symmetry-mapped endpoint disagrees')
    checked = {}
    for k, values in local_matrices.items():
        if not isinstance(k, Integral) or not 0 <= k < nk:
            raise ValueError('Direct MMN matrix k index is invalid')
        checked[k] = {}
        for g, value in values.items():
            if not isinstance(g, Integral) or not 0 <= g < ns:
                raise ValueError('Direct MMN matrix operation index is invalid')
            if kmap[g, k] != k:
                raise ValueError('Direct local matrix must belong to a stabilizing operation')
            value = np.asarray(value, dtype=complex)
            if value.shape != (nb, nb) or not np.isfinite(value).all():
                raise ValueError('Direct MMN endpoint matrix dimensions or values are invalid')
            checked[k][g] = value
    rows = []
    for g in range(ns):
        count, maximum, worst = 0, None, None
        for k, values in checked.items():
            if g not in values:
                continue
            for edge, neighbor in enumerate(neighbors[k]):
                if g not in checked.get(int(neighbor), {}):
                    continue
                mapped_k, mapped_edge = int(kmap[g, k]), int(edge_map[g, k, edge])
                lhs = values[g].conj().T @ mmn[mapped_k, mapped_edge] @ checked[int(neighbor)][g]
                rhs = phases[g, k, edge] * (mmn[k, edge].conj() if anti[g] else mmn[k, edge])
                difference = abs(lhs - rhs)
                error = float(difference.max())
                count += 1
                if maximum is None or error > maximum:
                    maximum = error
                    row, column = map(int, np.unravel_index(np.argmax(difference), difference.shape))
                    worst = {'source_k_1based': int(k) + 1,
                             'source_neighbor_k_1based': int(neighbor) + 1,
                             'source_neighbor_index_0based': edge,
                             'source_G': shifts[k, edge].tolist(),
                             'mapped_source_k_1based': mapped_k + 1,
                             'mapped_neighbor_index_0based': mapped_edge,
                             'mapped_G': shifts[mapped_k, mapped_edge].tolist(),
                             'row': row, 'column': column}
        rows.append({'operation_index': g, 'edges_checked': count,
                     'mmn_covariance_max': maximum, 'worst_edge': worst})
    return {
        'edges_checked': sum(row['edges_checked'] for row in rows),
        'mmn_covariance_max': max((row['mmn_covariance_max'] for row in rows
                                   if row['mmn_covariance_max'] is not None), default=None),
        'basis_assumption': 'Endpoint wavefunctions and native MMN must use the same Bloch basis; this check does not establish that independently.',
        'per_operation': rows,
    }
