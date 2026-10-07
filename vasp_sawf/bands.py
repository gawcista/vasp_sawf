"""Hamiltonian Fourier evaluation and optional DFT path comparison."""

import numpy as np
from numbers import Integral
from pathlib import Path
import re


def _hermiticity_guard_scale(system, matrices):
    return 256*np.finfo(float).eps*max(1, len(system.rvec.iRvec))*max(1.0, float(np.max(abs(matrices))))


def evaluate_hamiltonian(system, kpoints_reduced):
    kpoints = np.asarray(kpoints_reduced, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or not len(kpoints) or not np.isfinite(kpoints).all():
        raise ValueError("k points must be finite, nonempty [NK,3] reduced reciprocal coordinates")
    vectors = np.asarray(system.rvec.iRvec)
    matrices = np.asarray(system.get_R_mat("Ham"))
    if matrices.shape != (len(vectors), system.num_wann, system.num_wann) or not np.isfinite(matrices).all():
        raise ValueError("Invalid real-space Hamiltonian dimensions or values")
    # System_w90 has already applied pair-dependent 1/Ndegen weights.
    phases = np.exp(2j*np.pi*kpoints @ vectors.T)
    return np.einsum("kr,rnm->knm", phases, matrices, optimize=True)


def evaluate_bands(system, kpoints_reduced):
    matrices = evaluate_hamiltonian(system, kpoints_reduced)
    residual = float(np.max(abs(matrices-matrices.conj().transpose(0, 2, 1))))
    numerical_guard_scale = _hermiticity_guard_scale(system, matrices)
    if residual > numerical_guard_scale:
        raise ValueError(f"Hamiltonian Hermiticity exceeds the empirical floating-point guard scale: {residual} > {numerical_guard_scale} eV")
    return np.linalg.eigvalsh(matrices)


def read_dft_eigenval(path, bands_1based, *, spin_channel=1, source_ispin=1):
    bands = list(bands_1based)
    if (not bands or any(isinstance(b, bool) or not isinstance(b, Integral) or b < 1 for b in bands)
            or any(b <= a for a, b in zip(bands, bands[1:]))):
        raise ValueError('Original DFT band indices must be strictly increasing positive integers')
    from pymatgen.io.vasp.outputs import Eigenval
    from pymatgen.electronic_structure.core import Spin

    if (type(source_ispin) is not int or source_ispin not in (1, 2)
            or type(spin_channel) is not int or not 1 <= spin_channel <= source_ispin):
        raise ValueError('Select a valid explicit EIGENVAL spin channel and source ISPIN')

    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 6 or int(lines[0].split()[-1]) != source_ispin:
        raise ValueError('EIGENVAL spin-channel count disagrees with the selected model source ISPIN')
    _, nk, nb = map(int, lines[5].split())
    if nk <= 0 or nb <= 0 or bands[-1] > nb:
        raise ValueError('Invalid DFT dimensions or original target band indices out of range')
    cursor = 6
    for ik in range(nk):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        rows = [line.split() for line in lines[cursor:cursor+nb+1]]
        if len(rows) != nb+1:
            raise ValueError('EIGENVAL is missing a complete k-point block')
        if len(rows[0]) != 4:
            raise ValueError('EIGENVAL k-point records must have 4 columns')
        for ib, row in enumerate(rows[1:], 1):
            if len(row) != (3 if source_ispin == 1 else 5) or int(row[0]) != ib:
                raise ValueError('EIGENVAL band records are missing, reordered, or have incorrect column counts')
        if not all(np.isfinite(float(x)) for row in rows for x in row):
            raise ValueError('EIGENVAL contains nonfinite values')
        cursor += nb+1
    if any(line.strip() for line in lines[cursor:]):
        raise ValueError('EIGENVAL contains extra records')
    parsed = Eigenval(path)
    kpoints = np.array(parsed.kpoints)
    selected_spin = Spin.up if spin_channel == 1 else Spin.down
    if selected_spin not in parsed.eigenvalues:
        raise ValueError('EIGENVAL reader did not return the selected spin channel')
    energies = parsed.eigenvalues[selected_spin][:, np.array(bands) - 1, 0]
    if kpoints.shape != (nk, 3) or energies.shape != (nk, len(bands)):
        raise ValueError('Official reader dimensions disagree with the header')
    return {'kpoints': kpoints, 'eigenvalues_eV': energies,
            'bands_vasp_1based': np.array(bands), 'source_num_bands': nb,
            'source_ispin': source_ispin, 'spin_channel': spin_channel}


def read_win_path(path):
    text = '\n'.join(re.split(r'[!#]', line, maxsplit=1)[0] for line in Path(path).read_text().splitlines())
    blocks = re.findall(r'(?ims)^\s*begin\s+kpoint_path\s*$\n(.*?)^\s*end\s+kpoint_path\s*$', text)
    if len(blocks) != 1:
        raise ValueError('WIN must have exactly one kpoint_path block')
    result = []
    for line in blocks[0].splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError('Each WIN path segment must have two labels and two sets of 3D coordinates')
        start, end = list(map(float, fields[1:4])), list(map(float, fields[5:8]))
        if not np.isfinite(start + end).all():
            raise ValueError('WIN path contains nonfinite coordinates')
        result.append((fields[0], start, fields[4], end))
    if not result:
        raise ValueError('WIN path is empty')
    return result


def path_geometry(kpoints, segments, lattice, *, coordinate_atol=1e-7):
    """Check the actual k-point list against declared segments; tolerance reflects text precision only."""
    k = np.asarray(kpoints, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if (k.ndim != 2 or k.shape[1] != 3 or not len(k) or lattice.shape != (3, 3)
            or not np.isfinite(k).all() or not np.isfinite(lattice).all()
            or not np.isfinite(coordinate_atol) or coordinate_atol <= 0):
        raise ValueError('Invalid path or lattice dimensions or values')
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    distance = np.full(len(k), np.nan)
    labels, ticks, slices = [], [], []
    cursor, offset = 0, 0.
    close = lambda a, b: np.max(abs(a-b)) <= coordinate_atol
    for left, start, right, end in segments:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
            raise ValueError('Invalid path endpoint')
        delta = end - start
        if close(start, end):
            raise ValueError('Path segment has zero length')
        if cursor < len(k) and close(k[cursor], start):
            first = cursor
        elif cursor > 0 and close(k[cursor-1], start):
            first = cursor - 1
        else:
            raise ValueError(f'Actual k-point list does not match path start {left}')
        ends = [j for j in range(first+1, len(k)) if close(k[j], end)]
        if not ends:
            raise ValueError(f'Actual k-point list is missing path end {right}')
        last = ends[0]
        points = k[first:last+1]
        t = (points-start) @ delta / (delta @ delta)
        residual = points - (start + t[:, None] * delta)
        if (np.max(abs(residual)) > coordinate_atol or np.any(np.diff(t) <= 0)
                or t.min() < -coordinate_atol or t.max() > 1 + coordinate_atol):
            raise ValueError('Actual k-point list leaves the path segment or is out of order')
        segment_distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0) @ reciprocal, axis=1))]
        distance[first:last+1] = offset + segment_distance
        if not labels:
            labels.append(left)
            ticks.append(offset)
        elif labels[-1] != left:
            labels[-1] += '|' + left
        offset = float(distance[last])
        labels.append(right)
        ticks.append(offset)
        slices.append((first, last+1))
        cursor = last+1
    if cursor != len(k) or not np.isfinite(distance).all():
        raise ValueError('Path contains unconsumed k points')
    return {'distance_inv_angstrom': distance, 'segment_slices': slices,
            'tick_labels': labels, 'tick_positions_inv_angstrom': ticks,
            'coordinate_atol': coordinate_atol}
