"""Finite Fourier symmetry residual bounds for an unaveraged interpolant."""

import numpy as np


def _integers(value, name):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iuf' or not np.isfinite(array).all()
            or np.any(array != np.rint(array)) or np.any(np.abs(array.astype(object)) >= 2**52)):
        raise ValueError(f'{name} must contain finite, exactly representable integers')
    return array.astype(np.int64)


def _fourier_arrays(vectors, matrices):
    vectors = _integers(vectors, 'Fourier vectors')
    matrices = np.asarray(matrices, dtype=complex)
    if (vectors.ndim != 2 or vectors.shape[1] != 3 or not len(vectors)
            or matrices.ndim != 3 or matrices.shape[0] != len(vectors)
            or matrices.shape[1] != matrices.shape[2] or not matrices.shape[1]
            or not np.isfinite(matrices).all()):
        raise ValueError('Fourier arrays must have shapes (n,3), (n,w,w) and finite values')
    return vectors, matrices


def covariance_coefficients(R, H_R, Q, anti, translations, matrices):
    """Expand H(gk)-D(k) H(k)^(*) D(k)^dagger, without changing H_R.

    Conventions: H(k)=sum_R exp(2*pi*i*k.R) H_R, gk=k@Q, and
    D(k)=sum_t exp(2*pi*i*(k@Q).t) matrices[t]. Q includes the TR sign.
    Integer frequencies are coalesced exactly; no small terms are filtered.
    The caller must independently establish the physical target representation.
    """
    R, H_R = _fourier_arrays(R, H_R)
    translations, matrices = _fourier_arrays(translations, matrices)
    Q = _integers(Q, 'Q')
    if Q.shape != (3, 3) or matrices.shape[1:] != H_R.shape[1:]:
        raise ValueError('Q must be (3,3) and target and Hamiltonian dimensions must agree')
    if not isinstance(anti, (bool, np.bool_)):
        raise ValueError('anti must be boolean')
    q = Q.astype(object)
    determinant = sum(q[0, j]*(q[1, (j+1) % 3]*q[2, (j+2) % 3]
                              -q[1, (j+2) % 3]*q[2, (j+1) % 3]) for j in range(3))
    if abs(determinant) != 1:
        raise ValueError('Q must be integer unimodular')
    terms = {}

    def add(frequency, coefficient):
        key = tuple(int(x) for x in frequency)
        if any(abs(x) >= 2**52 for x in key):
            raise ValueError('Expanded Fourier frequency exceeds exact integer range')
        if key not in terms:
            terms[key] = np.zeros(H_R.shape[1:], dtype=complex)
        terms[key] += coefficient

    # Python integer arithmetic prevents overflow in frequency assembly.
    for r, h in zip(R.astype(object), H_R, strict=True):
        add(q @ r, h)
        for ti, pi in zip(translations.astype(object), matrices, strict=True):
            for tj, pj in zip(translations.astype(object), matrices, strict=True):
                add((-r if anti else r)+q @ (ti-tj),
                    -pi @ (h.conj() if anti else h) @ pj.conj().T)
    keys = sorted(terms)
    coefficients = np.array([terms[key] for key in keys])
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite expanded Fourier coefficients')
    return np.array(keys, dtype=np.int64), coefficients


def certify_covariance(frequencies, coefficients, *, active_axes, grid_side=161,
                       max_points=300000, chunk_size=2048):
    """Bound the spectral norm over a periodic coordinate plane (others zero).

    In exact arithmetic, L_a=2*pi*sum_n |n_a| ||delta_n||_2 bounds each
    derivative. Every point is within 1/(2*grid_side) of the periodic grid
    on each active axis, giving grid_max+sum_active L_a/(2*grid_side).
    The triangle bound is independent of the grid. Both concern only this
    finite Fourier interpolant, never the unsampled DFT Hamiltonian.

    Computation uses ordinary float64 arithmetic, not interval arithmetic.
    A separately reported heuristic roundoff allowance is added to the smaller
    bound; it is not a rigorous floating-point error certificate.
    """
    n, delta = _fourier_arrays(frequencies, coefficients)
    axes = np.asarray(active_axes)
    if (axes.ndim != 1 or (axes.size and axes.dtype.kind not in 'iu')
            or len(set(axes.tolist())) != len(axes) or np.any((axes < 0) | (axes > 2))):
        raise ValueError('active_axes must contain unique coordinate indices 0,1,2')
    axes = axes.astype(int)
    for name, value, limit in (('grid_side', grid_side, 300000), ('max_points', max_points, 300000),
                                ('chunk_size', chunk_size, 2048)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= limit:
            raise ValueError(f'{name} must be a positive integer at most {limit}')
    total = int(grid_side)**len(axes)
    if total > max_points:
        raise ValueError('Requested grid exceeds bounded max_points')
    norms = np.linalg.norm(delta, ord=2, axis=(-2, -1))
    triangle = float(norms.sum())
    derivative = 2*np.pi*(abs(n).T @ norms)
    maximum, worst = -1.0, np.zeros(3)
    for start in range(0, total, chunk_size):
        indices = np.arange(start, min(start+chunk_size, total))
        k = np.zeros((len(indices), 3))
        if len(axes):
            k[:, axes] = np.array(np.unravel_index(indices, (grid_side,)*len(axes))).T/grid_side-.5
        phase = np.exp(2j*np.pi*(k @ n.T))
        residual = np.einsum('kn,nij->kij', phase, delta, optimize=True)
        spectral = np.linalg.norm(residual, ord=2, axis=(-2, -1))
        index = int(spectral.argmax())
        if spectral[index] > maximum:
            maximum, worst = float(spectral[index]), k[index].copy()
    grid_bound = maximum+float(derivative[axes].sum())/(2*grid_side)
    allowance = 256*np.finfo(float).eps*len(n)*max(1.0, triangle)
    upper = min(triangle, grid_bound)+allowance
    if not np.isfinite([triangle, *derivative, maximum, grid_bound, allowance, upper]).all():
        raise ValueError('Nonfinite Fourier bound')
    return dict(coefficient_count=len(n), active_axes=axes.tolist(),
        fixed_inactive_coordinates={str(a): 0.0 for a in range(3) if a not in axes},
        grid_side=int(grid_side), grid_points=total, spectral_grid_max_ev=maximum,
        worst_kpoint=worst.tolist(), derivative_bounds_ev=derivative.tolist(),
        triangle_upper_ev=triangle, grid_lipschitz_upper_ev=grid_bound,
        uniform_upper_ev=upper, floating_point_allowance_ev=float(allowance),
        scope='Finite unaveraged Fourier interpolant on the stated coordinate plane; not DFT accuracy',
        arithmetic='Exact-arithmetic bounds evaluated in float64; added heuristic roundoff allowance '
                   '256*eps*coefficient_count*max(1,triangle_upper_ev); not interval arithmetic')
