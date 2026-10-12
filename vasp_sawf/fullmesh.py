"""Full native-MMN localization within exact unitary/antiunitary constraints.

This numerical kernel neither reads wavefunctions nor writes checkpoints. Its
stationarity report does not certify physical provenance or a global minimum.
"""

from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize


_MATRIX_TOL = 1e-10
_TANGENT_MAX_BYTES = 256*1024**2
_NEWTON_MAX_DOF = 1024
_NEWTON_MAX_STEPS = 3


class FullMeshConvergenceError(RuntimeError):
    """Numerical nonconvergence with a serializable diagnostic report."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            'Full-mesh localization did not converge: fresh-chart gradient '
            f"{report['final_fresh_chart_gradient_max']:.8g}, required "
            f"< {report['parameters']['gradient_tol']:.8g}")


def _adjoint(x):
    return x.conj().swapaxes(-1, -2)


def _maxabs(x):
    return float(np.max(np.abs(x), initial=0))


def _objective(u, mmn, neighbors, weights, bvectors, *, gradient=True, stable=False):
    """Return spread and G with dOmega = 2 Re <G,dU>, including both endpoints.

    The stable expression removes Omega_I on the unitary manifold. The
    completeness correction retains exact equivalence for imperfect shells.
    With gradient=False also return full-spread centers, spreads, and overlaps.
    """
    nk, nb = u.shape[:2]
    v = mmn @ u[neighbors]
    z = np.einsum('kin,kbin->kbn', u.conj(), v)
    if not np.isfinite(z).all() or np.min(abs(z)) < 1e-12:
        raise ValueError('Nonfinite or near-zero diagonal overlap in the spread objective')
    phase = np.angle(z)
    centers = -np.einsum('kbn,b,bc->nc', phase, weights, bvectors)/nk
    second = np.einsum('kbn,b->n', 1-abs(z)**2+phase**2, weights)/nk
    spreads = second-np.einsum('nc,nc->n', centers, centers)
    q = phase+(bvectors @ centers.T)[None]
    if stable:
        transformed = _adjoint(u)[:, None] @ v
        transformed[..., np.arange(nb), np.arange(nb)] = 0
        offdiagonal = np.einsum('kbij,b->', abs(transformed)**2, weights)/nk
        diagonal = np.einsum('kbn,b->', q**2, weights)/nk
        completeness = np.einsum('b,bi,bj->ij', weights, bvectors, bvectors)
        correction = np.einsum('ni,ij,nj->', centers, completeness-np.eye(3), centers)
        value = float(offdiagonal+diagonal-correction)
    else:
        value = float(spreads.sum())
    if not gradient:
        return value, centers, spreads, z
    if stable:
        transformed[..., np.arange(nb), np.arange(nb)] = 1j*q/z.conj()
        transformed *= weights[None, :, None, None]/nk
        grad = np.sum(v @ _adjoint(transformed), axis=1)
        incoming = _adjoint(mmn) @ u[:, None] @ transformed
    else:
        dz = weights[None, :, None]/nk*(-z+1j*q/z.conj())
        grad = np.einsum('kbin,kbn->kin', v, dz.conj())
        incoming = (_adjoint(mmn) @ u[:, None])*dz[:, :, None, :]
    np.add.at(grad, neighbors.ravel(), incoming.reshape(-1, nb, nb))
    if not np.isfinite(value) or not np.isfinite(grad).all():
        raise ValueError('Nonfinite full-mesh objective or gradient')
    return value, grad


def _commutant_basis(target, little, anti):
    nb = target.shape[-1]
    generators = []
    for i in range(nb):
        x = np.zeros((nb, nb), complex)
        x[i, i] = 1j
        generators.append(x)
    for i in range(nb):
        for j in range(i):
            x = np.zeros((nb, nb), complex)
            x[i, j], x[j, i] = 1/np.sqrt(2), -1/np.sqrt(2)
            generators.extend((x, 1j*abs(x)))
    generators = np.array(generators)
    projected = sum(target[g] @ (generators.conj() if anti[g] else generators)
                    @ _adjoint(target[g]) for g in little)/len(little)
    real_columns = np.concatenate([projected.real.reshape(nb*nb, -1).T,
                                   projected.imag.reshape(nb*nb, -1).T], axis=0)
    left, singular, _ = np.linalg.svd(real_columns, full_matrices=False)
    if np.any((singular > 1e-8) & (abs(singular-1) > 1e-8)):
        raise ValueError('Little-group Reynolds projector is not idempotent')
    count = np.count_nonzero(singular > .5)
    basis = (left[:nb*nb, :count]+1j*left[nb*nb:, :count]).T.reshape(count, nb, nb)
    if (_maxabs(basis+_adjoint(basis)) > _MATRIX_TOL
            or any(_maxabs(target[g] @ (basis.conj() if anti[g] else basis)
                          @ _adjoint(target[g])-basis) > _MATRIX_TOL for g in little)):
        raise ValueError('Little-group tangent basis violates its exact constraints')
    return basis


class _ConstrainedChart:
    """Real Cayley coordinates and their analytic pullback on full-grid gauges."""

    def __init__(self, u0, sym, target, mmn, neighbors, weights, bvectors):
        self.u0, self.sym = np.array(u0), sym
        self.mmn, self.neighbors = mmn, neighbors
        self.weights, self.bvectors = weights, bvectors
        # Conservative dense SVD workspace plus retained complex bases; not RSS.
        self.workspace_estimate = (128+16*sym.NKirr)*self.u0.shape[-1]**4
        if self.workspace_estimate > _TANGENT_MAX_BYTES:
            raise ValueError('Dense little-group tangent workspace estimate '
                             f'{self.workspace_estimate} exceeds {_TANGENT_MAX_BYTES} bytes')
        self.bases, self.maps = [], []
        for i, k in enumerate(sym.kptirr):
            self.bases.append(_commutant_basis(target[:, k], sym.isym_little[i], sym.time_reversals))
            unique = {}
            for g, image in enumerate(sym.kptirr2kpt[i]):
                unique.setdefault(int(image), g)
            self.maps.append(unique)
        self.slices = np.cumsum([0]+[len(b) for b in self.bases])
        self.ndof = int(self.slices[-1])

    def matrices(self, x):
        irr, inverses = [], []
        identity = np.eye(self.u0.shape[-1])
        for i, basis in enumerate(self.bases):
            generator = np.einsum('a,aij->ij', x[self.slices[i]:self.slices[i+1]], basis)
            inverse = np.linalg.solve(identity-generator/2, identity)
            irr.append(self.u0[i] @ (2*inverse-identity))
            inverses.append(inverse)
        return np.array(self.sym.U_to_full_BZ(irr)), np.array(inverses)

    def __call__(self, x):
        full, inverse = self.matrices(x)
        value, full_gradient = _objective(full, self.mmn, self.neighbors,
                                          self.weights, self.bvectors, stable=True)
        gradient = np.empty(self.ndof)
        for i, images in enumerate(self.maps):
            pulled = sum(self.sym.rotate_U(full_gradient[k], i, g, forward=False)
                         for k, g in images.items())
            gx = _adjoint(inverse[i]) @ _adjoint(self.u0[i]) @ pulled @ _adjoint(inverse[i])
            gradient[self.slices[i]:self.slices[i+1]] = 2*np.einsum(
                'ij,aij->a', gx.conj(), self.bases[i]).real
        return value, gradient


def _block_matrix(blocks, indices, nb):
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1:] != (2,) or len(indices) != len(blocks)
            or not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError('Invalid symmetry block indices')
    full, covered = np.zeros((nb, nb), complex), np.zeros(nb, int)
    for block, (start, stop) in zip(blocks, indices, strict=True):
        if not 0 <= start < stop <= nb or np.shape(block) != (stop-start, stop-start):
            raise ValueError('Symmetry blocks do not retain the complete band dimension')
        full[start:stop, start:stop] = block
        covered[start:stop] += 1
    if not np.all(covered == 1):
        raise ValueError('Symmetry blocks omit or duplicate bands')
    if not np.isfinite(full).all() or _maxabs(_adjoint(full) @ full-np.eye(nb)) > _MATRIX_TOL:
        raise ValueError('Symmetry action is not unitary; approximate representations must be handled explicitly')
    return full


def _validate(initial, sym, target, mmn, neighbors, weights, bvectors, maxiter, gradient_tol):
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError('maxiter must be a positive integer')
    if (isinstance(gradient_tol, bool) or not isinstance(gradient_tol, Real)
            or not np.isfinite(gradient_tol) or gradient_tol <= 0):
        raise ValueError('gradient_tol must be positive and finite')
    initial, target, mmn = [np.asarray(a, complex) for a in (initial, target, mmn)]
    neighbors = np.asarray(neighbors)
    if np.iscomplexobj(weights) or np.iscomplexobj(bvectors):
        raise ValueError('Neighbor weights and vectors must be real')
    weights, bvectors = np.asarray(weights, float), np.asarray(bvectors, float)
    nk, nb, ng = sym.NK, sym.NB, sym.Nsym
    if (nk < 1 or nb < 1 or ng < 1 or sym.num_wann != nb
            or initial.shape != (nk, nb, nb) or target.shape != (ng, nk, nb, nb)):
        raise ValueError('Full-grid square initial gauge and target dimensions must retain every band')
    if (mmn.ndim != 4 or mmn.shape[0] != nk or mmn.shape[2:] != (nb, nb)
            or mmn.shape[1] == 0):
        raise ValueError('MMN must contain every full-grid source and complete square band matrices')
    nnb = mmn.shape[1]
    if (neighbors.shape != (nk, nnb) or not np.issubdtype(neighbors.dtype, np.integer)
            or np.any((neighbors < 0) | (neighbors >= nk))):
        raise ValueError('Full-grid neighbor indices are invalid')
    if weights.shape != (nnb,) or bvectors.shape != (nnb, 3):
        raise ValueError('Invalid neighbor weight/vector dimensions')
    if any(not np.isfinite(a).all() for a in (initial, target, mmn, weights, bvectors)):
        raise ValueError('Full-mesh inputs must be finite')
    if (_maxabs(_adjoint(initial) @ initial-np.eye(nb)) > _MATRIX_TOL
            or _maxabs(_adjoint(target) @ target-np.eye(nb)) > _MATRIX_TOL):
        raise ValueError('Initial gauge and target must be unitary')
    irr, mapping, grid = map(np.asarray, (sym.kptirr, sym.kptirr2kpt, sym.grid))
    if (irr.shape != (sym.NKirr,) or sym.NKirr < 1 or mapping.shape != (sym.NKirr, ng)
            or grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid <= 0) or np.prod(grid) != nk
            or np.asarray(sym.time_reversals).shape != (ng,)
            or np.asarray(sym.time_reversals).dtype != bool):
        raise ValueError('Symmetrizer metadata does not describe the complete grid')
    for indices in (irr, mapping):
        if not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= nk)):
            raise ValueError('Invalid symmetry k-point indices')
    stars = [np.unique(row) for row in mapping]
    if not np.array_equal(np.sort(np.concatenate(stars)), np.arange(nk)):
        raise ValueError('Symmetry stars must partition the complete k grid')
    for i, k in enumerate(irr):
        little = np.flatnonzero(mapping[i] == k)
        if not len(little) or not np.array_equal(little, sym.isym_little[i]):
            raise ValueError('Missing or inconsistent little group')
        for g in range(ng):
            _block_matrix(sym.d_band_blocks[i][g], sym.d_band_block_indices[i], nb)
            stored = _block_matrix(sym.D_wann_blocks[i][g], sym.D_wann_block_indices, nb)
            if _maxabs(stored-target[g, k]) > _MATRIX_TOL:
                raise ValueError('Supplied target differs from the symmetrizer target')
    return initial, target, mmn, neighbors, weights, bvectors


def _constraint_errors(full, sym):
    unitarity = _maxabs(_adjoint(full) @ full-np.eye(full.shape[-1]))
    covariance = max(_maxabs(full[sym.kptirr2kpt[i, g]]-sym.rotate_U(full[k], i, g))
                     for i, k in enumerate(sym.kptirr) for g in range(sym.Nsym))
    return unitarity, covariance


def optimize_full_mesh(initial_gauge, sym, target, mmn, neighbors, weights, bvectors,
                       *, maxiter=1000, gradient_tol=1e-8):
    """Return ``(U, report)`` only after full-grid constrained stationarity.

    ``initial_gauge`` is a unitary NK x NB x NB copy of the caller's original
    AMN alignment. ``target`` has shape Nsym x NK x NB x NB and must agree with
    the exact, already validated WannierBerri symmetrizer (including integer-cell
    phases). MMN has shape NK x NNB x NB x NB, neighbors NK x NNB, weights NNB,
    and Cartesian bvectors NNB x 3. Their order and provenance belong to the caller.

    Little-group averaging and polar factors initialize complete square gauges;
    no band is excluded. Up to four re-centered L-BFGS-B charts share ``maxiter``.
    If they stop early, at most three dense finite-difference Newton refinements
    are allowed for at most 1024 real coordinates. These limits and all optimizer
    stops are reported. Dense tangent construction is refused before allocation
    if its conservative per-chart workspace estimate exceeds 256 MiB; this is
    not a total-process memory estimate and excludes caller-owned input arrays.
    A fresh zero-chart gradient, never SciPy's success flag,
    determines convergence. Failure raises :class:`FullMeshConvergenceError`
    with ``.report``; invalid or incompatible input raises ``ValueError``.

    Inputs and checkpoints are unchanged. Numerical stationarity is neither
    source-symmetry acceptance nor a global-minimum guarantee.
    """
    initial, target, mmn, neighbors, weights, bvectors = _validate(
        initial_gauge, sym, target, mmn, neighbors, weights, bvectors, maxiter, gradient_tol)
    irr, projected_singular = [], []
    for i, k in enumerate(sym.kptirr):
        little = sym.isym_little[i]
        projected = sum(sym.rotate_U(initial[k], i, g) for g in little)/len(little)
        left, singular, right = np.linalg.svd(projected)
        if singular[-1] < _MATRIX_TOL:
            raise ValueError('Projected initial intertwiner is rank deficient; band dropping is forbidden')
        irr.append(left @ right)
        projected_singular.append(singular.tolist())
    full = np.asarray(sym.U_to_full_BZ(irr))
    if max(_constraint_errors(full, sym)) > _MATRIX_TOL:
        raise ValueError('Projected initial gauge violates the supplied exact symmetry constraints')

    def new_chart(u):
        return _ConstrainedChart(u[np.asarray(sym.kptirr)], sym, target, mmn, neighbors, weights, bvectors)

    chart = new_chart(full)
    zero = np.zeros(chart.ndof)
    initial_value = _objective(full, mmn, neighbors, weights, bvectors, gradient=False)[0]
    invariant = float(np.einsum('kb,b->', sym.NB-np.sum(abs(mmn)**2, axis=(-2, -1)), weights)/sym.NK)
    blocks, refinements = [], []
    iterations = 0
    _, gradient = chart(zero)
    for block in range(4):
        if _maxabs(gradient) < gradient_tol or iterations >= maxiter:
            break
        result = minimize(chart, zero, jac=True, method='L-BFGS-B', options={
            'maxiter': min(400, maxiter-iterations), 'gtol': gradient_tol,
            'ftol': 1e-15, 'maxls': 40, 'maxcor': 20})
        iterations += int(result.nit)
        full, _ = chart.matrices(result.x)
        chart = new_chart(full)
        value, gradient = chart(zero)
        blocks.append(dict(block=block, scipy_success=bool(result.success),
                           message=str(result.message), iterations=int(result.nit),
                           evaluations=int(result.nfev), stable_objective=value,
                           fresh_chart_gradient_max=_maxabs(gradient)))

    refinement_status = 'already_stationary'
    if _maxabs(gradient) >= gradient_tol:
        refinement_status = ('iteration_budget_exhausted' if iterations >= maxiter else
                             'dimension_limit' if chart.ndof > _NEWTON_MAX_DOF else 'step_limit')
    if refinement_status == 'step_limit':
        for refinement in range(_NEWTON_MAX_STEPS):
            value, gradient = chart(zero)
            if _maxabs(gradient) < gradient_tol:
                refinement_status = 'stationary'
                break
            step = 1e-4
            hessian = np.empty((chart.ndof, chart.ndof))
            for j in range(chart.ndof):
                dx = np.zeros(chart.ndof)
                dx[j] = step
                hessian[:, j] = (chart(dx)[1]-chart(-dx)[1])/(2*step)
            asymmetry = _maxabs(hessian-hessian.T)
            hessian = (hessian+hessian.T)/2
            eig, vectors = np.linalg.eigh(hessian)
            floor = max(1e-8, _maxabs(eig)*1e-10)
            direction = -vectors @ ((vectors.T @ gradient)/np.maximum(eig, floor))
            direction *= min(1., .1/max(np.linalg.norm(direction), np.finfo(float).tiny))
            full_before = _objective(full, mmn, neighbors, weights, bvectors, gradient=False)[0]
            tolerance = 128*np.finfo(float).eps*max(1., abs(value))
            full_tolerance = 128*np.finfo(float).eps*max(1., abs(full_before))
            trials, accepted = [], False
            for scale in (1., .5, .25, .125):
                candidate, _ = chart.matrices(direction*scale)
                candidate_chart = new_chart(candidate)
                newvalue, newgradient = candidate_chart(zero)
                full_after = _objective(candidate, mmn, neighbors, weights, bvectors, gradient=False)[0]
                accepted = (_maxabs(newgradient) < _maxabs(gradient)
                            and newvalue <= value+tolerance and full_after <= full_before+full_tolerance)
                trials.append(dict(scale=scale, stable_objective_change=newvalue-value,
                                   full_objective_change=full_after-full_before,
                                   fresh_chart_gradient_max=_maxabs(newgradient), accepted=bool(accepted)))
                if accepted:
                    full, chart = candidate, candidate_chart
                    break
            refinements.append(dict(refinement=refinement, fd_step=step,
                                    hessian_asymmetry_max=asymmetry,
                                    hessian_eigenvalue_min=float(eig.min()), curvature_floor=floor,
                                    gradient_before_max=_maxabs(gradient),
                                    trust_step_norm=float(np.linalg.norm(direction)),
                                    numeric_roundoff_allowance_stable=tolerance,
                                    numeric_roundoff_allowance_full=full_tolerance,
                                    trials=trials, accepted=bool(accepted)))
            if not accepted:
                refinement_status = 'no_improving_step'
                break

    final_chart = new_chart(full)
    stable_value, gradient = final_chart(zero)
    final_value, centers, spreads, _ = _objective(full, mmn, neighbors, weights, bvectors, gradient=False)
    unitarity, covariance = _constraint_errors(full, sym)
    converged = (_maxabs(gradient) < gradient_tol and unitarity <= _MATRIX_TOL
                 and covariance <= _MATRIX_TOL)
    report = dict(schema='sawf-fullmesh-localization-v1', numerical_converged=bool(converged),
                  parameters=dict(maxiter=int(maxiter), gradient_tol=float(gradient_tol),
                                  matrix_tol=_MATRIX_TOL, max_charts=4,
                                  tangent_workspace_limit_bytes=_TANGENT_MAX_BYTES,
                                  newton_max_steps=_NEWTON_MAX_STEPS, newton_max_dof=_NEWTON_MAX_DOF),
                  shape_nk_nb_nw=list(full.shape), nk_irreducible=int(sym.NKirr),
                  real_parameters=chart.ndof, parameters_per_irreducible_point=[len(b) for b in chart.bases],
                  tangent_workspace_estimate_bytes=int(chart.workspace_estimate),
                  projected_intertwiner_singular_values=projected_singular,
                  initial_full_grid_total_spread_angstrom2=initial_value,
                  full_grid_total_spread_angstrom2=final_value,
                  centers_angstrom=centers.tolist(), spreads_angstrom2=spreads.tolist(),
                  omega_i_angstrom2=invariant, stable_objective=stable_value,
                  stable_objective_identity_error=abs(final_value-stable_value-invariant),
                  completeness_max=_maxabs(np.einsum('b,bi,bj->ij', weights, bvectors, bvectors)-np.eye(3)),
                  iterations=iterations, optimization_blocks=blocks, newton_refinement=refinements,
                  newton_status='stationary' if converged and refinements else refinement_status,
                  final_fresh_chart_gradient_max=_maxabs(gradient),
                  gauge_unitarity_max=unitarity, gauge_covariance_max=covariance,
                  convergence_criterion='Fresh zero-chart gradient max below gradient_tol and exact gauge constraints',
                  physical_acceptance=False, excluded_bands=[])
    if not converged:
        raise FullMeshConvergenceError(report)
    return full, report
