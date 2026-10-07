"""Align original SCDM guesses, run strict SAWF, and export models."""

from contextlib import redirect_stdout
from copy import copy
from importlib.metadata import version
import io
import numpy as np
from wannierberri.symmetry.sawf_kirr import Symmetrizer_Uirr, get_symmetrizer_Zirr
from wannierberri.wannierise.wannierise import update_chk
from wannierberri.wannierise.wannierizer import Wannierizer
import hashlib
import json
from pathlib import Path
import resource
import re
import time
from .inputs import read_inputs


def _square_dimension(symmetrizer):
    nb = symmetrizer.NB
    spinor = getattr(getattr(symmetrizer, 'spacegroup', None), 'spinor', True)
    if (not isinstance(nb, (int, np.integer)) or nb <= 0 or (spinor and nb % 2)
            or symmetrizer.num_wann != nb):
        raise ValueError("Strict entry requires positive NB=NW and an even dimension for spinors, retaining all target bands")
    return int(nb)


class StrictSymmetrizerUirr(Symmetrizer_Uirr):
    """Retain official averaging/polar decomposition; reject band dropping and treat nonconverged returns as failures."""

    def __init__(self, symmetrizer, ikirr, accuracy_threshold=1e-6):
        if not 0 < accuracy_threshold <= 1e-6:
            raise ValueError("The official local symmetrization tolerance of 1e-6 must not be relaxed")
        _square_dimension(symmetrizer)
        super().__init__(symmetrizer, ikirr, accuracy_threshold=accuracy_threshold)

    def check(self, U=None, verbose=False, accuracy_threshold=1e-6, no_exclude_bands=None):
        if no_exclude_bands not in (None, self.nb):
            raise ValueError("Automatic band-block exclusion must not be enabled")
        if not 0 < accuracy_threshold <= 1e-6:
            raise ValueError("The official local symmetrization tolerance of 1e-6 must not be relaxed")
        if U is None:
            rng = np.random.default_rng(0)
            U = rng.random((self.nb, self.num_wann)) + 1j*rng.random((self.nb, self.num_wann))
        error = super().check(U=U, verbose=verbose, accuracy_threshold=accuracy_threshold,
                              no_exclude_bands=self.nb)
        if not self.include_bands.all() or not np.isfinite(error) or error > accuracy_threshold:
            raise RuntimeError(f"Local symmetrization check failed for the complete target band set: {error}")
        return error

    def __call__(self, U, maxiter=100, tol=1e-6):
        if not 0 < tol <= 1e-6 or maxiter < 1:
            raise ValueError("Local iteration count must be positive and the official tolerance of 1e-6 must not be relaxed")
        if not self.include_bands.all():
            raise RuntimeError("Automatic band-block exclusion detected")
        if U.shape != (self.nb, self.num_wann) or not np.isfinite(U).all():
            raise ValueError(f"Local U must be a finite {self.nb}-by-{self.num_wann} matrix")
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                result = super().__call__(U, maxiter=maxiter, tol=tol)
        finally:
            message = output.getvalue()
            if message:
                print(message, end="")
        if "Warning: symmetrization did not converge" in message:
            raise RuntimeError("Official local symmetrization did not converge; the matrix returned after a warning is rejected")
        if not np.isfinite(result).all():
            raise RuntimeError("Local symmetrization produced nonfinite U")
        unitarity = float(np.max(abs(result.conj().T @ result - np.eye(self.num_wann))))
        covariance = max(float(np.max(abs(self.rotate_U(result, isym) - result)))
                         for isym in self.isym_little)
        if unitarity > tol or covariance > tol:
            raise RuntimeError(f"Local U failed the complete target-band check: unitarity {unitarity}, covariance {covariance}")
        return result


def _validated_inputs(data, symmetrizer):
    if version("wannierberri") != "1.7.0" or version("irrep") != "2.6.3":
        raise ValueError("The strict entry point was verified only for WannierBerri1.7.0/IrRep2.6.3")
    nb = _square_dimension(symmetrizer)
    nk = data.mmn.NK
    if data.irreducible or nk != int(np.prod(data.mp_grid)):
        raise ValueError("The full k-point grid is required; irreducible export is forbidden")
    if (data.mmn.NB, data.amn.NB, data.amn.NW, data.eig.NB,
            data.chk.num_bands, data.chk.num_wann) != (nb,)*6:
        raise ValueError("Original target bands must not be dropped and the Wannier dimension must not change")
    if (data.amn.NK, data.eig.NK, data.chk.num_kpts) != (nk,)*3:
        raise ValueError("Input files disagree on the number of k points")
    if (symmetrizer.NK != nk or not np.array_equal(symmetrizer.kpoints_all, data.chk.kpt_latt)
            or not np.array_equal(symmetrizer.grid, data.mp_grid)):
        raise ValueError("Symmetrizer and interface disagree on full k-point coordinates, order, or grid")
    if not np.any(symmetrizer.time_reversals):
        raise ValueError("Space-group/TR constrained entry is missing antiunitary operations")
    neighbour_input = data.bkvec.neighbours
    if isinstance(neighbour_input, dict) and set(neighbour_input) != set(range(nk)):
        raise ValueError("MMN neighbors are missing full k-grid indices")
    neighbours = np.asarray([neighbour_input[k] for k in range(nk)])
    if (neighbours.ndim != 2 or neighbours.shape[0] != nk
            or not np.issubdtype(neighbours.dtype, np.integer)
            or np.any(neighbours < 0) or np.any(neighbours >= nk)):
        raise ValueError("MMN neighbors do not cover the full k-point grid")
    nnb = neighbours.shape[1]
    if (np.shape(data.bkvec.wk) != (nnb,) or np.shape(data.bkvec.bk_cart) != (nnb, 3)
            or not np.isfinite(data.bkvec.wk).all() or not np.isfinite(data.bkvec.bk_cart).all()):
        raise ValueError("Invalid MMN neighbor weight/vector dimensions or values")
    originals = {}
    for name, shape in (("amn", (nb, nb)), ("mmn", (nnb, nb, nb)), ("eig", (nb,))):
        arrays = data.get_file(name).data
        if set(arrays) != set(range(nk)):
            raise ValueError(f"{name} is missing full k-grid data")
        originals[name] = {}
        for ik, array in arrays.items():
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} dimensions or values at k={ik}")
            originals[name][ik] = array.copy()
    for indices in symmetrizer.d_band_block_indices:
        if (np.shape(indices)[1:] != (2,) or not len(indices) or indices[0, 0] != 0
                or indices[-1, 1] != nb or np.any(indices[:, 1] <= indices[:, 0])
                or not np.array_equal(indices[:-1, 1], indices[1:, 0])):
            raise ValueError("Bloch representation band blocks do not cover all target bands")
    return originals, neighbours


def _assert_unchanged(data, originals):
    for name, arrays in originals.items():
        current = data.get_file(name).data
        if set(current) != set(arrays) or any(not np.array_equal(current[k], v) for k, v in arrays.items()):
            raise RuntimeError(f"SAWF changed the original {name} matrix")


def _square_matrices(values, nk, nb, name):
    if isinstance(values, dict):
        if set(values) != set(range(nk)):
            raise ValueError(f"{name} must cover all k points")
        values = [values[k] for k in range(nk)]
    matrices = np.asarray(values, dtype=complex)
    if matrices.shape != (nk, nb, nb) or not np.isfinite(matrices).all():
        raise ValueError(f"{name} must be a finite [NK,{nb},{nb}] array")
    return matrices.copy()


def _unitarity_error(matrices):
    return float(np.max(abs(matrices.conj().swapaxes(-1, -2) @ matrices - np.eye(matrices.shape[-1]))))


def _full_rank_polar(matrices, name):
    left, singular, right = np.linalg.svd(matrices, full_matrices=False)
    if np.any(singular[..., -1] <= matrices.shape[-1]*np.finfo(float).eps*singular[..., 0]):
        raise ValueError(f"{name} is rank deficient; polar decomposition cannot select missing initial-guess directions")
    return left @ right, singular


def _normalize_target_orbits(centers, orbitals):
    centers = np.asarray(centers, dtype=float)
    if centers.shape == (3,):
        centers = centers[None]
    orbitals = [orbitals] if isinstance(orbitals, str) else list(orbitals)
    if (centers.ndim != 2 or centers.shape[1:] != (3,) or not len(centers)
            or not np.isfinite(centers).all() or len(orbitals) != len(centers)
            or any(not isinstance(o, str) or not o.strip() for o in orbitals)):
        raise ValueError('Provide one finite fractional --center and one --orbital for each target orbit')
    return centers, orbitals


def _target_representation(sym, centers, orbitals):
    """Set target D only; orbit expansion and orbital/spin order come from WannierBerri."""
    from wannierberri.symmetry.projections import Projection
    from wannierberri.symmetry.Dwann import Dwann

    centers, orbitals = _normalize_target_orbits(centers, orbitals)
    spinor = bool(sym.spacegroup.spinor)
    projections = [Projection(position_num=c, spacegroup=sym.spacegroup, orbital=o, spinor=spinor)
                   for c, o in zip(centers, orbitals, strict=True)]
    sym.set_D_wann_from_projections(projections)
    blocks, positions, orbits = [], [], []
    for c, o, projection in zip(centers, orbitals, projections, strict=True):
        orbits.append(dict(position_fractional=c.tolist(), orbital=o,
                           expanded_positions_fractional=np.asarray(projection.positions).tolist()))
        for orbital in projection.orbitals:
            block = Dwann(spacegroup=sym.spacegroup, positions=projection.positions,
                          orbital=orbital, orbitalrotator=sym.orbitalrotator,
                          basis_list=projection.basis_list, spinor=spinor)
            blocks.append(block)
            positions.extend(np.repeat(np.asarray(block.orbit), block.num_orbitals, axis=0))
    positions = np.asarray(positions)
    if positions.shape != (sym.num_wann, 3):
        raise ValueError('Target column positions do not match the full Wannier dimension')
    descriptor = dict(orbits=orbits, column_centers_fractional=positions.tolist(),
        basis_convention='Input orbit order, then WannierBerri orbital block, orbit position, real orbital'
                         + (', interlaced spin' if spinor else ''),
        center_constraint='Affine center covariance under the declared site permutations and integer cell shifts; symmetry-allowed coordinates may relax')
    if len(orbits) == 1:
        descriptor.update(position_fractional=centers[0].tolist(), orbital=orbitals[0])
    return blocks, positions, descriptor


def _target_matrix(blocks, kpoint, image, operation):
    from scipy.linalg import block_diag
    return block_diag(*(block.get_on_points(kpoint, image, operation) for block in blocks))


def _center_covariance(sym, centers):
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (sym.num_wann, 3) or not np.isfinite(centers).all():
        raise ValueError('Invalid final Wannier centers')
    return float(np.max(abs(sym.symmetrize_WCC(centers)-centers)))


def _match_center_columns(ordinary_centers, target_centers, lattice):
    from pymatgen.core import Lattice
    from scipy.optimize import linear_sum_assignment

    metric = Lattice(lattice)
    ordinary = np.asarray(ordinary_centers) @ np.linalg.inv(lattice)
    target = np.asarray(target_centers) @ np.linalg.inv(lattice)
    nb = len(ordinary)
    distances, images = np.empty((nb, nb)), np.empty((nb, nb, 3), dtype=int)
    for i, source in enumerate(ordinary):
        for j, destination in enumerate(target):
            distances[i, j], images[i, j] = metric.get_distance_and_image(source, destination)
    rows, columns = linear_sum_assignment(distances)
    order = rows[np.argsort(columns)]
    return order, images[order, np.arange(nb)], float(distances[rows, columns].max())


def align_scdm_initial_gauge(original_amn, ordinary_gauge, ordinary_centers,
                           lattice, kpoints, sym, *, target_centers=None):
    """Return a reversible column/cell transform of polar(original AMN), with a full ledger.

    The caller checks run provenance of the ordinary gauge/centers. This function neither repeats SCDM nor changes AMN or D.
    Center assignment only chooses an initial guess; compatibility is checked by SAWF at every irreducible k point.
    """
    nb = _square_dimension(sym)
    kpoints = np.asarray(kpoints, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    centers = np.asarray(ordinary_centers, dtype=float)
    if (kpoints.ndim != 2 or kpoints.shape[1] != 3 or not len(kpoints)
            or lattice.shape != (3, 3) or centers.shape != (nb, 3)
            or not np.isfinite(kpoints).all() or not np.isfinite(lattice).all()
            or not np.isfinite(centers).all()):
        raise ValueError("Invalid k-point, lattice, or center dimensions/values for initial-guess alignment")
    nk = len(kpoints)
    if sym.NK != nk or not np.array_equal(sym.kpoints_all, kpoints):
        raise ValueError("Initial-guess alignment and canonical representation disagree on full k-point order")
    if np.linalg.matrix_rank(lattice) != 3:
        raise ValueError("Initial-guess alignment lattice is singular")
    amn = _square_matrices(original_amn, nk, nb, "original AMN")
    before = amn.copy()
    ordinary = _square_matrices(ordinary_gauge, nk, nb, "ordinary gauge")
    floating_tol = 256*np.finfo(float).eps*nb
    ordinary_error = _unitarity_error(ordinary)
    if ordinary_error > floating_tol:
        raise ValueError(f"Ordinary gauge failed the floating-point unitarity check: {ordinary_error}")
    inferred = target_centers is None
    target = np.asarray(sym.symmetrize_WCC(np.zeros((nb, 3))) if inferred else target_centers)
    if target.shape != (nb, 3) or not np.isfinite(target).all():
        raise ValueError("Invalid canonical target center")
    if inferred:
        if np.max(abs(target-target[0])) > 1e-6:
            raise ValueError("Inferring target centers requires the same canonical target center; supply explicit target_centers")
        for vector in lattice:
            if np.max(abs(sym.symmetrize_WCC(target+vector)-target)) > 1e-6:
                raise ValueError("Target centers have free coordinates; supply explicit target_centers")
        order = np.arange(nb)
        displacement = (centers-target) @ np.linalg.inv(lattice)
        shifts = np.rint(displacement).astype(int)
        center_error = float(np.max(abs((displacement-shifts) @ lattice)))
        if center_error > 1e-6:
            raise ValueError(f"Inferred target and ordinary centers do not differ by integer cells: {center_error} Å")
    else:
        if _center_covariance(sym, target) > 1e-6:
            raise ValueError('Declared target column centers violate the target representation')
        order, shifts, center_error = _match_center_columns(centers, target, lattice)
    gamma = np.flatnonzero(np.max(abs(kpoints-np.rint(kpoints)), axis=1) < 1e-12)
    if len(gamma) != 1:
        raise ValueError("Initial-guess alignment requires exactly one Gamma anchor")
    gamma = int(gamma[0])
    irr_index = np.flatnonzero(np.asarray(sym.kptirr) == gamma)
    if len(irr_index) != 1:
        raise ValueError("Gamma is not the unique stored irreducible anchor")
    igamma = int(irr_index[0])
    little = sym.isym_little[igamma]
    if len(little) != sym.Nsym or not np.any(sym.time_reversals[little]):
        raise ValueError("Gamma anchor lacks the full space group and antiunitary operations")
    ordinary_gamma = ordinary[gamma][:, order]
    projected = sum(sym.rotate_U(ordinary_gamma, igamma, s) for s in little)/len(little)
    if not np.isfinite(projected).all():
        raise ValueError("Gamma intertwiner is nonfinite")
    canonical_gamma, projected_singular = _full_rank_polar(
        projected, "Gamma initial-guess intertwiner (rank loss does not prove physical incompatibility)")
    gamma_error = max(float(np.max(abs(sym.rotate_U(canonical_gamma, igamma, s)-canonical_gamma)))
                      for s in little)
    if not np.isfinite(gamma_error) or gamma_error > 1e-6:
        raise ValueError(f"Gamma full-group/TR intertwining relation failed: {gamma_error}")
    q = ordinary_gamma.conj().T @ canonical_gamma
    q_error = _unitarity_error(q)
    raw_gauge, amn_singular = _full_rank_polar(amn, "original AMN")
    phase_dagger = np.exp(2j*np.pi*kpoints @ shifts.T)
    initial = (raw_gauge[:, :, order]*phase_dagger[:, None, :]) @ q
    initial_error = _unitarity_error(initial)
    recovered = ((initial @ q.conj().T)*phase_dagger.conj()[:, None, :])[:, :, np.argsort(order)]
    recovery_error = float(np.max(abs(recovered-raw_gauge)))
    if max(q_error, initial_error, recovery_error) > floating_tol or not np.isfinite(initial).all():
        raise ValueError("Initial-guess column transform failed unitarity/invertibility checks")
    if not np.array_equal(amn, before):
        raise RuntimeError("Initial-guess alignment changed the original AMN")
    report = dict(schema="sawf-bridge-scdm-initial-alignment-v1", formula="polar(AMN(k)) P S(k)^dagger Q",
                  permutation_source_columns=order.tolist(),
                  cell_shifts=shifts.tolist(), Q_real=q.real.tolist(), Q_imag=q.imag.tolist(),
                  target_centers_angstrom=target.tolist(), ordinary_centers_angstrom=centers.tolist(),
                  center_assignment_max_distance_angstrom=float(np.max(np.linalg.norm(
                      centers[order]-target-shifts @ lattice, axis=1))), gamma_index=gamma,
                  gamma_intertwiner_singular_values=projected_singular.tolist(),
                  gamma_intertwiner_covariance_max_abs=gamma_error,
                  original_amn_min_singular=float(amn_singular.min()),
                  ordinary_gauge_unitarity_max_abs=ordinary_error, Q_unitarity_max_abs=q_error,
                  initial_gauge_unitarity_max_abs=initial_error, inverse_transform_max_abs=recovery_error,
                  original_amn_unchanged=True, canonical_D_unchanged=True,
                  ordinary_provenance="must_be_verified_by_caller")
    report['center_assignment_role'] = 'Initialization only; ordinary centers need not equal the target and are not a compatibility certificate'
    return initial, report


def wannierise_strict(data, symmetrizer, *, num_iter=1000, conv_tol=1e-9, num_iter_converge=3,
                     frozen_all=False, mix_ratio_u=1.0, mix_ratio_z=0.5, initial_gauge=None):
    """Localize in memory without reading wavefunctions, writing checkpoint/hr files, or certifying physical provenance."""
    if not isinstance(num_iter, int) or num_iter < 1 or num_iter_converge < 2:
        raise ValueError("Invalid global iteration count or convergence-history length")
    if not 0 < conv_tol <= 1e-9:
        raise ValueError("The existing global numerical convergence setting of 1e-9 must not be relaxed")
    if not 0 < mix_ratio_u <= 1 or not 0 < mix_ratio_z <= 1:
        raise ValueError('Mixing fractions must lie in (0,1]')
    if mix_ratio_u != 1 and mix_ratio_u != mix_ratio_z:
        raise ValueError('WB1.7.0 uses the Z mixing fraction when U mixing is enabled; require u=1 or u=z to avoid misreporting parameters')
    originals, neighbours = _validated_inputs(data, symmetrizer)
    nb = symmetrizer.NB
    if initial_gauge is not None:
        initial_gauge = _square_matrices(initial_gauge, symmetrizer.NK, nb, "initial gauge copy")
        if _unitarity_error(initial_gauge) > 256*np.finfo(float).eps*nb:
            raise ValueError("Initial gauge copy failed the floating-point unitarity check")
    parameters = dict(init="amn" if initial_gauge is None else "provided_initial_gauge",
                      original_amn_used_without_basis_transform=initial_gauge is None,
                      sitesym=True, localise=True, parallel=False,
                      irreducible=False, symmetrize_hr=False, num_iter=num_iter,
                      conv_tol=conv_tol, num_iter_converge=num_iter_converge,
                      mix_ratio_z=mix_ratio_z, mix_ratio_u=mix_ratio_u, symmetrize_Z=True,
                      frozen_bands=nb if frozen_all else 0, local_symmetry_tol=1e-6, local_symmetry_maxiter=100,
                      center_recompute_tol_angstrom=1e-6, spread_recompute_tol_angstrom2=1e-2,
                      no_exclude_bands=nb, local_check_rng_seed=0)
    kptirr = np.asarray(symmetrizer.kptirr)
    frozen = np.full((symmetrizer.NKirr, nb), bool(frozen_all), dtype=bool)
    neighbour_irr = symmetrizer.kpt2kptirr[neighbours[kptirr]]
    wannierizer = Wannierizer(parallel=False, symmetrizer=symmetrizer)
    history = []
    try:
        for ikirr, ik in enumerate(kptirr):
            local = StrictSymmetrizerUirr(symmetrizer, ikirr)
            wannierizer.add_kpoint(Mmn=data.mmn.data[ik], frozen=frozen[ikirr],
                frozen_nb=frozen[neighbour_irr[ikirr]], wb=data.bkvec.wk, bk=data.bkvec.bk_cart,
                symmetrizer_Zirr=get_symmetrizer_Zirr(symmetrizer, ikirr, ~frozen[ikirr]),
                symmetrizer_Uirr=local, ikirr=ikirr,
                amn=data.amn.data[ik] if initial_gauge is None else initial_gauge[ik],
                weight=symmetrizer.ndegen(ikirr)/symmetrizer.NK)
        u_irr = wannierizer.get_U_opt_full()
        u_full = symmetrizer.U_to_full_BZ(u_irr)
        wannierizer.update_Unb_all([[u_full[j] for j in neighbours[k]] for k in kptirr])
        for iteration in range(num_iter):
            phase = np.exp(1j*wannierizer.wcc.dot(data.bkvec.bk_cart.T))
            u_irr = wannierizer.update_all([[u_full[j] for j in neighbours[k]] for k in kptirr],
                mix_ratio=mix_ratio_z, mix_ratio_u=mix_ratio_u, localise=True, wcc_bk_phase=phase)
            u_full = symmetrizer.U_to_full_BZ(u_irr)
            state = np.hstack((wannierizer.wcc, wannierizer.spreads[:, None]))
            if not np.isfinite(state).all():
                raise RuntimeError("SAWF centers/spreads are nonfinite")
            history.append(state.copy())
            delta_std = float(np.std(history[-num_iter_converge:], axis=0).max())
            if iteration % 50 == 0:
                print(f'SAWF iteration {iteration}, std={delta_std:.8g}, total spread={wannierizer.spreads.sum():.12g}')
            if iteration > num_iter_converge and delta_std < conv_tol:
                break
        else:
            error=RuntimeError(f"SAWF global iteration did not converge after {num_iter} iterations; final std={delta_std}")
            error.diagnostics={'parameters':parameters,'last_states':np.asarray(history[-10:]).tolist()}
            raise error
        gauge = np.asarray(u_full)
        if gauge.shape != (symmetrizer.NK, nb, nb) or not np.isfinite(gauge).all():
            raise RuntimeError("SAWF did not return U for all target bands on the full grid")
        if any(not point.symmetrizer_Uirr.include_bands.all() for point in wannierizer.kpoints):
            raise RuntimeError("SAWF automatically excluded a band block")
        unitarity = float(np.max(abs(gauge.conj().transpose(0, 2, 1) @ gauge - np.eye(nb))))
        covariance = max(float(np.max(abs(gauge[symmetrizer.kptirr2kpt[i, s]]
                              - symmetrizer.rotate_U(gauge[k], i, s))))
                         for i, k in enumerate(kptirr) for s in range(symmetrizer.Nsym))
        if unitarity > 1e-6 or covariance > 1e-6:
            raise RuntimeError(f"Final complete target-band U check failed: unitarity {unitarity}, covariance {covariance}")
        candidate_chk = copy(data.chk)
        candidate_chk.v_matrix = {ik: u for ik, u in enumerate(u_full)}
        centers_chk, spreads_chk = candidate_chk.get_wannier_centers(data.bkvec, data.mmn, spreads=True)
        center_error = float(np.max(abs(centers_chk - wannierizer.wcc)))
        spread_error = float(np.max(abs(spreads_chk - wannierizer.spreads)))
        if (not np.isfinite(centers_chk).all() or not np.isfinite(spreads_chk).all()
                or center_error > 1e-6 or spread_error > 1e-2):
            raise RuntimeError(f"Final-gauge full-BZ center/spread recomputation failed: {center_error} Å, {spread_error} Å²")
    finally:
        _assert_unchanged(data, originals)
    update_chk(w90data=data, U_opt_full_BZ=u_full, wcc=wannierizer.wcc, spreads=wannierizer.spreads)
    data.set_symmetrizer(symmetrizer, read_npz=False)
    data.wannierised = True
    return dict(schema="sawf-bridge-strict-localisation-v1", parameters=parameters,
                shape_nk_nb_nw=list(gauge.shape), converged=True, iterations_executed=len(history),
                converged_iteration_zero_based=iteration, final_delta_std=delta_std,
                convergence_history=np.asarray(history).tolist(), gauge_unitarity_max_abs=unitarity,
                full_symmetry_covariance_max_abs=covariance, excluded_bands=[],
                centers_angstrom=wannierizer.wcc.tolist(), spreads_angstrom2=wannierizer.spreads.tolist(),
                chk_centers_angstrom=centers_chk.tolist(), chk_spreads_angstrom2=spreads_chk.tolist(),
                center_recompute_max_abs_angstrom=center_error, spread_recompute_max_abs_angstrom2=spread_error,
                input_matrix_max_abs_changes={name: 0.0 for name in originals},
                physical_source_and_target_validation="required_before_call_not_certified_by_driver")


def _validate_spin_metadata(arrays, report, *, win_text=None):
    """Bind explicit spin semantics; older packages describe the original SOC workflow."""
    keys = ('spinor', 'source_ispin', 'spin_channel', 'antiunitary_kind')
    spin = report.get('spin')
    if spin is None:
        if any(key in arrays for key in (*keys, 'time_reversal_square')):
            raise ValueError('Spin arrays require their complete report metadata')
        spin = dict(spinor=True, source_ispin=1, spin_channel=1,
                    antiunitary_kind='physical_time_reversal', time_reversal_square=-1)
    else:
        if (not isinstance(spin, dict) or type(spin.get('spinor')) is not bool
                or type(spin.get('source_ispin')) is not int or spin['source_ispin'] not in (1, 2)
                or type(spin.get('spin_channel')) is not int
                or not 1 <= spin['spin_channel'] <= spin['source_ispin']):
            raise ValueError('Invalid source spin mode or selected channel in the symmetry report')
        if spin['spinor'] and spin['source_ispin'] != 1:
            raise ValueError('Spinor WAVECAR requires a single VASP spin channel')
        kind = ('physical_time_reversal' if spin['spinor'] else
                'orbital_complex_conjugation' if spin['source_ispin'] == 1 else
                'channel_complex_conjugation')
        square = -1 if spin['spinor'] else 1
        if (spin.get('antiunitary_kind') != kind or type(spin.get('time_reversal_square')) is not int
                or spin['time_reversal_square'] != square):
            raise ValueError('Antiunitary meaning or square disagrees with the source spin mode')
        for key in keys:
            value = arrays.get(key)
            if (value is None or value.ndim != 0 or type(value.item()) is not type(spin[key])
                    or value.item() != spin[key]):
                raise ValueError(f'Packaged spin metadata disagrees on {key}')
        if 'time_reversal_square' in arrays:
            value = arrays['time_reversal_square']
            if value.ndim != 0 or type(value.item()) is not int or value.item() != square:
                raise ValueError('Packaged antiunitary square disagrees with the spin mode')
        if 'spacegroup_spinor' not in arrays:
            raise ValueError('Explicit spin metadata requires the saved space-group spinor flag')
        context = report.get('outcar', {}).get('spin_context', {})
        if (not spin['spinor'] and 'ISPIN' in context and context['ISPIN'] != spin['source_ispin']
                or 'LNONCOLLINEAR' in context and context['LNONCOLLINEAR'] != spin['spinor']):
            raise ValueError('OUTCAR spin context disagrees with the packaged spin mode')
        channels = report.get('wavecar', {}).get('spin_channels', spin['source_ispin'])
        if channels != spin['source_ispin']:
            raise ValueError('WAVECAR channel count disagrees with the packaged spin mode')
    if 'spacegroup_spinor' in arrays:
        group_spinor = np.asarray(arrays['spacegroup_spinor'])
        if (group_spinor.ndim != 0 or group_spinor.dtype != np.dtype(bool)
                or group_spinor.item() != spin['spinor']):
            raise ValueError('Space-group spinor flag disagrees with packaged spin semantics')
    if win_text is not None:
        uncommented = '\n'.join(re.split(r'[!#]', row, maxsplit=1)[0] for row in win_text.splitlines())
        values = re.findall(r'^[ \t]*spinors[ \t]*[=:][ \t]*([^\n]*)$', uncommented, re.I | re.M)
        if len(values) > 1:
            raise ValueError('WIN spinors must not be duplicated')
        if values:
            booleans = {'true': True, 't': True, 'false': False, 'f': False}
            value = booleans.get(values[0].lower().strip().strip('.'))
            if value is None or value != spin['spinor']:
                raise ValueError('WIN spinors disagrees with the packaged spin mode')
        channels = re.findall(r'^[ \t]*spin[ \t]*[=:][ \t]*([^\n]*)$', uncommented, re.I | re.M)
        if len(channels) > 1:
            raise ValueError('WIN spin must not be duplicated')
        if channels:
            selected = {'up': 1, 'down': 2}.get(channels[0].lower().strip())
            if selected is None:
                raise ValueError('WIN spin must be up or down')
            if spin['source_ispin'] == 2 and selected != spin['spin_channel']:
                raise ValueError('WIN spin disagrees with the packaged selected channel')
    return dict(spin)


def _read_bound_package(folder, source_hashes):
    from .symmetry import _check_coefficient_closure
    folder = Path(folder).resolve()
    report_bytes = (folder / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    ready = report.get('status') == 'ready' and report.get('sawf_ready') is True
    previous_acceptance = report.get('physical_acceptance_status')
    if ready and previous_acceptance == 'pending_coefficient_closure':
        raise ValueError('Symmetry package claims both ready status and unaccepted coefficient closure')
    legacy = (report.get('status') == 'numerical_trial' and report.get('sawf_ready') is False
              and previous_acceptance == 'pending_coefficient_closure')
    if (report.get('schema') != 'sawf-bridge-bloch-v1' or not (ready or legacy)
            or report.get('numerical_checks_passed') is not True
            or report.get('source_hashes') != source_hashes):
        raise ValueError('Symmetry package is not fully validated or does not match this interface provenance')
    content = (folder / 'bloch.npz').read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != report.get('bloch_sha256') or len(content) != report.get('bloch_bytes'):
        raise ValueError('Bloch symmetry matrix package SHA256 or byte count does not match')
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    for key, value in arrays.items():
        if value.dtype.hasobject or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
            raise ValueError(f'Symmetry package field {key} is nonfinite or not a safe array')
    _validate_spin_metadata(arrays, report)
    closure = report.get('residuals',{}).get('ibz_coefficient_closure_relative_max')
    if not isinstance(closure,(int,float)) or isinstance(closure,bool):
        raise ValueError('Symmetry package is missing a valid coefficient-closure record')
    recorded = report.get('coefficient_closure',{})
    if 'raw_value' in recorded and recorded['raw_value'] != closure:
        raise ValueError('Duplicate coefficient-closure residual records in the symmetry package disagree')
    decision = _check_coefficient_closure(report,closure)
    if previous_acceptance == 'accepted_for_single_particle_model' and decision is None:
        raise ValueError('The claimed physical acceptance in the symmetry package does not match its actual provenance')
    if legacy and (decision is None or digest != decision.get('reviewed_legacy_bloch_sha256')):
        raise ValueError('The legacy trial package is not covered by the current approved decision')
    report['source_report_status'] = report['status']
    report.update(status='ready',sawf_ready=True,gauge_status='verified_mmn_anchor_transport')
    report['_read_report_sha256'] = hashlib.sha256(report_bytes).hexdigest()
    return arrays, report


def _gate(report, key, value, tolerance=1e-6):
    report[key] = float(value)
    if not np.isfinite(value) or value > tolerance:
        raise ValueError(f'{key} failed the numerical check: {value} > {tolerance}')


def _new_output(output_dir, source_files):
    requested = Path(output_dir)
    output = requested.resolve()
    parents = [Path(p).resolve().parent for p in source_files]
    if requested.is_symlink() or output.exists() or any(output.is_relative_to(p) for p in parents):
        raise ValueError('Run output must be a new directory outside protected input directories; overwriting is forbidden')
    output.mkdir(parents=True, exist_ok=False)
    return output


def _ordinary_initial_gauge(data, *, num_iter=1000):
    """Localize original AMN in memory to supply U and centers for initial-guess alignment without additional DFT input."""
    from wannierberri.wannierise.wannierise import wannierise
    originals = {name: {k: a.copy() for k, a in data.get_file(name).data.items()}
                 for name in ('amn', 'mmn', 'eig')}
    transcript = io.StringIO()
    with redirect_stdout(transcript):
        wannierise(data, init='amn', sitesym=False, parallel=False, irreducible=False,
                    localise=True, num_iter=num_iter, conv_tol=1e-9)
    matches = re.findall(r'^Converged after (\d+) iterations$', transcript.getvalue(), re.M)
    deviations = re.findall(r'^standard deviation = (\S+)$', transcript.getvalue(), re.M)
    delta = float(deviations[-1]) if deviations else float('nan')
    if len(matches) != 1 or not np.isfinite(delta) or delta >= 1e-9:
        raise ValueError('Initial localization of original AMN did not converge')
    nk, nb = data.mmn.NK, data.mmn.NB
    if set(data.chk.v_matrix) != set(range(nk)):
        raise ValueError('Initialization is missing the full k-grid gauge')
    gauge = np.array([data.chk.v_matrix[k] for k in range(nk)])
    centers = data.chk.wannier_centers_cart.copy()
    spreads = data.chk.wannier_spreads.copy()
    checked_centers, checked_spreads = data.chk.get_wannier_centers(data.bkvec,data.mmn,spreads=True)
    if gauge.shape != (nk,nb,nb) or not np.isfinite(gauge).all():
        raise ValueError('Invalid initial gauge dimensions or values')
    report = {'converged':True,'iterations':int(matches[0])+1,'final_delta_std':delta,
              'total_spread_angstrom2':float(spreads.sum())}
    _gate(report,'gauge_unitarity',np.max(abs(gauge.conj().swapaxes(-1,-2)@gauge-np.eye(nb))),
          256*np.finfo(float).eps*nb)
    _gate(report,'center_recompute_angstrom',np.max(abs(checked_centers-centers)))
    _gate(report,'spread_recompute_angstrom2',np.max(abs(checked_spreads-spreads)),1e-2)
    for name, before in originals.items():
        after = data.get_file(name).data
        if set(after) != set(before) or any(not np.array_equal(a,after[k]) for k,a in before.items()):
            raise ValueError(f'Initialization changed the original {name} matrix')
    return gauge, centers, report


def run_sawf(seed, symmetry_dir, output_dir, *, center, orbital, num_iter=1000,
             dft_eigenval=None, energy_reference_ev=0.):
    """Initialize from original AMN, align to the target representation, and run SAWF; the DFT path is optional plotting data."""
    center, orbital = _normalize_target_orbits(center, orbital)
    if not np.isfinite(energy_reference_ev):
        raise ValueError('The plot energy reference must be finite')
    seed, symmetry_dir = [Path(p).resolve() for p in (seed, symmetry_dir)]
    sources = [Path(f'{seed}.{ext}') for ext in ('win','amn','mmn','eig')]
    sources += [symmetry_dir/'bloch.npz',symmetry_dir/'report.json']
    if dft_eigenval is not None:
        sources.append(Path(dft_eigenval))
    output = _new_output(output_dir,sources)
    report = dict(schema='sawf-bridge-model-v2',status='not_ready',sawf_ready=False,converged=False,
                  source_seed=str(seed),symmetrize=False,irreducible=False,wavefunction_bytes_read=0)
    started = time.perf_counter()
    transcript = io.StringIO()
    try:
        with redirect_stdout(transcript):
            _run_sawf(seed,symmetry_dir,output,report,num_iter,
                      center,orbital,dft_eigenval,energy_reference_ev)
    except Exception as error:
        report.update(error=str(error),last_output_lines=transcript.getvalue().splitlines()[-15:])
        raise
    finally:
        report.update(seconds=time.perf_counter()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def _run_sawf(seed, symmetry_dir, output, report, num_iter,
              center, orbital, dft_eigenval, energy_reference_ev):
    from irrep.spacegroup import SpaceGroup
    from wannierberri.system.system_w90 import System_w90
    from .bands import evaluate_hamiltonian, evaluate_bands
    from .symmetry import _cell_from_win, _validate_grey_group, _spacegroup_from_context
    from .symmetry import build_symmetry_maps, make_symmetrizer, group_residuals, mmn_translation_phases

    from .inputs import load_wannier_data

    metadata = json.loads((symmetry_dir/'report.json').read_text())
    source_nb = metadata.get('source_num_bands',metadata.get('wavecar',{}).get('num_bands'))
    if not isinstance(source_nb,int) or source_nb < 1:
        raise ValueError('Symmetry package is missing the original WAVECAR band count')
    bundle = read_inputs(seed, source_nb=source_nb)
    arrays, source_report = _read_bound_package(symmetry_dir, bundle.hashes)
    win_text = Path(f'{seed}.win').read_text()
    spin = _validate_spin_metadata(arrays, source_report, win_text=win_text)
    spinor = spin['spinor']
    report['spin'] = spin
    report['antiunitary_scope'] = ('Physical spinful time reversal' if spinor else
        'Orbital complex conjugation with square +1; the implicit spin degeneracy is not expanded'
        if spin['source_ispin'] == 1 else
        'Complex conjugation within the selected collinear spin channel; physical spin-exchanging '
        'time reversal and cross-channel magnetic operations are not imposed')
    report.update(source_hashes=bundle.hashes, bands_vasp_1based=bundle.bands_vasp_1based.tolist(),
                  source_num_bands=source_nb, mesh=bundle.mesh.tolist(),
                  symmetry_package_sha256=source_report['bloch_sha256'], numerical_gate=1e-6,
                  numerical_gate_source='WB1.7.0 Symmetrizer_Uirr default tolerance; not a user-specified physical error criterion')
    report['coefficient_closure_relative_max'] = source_report['residuals']['ibz_coefficient_closure_relative_max']
    report['coefficient_closure_evidence'] = dict(
        relative_frobenius_max=report['coefficient_closure_relative_max'],
        definition='max ||d_g(k)^T WF(k) - transformed_WF(k)||_F / ||transformed_WF(k)||_F',
        independent_ibz_little_checks=source_report['independent_ibz_little_checks'],
        independent_transform='Integer-G dictionary mapping of original full-G/spinor coefficients, independent of IrRep transforms; Gamma also checked against lstsq',
        source_report_sha256=source_report['_read_report_sha256'])
    report['physical_acceptance_status'] = source_report.get('physical_acceptance_status', 'not_specified')
    report['coefficient_closure'] = source_report['coefficient_closure']
    report['symmetry_source_report_status'] = source_report['source_report_status']
    nk,nb,nw = bundle.amn.shape
    if nb != nw or nb < 1 or (spinor and nb % 2):
        raise ValueError('SAWF requires positive NB=NW, with even dimension for a spinor subspace')
    for key, expected in [('kpoints', bundle.kpoints), ('eig', bundle.eig),
                          ('bands_vasp_1based', bundle.bands_vasp_1based)]:
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f'Symmetry package and interface disagree on {key}')
    group_fields = {k[len('spacegroup_'):]: v.item() if v.ndim == 0 else v
                    for k, v in arrays.items() if k.startswith('spacegroup_')}
    sg = SpaceGroup(**group_fields)
    positions, types, _ = _cell_from_win(win_text, bundle.lattice)
    if 'spin' in source_report:
        actual_sg = _spacegroup_from_context(bundle, positions, types, source_report['outcar']['spin_context'])
    else:
        actual_sg = SpaceGroup.from_cell(cell=(bundle.lattice,positions,types), spinor=True,
                                         magmom=True, include_TR=True, verbosity=0)
    for key, value in actual_sg.as_dict().items():
        if not np.array_equal(np.asarray(value), arrays[f'spacegroup_{key}']):
            raise ValueError(f'Packaged space group disagrees with independent reconstruction from this WIN: {key}')
    for key in ('rotations', 'translations', 'time_reversals', 'spinor_rotations'):
        if not np.array_equal(arrays[key], arrays[f'spacegroup_{key}']):
            raise ValueError(f'Duplicate symmetry-operation field records disagree: {key}')
    kmap, edge_map = build_symmetry_maps(bundle, sg)
    if not np.array_equal(kmap, arrays['kmap']):
        raise ValueError('Packaged k-point/G mappings disagree with the actual interface')
    d = arrays['d']
    edge_phases = mmn_translation_phases(bundle, sg)
    covariance = []
    for s,anti in enumerate(arrays['time_reversals']):
        mg = bundle.mmn[kmap[s,:,None],edge_map[s]]
        reconstructed = d[s,:,None].conj().swapaxes(-1,-2) @ mg @ d[s,bundle.neighbor_indices]
        expected = edge_phases[s, :, :, None, None]*(bundle.mmn.conj() if anti else bundle.mmn)
        covariance.append(np.max(abs(reconstructed-expected)))
    _gate(report, 'bloch_all_native_MMN_covariance_max', np.max(covariance))
    _gate(report, 'bloch_energy_covariance_eV',
          np.max(abs(bundle.eig[kmap,:,None]*d-d*bundle.eig[None,:,None,:])))
    _gate(report, 'bloch_unitarity_max', np.max(abs(d.conj().swapaxes(-1,-2) @ d - np.eye(nb))))
    sym = make_symmetrizer(bundle, sg, d)
    product = group_residuals(d, kmap, sym.time_reversals, sym.sym_product_table, sym.spinor_factors,
                              kpoints=bundle.kpoints, translations_diff=sym.translations_diff)
    _gate(report, 'bloch_group_composition_max', product['group_composition_max'])
    t = _validate_grey_group(sg)
    square_key = 'bloch_TR_squared_plus_identity_max' if spinor else 'bloch_K_squared_minus_identity_max'
    _gate(report, square_key, np.max(abs(d[t, kmap[t]] @ d[t].conj()-spin['time_reversal_square']*np.eye(nb))))
    target_blocks, target_centers, target_report = _target_representation(sym, center, orbital)
    if sym.num_wann != nb:
        raise ValueError(f'Target representation produces {sym.num_wann} functions, inconsistent with {nb} target bands')
    chars = [abs(np.trace(d[s,k])-sum(np.trace(block) for block in sym.D_wann_blocks[i][s]))
             for i,k in enumerate(sym.kptirr) for s in sym.isym_little[i] if not sym.time_reversals[s]]
    _gate(report, 'target_unitary_character_max', np.max(chars))
    report['target'] = target_report
    D = np.array([[_target_matrix(target_blocks, bundle.kpoints[k], bundle.kpoints[kmap[s,k]], s)
                   for k in range(sym.NK)] for s in range(sym.Nsym)])
    target_product = group_residuals(D, kmap, sym.time_reversals, sym.sym_product_table, sym.spinor_factors,
                                    kpoints=bundle.kpoints, translations_diff=sym.translations_diff)
    _gate(report, 'target_group_composition_max', target_product['group_composition_max'])
    path = geometry = None
    if dft_eigenval is not None:
        from .bands import read_dft_eigenval, read_win_path, path_geometry
        path = read_dft_eigenval(dft_eigenval,bundle.bands_vasp_1based,
                                spin_channel=spin['spin_channel'], source_ispin=spin['source_ispin'])
        geometry = path_geometry(path['kpoints'],read_win_path(f'{seed}.win'),bundle.lattice)
    data, _ = load_wannier_data(seed,source_nb=source_nb)
    ordinary_u, ordinary_centers, ordinary_report = _ordinary_initial_gauge(data,num_iter=num_iter)
    ordinary_system = None
    if dft_eigenval is not None:
        ordinary_system = System_w90(data,symmetrize=False,fftlib='numpy',spinor=spinor,
                                     berry=False,morb=False,spin=False,wannier_centers_from_chk=True)
    initial, alignment = align_scdm_initial_gauge(bundle.amn,ordinary_u,ordinary_centers,
                                                 bundle.lattice,bundle.kpoints,sym,
                                                 target_centers=target_centers @ bundle.lattice)
    alignment['ordinary_provenance'] = 'Official unconstrained localization from original AMN in the same process; all three input matrices remain elementwise unchanged'
    report['initial_alignment'] = alignment
    report['ordinary_initialization'] = ordinary_report
    report['localisation'] = wannierise_strict(data, sym, frozen_all=True, num_iter=num_iter, initial_gauge=initial)
    u = np.array([data.chk.v_matrix[k] for k in range(sym.NK)])
    cov, hcov = [], []
    hk = u.conj().swapaxes(-1,-2) @ (bundle.eig[:,:,None]*u)
    for s,anti in enumerate(sym.time_reversals):
        predicted = d[s] @ (u.conj() if anti else u) @ D[s].conj().swapaxes(-1,-2)
        cov.append(float(np.max(abs(u[kmap[s]]-predicted))))
        hpred = D[s] @ (hk.conj() if anti else hk) @ D[s].conj().swapaxes(-1,-2)
        hcov.append(float(np.max(abs(hk[kmap[s]]-hpred))))
    _gate(report, 'all_k_gauge_covariance_max', np.max(cov))
    _gate(report, 'all_k_H_covariance_eV', np.max(hcov))
    centers, spreads = data.chk.get_wannier_centers(data.bkvec, data.mmn, spreads=True)
    report.update(centers_angstrom=centers.tolist(), spreads_angstrom2=spreads.tolist(),
                  total_spread_angstrom2=float(sum(spreads)))
    data.irreducible = False
    system = System_w90(data, symmetrize=False, fftlib='numpy', spinor=spinor,
                        berry=False, morb=False, spin=False, wannier_centers_from_chk=True)
    exact_mesh = np.rint(bundle.kpoints*bundle.mesh)/bundle.mesh
    _gate(report, 'mesh_H_roundtrip_eV', np.max(abs(evaluate_hamiltonian(system, exact_mesh)-hk)))
    _gate(report, 'mesh_spectrum_error_eV', np.max(abs(evaluate_bands(system, exact_mesh)-bundle.eig)))
    q = np.random.default_rng(7).random((13,3))
    hq = evaluate_hamiltonian(system, q)
    residuals = []
    for s,anti in enumerate(sym.time_reversals):
        qg = np.array([sg.symmetries[s].transform_k(k) for k in q])
        Dq = np.array([_target_matrix(target_blocks,k,l,s) for k,l in zip(q,qg)])
        predicted = Dq @ (hq.conj() if anti else hq) @ Dq.conj().swapaxes(-1,-2)
        residuals.append(float(np.max(abs(evaluate_hamiltonian(system,qg)-predicted))))
    _gate(report, 'off_mesh_H_covariance_eV', np.max(residuals))
    report['off_mesh_H_covariance_per_operation_eV'] = residuals
    if spinor:
        trim = np.array([[a,b,c] for a in (0,.5) for b in (0,.5) for c in (0,.5)])
        energies = evaluate_bands(system, trim)
        _gate(report, 'TRIM_Kramers_max_split_eV', np.max(abs(energies[:,1::2]-energies[:,::2])))
    _gate(report, 'target_center_covariance_max_abs_angstrom', _center_covariance(sym, centers))
    report['target_center_displacement_max_abs_angstrom'] = float(np.max(abs(centers-target_centers @ bundle.lattice)))
    if not np.isfinite(spreads).all() or np.any(spreads < 0):
        raise ValueError('Final spreads are nonfinite or negative')
    for ext, expected in bundle.hashes.items():
        if hashlib.sha256(Path(f'{seed}.{ext}').read_bytes()).hexdigest() != expected:
            raise ValueError('Inputs changed during localization')
    model_file = output/'model.npz'
    values = dict(U=u,kpoints=bundle.kpoints,eigenvalues_eV=bundle.eig,
                  centers_angstrom=centers,spreads_angstrom2=spreads,lattice=bundle.lattice,
                  R=system.rvec.iRvec,H_R=system.get_R_mat('Ham'))
    np.savez_compressed(model_file,**values)
    with np.load(model_file,allow_pickle=False) as restored:
        if any(not np.array_equal(restored[k],value) for k,value in values.items()):
            raise ValueError('Model save/readback changed arrays')
    report['model_readback_exact'] = True
    report['fourier_convention'] = 'H(k)=sum_R exp(+2πi k·R)H_R; k uses reduced reciprocal coordinates and R integer lattice coordinates; H_R already includes pair-dependent WS weights and must not be divided by degeneracies again; energies in eV, lattice and centers in angstroms'
    report['symmetry_package_required_for_representation'] = True
    if dft_eigenval is not None:
        eordinary = evaluate_bands(ordinary_system,path['kpoints'])
        esawf = evaluate_bands(system,path['kpoints'])
        np.savez_compressed(output/'bands.npz',distance=geometry['distance_inv_angstrom'],
            segment_slices=np.array(geometry['segment_slices']),
            tick_positions=geometry['tick_positions_inv_angstrom'],tick_labels=geometry['tick_labels'],
            kpoints=path['kpoints'],dft=path['eigenvalues_eV'],wannier=eordinary,sawf=esawf,
            energy_reference_ev=energy_reference_ev,sawf_note='',
            source_ispin=spin['source_ispin'], spin_channel=spin['spin_channel'], spinor=spinor)
        for name,energy in [('wannier',eordinary),('sawf',esawf)]:
            error = np.sort(energy,axis=1)-np.sort(path['eigenvalues_eV'],axis=1)
            report[name+'_path_error_ev'] = dict(max_abs=float(np.max(abs(error))),
                                                rms=float(np.sqrt(np.mean(error**2))))
    report['localisation'].pop('convergence_history',None)
    from .wanproj import write_wanproj
    if spin['source_ispin'] == 2:
        report['wanproj'] = dict(status='not_exported', reason='A standalone collinear spin-channel '
            'model is not a complete two-channel VASP WANPROJ; both channel gauges are required')
    else:
        report['wanproj'] = write_wanproj(
            output/'WANPROJ', U=u, kpoints=bundle.kpoints,
            bands_vasp_1based=bundle.bands_vasp_1based, source_nb=source_nb, mesh=bundle.mesh)
    report.update(model_file='model.npz',model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                  status='ready',sawf_ready=True,
                  numerical_checks_passed=True,converged=True)
