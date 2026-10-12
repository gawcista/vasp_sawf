"""Algebraic reconstruction of finite unitary/antiunitary symmetry actions.

This changes the symmetry matrices, not the Bloch basis. It establishes no
physical closure, target compatibility, Hamiltonian symmetry, or acceptance.
"""

import numpy as np


def _adjoint(value):
    return value.conj().swapaxes(-1, -2)


def _isometry(matrix):
    gram = _adjoint(matrix) @ matrix
    eigenvalues, vectors = np.linalg.eigh((gram + _adjoint(gram)) / 2)
    if eigenvalues[0] <= 100 * np.finfo(float).eps * max(1., eigenvalues[-1]):
        raise ValueError("Embedding loses rank; reconstruction is ambiguous")
    inverse_sqrt = (vectors * eigenvalues**-.5) @ _adjoint(vectors)
    return matrix @ inverse_sqrt


def _integer_array(value, shape, bound, name):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= bound)):
        raise ValueError(f"{name} must contain integer indices with shape {shape}")
    return result.astype(np.intp, copy=False)


def _validate(raw, kmap, anti, product, cocycle):
    try:
        raw = np.asarray(raw, dtype=complex)
        cocycle = np.asarray(cocycle, dtype=complex)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Representation and cocycle must be numeric arrays") from exc
    if (raw.ndim != 4 or min(raw.shape) < 1 or raw.shape[-1] != raw.shape[-2]
            or not np.isfinite(raw).all()):
        raise ValueError("Representation must be finite with shape (ng, nk, nb, nb)")
    ng, nk, nb, _ = raw.shape
    if (cocycle.shape != (ng, ng, nk) or not np.isfinite(cocycle).all()
            or np.max(abs(abs(cocycle) - 1)) > 1e-12):
        raise ValueError("Invalid cocycle shape, finiteness, or unit modulus")
    kmap = _integer_array(kmap, (ng, nk), nk, "k-point map")
    product = _integer_array(product, (ng, ng), ng, "Group product")
    anti = np.asarray(anti)
    if (anti.shape != (ng,) or anti.dtype.kind not in "biuf"
            or not np.isfinite(anti).all() or not np.all((anti == 0) | (anti == 1))):
        raise ValueError("Antiunitary flags must be boolean with shape (ng,)")
    anti = anti.astype(bool, copy=False)
    indices = np.arange(ng)
    identities = np.flatnonzero(np.all(product == indices[None, :], axis=1)
                               & np.all(product == indices[:, None], axis=0))
    if len(identities) != 1:
        raise ValueError("Group product must have a unique two-sided identity")
    if (not np.all(np.sort(product, axis=1) == indices)
            or not np.all(np.sort(product, axis=0) == indices[:, None])):
        raise ValueError("Invalid group multiplication table: inverses are required")
    if not np.all(np.sort(kmap, axis=1) == np.arange(nk)):
        raise ValueError("Invalid k-point permutation")
    if not np.array_equal(anti[product], anti[:, None] ^ anti[None, :]):
        raise ValueError("Inconsistent antiunitary multiplication")
    cocycle_error = 0.
    for g in range(ng):
        for h in range(ng):
            gh = product[g, h]
            if not np.array_equal(product[gh], product[g, product[h]]):
                raise ValueError("Group multiplication is not associative")
            if not np.array_equal(kmap[g, kmap[h]], kmap[gh]):
                raise ValueError("Group action and k-point map disagree")
            # omega(g,h,l k) omega(gh,l,k) = C_g(omega(h,l,k)) omega(g,hl,k).
            left = cocycle[g, h, kmap] * cocycle[gh]
            right = (cocycle[h].conj() if anti[g] else cocycle[h]) * cocycle[g, product[h]]
            cocycle_error = max(cocycle_error, float(np.max(abs(left - right))))
    if cocycle_error > 1e-11:
        raise ValueError("Inconsistent twisted group cocycle")
    return raw, kmap, anti, product, cocycle, cocycle_error


def _algebra_metrics(matrices, kmap, anti, product, cocycle):
    group_max = group_frobenius = unitary_max = 0.
    identity = np.eye(matrices.shape[-1])
    for g in range(len(anti)):
        unitary_max = max(unitary_max, float(np.max(abs(_adjoint(matrices[g]) @ matrices[g] - identity))))
        for h in range(len(anti)):
            residual = matrices[g, kmap[h]] @ (matrices[h].conj() if anti[g] else matrices[h])
            residual -= cocycle[g, h, :, None, None] * matrices[product[g, h]]
            group_max = max(group_max, float(np.max(abs(residual))))
            group_frobenius = max(group_frobenius, float(np.linalg.norm(residual, axis=(-2, -1)).max()))
    return {"group_entry_max": group_max, "group_frobenius_max": group_frobenius,
            "unitarity_entry_max": unitary_max}


def reconstruct_representation(raw, kmap, anti, product, cocycle):
    """Return a nearby algebraically consistent action and a diagnostic report.

    ``raw[g,k]`` is an ``nb x nb`` matrix from k to ``kmap[g,k]``. The inputs
    have shapes ``(ng,nk,nb,nb)``, ``(ng,nk)``, ``(ng,)``, ``(ng,ng)`` and
    ``(ng,ng,nk)`` respectively. ``kmap`` and ``product`` contain integer
    indices; ``anti`` gives complex conjugation C_g. Composition is

        B_g(h k) C_g(B_h(k)) = cocycle[g,h,k] B_product[g,h](k).

    The finite group table, its k action, and the unit-modulus semilinear
    cocycle are validated. The identity may occur at any index; a consistent
    non-normalized identity phase is retained. No input array is modified.

    The approximate action defines an isometric embedding into a twisted
    regular representation with nb copies. Equal-weight group averaging of
    its projector, an isolated rank-nb spectral projection, and polar
    alignment to that embedding give the corrected matrices. Rank loss,
    absent spectral separation, or unreliable numerical algebra raise
    ValueError. All bands are retained; exact actions are fixed to rounding.

    Regular actions are applied as block permutations. Only one dense
    ``(ng*nb,ng*nb)`` fiber is diagonalized at a time. Additional storage is
    O(nk*ng*nb**2 + (ng*nb)**2), including embeddings but excluding inputs,
    outputs and LAPACK workspace. Dense eigensolves cost O(nk*(ng*nb)**3);
    this method is intended for modest group and band counts. The report's
    storage figures describe arrays, not a measured or upper-bound RSS.

    This is not a gauge change, a globally nearest action, or a physical
    acceptance decision. Equal group weights do not define physical error
    priorities. Native MMN/EIG and source-wavefunction errors need separate
    assessment; this function does not receive or modify those inputs.
    """
    raw, kmap, anti, product, cocycle, cocycle_error = _validate(raw, kmap, anti, product, cocycle)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _reconstruct(raw, kmap, anti, product, cocycle, cocycle_error)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError("Representation reconstruction is numerically ill-conditioned") from exc


def _reconstruct(raw, kmap, anti, product, cocycle, cocycle_error):
    ng, nk, nb, _ = raw.shape
    dimension = ng * nb
    inverse_kmap = np.argsort(kmap, axis=1)
    operations = np.arange(ng)

    def apply_regular(g, k, matrix):
        blocks = matrix.reshape(ng, nb, -1)
        phases = cocycle[g, operations, inverse_kmap[:, k]]
        result = np.empty_like(blocks)
        result[product[g]] = phases[:, None, None] * (blocks.conj() if anti[g] else blocks)
        return result.reshape(matrix.shape)

    embedding = np.empty((nk, dimension, nb), complex)
    for k in range(nk):
        blocks = _adjoint(raw[operations, inverse_kmap[:, k]])
        embedding[k] = _isometry(blocks.reshape(dimension, nb) / np.sqrt(ng))

    nearest = np.empty_like(embedding)
    eigenvalues = np.empty((nk, dimension))
    gaps = np.ones(nk)
    for k in range(nk):
        averaged = np.zeros((dimension, dimension), complex)
        for g in range(ng):
            source = inverse_kmap[g, k]
            transformed = apply_regular(g, source, embedding[source])
            averaged += (transformed @ _adjoint(transformed)) / ng
        values, vectors = np.linalg.eigh((averaged + _adjoint(averaged)) / 2)
        eigenvalues[k] = values
        if ng > 1:
            gaps[k] = values[-nb] - values[-nb - 1]
            if gaps[k] <= 100 * np.finfo(float).eps * dimension:
                raise ValueError("Invariant spectral cluster has no isolated rank-NB gap")
        selected = vectors[:, -nb:]
        nearest[k] = _isometry(selected @ (_adjoint(selected) @ embedding[k]))

    corrected = np.empty_like(raw)
    projector_error = isometry_error = 0.
    for g in range(ng):
        for k in range(nk):
            transformed = apply_regular(g, k, nearest[k])
            target = nearest[kmap[g, k]]
            corrected[g, k] = _adjoint(target) @ transformed
            # Evaluate the projector residual by block rows without storing
            # a dense projector for every k point or every operation.
            for start in range(0, dimension, nb):
                stop = start + nb
                residual = transformed[start:stop] @ _adjoint(transformed)
                residual -= target[start:stop] @ _adjoint(target)
                projector_error = max(projector_error, float(np.max(abs(residual))))
    for value in nearest:
        isometry_error = max(isometry_error, float(np.max(abs(_adjoint(value) @ value - np.eye(nb)))))
    raw_metrics = _algebra_metrics(raw, kmap, anti, product, cocycle)
    corrected_metrics = _algebra_metrics(corrected, kmap, anti, product, cocycle)
    numerical_tolerance = max(1e-10, 100 * np.finfo(float).eps * dimension)
    if (corrected_metrics["group_entry_max"] > numerical_tolerance
            or corrected_metrics["unitarity_entry_max"] > numerical_tolerance
            or projector_error > numerical_tolerance):
        raise ValueError("Invariant reconstruction fails numerical group/unitarity checks")
    correction = corrected - raw
    return corrected, {
        "raw": raw_metrics,
        "corrected": corrected_metrics,
        "regular_group_error": cocycle_error,
        "cocycle_relation_max": cocycle_error,
        "projector_gap_min": float(gaps.min()),
        "projector_eigenvalues": eigenvalues.tolist(),
        "selected_cluster_min": float(eigenvalues[:, -nb:].min()),
        "discarded_cluster_max": float(eigenvalues[:, :-nb].max()) if ng > 1 else None,
        "invariant_projector_covariance_max": projector_error,
        "embedding_isometry_max": isometry_error,
        "correction_entry_max": float(np.max(abs(correction))),
        "correction_frobenius_max": float(np.linalg.norm(correction, axis=(-2, -1)).max()),
        "algebra_numerical_tolerance": numerical_tolerance,
        "regular_fiber_dimension": dimension,
        "dense_fiber_array_bytes": dimension**2 * np.dtype(complex).itemsize,
        "embedding_array_bytes": embedding.nbytes,
        "diagnostic_only": True,
        "physical_acceptance": "not_assessed",
        "interpretation": "Algebraically consistent nearby symmetry model, not a Bloch basis change or a physical acceptance decision.",
    }
