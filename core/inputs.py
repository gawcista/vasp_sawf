"""Original interface parsing, explicit neighbors, and WannierBerri adaptation."""

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
from pathlib import Path
import re
import numpy as np
import importlib.metadata


# Coordinate serialization tolerance in grid units, not a physical residual gate.
_GRID_TOL = 1e-8
# CODATA 2022, matching the audited SciPy 1.17.0 / WannierBerri conversion.
_BOHR_ANGSTROM = 0.529177210544


class InputValidationError(ValueError):
    """Missing input, ambiguous format, or inconsistent internal indices."""


@dataclass(frozen=True)
class InputBundle:
    """Lattice units are angstroms; matrices retain input precision and neighbor indices are zero-based."""

    seed: Path
    lattice: np.ndarray
    kpoints: np.ndarray
    mesh: np.ndarray
    amn: np.ndarray
    eig: np.ndarray
    mmn: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_shifts: np.ndarray
    bands_vasp_1based: np.ndarray | None
    hashes: dict[str, str]
    warnings: tuple[str, ...]
    mmn_reverse_max_abs: float
    source_status: str = "unproven"
    gauge_status: str = "unproven"


def _fail(message):
    raise InputValidationError(message)


def _integer(token, context):
    if not re.fullmatch(r"[+-]?\d+", token):
        _fail(f"{context}: expected an integer, got {token!r}")
    return int(token)


def _numbers(line, count, context, *, integers=False):
    tokens = line.split()
    if len(tokens) != count:
        _fail(f"{context}: expected {count} columns, got {len(tokens)}")
    if integers:
        return [_integer(t, context) for t in tokens]
    try:
        result = [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
    except ValueError:
        _fail(f"{context}: contains an unparseable number")
    if not all(math.isfinite(v) for v in result):
        _fail(f"{context}: contains nonfinite values")
    return result


def _win_sections(text):
    scalars, blocks = {}, {}
    current = None
    relevant = {"num_bands", "num_wann", "mp_grid", "exclude_bands"}
    for original in text.splitlines():
        line = re.split(r"[!#]", original, maxsplit=1)[0].strip()
        if not line:
            continue
        delimiter = re.fullmatch(r"(begin|end)\s*:?\s+(\w+)", line, re.I)
        if delimiter:
            action, name = (part.lower() for part in delimiter.groups())
            if action == "begin":
                if current is not None or name in blocks:
                    _fail(f"WIN: duplicate or nested block {name}")
                current = name
                blocks[name] = []
            else:
                if name != current:
                    _fail(f"WIN: mismatched end marker for block {name}")
                current = None
        elif current is not None:
            blocks[current].append(line)
        else:
            assignment = re.fullmatch(r"(\w+)\s*[=:]\s*(.*)", line)
            if assignment and assignment[1].lower() in relevant:
                key, value = assignment[1].lower(), assignment[2]
                if key in scalars:
                    _fail(f"WIN: duplicate parameter {key}")
                scalars[key] = value
    if current is not None:
        _fail(f"WIN: unterminated block {current}")
    for name in ("num_bands", "num_wann", "mp_grid"):
        if name not in scalars:
            _fail(f"WIN: missing parameter {name}")
    for name in ("unit_cell_cart", "kpoints"):
        if name not in blocks:
            _fail(f"WIN: missing block {name}")
    return scalars, blocks


def _read_win(text):
    scalars, blocks = _win_sections(text)
    nb = _integer(scalars["num_bands"], "WIN num_bands")
    nw = _integer(scalars["num_wann"], "WIN num_wann")
    if not 0 < nw <= nb:
        _fail("WIN: requires 0 < num_wann <= num_bands")
    mesh = np.array(_numbers(scalars["mp_grid"], 3, "WIN mp_grid", integers=True), dtype=int)
    if np.any(mesh <= 0):
        _fail("WIN: each mp_grid dimension must be a positive integer")
    rows = blocks["unit_cell_cart"]
    unit = "ang"
    if rows and len(rows[0].split()) == 1:
        unit, rows = rows[0].lower(), rows[1:]
    if unit not in ("ang", "bohr"):
        _fail(f"WIN: unsupported lattice unit {unit!r}; only ang or bohr is supported")
    if len(rows) != 3:
        _fail("WIN: lattice must contain exactly three row vectors")
    lattice = np.array([_numbers(row, 3, "WIN lattice") for row in rows])
    if unit == "bohr":
        lattice *= _BOHR_ANGSTROM
    if np.linalg.slogdet(lattice)[0] == 0:
        _fail("WIN: lattice matrix is singular")
    kpoints = np.array([_numbers(row, 3, "WIN kpoints") for row in blocks["kpoints"]])
    nk = math.prod(int(n) for n in mesh)
    if kpoints.shape != (nk, 3):
        _fail(f"WIN: full grid requires {nk} three-dimensional k points, got shape {kpoints.shape}")
    scaled = (kpoints - kpoints[0]) * mesh
    rounded = np.rint(scaled)
    if np.max(np.abs(scaled - rounded)) > _GRID_TOL:
        _fail("WIN: k points do not form the uniformly shifted grid specified by mp_grid")
    addresses = rounded.astype(np.int64) % mesh
    if len(set(map(tuple, addresses))) != nk:
        _fail("WIN: reciprocal-equivalent k points are duplicated, leaving points missing from the full grid")
    return lattice, kpoints, mesh, nb, nw, scalars.get("exclude_bands", "")


def _source_bands(exclude, source_nb, nb):
    excluded = set()
    if exclude.strip():
        normalized = re.sub(r"\s*([-:])\s*", r"\1", exclude.strip())
        for token in re.split(r"[,;\s]+", normalized):
            match = re.fullmatch(r"(\d+)(?:[-:](\d+))?", token)
            if match is None:
                _fail(f"WIN exclude_bands: invalid range {token!r}")
            start = int(match[1])
            end = int(match[2] or match[1])
            if not 1 <= start <= end:
                _fail("WIN exclude_bands: ranges must be ascending and indexed from 1")
            excluded.update(range(start, end + 1))
    if source_nb is None:
        return None
    if isinstance(source_nb, bool) or not isinstance(source_nb, Integral) or source_nb <= 0:
        _fail("source_nb: original band count must be a positive integer")
    if any(i > source_nb for i in excluded):
        _fail("WIN exclude_bands: excluded bands exceed the explicitly specified original band count")
    bands = np.array([i for i in range(1, source_nb + 1) if i not in excluded], dtype=int)
    if len(bands) != nb:
        _fail(f"Original band mapping mismatch: {len(bands)} bands after exclusion, {nb} bands in the compact interface")
    return bands


def _body(text, name, dimensions):
    lines = text.splitlines()
    if len(lines) < 2:
        _fail(f"{name}: missing title or dimension line")
    shape = _numbers(lines[1], len(dimensions), f"{name} dimensions", integers=True)
    if tuple(shape) != tuple(dimensions):
        _fail(f"{name}: dimensions {shape} disagree with WIN dimensions {list(dimensions)}")
    return [line for line in lines[2:] if line.strip()]


def _read_amn(text, nb, nk, nw):
    lines = _body(text, "AMN", (nb, nk, nw))
    if len(lines) != nk * nb * nw:
        _fail("AMN: missing or extra data lines")
    result = np.empty((nk, nb, nw), dtype=np.complex128)
    seen = set()
    for line in lines:
        tokens = line.split()
        if len(tokens) != 5:
            _fail("AMN: each line must have exactly five columns")
        b, w, k = [_integer(t, "AMN indices") for t in tokens[:3]]
        if not (1 <= b <= nb and 1 <= w <= nw and 1 <= k <= nk):
            _fail("AMN: index out of bounds")
        key = (k - 1, b - 1, w - 1)
        if key in seen:
            _fail(f"AMN: duplicate index {(b, w, k)}")
        seen.add(key)
        real, imag = _numbers(" ".join(tokens[3:]), 2, "AMN matrix")
        result[key] = complex(real, imag)
    return result


def _read_eig(text, nb, nk):
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != nk * nb:
        _fail("EIG: missing or extra data lines")
    result = np.empty((nk, nb))
    seen = set()
    for line in lines:
        tokens = line.split()
        if len(tokens) != 3:
            _fail("EIG: each line must have exactly three columns")
        b, k = [_integer(t, "EIG indices") for t in tokens[:2]]
        if not (1 <= b <= nb and 1 <= k <= nk):
            _fail("EIG: index out of bounds")
        key = (k - 1, b - 1)
        if key in seen:
            _fail(f"EIG: duplicate index {(b, k)}")
        seen.add(key)
        result[key] = _numbers(tokens[2], 1, "EIG energy")[0]
    return result


def _read_mmn(text, nb, kpoints, mesh):
    nk = len(kpoints)
    header = text.splitlines()
    if len(header) < 2:
        _fail("MMN: missing title or dimension line")
    _, _, nnb = _numbers(header[1], 3, "MMN dimensions", integers=True)
    if nnb <= 0:
        _fail("MMN: neighbor count must be a positive integer")
    lines = _body(text, "MMN", (nb, nk, nnb))
    block_size = 1 + nb * nb
    if len(lines) != nk * nnb * block_size:
        _fail("MMN: neighbor blocks are missing, truncated, or contain extra lines")
    matrices = np.empty((nk, nnb, nb, nb), dtype=np.complex128)
    neighbors = np.empty((nk, nnb), dtype=int)
    shifts = np.empty((nk, nnb, 3), dtype=int)
    counts = np.zeros(nk, dtype=int)
    edges = {}
    displacement_sets = [set() for _ in range(nk)]
    for offset in range(0, len(lines), block_size):
        source, dest, *shift = _numbers(lines[offset], 5, "MMN neighbor indices", integers=True)
        if not (1 <= source <= nk and 1 <= dest <= nk):
            _fail("MMN: neighbor k-point index out of bounds")
        key = (source, dest, *shift)
        if key in edges:
            _fail(f"MMN: duplicate neighbor edge {key}")
        k, n = source - 1, counts[source - 1]
        if n >= nnb:
            _fail(f"MMN: source k point {source} has more neighbors than declared")
        values = np.array([_numbers(row, 2, "MMN matrix") for row in lines[offset + 1:offset + block_size]])
        matrix = (values[:, 0] + 1j * values[:, 1]).reshape(nb, nb, order="F")
        matrices[k, n] = matrix
        neighbors[k, n] = dest - 1
        shifts[k, n] = shift
        counts[k] += 1
        edges[key] = matrix
        displacement = (kpoints[dest - 1] + shift - kpoints[k]) * mesh
        integer_displacement = np.rint(displacement)
        if np.max(np.abs(displacement - integer_displacement)) > _GRID_TOL:
            _fail("MMN: neighbor displacement disagrees with the full k-point grid")
        step = tuple(integer_displacement.astype(np.int64))
        if not any(step) or step in displacement_sets[k]:
            _fail("MMN: zero displacement or duplicate neighbor displacement")
        displacement_sets[k].add(step)
    if np.any(counts != nnb):
        _fail("MMN: not every k point has a complete neighbor set")
    if any(steps != displacement_sets[0] for steps in displacement_sets):
        _fail("MMN: full neighbor/G displacement sets differ between k points; check k-point order and G")
    reverse_max = 0.0
    for (source, dest, x, y, z), matrix in edges.items():
        reverse = edges.get((dest, source, -x, -y, -z))
        if reverse is None:
            _fail(f"MMN: neighbor edge {(source, dest, x, y, z)} has no reverse edge")
        reverse_max = max(reverse_max, float(np.max(np.abs(matrix - reverse.conj().T))))
    return matrices, neighbors, shifts, reverse_max


def read_inputs(seed: str | Path, *, source_nb: int | None = None) -> InputBundle:
    """Read four interface files from an absolute seed; source_nb only determines original band indices.

    Do not reapply exclude_bands to compact AMN/EIG/MMN. Without the original band count,
    bands_vasp_1based is None; the largest excluded index does not determine the band count.
    Reverse-edge matrix differences are diagnostic only, without physical acceptance thresholds.
    """
    seed = Path(seed)
    if not seed.is_absolute():
        _fail("seed must be an absolute prefix path including its directory")
    files, hashes = {}, {}
    for suffix in ("win", "amn", "eig", "mmn"):
        path = Path(f"{seed}.{suffix}")
        try:
            raw = path.read_bytes()
            files[suffix] = raw.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise InputValidationError(f"Cannot read interface file {path}: {error}") from error
        hashes[suffix] = hashlib.sha256(raw).hexdigest()
    lattice, kpoints, mesh, nb, nw, exclude = _read_win(files["win"])
    bands = _source_bands(exclude, source_nb, nb)
    amn = _read_amn(files["amn"], nb, len(kpoints), nw)
    eig = _read_eig(files["eig"], nb, len(kpoints))
    mmn, neighbors, shifts, reverse_max = _read_mmn(files["mmn"], nb, kpoints, mesh)
    warnings = [
        "Consistent input indices and grids do not prove wavefunction provenance or Bloch gauge; neither is verified here.",
        "MMN reverse-conjugation residuals are diagnostic only; no physical acceptance threshold is set.",
        "Interface arrays use compact dimensions; exclude_bands only maps source indices and is not reapplied.",
    ]
    if bands is None:
        warnings.append("source_nb was not provided, so the original VASP band mapping is unknown; the final band index is not inferred from exclude_bands.")
    return InputBundle(
        seed=seed, lattice=lattice, kpoints=kpoints, mesh=mesh,
        amn=amn, eig=eig, mmn=mmn, neighbor_indices=neighbors,
        neighbor_shifts=shifts, bands_vasp_1based=bands, hashes=hashes,
        warnings=tuple(warnings), mmn_reverse_max_abs=reverse_max,
    )


def from_mmn(bundle, reciprocal):
    from wannierberri.w90files.bkvectors import BKVectors

    # These are the pinned WB defaults, not project physical acceptance gates.
    shell_tolerance = 1e-7
    completeness_tolerance = 1e-5
    reduced = bundle.kpoints[bundle.neighbor_indices] + bundle.neighbor_shifts - bundle.kpoints[:, None, :]
    scaled = reduced * bundle.mesh
    steps = np.rint(scaled).astype(np.int64)
    if np.max(abs(scaled-steps)) > _GRID_TOL:
        raise InputValidationError("MMN displacements cannot be mapped to integer grid steps")
    reference = set(map(tuple, steps[0]))
    nnb = steps.shape[1]
    if (0, 0, 0) in reference or len(reference) != nnb:
        raise InputValidationError("MMN contains zero displacement or duplicate directions")
    if any(set(map(tuple, row)) != reference for row in steps):
        raise InputValidationError("MMN direction sets differ between k points")

    cartesian = steps[0] @ (reciprocal / bundle.mesh[:, None])
    if not np.all(np.isfinite(cartesian)):
        raise InputValidationError("MMN neighbor geometry contains nonfinite values")
    shells_grid, shells_cart = BKVectors.k_to_shells(steps[0], cartesian, kmesh_tol=shell_tolerance)
    flattened = np.vstack(shells_grid)
    if len(flattened) != nnb or set(map(tuple, flattened)) != reference:
        raise InputValidationError("Official shell grouping lost original MMN directions; refusing to drop neighbors silently")
    design = np.array([vectors.T @ vectors for vectors in shells_cart]).reshape(len(shells_cart), 9)
    singular = np.linalg.svd(design, compute_uv=False)
    rank_tolerance = singular[0] * max(design.shape) * np.finfo(design.dtype).eps
    rank = int(np.count_nonzero(singular > rank_tolerance))
    if rank != len(shells_cart):
        raise InputValidationError("Original MMN shell matrix is rank deficient and weights are not unique; refusing to invert tiny singular values or drop shells")
    try:
        weights, _, ordered_steps = BKVectors.get_shell_weights(
            shells_grid, shells_cart, bk_complete_tol=completeness_tolerance,
        )
    except RuntimeError as exc:
        raise InputValidationError("Original MMN directions fail the official finite-difference completeness check") from exc
    if not np.all(np.isfinite(weights)):
        raise InputValidationError("MMN weights contain nonfinite values")
    if len(ordered_steps) != nnb or set(map(tuple, ordered_steps)) != reference:
        raise InputValidationError("Official weight solver changed the original MMN direction set")

    neighbors, shifts = {}, {}
    for ik, row in enumerate(steps):
        lookup = {tuple(step): index for index, step in enumerate(row)}
        order = [lookup[tuple(step)] for step in ordered_steps]
        neighbors[ik] = bundle.neighbor_indices[ik, order].copy()
        shifts[ik] = bundle.neighbor_shifts[ik, order].copy()
    result = BKVectors(
        recip_lattice=reciprocal, mp_grid=bundle.mesh, wk=weights, bk_latt=ordered_steps,
        G=shifts, neighbours=neighbors, kpt_latt=np.rint(bundle.kpoints*bundle.mesh).astype(int),
    )
    complete = np.einsum("b,bi,bj->ij", result.wk, result.bk_cart, result.bk_cart)
    residual = float(np.linalg.norm(complete-np.eye(3)))
    if not np.isfinite(residual) or residual > completeness_tolerance:
        raise InputValidationError("Constructed WB neighbors fail the official completeness check")
    return result, {
        "method": "WannierBerri_1.7.0_get_shell_weights_existing_MMN_only",
        "shell_count": len(shells_cart), "rank": rank,
        "singular_values": singular.tolist(), "rank_tolerance": float(rank_tolerance),
        "rank_tolerance_basis": "numpy_matrix_rank_default_max_shape_times_machine_epsilon_times_max_singular",
        "shell_tolerance_inv_angstrom": shell_tolerance,
        "software_completeness_tolerance": completeness_tolerance,
        "physical_acceptance_threshold_set": False,
        "completeness_frobenius_residual": residual,
    }


def load_wannier_data(seed: str | Path, *, source_nb: int | None = None):
    bundle = read_inputs(seed, source_nb=source_nb)
    address = bundle.kpoints * bundle.mesh
    if np.max(abs(address - np.rint(address))) > _GRID_TOL:
        raise InputValidationError("The current WannierBerri adapter does not support shifted grids; independent input parsing still supports valid shifted grids")
    if importlib.metadata.version("wannierberri") != "1.7.0":
        raise InputValidationError("This adapter was audited and validated only with WannierBerri 1.7.0; use the pinned environment")
    from wannierberri.w90files.amn import AMN
    from wannierberri.w90files.eig import EIG
    from wannierberri.w90files.mmn import MMN
    from wannierberri.w90files.w90data import Wannier90data
    from wannierberri.w90files.win import WIN

    data = Wannier90data()
    data.seedname = str(bundle.seed)
    data.write_npz_list = set()
    data.formatted_list = []
    data.set_file("win", WIN(seedname=data.seedname, autoread=True), read_npz=False)
    data.set_chk(read=False)
    if not np.array_equal(data.chk.kpt_latt, bundle.kpoints):
        raise InputValidationError("WannierBerri and independent reader disagree on k-point coordinates or order")
    if not np.array_equal(data.chk.real_lattice, bundle.lattice):
        raise InputValidationError("WannierBerri and independent reader disagree on the lattice")
    nk, nb, nw = bundle.amn.shape
    if (data.chk.num_bands, data.chk.num_wann) != (nb, nw):
        raise InputValidationError("WannierBerri changed the compact band count or Wannier dimension")
    bkvec, weight_report = from_mmn(bundle, data.chk.recip_lattice)
    data.set_file("bkvec", bkvec)
    if data.bkvec.NNB != bundle.mmn.shape[1]:
        raise InputValidationError("WannierBerri neighbor count differs from the original MMN")

    permutations = {}
    reordered = {}
    for ik in range(nk):
        original = [
            (int(neighbor), *map(int, shift))
            for neighbor, shift in zip(bundle.neighbor_indices[ik], bundle.neighbor_shifts[ik])
        ]
        generated = [
            (int(neighbor), *map(int, shift))
            for neighbor, shift in zip(data.bkvec.neighbours[ik], data.bkvec.G[ik])
        ]
        if len(set(generated)) != len(generated) or set(original) != set(generated):
            raise InputValidationError(f"MMN neighbor indices/reciprocal shifts disagree with WannierBerri at k point {ik+1}")
        permutation = np.array([original.index(edge) for edge in generated], dtype=int)
        permutations[ik] = permutation
        reordered[ik] = bundle.mmn[ik, permutation].copy()

    data.set_file("mmn", MMN(data=reordered, bk_reorder=permutations, NK=nk))
    data.set_file("eig", EIG(data=bundle.eig.copy(), NK=nk))
    data.set_file("amn", AMN(data=bundle.amn.copy(), NK=nk))
    differences = {
        "amn_max_abs_difference": float(max(np.max(abs(data.amn.data[k]-bundle.amn[k])) for k in range(nk))),
        "eig_max_abs_difference": float(max(np.max(abs(data.eig.data[k]-bundle.eig[k])) for k in range(nk))),
        "mmn_max_abs_difference": float(max(np.max(abs(data.mmn.data[k]-bundle.mmn[k, permutations[k]])) for k in range(nk))),
    }
    if any(value != 0 for value in differences.values()):
        raise InputValidationError("The adapter changed original AMN/EIG/MMN values")
    after = {
        ext: hashlib.sha256(Path(str(bundle.seed)+"."+ext).read_bytes()).hexdigest()
        for ext in bundle.hashes
    }
    if after != bundle.hashes:
        raise InputValidationError("Source files changed during reading")
    complete = np.einsum("b,bi,bj->ij", data.bkvec.wk, data.bkvec.bk_cart, data.bkvec.bk_cart)
    report = {
        "source_seed": str(bundle.seed), "shape_nk_nb_nw": [nk, nb, nw],
        "mesh": bundle.mesh.tolist(), "neighbor_source": "mmn_explicit_edges_with_win_geometry",
        "weight_solver": weight_report,
        "gauge_status": "unproven", "symmetry_calculation_performed": False,
        "bands_vasp_1based": None if bundle.bands_vasp_1based is None else bundle.bands_vasp_1based.tolist(),
        "neighbor_permutation_wb_to_input": {str(k): v.tolist() for k, v in permutations.items()},
        "bk_grid_units": data.bkvec.bk_latt.tolist(),
        "weights_angstrom2": data.bkvec.wk.tolist(),
        "weight_completeness_max_abs": float(np.max(abs(complete-np.eye(3)))),
        "source_hashes_before": bundle.hashes, "source_hashes_after": after,
        "mmn_reverse_max_abs": bundle.mmn_reverse_max_abs,
        "warnings": list(bundle.warnings), **differences,
    }
    return data, report
