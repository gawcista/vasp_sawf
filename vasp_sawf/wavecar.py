"""Selected-band random access and independently checked G-to-coefficient reassociation."""

from dataclasses import dataclass
from importlib.metadata import version
import numpy as np
from numbers import Integral
import os
from pathlib import Path


class GAssociationError(ValueError):
    pass


@dataclass(frozen=True)
class GAssociation:
    ig: np.ndarray
    coefficients: np.ndarray
    native_to_record: np.ndarray
    corrected_to_record: np.ndarray
    corrected_to_native: np.ndarray


def repair_vasp_g_association(ig, coefficients, *, coefficient_count, rtag, spinor=True):
    """Reassociate coefficients in standard VASP record order, returning independent arrays.

    Input must come from native reading without normalization, rotation, or G truncation; count comes from
    the WAVECAR k record. This checks ordering only; counts alone do not prove correct G enumeration.
    VASP order follows pinned pymatgen 2025.10.7 Wavecar._generate_G_points:
    z outermost, x innermost; each axis lists ascending nonnegative then ascending negative values.
    A sufficiently large FFT box preserves the relative order of retained G points. Only RTAG45200 is supported.
    """
    ig = np.asarray(ig)
    coefficients = np.asarray(coefficients)
    if rtag != 45200:
        raise GAssociationError('This adapter was validated only for RTAG45200; refusing to guess other precisions or formats')
    if not isinstance(spinor, (bool, np.bool_)):
        raise GAssociationError('spinor must be an explicit boolean')
    components = 2 if spinor else 1
    if (ig.ndim != 2 or ig.shape[1] != 6 or len(ig) == 0
            or not np.issubdtype(ig.dtype, np.integer)):
        raise GAssociationError('ig must be a nonempty integer NG-by-6 array')
    ng = len(ig)
    if coefficient_count != components * ng:
        raise GAssociationError('Header coefficient count disagrees with the full G/component count; G truncation is forbidden')
    if (coefficients.ndim != 3 or coefficients.shape[0] == 0
            or coefficients.shape[1:] != (ng, components)
            or coefficients.dtype != np.dtype('complex64')):
        raise GAssociationError('RTAG45200 coefficients must retain complex64 precision and NB-by-NG-by-component layout')
    if not np.all(np.isfinite(coefficients)):
        raise GAssociationError('Original coefficients contain nonfinite values')
    if len(np.unique(ig[:, :3], axis=0)) != ng:
        raise GAssociationError('G list contains duplicates')
    native_to_record = ig[:, 3].copy()
    if not np.array_equal(np.sort(native_to_record), np.arange(ng)):
        raise GAssociationError('Native record indices are not a complete bijection; missing coefficients cannot be recovered')
    row = np.arange(ng)
    if np.any((ig[:, 4] < 0) | (ig[:, 4] > row) | (ig[:, 5] <= row) | (ig[:, 5] > ng)):
        raise GAssociationError('Invalid IrRep energy-shell boundaries')

    g = ig[:, :3]
    record_to_corrected = np.lexsort((g[:, 0], g[:, 0] < 0,
                                       g[:, 1], g[:, 1] < 0,
                                       g[:, 2], g[:, 2] < 0))
    corrected_to_record = np.argsort(record_to_corrected)
    record_to_native = np.argsort(native_to_record)
    corrected_to_native = record_to_native[corrected_to_record]
    corrected_ig = ig.copy()
    corrected_ig[:, 3] = corrected_to_record
    corrected_coefficients = coefficients[:, corrected_to_native, :].copy()
    return GAssociation(corrected_ig, corrected_coefficients, native_to_record,
                        corrected_to_record, corrected_to_native)


def adapt_irrep_kpoint(kpoint, *, coefficient_count, rtag):
    """Return a new Kpoint and an invertible permutation without changing the original object.

    The new object initializes wavefunctions and geometry for explicit symm_matrix calls. It neither copies nor
    recomputes little_group, characters, or irreps, and is not a transparent replacement for a full character workflow.
    """
    if version('irrep') != '2.6.3':
        raise GAssociationError('This adapter was audited only for IrRep2.6.3')
    if kpoint.ik0 is None:
        raise GAssociationError('A Kpoint with its original k-point index is required')
    from irrep.kpoint import Kpoint

    association = repair_vasp_g_association(kpoint.ig, kpoint.WF,
                                           coefficient_count=coefficient_count, rtag=rtag, spinor=kpoint.spinor)
    result = Kpoint(ik=int(kpoint.ik0) - 1, num_bands=kpoint.num_bands,
                    RecLattice=kpoint.RecLattice.copy(), spinor=kpoint.spinor,
                    kpt=kpoint.k.copy(), WF=association.coefficients,
                    Energy=kpoint.Energy_raw.copy(), ig=association.ig,
                    upper=kpoint.upper, normalize=False, eKG=kpoint.eKG.copy())
    result.weight = kpoint.weight
    return result, association


class WavecarReadError(ValueError):
    pass


@dataclass(frozen=True)
class WavecarHeader:
    record_bytes: int
    spin_channels: int
    rtag: int
    num_kpoints: int
    num_bands: int
    cutoff_ev: float
    lattice: np.ndarray
    fermi_ev: float
    file_bytes: int


@dataclass(frozen=True)
class SelectedWavecar:
    header: WavecarHeader
    kpoints: tuple
    bands_1based: tuple[int, ...]
    kpoints_1based: tuple[int, ...]
    read_ledger: tuple[dict, ...]
    source_status: str = 'unproven'
    gauge_status: str = 'unproven'
    spinor: bool = True
    spin_channel: int = 1


@dataclass(frozen=True)
class WavecarMetadata:
    header: WavecarHeader
    bands_1based: tuple[int, ...]
    kpoints_1based: tuple[int, ...]
    kpoints: np.ndarray
    energies: np.ndarray
    coefficient_counts: tuple[int, ...]
    source_identity: dict
    read_ledger: tuple[dict, ...]
    spinor: bool = True
    spin_channel: int = 1


def _source_identity(path, stat):
    return {'resolved_path': str(path), **{field: int(getattr(stat, field)) for field in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}}


def verify_wavecar_source(path, expected_source_identity):
    """Check cheap file identity shared by workers; this is not a content hash."""
    path = Path(path).resolve(strict=True)
    if _source_identity(path, path.stat()) != expected_source_identity:
        raise WavecarReadError('WAVECAR source changed since metadata inspection')


def _verify_open_source(reader, path, expected_source_identity):
    if _source_identity(path, os.fstat(reader.f.fileno())) != expected_source_identity:
        raise WavecarReadError('WAVECAR source changed during reading')
    verify_wavecar_source(path, expected_source_identity)


def _validate_lattice(lattice):
    lattice = np.asarray(lattice, dtype=float)
    if (lattice.shape != (3, 3) or not np.isfinite(lattice).all()
            or np.linalg.slogdet(lattice)[0] == 0):
        raise WavecarReadError('Valid WIN lattice row vectors are required')
    return lattice


def _positive_integer(value, name):
    if not np.isfinite(value) or value < 1 or value != int(value):
        raise WavecarReadError(f'{name} must be a positive integer')
    return int(value)


def _indices(values, limit, name):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise WavecarReadError(f'{name} requires an explicit nonempty integer sequence') from exc
    if (not result or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                          or not 1 <= v <= limit for v in result)
            or len(set(result)) != len(result)):
        raise WavecarReadError(f'{name} must contain distinct in-range 1-based integers')
    return tuple(int(v) for v in result)


def _open_wavecar(path):
    if version('irrep') != '2.6.3':
        raise WavecarReadError('This reader was validated only with IrRep2.6.3')
    from irrep.readfiles import WAVECARFILE

    path = Path(path).resolve(strict=True)
    original_stat = path.stat()

    class RecordedWavecar(WAVECARFILE):
        def __init__(self, filename):
            self.ledger = []
            self.context = {'kind': 'file_header'}
            self.bootstrap = True
            self.verbosity = 0
            self.f = open(filename, 'rb')
            self.rl = 3
            try:
                # IrRep's constructor rejects collinear channels; retain its record methods.
                self.rl, self.ispin, self.iprec = (int(value) for value in self.record(0))
                self.nrec_enocc = None
                self.nrec_kpoint = None
                self.nrec_header = 2
            except Exception:
                if hasattr(self, 'f'):
                    self.f.close()
                raise
            self.bootstrap = False

        def record(self, irec, cnt=np.inf, dtype=float):
            if not self.bootstrap and (not np.isfinite(cnt) or cnt * np.dtype(dtype).itemsize > self.rl):
                raise WavecarReadError('Refusing unbounded reads or reads across records')
            data = super().record(irec, cnt=cnt, dtype=dtype)
            offset = int(irec * self.rl)
            self.ledger.append({**self.context, 'record_index': int(irec), 'offset_bytes': offset,
                                'bytes_returned': int(data.nbytes), 'elements_returned': int(data.size),
                                'dtype': str(data.dtype)})
            expected_count = 3 if self.bootstrap else int(cnt)
            if data.size != expected_count or self.f.tell() != offset + data.nbytes:
                raise WavecarReadError('WAVECAR record is truncated or the read range is inconsistent')
            if self.bootstrap:
                recl, nspin, rtag = (_positive_integer(v, name) for v, name in
                                     zip(data, ('RECL', 'NSPIN', 'RTAG'), strict=True))
                if recl % 8 or recl < 104 or nspin not in (1, 2) or rtag != 45200:
                    raise WavecarReadError('Only valid RTAG45200 WAVECAR files with one or two spin channels are supported')
            return data

    reader = RecordedWavecar(str(path))
    try:
        reader.context = {'kind': 'dimension_header'}
        raw_header = reader.record(1, cnt=13)
        if not np.isfinite(raw_header).all():
            raise WavecarReadError('WAVECAR header contains nonfinite values')
        nk = _positive_integer(raw_header[0], 'NKPTS')
        nb = _positive_integer(raw_header[1], 'NBANDS')
        cutoff = float(raw_header[2])
        wave_lattice = raw_header[3:12].reshape(3, 3).copy()
        if cutoff <= 0 or np.linalg.slogdet(wave_lattice)[0] == 0:
            raise WavecarReadError('Invalid WAVECAR cutoff or lattice')
        if (4 + 3 * nb) * 8 > reader.rl:
            raise WavecarReadError('WAVECAR files with multiple energy records have not been validated; refusing to read')
        reader.set_nrec_kpoint(NBin=nb)
        if reader.nrec_enocc != 1 or original_stat.st_size != reader.rl * (2 + reader.ispin * nk * (1 + nb)):
            raise WavecarReadError('WAVECAR file size disagrees with the supported record layout')
        header = WavecarHeader(reader.rl, reader.ispin, reader.iprec, nk, nb, cutoff,
                               wave_lattice, float(raw_header[12]), original_stat.st_size)
        _verify_open_source(reader, path, _source_identity(path, original_stat))
        return reader, header, original_stat, path
    except Exception:
        reader.f.close()
        raise


def inspect_wavecar(path):
    """Read only 128 logical bytes from two headers to check the supported record layout."""
    reader, header, _, _ = _open_wavecar(path)
    reader.f.close()
    return header


def _spin_components(header, spinor, spin_channel):
    if not isinstance(spinor, (bool, np.bool_)):
        raise WavecarReadError('spinor must be an explicit boolean')
    _indices((spin_channel,), header.spin_channels, 'spin_channel')
    if spinor and header.spin_channels != 1:
        raise WavecarReadError('Two-component spinors require one WAVECAR spin channel')
    return 2 if spinor else 1


def _read_kpoint_metadata(reader, header, kindex, bands, components=2, spin_channel=1):
    reader.context = {'kind': 'kpoint_header', 'kpoint_1based': kindex,
                      'spin_channel_1based': int(spin_channel)}
    stored_index = (spin_channel - 1) * header.num_kpoints + kindex - 1
    record = reader.record(reader.irec_start_k(stored_index), cnt=4 + 3 * header.num_bands)
    if not np.isfinite(record).all():
        raise WavecarReadError('WAVECAR k record contains nonfinite values')
    count = _positive_integer(record[0], 'coefficient count')
    if count % components or count * 8 > reader.rl:
        raise WavecarReadError('Invalid two-component coefficient count or record byte range' if components == 2
                                else 'Invalid scalar coefficient count or record byte range')
    k = record[1:4].copy()
    energies = record[4:].reshape(header.num_bands, 3)[np.array(bands) - 1, 0].copy()
    return count, k, energies


def inspect_selected_wavecar(path, *, bands_1based, lattice, kpoints_1based=None,
                             spinor=True, spin_channel=1):
    """Read headers and selected-channel energy records, preserving original band/k order."""
    lattice = _validate_lattice(lattice)
    reader, header, original_stat, path = _open_wavecar(path)
    identity = _source_identity(path, original_stat)
    try:
        components = _spin_components(header, spinor, spin_channel)
        if not np.allclose(header.lattice, lattice, rtol=0, atol=1e-8):
            raise WavecarReadError('WAVECAR lattice disagrees with interface WIN')
        bands = _indices(bands_1based, header.num_bands, 'bands_1based')
        kindices = (tuple(range(1, header.num_kpoints + 1)) if kpoints_1based is None else
                    _indices(kpoints_1based, header.num_kpoints, 'kpoints_1based'))
        records = [_read_kpoint_metadata(reader, header, ik, bands, components, spin_channel) for ik in kindices]
        _verify_open_source(reader, path, identity)
        return WavecarMetadata(header, bands, kindices, np.array([row[1] for row in records]),
                               np.array([row[2] for row in records]), tuple(row[0] for row in records),
                               identity, tuple(reader.ledger), bool(spinor), int(spin_channel))
    finally:
        reader.f.close()


def estimate_wavecar_memory(metadata, *, workers=1, working_complex128_copies=6):
    """Estimate one-k worker storage; allocator, G enumeration and LAPACK peaks need measurement."""
    if (isinstance(workers, (bool, np.bool_)) or not isinstance(workers, Integral)
            or workers < 1):
        raise WavecarReadError('workers must be a positive integer')
    if (isinstance(working_complex128_copies, (bool, np.bool_))
            or not isinstance(working_complex128_copies, Integral) or working_complex128_copies < 1):
        raise WavecarReadError('working_complex128_copies must be a positive integer')
    nb = len(metadata.bands_1based)
    counts = metadata.coefficient_counts
    max_count = max(counts)
    components = 2 if metadata.spinor else 1
    ng = max_count // components
    active_workers = min(int(workers), len(counts))
    per_worker = {
        'raw_complex64_bytes': nb * max_count * 8,
        'one_complex128_copy_bytes': nb * max_count * 16,
        'working_complex128_copies': int(working_complex128_copies),
        'g_geometry_bytes': ng * (6 * 8 + 8),
        'g_index_workspace_bytes': ng * 8 * 8,
        'python_g_map_allowance_bytes': ng * 320,
        'interpreter_allowance_bytes': 1024**3,
    }
    per_worker_total = (per_worker['raw_complex64_bytes']
                        + working_complex128_copies * per_worker['one_complex128_copy_bytes']
                        + sum(per_worker[key] for key in ('g_geometry_bytes', 'g_index_workspace_bytes',
                                                         'python_g_map_allowance_bytes',
                                                         'interpreter_allowance_bytes')))
    return {'model': 'selected-k-worker-v1', 'is_estimate': True,
            'workers_requested': int(workers), 'workers': active_workers,
            'stored_kpoints': len(counts), 'selected_bands': nb, 'spinor_components': components,
            'spin_channel': metadata.spin_channel,
            'max_gvectors': ng, 'coefficient_count_includes_spinor': True,
            'selected_coefficient_read_bytes': sum(counts) * nb * 8,
            'per_worker': per_worker, 'per_worker_estimated_bytes': int(per_worker_total),
            'all_workers_estimated_bytes': int(per_worker_total * active_workers),
            'limitations': 'Heuristic working-copy and Python allowances; not a guaranteed peak. '
                           'The 1 GiB import allowance exceeds the measured 588 MiB small-fixture metadata process. '
                           'Excludes parent-process state and filesystem cache. Measure RSS on allocated hardware.'}


def read_selected_wavecar(path, *, bands_1based, kpoints_1based, lattice, expected_source_identity=None,
                          spinor=True, spin_channel=1):
    """Read selected k/band records only; return unnormalized official Kpoints and a read ledger.

    lattice contains row vectors in angstroms read by the caller from interface WIN; it must match WAVECAR.
    Explicit kpoints_1based=None selects all stored k points; band indices must always be explicit.
    No POSCAR dependency, degenerate-state rotation, band dropping, G truncation, PAW overlap, or gauge certification.
    Scalar spin channels are selected explicitly using 1-based spin_channel. Spinors require NSPIN=1.
    Formats other than RTAG45200 and multiple energy records are rejected.
    """
    from irrep.gvectors import calc_gvectors
    from irrep.kpoint import Kpoint

    lattice = _validate_lattice(lattice)
    if expected_source_identity is not None:
        verify_wavecar_source(path, expected_source_identity)
    reader, header, original_stat, path = _open_wavecar(path)
    try:
        components = _spin_components(header, spinor, spin_channel)
        identity = _source_identity(path, original_stat)
        if expected_source_identity is not None and identity != expected_source_identity:
            raise WavecarReadError('WAVECAR source changed since metadata inspection')
        if not np.allclose(header.lattice, lattice, rtol=0, atol=1e-8):
            raise WavecarReadError('WAVECAR lattice disagrees with interface WIN')
        nb, nk, cutoff = header.num_bands, header.num_kpoints, header.cutoff_ev
        bands = _indices(bands_1based, nb, 'bands_1based')
        kindices = tuple(range(1, nk + 1)) if kpoints_1based is None else _indices(kpoints_1based, nk, 'kpoints_1based')
        reciprocal = np.linalg.inv(header.lattice).T * (2 * np.pi)
        kpoints = []
        for kindex in kindices:
            count, k, energy = _read_kpoint_metadata(reader, header, kindex, bands, components, spin_channel)
            ig, e_kg = calc_gvectors(k, reciprocal, cutoff, nplane=count // components,
                                     Ecut1=cutoff, spinor=spinor, verbosity=0)
            if len(ig) * components != count:
                raise WavecarReadError('Independent G enumeration count disagrees with the complete coefficient record')
            raw = np.empty((len(bands), count // components, components), dtype=np.complex64)
            for index, band in enumerate(bands):
                reader.context = {'kind': 'coefficients', 'kpoint_1based': kindex, 'band_1based': band,
                                  'spin_channel_1based': int(spin_channel)}
                stored_index = (spin_channel - 1) * nk + kindex - 1
                coefficients = reader.record_k_band(stored_index, band - 1, cnt=count)
                raw[index] = coefficients.reshape(count // components, components, order='F')
            association = repair_vasp_g_association(ig, raw[:, ig[:, 3], :],
                                                    coefficient_count=count, rtag=reader.iprec, spinor=spinor)
            point = Kpoint(ik=kindex - 1, num_bands=len(bands), RecLattice=reciprocal.copy(),
                           spinor=bool(spinor), kpt=k, WF=association.coefficients, Energy=energy,
                           ig=association.ig, upper=None, normalize=False, eKG=e_kg)
            kpoints.append(point)
        _verify_open_source(reader, path, identity)
        return SelectedWavecar(header, tuple(kpoints), bands, kindices, tuple(reader.ledger),
                               spinor=bool(spinor), spin_channel=int(spin_channel))
    finally:
        reader.f.close()
