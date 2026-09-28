"""Selected-band random access and independently checked G-to-coefficient reassociation."""

from dataclasses import dataclass
from importlib.metadata import version
import numpy as np
from numbers import Integral
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


def repair_vasp_g_association(ig, coefficients, *, coefficient_count, rtag):
    """Reassociate coefficients in standard noncollinear VASP record order, returning independent arrays.

    Input must come from native reading without normalization, rotation, or G truncation; count comes from
    the WAVECAR k record. This checks ordering only; counts alone do not prove correct G enumeration.
    VASP order follows pinned pymatgen 2025.10.7 Wavecar._generate_G_points:
    z outermost, x innermost; each axis lists ascending nonnegative then ascending negative values.
    A sufficiently large FFT box preserves the relative order of retained G points. Only two-component RTAG45200 is supported.
    """
    ig = np.asarray(ig)
    coefficients = np.asarray(coefficients)
    if rtag != 45200:
        raise GAssociationError('This adapter was validated only for RTAG45200; refusing to guess other precisions or formats')
    if (ig.ndim != 2 or ig.shape[1] != 6 or len(ig) == 0
            or not np.issubdtype(ig.dtype, np.integer)):
        raise GAssociationError('ig must be a nonempty integer NG-by-6 array')
    ng = len(ig)
    if coefficient_count != 2 * ng:
        raise GAssociationError('Header coefficient count disagrees with the full two-component G count; G truncation is forbidden')
    if (coefficients.ndim != 3 or coefficients.shape[0] == 0
            or coefficients.shape[1:] != (ng, 2)
            or coefficients.dtype != np.dtype('complex64')):
        raise GAssociationError('RTAG45200 coefficients must retain complex64 precision and NB-by-NG-by-2 layout')
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
    if not kpoint.spinor or kpoint.ik0 is None:
        raise GAssociationError('A two-component Kpoint with its original k-point index is required')
    from irrep.kpoint import Kpoint

    association = repair_vasp_g_association(kpoint.ig, kpoint.WF,
                                           coefficient_count=coefficient_count, rtag=rtag)
    result = Kpoint(ik=int(kpoint.ik0) - 1, num_bands=kpoint.num_bands,
                    RecLattice=kpoint.RecLattice.copy(), spinor=True,
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
            try:
                super().__init__(filename, verbosity=0)
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
                if recl % 8 or recl < 104 or nspin != 1 or rtag != 45200:
                    raise WavecarReadError('Only valid RTAG45200 single-channel spinor WAVECAR files are supported')
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
        if reader.nrec_enocc != 1 or original_stat.st_size != reader.rl * (2 + nk * (1 + nb)):
            raise WavecarReadError('WAVECAR file size disagrees with the supported record layout')
        header = WavecarHeader(reader.rl, 1, reader.iprec, nk, nb, cutoff,
                               wave_lattice, float(raw_header[12]), original_stat.st_size)
        return reader, header, original_stat, path
    except Exception:
        reader.f.close()
        raise


def inspect_wavecar(path):
    """Read only 128 logical bytes from two headers to check the supported record layout."""
    reader, header, _, _ = _open_wavecar(path)
    reader.f.close()
    return header


def read_selected_wavecar(path, *, bands_1based, kpoints_1based, lattice):
    """Read selected k/band records only; return unnormalized official Kpoints and a read ledger.

    lattice contains row vectors in angstroms read by the caller from interface WIN; it must match WAVECAR.
    Explicit kpoints_1based=None selects all stored k points; band indices must always be explicit.
    No POSCAR dependency, degenerate-state rotation, band dropping, G truncation, PAW overlap, or gauge certification.
    Formats other than RTAG45200, multiple spin channels, and multiple energy records are rejected.
    """
    from irrep.gvectors import calc_gvectors
    from irrep.kpoint import Kpoint

    lattice = np.asarray(lattice, dtype=float)
    if (lattice.shape != (3, 3) or not np.isfinite(lattice).all()
            or np.linalg.slogdet(lattice)[0] == 0):
        raise WavecarReadError('Valid WIN lattice row vectors are required')
    reader, header, original_stat, path = _open_wavecar(path)
    try:
        if not np.allclose(header.lattice, lattice, rtol=0, atol=1e-8):
            raise WavecarReadError('WAVECAR lattice disagrees with interface WIN')
        nb, nk, cutoff = header.num_bands, header.num_kpoints, header.cutoff_ev
        bands = _indices(bands_1based, nb, 'bands_1based')
        kindices = tuple(range(1, nk + 1)) if kpoints_1based is None else _indices(kpoints_1based, nk, 'kpoints_1based')
        reciprocal = np.linalg.inv(header.lattice).T * (2 * np.pi)
        kpoints = []
        for kindex in kindices:
            reader.context = {'kind': 'kpoint_header', 'kpoint_1based': kindex}
            record = reader.record(reader.irec_start_k(kindex - 1), cnt=4 + 3 * nb)
            if not np.isfinite(record).all():
                raise WavecarReadError('WAVECAR k record contains nonfinite values')
            count = _positive_integer(record[0], 'spinor coefficient count')
            if count % 2 or count * 8 > reader.rl:
                raise WavecarReadError('Invalid two-component coefficient count or record byte range')
            k = record[1:4].copy()
            energy = record[4:].reshape(nb, 3)[np.array(bands) - 1, 0].copy()
            ig, e_kg = calc_gvectors(k, reciprocal, cutoff, nplane=count // 2,
                                     Ecut1=cutoff, spinor=True, verbosity=0)
            if len(ig) * 2 != count:
                raise WavecarReadError('Independent G enumeration count disagrees with the complete spinor record')
            raw = np.empty((len(bands), count // 2, 2), dtype=np.complex64)
            for index, band in enumerate(bands):
                reader.context = {'kind': 'coefficients', 'kpoint_1based': kindex, 'band_1based': band}
                coefficients = reader.record_k_band(kindex - 1, band - 1, cnt=count)
                raw[index] = coefficients.reshape(count // 2, 2, order='F')
            association = repair_vasp_g_association(ig, raw[:, ig[:, 3], :],
                                                    coefficient_count=count, rtag=reader.iprec)
            point = Kpoint(ik=kindex - 1, num_bands=len(bands), RecLattice=reciprocal.copy(),
                           spinor=True, kpt=k, WF=association.coefficients, Energy=energy,
                           ig=association.ig, upper=None, normalize=False, eKG=e_kg)
            kpoints.append(point)
        final_stat = path.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')
        if any(getattr(original_stat, field) != getattr(final_stat, field) for field in fields):
            raise WavecarReadError('WAVECAR source changed during reading')
        return SelectedWavecar(header, tuple(kpoints), bands, kindices, tuple(reader.ledger))
    finally:
        reader.f.close()
