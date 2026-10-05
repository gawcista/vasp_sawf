"""Export a validated SAWF gauge to VASP's formatted WANPROJ without changing phases."""

import argparse
import hashlib
import io
import json
import math
from numbers import Integral
from pathlib import Path
import re
import sys

import numpy as np


def _validate_payload(U, kpoints, bands_vasp_1based, source_nb, mesh):
    u, kpoints, bands, mesh = map(np.asarray, (U, kpoints, bands_vasp_1based, mesh))
    if (u.ndim != 3 or u.shape[0] < 1 or u.shape[1] < 2
            or u.shape[1] != u.shape[2] or u.shape[1] % 2
            or not np.issubdtype(u.dtype, np.number) or not np.isfinite(u).all()):
        raise ValueError('WANPROJ requires finite full-grid square U with an even spinor dimension')
    nk, nb, nw = u.shape
    if (isinstance(source_nb, bool) or not isinstance(source_nb, Integral) or source_nb < nb):
        raise ValueError('The original source_nb must be a positive integer covering all selected bands')
    if (bands.shape != (nb,) or not np.issubdtype(bands.dtype, np.integer)
            or len(np.unique(bands)) != nb or np.any(bands < 1) or np.any(bands > source_nb)):
        raise ValueError('Original VASP band indices must be unique integers within source_nb')
    if (mesh.shape != (3,) or not np.issubdtype(mesh.dtype, np.integer)
            or np.any(mesh < 1) or math.prod(map(int, mesh)) != nk):
        raise ValueError('WANPROJ requires the complete three-dimensional Gamma mesh')
    if (kpoints.shape != (nk, 3) or not np.issubdtype(kpoints.dtype, np.number)
            or np.issubdtype(kpoints.dtype, np.complexfloating)
            or not np.isfinite(kpoints).all()):
        raise ValueError('WANPROJ requires finite reduced k coordinates for every full-grid point')
    scaled = (kpoints - np.floor(kpoints)) * mesh
    if np.max(abs(scaled - np.rint(scaled))) > 1e-10:
        raise ValueError('WANPROJ k points do not lie on the Gamma-centered mesh')
    nodes = np.rint(scaled).astype(np.int64) % mesh
    if len(np.unique(nodes, axis=0)) != nk:
        raise ValueError('WANPROJ k points contain duplicate periodic mesh nodes')
    residual = float(np.max(abs(u.conj().swapaxes(-1, -2) @ u - np.eye(nw))))
    if not np.isfinite(residual) or residual > 1e-6:
        raise ValueError(f'WANPROJ U failed the unitarity gate: {residual} > 1e-6')
    return u, kpoints, bands, mesh, residual


def read_wanproj(path):
    """Read the single-channel formatted VASP layout, retaining band and Wannier order."""
    lines = iter(Path(path).read_text(encoding='ascii').splitlines())

    def fields(count):
        tokens = next(lines).split()
        if len(tokens) != count:
            raise ValueError('Invalid WANPROJ record column count')
        return tokens

    try:
        if not next(lines).lstrip().startswith('#'):
            raise ValueError('WANPROJ is missing its comment header')
        ispin, nk, source_nb, nw = map(int, fields(4))
        if ispin != 1 or nk < 1 or nw < 1 or source_nb < nw:
            raise ValueError('Invalid single-channel WANPROJ dimensions')
        kpoints = np.empty((nk, 3))
        for k in range(nk):
            row = fields(4)
            if int(row[0]) != k + 1:
                raise ValueError('WANPROJ k-point indices are not sequential')
            kpoints[k] = list(map(float, row[1:]))
        u, bands = None, None
        for k in range(nk):
            row = fields(5)
            spin, nb = map(int, row[:2])
            if spin != 1 or nb < 1 or nb > source_nb:
                raise ValueError('Invalid WANPROJ spin or selected-band count')
            if not np.array_equal(np.array(list(map(float, row[2:]))), kpoints[k]):
                raise ValueError('WANPROJ block coordinates disagree with its k-point list')
            if u is None:
                u, bands = np.empty((nk, nb, nw), dtype=complex), np.empty(nb, dtype=int)
            elif nb != u.shape[1]:
                raise ValueError('WANPROJ changes the selected-band count across k points')
            for band in range(nb):
                for wannier in range(nw):
                    row = fields(4)
                    original, column = map(int, row[:2])
                    if k == 0 and wannier == 0:
                        bands[band] = original
                    if original != bands[band] or column != wannier + 1:
                        raise ValueError('WANPROJ band or Wannier indices disagree with record order')
                    u[k, band, wannier] = complex(float(row[2]), float(row[3]))
        if any(line.strip() for line in lines):
            raise ValueError('WANPROJ contains unexpected trailing records')
        if (not np.isfinite(kpoints).all() or not np.isfinite(u).all()
                or len(np.unique(bands)) != len(bands)
                or np.any(bands < 1) or np.any(bands > source_nb)):
            raise ValueError('WANPROJ contains invalid band indices or nonfinite values')
    except (StopIteration, OverflowError) as error:
        raise ValueError('WANPROJ contains incomplete or invalid records') from error
    return dict(U=u, kpoints=kpoints, bands_vasp_1based=bands,
                source_num_bands=source_nb, num_wann=nw, ispin=ispin)


def write_wanproj(path, *, U, kpoints, bands_vasp_1based, source_nb, mesh):
    """Write U[k, original-band-row, Wannier-column] directly; no transpose or conjugation."""
    u, kpoints, bands, mesh, residual = _validate_payload(
        U, kpoints, bands_vasp_1based, source_nb, mesh)
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise ValueError('WANPROJ must be a new file; overwriting and symlink destinations are forbidden')
    nk, nb, nw = u.shape
    with path.open('x', encoding='ascii', newline='\n') as handle:
        handle.write('# Created by vasp_sawf; Bloch-to-Wannier U is stored without phase changes.\n')
        handle.write(f'1 {nk} {source_nb} {nw}\n')
        for k, point in enumerate(kpoints, 1):
            handle.write(f'{k} ' + ' '.join(f'{x:.17e}' for x in point) + '\n')
        for k, point in enumerate(kpoints):
            handle.write(f'1 {nb} ' + ' '.join(f'{x:.17e}' for x in point) + '\n')
            for row, original in enumerate(bands):
                for column in range(nw):
                    value = u[k, row, column]
                    handle.write(f'{original} {column + 1} {value.real:.17e} {value.imag:.17e}\n')
    restored = read_wanproj(path)
    if (restored['source_num_bands'] != source_nb or restored['ispin'] != 1
            or not np.array_equal(restored['U'], u)
            or not np.array_equal(restored['kpoints'], kpoints)
            or not np.array_equal(restored['bands_vasp_1based'], bands)):
        raise ValueError('WANPROJ save/readback changed its source dimensions, indices or complex gauge')
    content = path.read_bytes()
    return dict(file=path.name, format='vasp-formatted-wanproj-v1', ispin=1,
                num_kpoints=nk, source_num_bands=int(source_nb), num_selected_bands=nb,
                num_wann=nw, mesh=mesh.tolist(), bands_vasp_1based=bands.tolist(),
                unitarity_max=residual, readback_exact=True, bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                convention='Psi_W(k,w) = sum_n Psi_B(k,n) U(k,n,w); original complex U and all cell phases retained')


def export_wanproj(model_dir, symmetry_dir, output_dir):
    """Export an already accepted model bound to its symmetry package without wavefunction I/O."""
    from .localize import _new_output, _read_bound_package

    model_dir, symmetry_dir = [Path(p).resolve() for p in (model_dir, symmetry_dir)]
    summary_bytes = (model_dir / 'summary.json').read_bytes()
    summary = json.loads(summary_bytes)
    if (summary.get('schema') != 'sawf-bridge-model-v2' or summary.get('status') != 'ready'
            or any(summary.get(key) is not True for key in
                   ('converged', 'sawf_ready', 'numerical_checks_passed'))):
        raise ValueError('WANPROJ export requires a ready, converged model with all numerical checks passed')
    hashes = summary.get('source_hashes')
    if (not isinstance(hashes, dict) or set(hashes) != {'win', 'amn', 'mmn', 'eig'}
            or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
                   for v in hashes.values())):
        raise ValueError('Model summary is missing complete original interface source hashes')
    if summary.get('model_file') != 'model.npz':
        raise ValueError('Model summary must identify its bound model.npz')
    model_file = model_dir / 'model.npz'
    content = model_file.read_bytes()
    model_digest = hashlib.sha256(content).hexdigest()
    if model_digest != summary.get('model_sha256'):
        raise ValueError('Model SHA256 does not match its validated summary')
    arrays, symreport = _read_bound_package(symmetry_dir, hashes)
    if symreport['bloch_sha256'] != summary.get('symmetry_package_sha256'):
        raise ValueError('Bound symmetry package SHA256 disagrees with the model summary')
    source_nb = symreport.get('source_num_bands', symreport.get('wavecar', {}).get('num_bands'))
    wavecar_nb = symreport.get('wavecar', {}).get('num_bands', source_nb)
    if wavecar_nb != source_nb or summary.get('source_num_bands', source_nb) != source_nb:
        raise ValueError('Original source band counts disagree across provenance records')
    mesh = symreport.get('mesh')
    if 'mesh' in summary and not np.array_equal(summary['mesh'], mesh):
        raise ValueError('Model and symmetry reports disagree on the full mesh')
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        required = {'U', 'kpoints', 'eigenvalues_eV', 'centers_angstrom',
                    'spreads_angstrom2', 'lattice', 'R', 'H_R'}
        if not required.issubset(archive.files):
            raise ValueError('The saved model is missing required final model arrays')
        model = {key: archive[key].copy() for key in archive.files}
    for key, value in model.items():
        if (value.dtype.hasobject or not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value).all()):
            raise ValueError(f'Model array {key} is nonfinite or unsafe')
    bands = np.asarray(summary.get('bands_vasp_1based'))
    for key, expected in [('kpoints', model['kpoints']), ('eig', model['eigenvalues_eV']),
                          ('bands_vasp_1based', bands)]:
        if key not in arrays or not np.array_equal(arrays[key], expected):
            raise ValueError(f'Model and bound symmetry arrays disagree on {key}')
    if ('bands_vasp_1based' in symreport
            and not np.array_equal(symreport['bands_vasp_1based'], bands)):
        raise ValueError('Original bands disagree with the symmetry report')
    if model['eigenvalues_eV'].shape != model['U'].shape[:2]:
        raise ValueError('Model eigenvalues do not cover every selected band and k point')
    _validate_payload(model['U'], model['kpoints'], bands, source_nb, mesh)
    source_seed = summary.get('source_seed')
    if not isinstance(source_seed, str) or not source_seed or not Path(source_seed).is_absolute():
        raise ValueError('Model summary is missing its absolute original interface source_seed')
    protected = [model_file, model_dir / 'summary.json', symmetry_dir / 'bloch.npz',
                 symmetry_dir / 'report.json']
    protected += [Path(f'{source_seed}.{ext}') for ext in hashes]
    for key in ('wavecar', 'outcar'):
        record = symreport.get(key, {})
        if isinstance(record, dict) and isinstance(record.get('path'), str):
            protected.append(Path(record['path']))
    output = _new_output(output_dir, protected)
    wanproj = write_wanproj(output / 'WANPROJ', U=model['U'], kpoints=model['kpoints'],
                           bands_vasp_1based=bands, source_nb=source_nb, mesh=mesh)
    report = dict(schema='sawf-bridge-wanproj-v1', status='ready', wavefunction_bytes_read=0,
                  model_sha256=model_digest, model_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
                  symmetry_package_sha256=symreport['bloch_sha256'], source_hashes=hashes,
                  source_seed=source_seed, source_num_bands=int(source_nb), mesh=list(mesh),
                  bands_vasp_1based=bands.tolist(), wanproj=wanproj)
    with (output / 'report.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='Ready model directory containing model.npz and summary.json')
    parser.add_argument('--symmetry', required=True, type=Path, help='Bound symmetry package directory')
    parser.add_argument('--output', required=True, type=Path, help='New export directory outside all protected input directories')
    args = parser.parse_args()
    try:
        report = export_wanproj(args.model, args.symmetry, args.output)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'WANPROJ export failed: {error}', file=sys.stderr)
        return 1
    print(f"WANPROJ: {args.output.resolve() / 'WANPROJ'}; status: {report['status']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
