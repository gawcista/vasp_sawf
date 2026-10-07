"""Bounded per-k-point evaluation and process scheduling for symmetry extraction."""

from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
import os
from pathlib import Path
import re
import resource
import time

import numpy as np


def _nonnegative_memory_value(value):
    result = int(value)
    if result < 0:
        raise ValueError('Memory values must be nonnegative')
    return result


def _cgroup_memory_limits():
    """Read the process hierarchy through its actual v1/v2 mount, including ancestor limits."""
    try:
        membership = Path('/proc/self/cgroup').read_text().splitlines()
    except FileNotFoundError:
        return []
    groups = []
    for line in membership:
        identifier, controllers, directory = line.split(':', 2)
        if identifier == '0' and not controllers:
            groups.append(('cgroup2', Path(directory)))
        elif 'memory' in controllers.split(','):
            groups.append(('cgroup', Path(directory)))
    if not groups:
        return []
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        fields, filesystem = before.split(), after.split()
        if filesystem[0] == 'cgroup2' or (filesystem[0] == 'cgroup' and 'memory' in filesystem[2].split(',')):
            paths = [Path(re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value))
                     for value in fields[3:5]]
            mounts.append((filesystem[0], *paths))
    limits = []
    for kind, directory in groups:
        if not directory.is_absolute() or '..' in directory.parts:
            raise ValueError('Unresolved cgroup namespace path')
        matching = [(root, mount) for fs, root, mount in mounts
                    if fs == kind and directory.is_relative_to(root)]
        if not matching:
            raise ValueError('Cannot locate the process memory-controller mount')
        for root, mount in matching:
            leaf = mount / directory.relative_to(root)
            maximum_name, used_name = (('memory.max', 'memory.current') if kind == 'cgroup2' else
                                      ('memory.limit_in_bytes', 'memory.usage_in_bytes'))
            observed = False
            for folder in (leaf, *leaf.parents):
                if not folder.is_relative_to(mount):
                    break
                try:
                    maximum = (folder / maximum_name).read_text().strip()
                except FileNotFoundError:
                    if folder == mount and root == Path('/'):
                        continue
                    if kind == 'cgroup2':
                        continue
                    raise
                observed = True
                if kind == 'cgroup2' and maximum == 'max':
                    continue
                maximum = _nonnegative_memory_value(maximum)
                used = _nonnegative_memory_value((folder / used_name).read_text())
                limits.append(max(0, maximum - used))
            if not observed and directory != Path('/'):
                raise ValueError('Cannot inspect any memory-controller limit in the process hierarchy')
    return limits


def available_memory_bytes():
    """Use the smallest available host, cgroup v1/v2, and allocated Slurm memory budget."""
    limits = []
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                limits.append(_nonnegative_memory_value(line.split()[1]) * 1024)
    except OSError:
        pass
    try:
        limits.extend(_cgroup_memory_limits())
    except (OSError, ValueError) as error:
        raise ValueError('Cannot inspect the process memory controller; specify --memory-gb explicitly') from error
    try:
        per_node = _nonnegative_memory_value(os.environ.get('SLURM_MEM_PER_NODE', '0'))
        if per_node > 0:
            limits.append(per_node * 1024**2)
        per_cpu = _nonnegative_memory_value(os.environ.get('SLURM_MEM_PER_CPU', '0'))
        if per_cpu > 0:
            allocated = [int(os.environ[key]) for key in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE')
                         if key in os.environ]
            if not allocated or min(allocated) < 1:
                raise ValueError('Per-CPU memory requires an explicit Slurm CPU allocation')
            if hasattr(os, 'sched_getaffinity'):
                allocated.append(len(os.sched_getaffinity(0)))
            limits.append(per_cpu * min(allocated) * 1024**2)
    except (OSError, ValueError) as error:
        raise ValueError('Cannot determine the Slurm memory allocation; specify --memory-gb explicitly') from error
    if not limits:
        raise ValueError('Cannot determine available memory; specify --memory-gb')
    return min(limits)


def worker_plan(*, requested, allocated_cpus, jobs, memory_bytes, per_worker_bytes):
    """Cap concurrency, reserving 25% of the budget beyond the worker estimate."""
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (requested, allocated_cpus, jobs, memory_bytes, per_worker_bytes)):
        raise ValueError('CPU counts, jobs, and memory budgets must be positive integers')
    by_memory = memory_bytes * 3 // 4 // per_worker_bytes
    if by_memory < 1:
        raise ValueError('The estimated one-k-point working set exceeds the memory budget')
    return dict(workers=min(requested, allocated_cpus, jobs, by_memory), requested_workers=requested,
                allocated_cpus=allocated_cpus, jobs=jobs, memory_budget_bytes=memory_bytes,
                per_worker_estimated_bytes=per_worker_bytes, memory_reserve_fraction=.25,
                memory_policy='Estimate with headroom, not a guaranteed RSS bound; calibrate on allocated hardware')


def evaluate_kpoint(wavecar, bands, lattice, kindex, group_dict, little_indices, source_identity, spin_channel=1, *, gamma=False):
    """Read one complete selected-band k point and return only small numerical results."""
    from irrep.spacegroup import SpaceGroup
    from .wavecar import read_selected_wavecar
    from .symmetry import _independent_transform, _finite_max

    started = time.perf_counter()
    stored = read_selected_wavecar(wavecar, bands_1based=bands, kpoints_1based=[kindex],
                                   lattice=lattice, expected_source_identity=source_identity,
                                   spinor=bool(group_dict['spinor']), spin_channel=spin_channel)
    read_seconds = time.perf_counter() - started
    raw = stored.kpoints[0]
    point = copy.copy(raw)
    point.WF = raw.WF.astype(np.complex128)
    if not np.array_equal(point.WF.astype(np.complex64), raw.WF):
        raise ValueError('Precision conversion changed original coefficients')
    sg = SpaceGroup(**group_dict)
    nb = len(bands)
    wf = point.WF.reshape(nb, -1)
    closure = gamma_lstsq = 0.
    matrices = []
    for isym in little_indices:
        operation = sg.symmetries[isym]
        local = point.symm_matrix(point, operation, block_indices=np.array([[0, nb]]), unitary=False)[0]
        transformed = _independent_transform(point, operation).reshape(nb, -1)
        if not np.isfinite(local).all() or not np.isfinite(transformed).all():
            raise ValueError('Nonfinite sewing matrix or independently transformed coefficients')
        norm = np.linalg.norm(transformed)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Independently transformed coefficient norms are nonfinite or zero')
        closure = _finite_max(closure, float(np.linalg.norm(local.T @ wf - transformed) / norm))
        if gamma:
            direct = np.linalg.lstsq(wf.T, transformed.T, rcond=None)[0]
            gamma_lstsq = _finite_max(gamma_lstsq, float(np.max(abs(local - direct))))
        matrices.append(local)
    return dict(kpoint_1based=int(kindex), little_indices=list(map(int, little_indices)),
                matrices=np.array(matrices), closure=closure, gamma_lstsq=gamma_lstsq,
                read_ledger=list(stored.read_ledger), read_seconds=read_seconds,
                compute_seconds=time.perf_counter() - started - read_seconds,
                peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def _evaluate_task(arguments):
    return evaluate_kpoint(*arguments)


def record_kpoint_result(report, result):
    """Keep completed work visible even when a later stage fails."""
    report.setdefault('read_ledger', []).extend(result['read_ledger'])
    report['read_bytes'] = report.get('read_bytes', 0) + sum(row['bytes_returned'] for row in result['read_ledger'])
    report['total_logical_read_bytes'] = (report['read_bytes'] + report['metadata_read_bytes']
                                         + report['header_preflight_read_bytes'])
    report.setdefault('kpoint_resources', []).append({key: result[key] for key in
        ('kpoint_1based', 'read_seconds', 'compute_seconds', 'peak_rss_kib')})


def evaluate_kpoints(tasks, *, workers):
    """Spawn independent readers; never send wavefunction arrays between processes."""
    if workers == 1:
        for task in tasks:
            yield _evaluate_task(task)
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
    try:
        yield from executor.map(_evaluate_task, tasks, chunksize=1)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
