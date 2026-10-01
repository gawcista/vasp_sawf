from pathlib import Path

import pytest


def _mock_host(monkeypatch, files, **environment):
    import vasp_sawf.extraction as extraction

    files = {'/proc/meminfo': 'MemAvailable: 104857600 kB\n',
             '/proc/self/cgroup': '', '/proc/self/mountinfo': '', **files}

    def read(path, *args, **kwargs):
        value = files.get(str(path), FileNotFoundError(str(path)))
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(extraction.os, 'sched_getaffinity', lambda pid: set(range(72)))
    for name in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK',
                 'SLURM_CPUS_ON_NODE'):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, str(value))


def test_cgroup_v1_ancestor_remaining_memory_caps_host(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {
        '/proc/self/cgroup': '7:memory:/slurm/job123/step0\n',
        '/proc/self/mountinfo': '30 20 0:30 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n',
        '/sys/fs/cgroup/memory/slurm/job123/step0/memory.limit_in_bytes': '8589934592',
        '/sys/fs/cgroup/memory/slurm/job123/step0/memory.usage_in_bytes': '1073741824',
        '/sys/fs/cgroup/memory/slurm/job123/memory.limit_in_bytes': '4294967296',
        '/sys/fs/cgroup/memory/slurm/job123/memory.usage_in_bytes': '2147483648',
        '/sys/fs/cgroup/memory/slurm/memory.limit_in_bytes': '9223372036854771712',
        '/sys/fs/cgroup/memory/slurm/memory.usage_in_bytes': '0',
    })
    assert available_memory_bytes() == 2147483648


def test_cgroup_v1_mounted_subtree_uses_actual_mount_root(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {
        '/proc/self/cgroup': '5:memory:/slurm/job123/step0\n',
        '/proc/self/mountinfo': '30 20 0:30 /slurm/job123 /custom/memory rw - cgroup cgroup rw,memory\n',
        '/custom/memory/step0/memory.limit_in_bytes': '8589934592',
        '/custom/memory/step0/memory.usage_in_bytes': '1073741824',
        '/custom/memory/memory.limit_in_bytes': '4294967296',
        '/custom/memory/memory.usage_in_bytes': '1073741824',
    })
    assert available_memory_bytes() == 3221225472


def test_cgroup_v2_parent_limit_and_unlimited_child(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {
        '/proc/self/cgroup': '0::/slurm/job123\n',
        '/proc/self/mountinfo': '30 20 0:30 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n',
        '/sys/fs/cgroup/slurm/job123/memory.max': 'max',
        '/sys/fs/cgroup/slurm/memory.max': '6442450944',
        '/sys/fs/cgroup/slurm/memory.current': '2147483648',
    })
    assert available_memory_bytes() == 4294967296


def test_slurm_per_cpu_memory_uses_allocated_task_not_entire_node(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {}, SLURM_MEM_PER_CPU=1024,
               SLURM_CPUS_PER_TASK=4, SLURM_CPUS_ON_NODE=72)
    assert available_memory_bytes() == 4 * 1024**3


def test_slurm_per_cpu_memory_obeys_actual_cpu_affinity(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes
    import vasp_sawf.extraction as extraction

    _mock_host(monkeypatch, {}, SLURM_MEM_PER_CPU=1024, SLURM_CPUS_PER_TASK=8)
    monkeypatch.setattr(extraction.os, 'sched_getaffinity', lambda pid: {0, 1})
    assert available_memory_bytes() == 2 * 1024**3


def test_slurm_per_node_and_per_cpu_take_smaller_budget(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {}, SLURM_MEM_PER_CPU=1024, SLURM_CPUS_PER_TASK=8,
               SLURM_MEM_PER_NODE=3072)
    assert available_memory_bytes() == 3 * 1024**3


def test_slurm_exclusive_zero_means_all_available_host_memory(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {}, SLURM_MEM_PER_NODE=0)
    assert available_memory_bytes() == 100 * 1024**3


@pytest.mark.parametrize('problem', ['missing_mount', 'unreadable_limit', 'unreadable_usage', 'negative_limit'])
def test_unreadable_or_invalid_memory_controller_requires_explicit_budget(monkeypatch, problem):
    from vasp_sawf.extraction import available_memory_bytes

    files = {
        '/proc/self/cgroup': '7:memory:/job123\n',
        '/proc/self/mountinfo': '30 20 0:30 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n',
        '/sys/fs/cgroup/memory/job123/memory.limit_in_bytes': '8589934592',
        '/sys/fs/cgroup/memory/job123/memory.usage_in_bytes': '1073741824',
    }
    if problem == 'missing_mount':
        files['/proc/self/mountinfo'] = ''
    elif problem == 'unreadable_limit':
        files['/sys/fs/cgroup/memory/job123/memory.limit_in_bytes'] = PermissionError('denied')
    elif problem == 'unreadable_usage':
        del files['/sys/fs/cgroup/memory/job123/memory.usage_in_bytes']
    else:
        files['/sys/fs/cgroup/memory/job123/memory.limit_in_bytes'] = '-1'
    _mock_host(monkeypatch, files)
    with pytest.raises(ValueError, match='memory-gb'):
        available_memory_bytes()


def test_per_cpu_memory_without_cpu_allocation_does_not_guess_host_count(monkeypatch):
    from vasp_sawf.extraction import available_memory_bytes

    _mock_host(monkeypatch, {}, SLURM_MEM_PER_CPU=1024)
    with pytest.raises(ValueError, match='memory-gb'):
        available_memory_bytes()
