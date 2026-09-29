"""Exercise installed entry points without relying on the checkout directory."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


@pytest.mark.parametrize('command', [
    'sawf-extract', 'sawf-run', 'sawf-plot-bands', 'sawf-plot-symmetry',
])
def test_installed_command_help_outside_checkout(command, tmp_path):
    executable = Path(sysconfig.get_path('scripts')) / command
    assert executable.is_file(), f'{command} was not installed in this environment'
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'matplotlib'),
                       PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([str(executable), '--help'], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--output' in result.stdout


def test_installed_package_contains_acceptance_record(tmp_path):
    result = subprocess.run([
        sys.executable, '-I', '-c',
        'import json; from importlib.resources import files; '
        'record = json.loads(files("vasp_sawf").joinpath("accepted_closure.json").read_text()); '
        'assert record; print("acceptance record available")',
    ], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'acceptance record available'


def test_installed_computation_modules_import_outside_checkout(tmp_path):
    result = subprocess.run([
        sys.executable, '-I', '-c',
        'import vasp_sawf.inputs, vasp_sawf.wavecar, vasp_sawf.symmetry, vasp_sawf.localize',
    ], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
