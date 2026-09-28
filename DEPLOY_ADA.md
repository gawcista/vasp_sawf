# Deploy and run on ADA

Clone the repository and run the two scripts in an isolated Python environment. There is no need to install this project as a Python package or transfer the local environment, results, or WAVECAR. Run step 1 on ADA where WAVECAR is stored; step 2 can run on ADA or locally.

The current dependency lock targets Linux x86_64 and CPython 3.13. Core versions are WannierBerri 1.7.0, IrRep 2.6.3, NumPy 2.3.5, SciPy 1.17.0, and Numba 0.62.1. Internal library interfaces were checked against these versions; do not replace them with the latest releases. Installation and real-data reproduction on ADA have not yet been verified. Local validation does not replace that check.

## 1. Clone and create an environment

Run these commands where you keep software. If `vasp_sawf` already exists, inspect it first; do not overwrite it.

```bash
git clone https://github.com/gawcista/vasp_sawf.git
cd vasp_sawf
git rev-parse HEAD
find-module python-waterboa
```

ADA uses explicitly versioned environment modules. MPCDF's `python-waterboa/2025.06` is based on CPython 3.13 and is a suitable first candidate to check. Availability must be confirmed with `find-module` on ADA. If that version is available:

```bash
module load python-waterboa/2025.06
python -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 13)'
getconf GNU_LIBC_VERSION
python -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install --require-hashes -r requirements.lock
python -m pip check
```

Dependencies are installed into the new `.venv`, without `--user` or `--system-site-packages`. If `.venv` already exists, identify its purpose before recreating it. Include the module load and environment activation in the actual job script; running them once in a login shell does not initialize every batch job.

`requirements.lock` pins runtime package versions and the SHA256 hashes of the artifacts downloaded locally. It is not a cross-platform lock: binary artifacts must match the Python ABI, CPU architecture, and glibc. For example, the SciPy artifact carries manylinux 2.27/2.28 tags. If pip reports no compatible artifact or a hash mismatch, retain the error and check these conditions. Do not bypass it by removing hashes or upgrading dependencies. `pylatexenc` is a source distribution, and its temporary build tools are not included in the lock; a fully closed, reproducible installation has not yet been demonstrated.

Module references: [MPCDF environment modules](https://docs.mpcdf.mpg.de/faq/hpc_software.html), [ADA documentation](https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/MPSD_PKS_ADA.html), and the [Python 3.13 stack announcement](https://docs.mpcdf.mpg.de/bnb/pdf/bits_and_bytes_issue_219.pdf). Compiler, MPI, or GPU settings in unrelated cluster examples are not requirements of this program.

## 2. Check the installation

Check dependencies and command entry points without reading wavefunctions:

```bash
python - <<'PYTHON'
from importlib.metadata import version
import core.inputs
import core.wavecar
import core.symmetry
import core.localize
for name in ('numpy', 'scipy', 'numba', 'irrep', 'wannierberri'):
    print(name, version(name))
assert version('irrep') == '2.6.3'
assert version('wannierberri') == '1.7.0'
PYTHON
python extract_symmetry.py --help
python run_sawf.py --help
```

Run regression tests that require no external DFT data on allocated compute resources:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -m 'not real_data' -p no:cacheprovider
```

This checks software behavior; it does not establish that the SrVO₃ calculation has been reproduced on ADA. Real-data regression requires separately supplied read-only inputs, using the environment variables in [README.md](README.md). Those data are not included in the repository. Record any skipped or deselected real-data tests accurately.

## 3. Extract symmetry information on ADA

Run on allocated compute resources, replacing the example paths:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python extract_symmetry.py \
  --seed /path/to/interface/wannier90 \
  --wavecar /path/to/interface/WAVECAR \
  --outcar /path/to/interface/OUTCAR \
  --output /path/to/results/symmetry
```

Required inputs are one consistent set of `.win/.mmn/.amn/.eig`, the OUTCAR from that interface calculation, and the WAVECAR saved at its end. IBZ symmetry reduction has been validated; full-BZ WAVECAR is not required. An earlier SCF input WAVECAR alone is outside the validated entry point. See README for the full scope, including the single-center target and Γ-centered mesh restrictions. SrVO₃ validation does not imply tSnS validation.

The only outputs are `bloch.npz` and `report.json`. To run step 2 locally, download the complete `symmetry` directory and retain the original interface files with identical contents. WAVECAR and UNK do not need to be downloaded.

The program currently uses one Python process with NumPy/SciPy numerical kernels. Do not start the same extraction command on multiple MPI ranks; GPUs are not required. Begin with the BLAS thread settings used in small-system validation, then measure scaling separately. Large-system memory and wall time must be estimated from actual NG/NK/NB and calibrated with small tests. This guide provides no uncalibrated Slurm resource values, and the program does not submit jobs automatically.

## 4. Run SAWF

Use a machine with the same dependencies. The center and orbital below apply only to the validated SrVO₃ example:

```bash
python run_sawf.py \
  --seed /path/to/interface/wannier90 \
  --symmetry /path/to/results/symmetry \
  --center 0.5 0.5 0.5 --orbital t2g \
  --output /path/to/results/model
```

Both computational entry points refuse to overwrite existing output directories. Keep outputs separate from original inputs. Another material requires its own established target center and local representation; do not copy the SrVO₃ `t2g` target without justification.

See README for the optional DFT path, three band plots, and replotting commands. Keep data outside the repository so that updating code does not change existing results.

## Update the code

Record `git rev-parse HEAD` when installing. Later, run `git pull --ff-only` only with a clean working tree and no calculation currently using that source checkout. If the dependency lock changes, reinstall into the isolated environment and repeat the checks above. Use `git switch --detach <full-commit>` to fix the source version for reproduction. To modify source, create a separate worktree from the relevant commit and keep original calculation inputs read-only.

## Release validation status

Local Linux/Python 3.13 checks on 2026-09-28 passed `pip check`, actual core-module imports, both computational entry points' `--help`, and 184 portable tests; 16 external-data cases were deselected. The final English source was tested from a temporary copy containing only the release files. Four additional tests passed against the existing SrVO₃ symmetry bundle, checking legacy compatibility and rejection of altered evidence without reading WAVECAR. The temporary copy was removed after testing. The optional pyFFTW package is not installed, so WannierBerri uses its official NumPy FFT fallback. Installation and calculation reproduction on ADA remain untested. No cluster jobs have been submitted.
