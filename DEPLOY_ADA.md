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

ADA uses explicitly versioned environment modules. MPCDF's `python-waterboa/2025.06` is based on CPython 3.13 and is a suitable first candidate to check. Availability must be confirmed with `find-module` on ADA. If a virtual environment is already active, run `deactivate` before loading the module. Loading a module does not replace the interpreter of an existing virtual environment. If that module version is available:

```bash
module load python-waterboa/2025.06 &&
python3.13 -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 13)' &&
getconf GNU_LIBC_VERSION &&
test ! -e .venv && test ! -L .venv &&
python3.13 -m venv .venv &&
source .venv/bin/activate &&
export PYTHONNOUSERSITE=1 &&
python -c 'import sys; print(sys.version); print(sys.executable); assert sys.version_info[:2] == (3, 13) and sys.prefix != sys.base_prefix' &&
python -m pip install --require-hashes -r requirements.lock &&
python -m pip check
```

Stop if the module load or either interpreter check fails. Dependencies are installed into the new `.venv`, without `--user` or `--system-site-packages`. If `.venv` already exists, identify its purpose before recreating it; the recovery instructions below preserve an existing environment. Include the module load and environment activation in the actual job script; running them once in a login shell does not initialize every batch job.

`requirements.lock` pins runtime package versions and the SHA256 hashes of the artifacts downloaded locally. It is not a cross-platform lock: binary artifacts must match the Python ABI, CPU architecture, and glibc. For example, the SciPy artifact carries manylinux 2.27/2.28 tags. If pip reports no compatible artifact or a hash mismatch, retain the error and check these conditions. Do not bypass it by removing hashes or upgrading dependencies. `pylatexenc` is a source distribution, and its temporary build tools are not included in the lock; a fully closed, reproducible installation has not yet been demonstrated.

Module references: [MPCDF environment modules](https://docs.mpcdf.mpg.de/faq/hpc_software.html), [ADA documentation](https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/MPSD_PKS_ADA.html), and the [Python 3.13 stack announcement](https://docs.mpcdf.mpg.de/bnb/pdf/bits_and_bytes_issue_219.pdf). Compiler, MPI, or GPU settings in unrelated cluster examples are not requirements of this program.

### Recover from a virtual environment created with system Python

An ADA installation attempt used Python 3.6.15 inside `.venv`. Pip reached PyPI, but its verbose output rejected every `annotated-types` release because of `Requires-Python`. The locked version 0.8.0 requires Python >=3.10, while this project's full artifact lock targets CPython 3.13. The final `from versions: none` message did not mean that the release was absent from PyPI.

Recreate the environment using Python 3.13, rather than upgrading pip inside the Python 3.6 environment or changing package pins. A virtual environment uses the base interpreter with which it was created; see the [Python venv documentation](https://docs.python.org/3.13/library/venv.html).

From the repository directory, with the old environment active, first run:

```bash
deactivate
module load python-waterboa/2025.06
python3.13 --version
```

Continue only if the module loads and the interpreter reports Python 3.13.x. If the module is unavailable, use `find-module python-waterboa` to identify an available Python 3.13 module; do not fall back to system `python3`.

Use a new `.venv-py313` directory, leaving the old `.venv` untouched. If that new path already exists, inspect it instead of recreating it:

```bash
test ! -e .venv-py313 && test ! -L .venv-py313 &&
python3.13 -m venv .venv-py313 &&
source .venv-py313/bin/activate &&
export PYTHONNOUSERSITE=1 &&
python -c 'import sys; print(sys.version); print(sys.executable); assert sys.version_info[:2] == (3, 13) and sys.prefix != sys.base_prefix' &&
python -m pip install --require-hashes -r requirements.lock &&
python -m pip check
```

Stop if environment creation, activation, or the interpreter check fails. Subsequent sessions and batch scripts must activate `.venv-py313/bin/activate` or explicitly invoke `.venv-py313/bin/python` when using this recovery environment. No package version or hash needs to change for this error. All 76 pinned versions and hashes were checked against official PyPI release metadata, and the 0.8.0 wheel was downloaded and hash-verified independently. These checks establish artifact availability, not a completed installation on ADA.

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

The program currently uses one Python process with NumPy/SciPy numerical kernels. Do not start the same extraction command on multiple MPI ranks; GPUs are not required. Begin with the BLAS thread settings used in small-system validation, then measure scaling separately. Large-system memory and wall time must be estimated from actual NG/NK/NB and calibrated with small tests. The program does not submit jobs automatically.

### ADA batch template and the current tSnS limitation

[extract_ada.sbatch](extract_ada.sbatch) uses the following paths:

| Purpose | Path |
| --- | --- |
| Source checkout | `$HOME/.src/vasp_sawf` |
| Python environment | `$HOME/.src/vasp_sawf/.venv-py313` |
| Read-only interface inputs | `/ptmp/tSnS/scdm661` |
| Job working directory and logs | `/ptmp/tSnS/sawf661` |
| New extraction output | `/ptmp/tSnS/sawf661/symmetry` |

The Python module must match the module used to create the environment. The template uses `python-waterboa/2025.06`; confirm its availability and complete installation first. It intentionally uses the Python 3.13 recovery environment, not the old Python 3.6 `.venv`.

**The current tSnS structure is not supported by the extraction algorithm yet.** A lightweight local check on 2026-09-28, using the actual WIN structure and fixed IrRep 2.6.3, found the candidate structural grey group `P2_11'`, with four operations. The nontrivial unitary operation has fractional rotation `diag(-1, 1, -1)` and translation `(0, 0.5, 0.3371566930075367)`. Its square is translation by `(0, 1, 0)`: the half translation along b cannot be removed by shifting the origin. Constructing a candidate grey group from structure does not establish the magnetic state of the calculation.

`_validate_grey_group` and `build_symmetry_maps` currently reject nonzero spatial translations. Simply removing those checks would be incorrect: MMN transport, independent coefficient transformations, and group-composition checks need the corresponding translation phases. The SAWF verification must use the same conventions. The screw has no individually fixed center modulo lattice translations, so the current single-center target and initial-guess alignment also require extension to multiple symmetry-related centers. The specific centers and local representation remain undetermined. These are implementation limits; the accepted SrVO3 numerical workflow is unchanged. No target Wannier representation is inferred from the tSnS SCDM AMN.

The template therefore performs the existing structural checks before coefficient I/O. For the local tSnS structure it is expected to stop at that check. The check reads only 128 logical WAVECAR header bytes plus the small interface files; it does not read coefficients. Do not submit a full tSnS extraction expecting it to succeed until translation support is implemented and independently tested. The local historical OUTCAR also has `LWAVE=F`; the contents of the remote directory have not been inspected. The actual remote WAVECAR/OUTCAR must meet the documented interface-output requirements; do not assume that an earlier SCF WAVECAR is the interface output.

After these prerequisites have been resolved, submission would be:

```bash
cd "$HOME/.src/vasp_sawf"
git pull --ff-only
mkdir -p /ptmp/tSnS/sawf661
sbatch extract_ada.sbatch
```

Update source only when the checkout is clean and no job uses it. The working directory must exist before `sbatch`, because Slurm opens its logs before executing the script. Logs append to fixed `extract.out` and `extract.err` files; extraction refuses to overwrite the existing `symmetry` directory. No input files are copied or modified. If moving the template to another system, change its input and working-directory paths. `#SBATCH` directives do not expand `$HOME` or shell variables; this is why the source path is set in the shell body and the working directory is literal. See the [Slurm sbatch manual](https://slurm.schedmd.com/sbatch.html).

The template requests one node, one process, one CPU, and 64 GiB on `p.large`, without MPI, GPUs, or an exclusive node reservation. ADA's `p.large` nodes have 2 TB each and can be shared; an explicit `--mem` is needed to avoid the default allocation of all memory. The 24-hour setting is the documented partition time limit, not a runtime estimate. See the [official ADA partition documentation](https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/MPSD_PKS_ADA.html).

**Resource values are provisional trial limits, not measured tSnS requirements.** The local OUTCAR reports 24 stored k points, 4320 original bands, and at most 1,372,860 "plane waves". WIN selects bands 3633-3640 on the full 6x6x1 mesh. Let `G_k` count G vectors for one spinor component. The reader retains `complex64` coefficients for the eight selected bands, and extraction also retains a `complex128` copy; together these occupy `48 * 8 * sum(G_k)` bytes. The shared G tables and kinetic energies add approximately `56 * sum(G_k)` bytes. This is not a streaming implementation.

The SOC meaning of OUTCAR's plane-wave count must be resolved using the actual WAVECAR k-record coefficient count (`C_k = 2 G_k`). Conservatively treating the reported maximum as a single-component G count gives about 11.78 GiB for both coefficient copies and 1.72 GiB for G tables and kinetic energies. If the printed count already includes both spinor components, these estimates halve. Temporary transforms, Python G-vector dictionaries, least-squares work arrays, and library overhead add to this baseline. Thus 64 GiB is a trial budget with workspace headroom, not evidence that the measured peak fits it. A 1.1 TB file does not imply 1.1 TB resident memory: unselected band coefficient records are not read.

Serial G enumeration and mapping can dominate runtime; allocating 72 MPI ranks would launch duplicate work, and more BLAS threads do not parallelize those Python loops. After the symmetry extension, calibrate one stored k point with all eight bands and all G vectors on allocated resources before attempting the full extraction; such a calibration is not a replacement or downsampling of the required 6x6x1 production mesh. No calibration mode or parallel algorithm is added by this batch template.

`/usr/bin/time -v` records extraction wall time and peak RSS in `extract.err`; `symmetry/report.json` records the process peak RSS, elapsed time, and logical record-read ledger. Its `read_bytes` excludes the extractor's own 128-byte header preflight (recorded separately as `header_preflight_read_bytes`) and the additional 128-byte batch preflight. Slurm provides an independent accounting view after a job:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,AllocCPUS
```

Inspect the individual `srun` steps as well as the job row. Logical bytes read by the program are not identical to physical filesystem I/O. Neither tSnS peak memory nor extraction time has been measured, and this template has not been run on ADA.

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
