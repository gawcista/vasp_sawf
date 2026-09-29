# Install and run on ADA

Install `vasp-sawf` into your chosen current Python environment and use its commands from any working directory. No project-specific virtual environment or fixed environment module is required. Run extraction on ADA where WAVECAR is stored; SAWF can run on ADA or locally without WAVECAR.

The current release requires Python 3.13 (`>=3.13,<3.14`), matching the validated range. Critical versions are pinned in `pyproject.toml`, including WannierBerri 1.7.0, IrRep 2.6.3, NumPy 2.3.5, SciPy 1.17.0, and Numba 0.62.1. Installation and calculation reproduction on ADA have not yet been completed. A successful local test is not an ADA result.

## 1. Install in the current environment

Activate the compatible environment you want to use. If a `vasp_sawf` checkout already exists, inspect it rather than overwriting it.

```bash
git clone https://github.com/gawcista/vasp_sawf.git
cd vasp_sawf
git rev-parse HEAD
python -c 'import sys; print(sys.version); print(sys.executable); assert sys.version_info[:2] == (3, 13)'
python -m pip install -e .
python -m pip check
```

Stop if the interpreter check fails. Pip installs the package and dependencies into the environment selected by `python`; do not substitute another pip executable. The installed commands are `sawf-extract`, `sawf-run`, `sawf-plot-bands`, and `sawf-plot-symmetry`. Editable installation refers to this checkout directly, so keep it in place. There is no need to set `PYTHONPATH` or run from the source directory.

An existing Python 3.6 environment cannot run this release. Activate an existing Python 3.13 environment instead. If none is available, `find-module python-waterboa` can identify suitable ADA modules. MPCDF documents `python-waterboa/2025.06` as based on CPython 3.13; use it only if available and appropriate for your setup. Loading a module does not change the interpreter with which an existing virtual environment was created. If you prefer a new environment, create it with the selected Python 3.13 interpreter using `python -m venv /path/to/new/environment`, without overwriting an existing path. These are optional ways to obtain a compatible interpreter, not project requirements.

References: [MPCDF environment modules](https://docs.mpcdf.mpg.de/faq/hpc_software.html), [ADA documentation](https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/MPSD_PKS_ADA.html), [Python 3.13 stack announcement](https://docs.mpcdf.mpg.de/bnb/pdf/bits_and_bytes_issue_219.pdf), and [Python virtual environments](https://docs.python.org/3.13/library/venv.html).

### Optional: reproduce the full artifact lock

For the audited Linux x86_64 / CPython 3.13 dependency set, install the lock first and then install this project without resolving dependencies again:

```bash
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

The lock pins downloaded artifacts by version and SHA256. It is not a cross-platform lock: binary artifacts must match the Python ABI, CPU architecture, and glibc. For example, the SciPy artifact carries manylinux 2.27/2.28 tags. If pip reports no compatible artifact or a hash mismatch, retain the error and inspect these conditions instead of deleting hashes or upgrading dependencies. `pylatexenc` is a source distribution; its temporary build tools are outside this runtime lock. The project's build backend also has a separate build environment. This procedure does not claim a completely closed build-toolchain reproduction.

An earlier ADA attempt used Python 3.6.15. Pip could reach PyPI but rejected `annotated-types` releases because of `Requires-Python`; the pinned 0.8.0 requires Python >=3.10, while the complete lock targets CPython 3.13. All 76 pinned versions and hashes were checked against official PyPI release metadata, and the 0.8.0 wheel was independently downloaded and hash-verified. Those historical checks establish artifact availability, not a completed ADA installation.

## 2. Check the installation

Check actual computational imports, packaged data, and entry points without reading wavefunctions. These commands can be run outside the checkout:

```bash
python - <<'PYTHON'
from importlib.metadata import version
from importlib.resources import files
import json
import vasp_sawf.inputs
import vasp_sawf.wavecar
import vasp_sawf.symmetry
import vasp_sawf.localize
for name in ('numpy', 'scipy', 'numba', 'irrep', 'wannierberri'):
    print(name, version(name))
assert version('irrep') == '2.6.3'
assert version('wannierberri') == '1.7.0'
decision = json.loads(files('vasp_sawf').joinpath('accepted_closure.json').read_text())
print('Packaged acceptance decision:', decision['id'])
PYTHON
sawf-extract --help
sawf-run --help
sawf-plot-bands --help
sawf-plot-symmetry --help
```

For development tests, install `python -m pip install -e ".[test]"` from the checkout. Then run tests without external DFT data on allocated compute resources:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -m 'not real_data' -p no:cacheprovider
```

Tests run from the checkout. They check software behavior, not reproduction of the SrVO₃ calculation on ADA. Real-data regression requires separately supplied read-only inputs, using the environment variables in [README.md](README.md). Record skipped or deselected real-data cases accurately.

## 3. Extract symmetry information on ADA

Run on allocated compute resources, replacing the example paths:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sawf-extract \
  --seed /path/to/interface/wannier90 \
  --wavecar /path/to/interface/WAVECAR \
  --outcar /path/to/interface/OUTCAR \
  --output /path/to/results/symmetry
```

Required inputs are one consistent set of `.win/.mmn/.amn/.eig`, the OUTCAR from that interface calculation, and the WAVECAR saved at its end. IBZ symmetry reduction has been validated; full-BZ WAVECAR is not required. An earlier SCF input WAVECAR alone is outside the validated entry point. See README for the full scope, including the single-center target and Γ-centered mesh restrictions. SrVO₃ validation does not imply tSnS validation.

The only outputs are `bloch.npz` and `report.json`. To run step 2 locally, download the complete `symmetry` directory and retain the original interface files with identical contents. WAVECAR and UNK do not need to be downloaded.

The program uses one Python process with NumPy/SciPy numerical kernels. Do not start the same extraction command on multiple MPI ranks; GPUs are not required. The pinned WannierBerri implementation imports Ray even in serial mode, so Ray is a runtime dependency; the workflow does not start a Ray service or enable parallel execution. Begin with the BLAS thread settings used in small-system validation, then measure scaling separately. Large-system memory and wall time must be estimated from actual NG/NK/NB and calibrated with small tests. Jobs are not submitted automatically.

### ADA batch template and the current tSnS limitation

[extract_ada.sbatch](extract_ada.sbatch) uses the following paths:

| Purpose | Path |
| --- | --- |
| Source checkout | `$HOME/.src/vasp_sawf` |
| Python environment | The compatible environment active at submission, with `vasp-sawf` installed |
| Read-only interface inputs | `/ptmp/tSnS/scdm661` |
| Job working directory and logs | `/ptmp/tSnS/sawf661` |
| New extraction output | `/ptmp/tSnS/sawf661/symmetry` |

Use the Python 3.13 environment into which you installed the package when submitting. The template inherits that environment and binds both preflight and extraction to its `python`. Extraction uses `python -m vasp_sawf.extract_symmetry`, the module behind `sawf-extract`, so an unrelated command elsewhere on `PATH` cannot select a different interpreter. The template does not purge or load modules, activate a fixed virtual environment, or set `PYTHONPATH`. Any runtime libraries or module settings needed by your chosen interpreter must therefore be present in the submitted environment.

**The current tSnS structure is not supported by the extraction algorithm yet.** A lightweight local check on 2026-09-28, using the actual WIN structure and fixed IrRep 2.6.3, found the candidate structural grey group `P2_11'`, with four operations. The nontrivial unitary operation has fractional rotation `diag(-1, 1, -1)` and translation `(0, 0.5, 0.3371566930075367)`. Its square is translation by `(0, 1, 0)`: the half translation along b cannot be removed by shifting the origin. Constructing a candidate grey group from structure does not establish the magnetic state of the calculation.

`_validate_grey_group` and `build_symmetry_maps` currently reject nonzero spatial translations. Simply removing those checks would be incorrect: MMN transport, independent coefficient transformations, and group-composition checks need the corresponding translation phases. The SAWF verification must use the same conventions. The screw has no individually fixed center modulo lattice translations, so the current single-center target and initial-guess alignment also require extension to multiple symmetry-related centers. The specific centers and local representation remain undetermined. These are implementation limits; the accepted SrVO3 numerical workflow is unchanged. No target Wannier representation is inferred from the tSnS SCDM AMN.

The template therefore performs the existing structural checks before coefficient I/O. For the local tSnS structure it is expected to stop at that check. The check reads only 128 logical WAVECAR header bytes plus the small interface files; it does not read coefficients. Do not submit a full tSnS extraction expecting it to succeed until translation support is implemented and independently tested. The local historical OUTCAR also has `LWAVE=F`; the contents of the remote directory have not been inspected. The actual remote WAVECAR/OUTCAR must meet the documented interface-output requirements; do not assume that an earlier SCF WAVECAR is the interface output.

After these prerequisites have been resolved, submission would be:

```bash
cd "$HOME/.src/vasp_sawf"
git pull --ff-only
python -m pip install -e .
mkdir -p /ptmp/tSnS/sawf661
sbatch extract_ada.sbatch
```

Update source only when the checkout is clean and no job uses it. The working directory must exist before `sbatch`, because Slurm opens its logs before executing the script. Logs append to fixed `extract.out` and `extract.err` files; extraction refuses to overwrite the existing `symmetry` directory. No input files are copied or modified. If moving the template to another system, change its input and working-directory paths. `#SBATCH` directives do not expand `$HOME` or shell variables; the working-directory directive therefore uses a literal path. See the [Slurm sbatch manual](https://slurm.schedmd.com/sbatch.html).

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
sawf-run \
  --seed /path/to/interface/wannier90 \
  --symmetry /path/to/results/symmetry \
  --center 0.5 0.5 0.5 --orbital t2g \
  --output /path/to/results/model
```

Both computational entry points refuse to overwrite existing output directories. Keep outputs separate from original inputs. Another material requires its own established target center and local representation; do not copy the SrVO₃ `t2g` target without justification.

See README for the optional DFT path, three band plots, and replotting commands. Keep data outside the repository so that updating code does not change existing results.

## Update the code

Record `git rev-parse HEAD` when installing. Later, run `git pull --ff-only` only with a clean working tree and no calculation currently using that source checkout. If package metadata or dependencies change, rerun `python -m pip install -e .` in the chosen environment and repeat the checks above. Editable source changes are otherwise visible immediately, so do not update a checkout used by a running job. Use `git switch --detach <full-commit>` to fix the source version for reproduction. To modify source, create a separate worktree from the relevant commit and keep original calculation inputs read-only.

## Validation status

The 2026-09-29 package installation was tested in a fresh temporary local Python 3.13 environment, with runtime dependencies resolved from official PyPI. Editable installation and `pip check` passed. From a directory outside the checkout, all four commands, actual computational imports, the packaged acceptance record, and the full regression suite passed: 190 tests passed and 16 external-data cases were skipped. Replacing the editable install with a built wheel produced the same result, with imports resolving to `site-packages`. The wheel contains only the package and its distribution metadata; its acceptance JSON is byte-identical to the source record. Slurm shell and embedded-Python syntax checks also passed. No production environment was modified, no original WAVECAR was read, and no ADA job was submitted.

Before the packaging change, local Linux/Python 3.13 checks on 2026-09-28 passed `pip check`, actual computational-module imports, both computational entry points' `--help`, and 184 portable tests; 16 external-data cases were deselected. The final English source was tested from a temporary copy containing only the release files. Four additional tests passed against the existing SrVO₃ symmetry bundle, checking legacy compatibility and rejection of altered evidence without reading WAVECAR. The temporary copy was removed after testing. The optional pyFFTW package is not installed, so WannierBerri uses its official NumPy FFT fallback. Installation and calculation reproduction on ADA remain untested. No cluster jobs have been submitted.
