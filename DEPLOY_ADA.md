# Install and run on ADA

Install `vasp-sawf` into your chosen current Python environment and use its commands from any working directory. No project-specific virtual environment or fixed environment module is required. Run extraction on ADA where WAVECAR is stored; SAWF can run on ADA or locally without WAVECAR.

The current release requires Python 3.13 (`>=3.13,<3.14`), matching the validated range. Critical versions are pinned in `pyproject.toml`, including WannierBerri 1.7.0, IrRep 2.6.3, NumPy 2.3.5, SciPy 1.17.0, and Numba 0.62.1. The user has reported completing installation on ADA in Python 3.13. Calculation reproduction and resource measurements on ADA have not yet been verified.

## Submit once, download the result, continue locally

With the installed Python 3.13 environment active on ADA:

```bash
mkdir -p /ptmp/tSnS/sawf661
sbatch "$HOME/.src/vasp_sawf/extract_ada.sbatch"
```

The batch file runs one extraction command using the inherited environment. Input and symmetry checks are performed inside the program; they do not need a separate batch-script preflight. A successful extraction creates:

```text
/ptmp/tSnS/sawf661/symmetry/
├── bloch.npz
└── report.json
```

After the job succeeds, run the following on your local machine, replacing `YOUR_ADA_SSH_HOST` with your working ADA SSH host or alias:

```bash
scp -r YOUR_ADA_SSH_HOST:/ptmp/tSnS/sawf661/symmetry ./
```

Use this bundle with the identical original `.win/.mmn/.amn/.eig` files already stored locally, then run `sawf-run` with the material's established target representation. The [local SAWF command below](#4-run-sawf) shows the validated SrVO₃ example. Neither WAVECAR nor UNK needs to be downloaded.

The template reserves an exclusive large-memory node and lets the extractor choose a bounded number of workers. General rotation-plus-translation operations are handled by the code; this does not establish that the actual tSnS eight-band subspace or an unspecified target representation is compatible. tSnS extraction, localization, and ADA resource use remain to be verified with its actual inputs.

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

### Why Python 3.14 is rejected

The reported error `Python: 3.14.6 not in '<3.14,>=3.13'` cannot be fixed by removing this project's upper bound alone. The current dependencies have independent incompatibilities:

- Numba 0.62.1 explicitly rejects Python >=3.14 in its [version guard](https://github.com/numba/numba/blob/0.62.1/setup.py#L20-L48), including source builds. Its [official support table](https://numba.readthedocs.io/en/0.62.1/user/installing.html#version-support-information) lists Python >=3.10,<3.14. PyPI's less restrictive `Requires-Python: >=3.10` metadata does not override that build-time guard.
- Ray 2.51.1 provides no CPython 3.14 wheel and no source distribution on [PyPI](https://pypi.org/project/ray/2.51.1/#files). Its [fixed-version build script](https://github.com/ray-project/ray/blob/ray-2.51.1/python/setup.py) also limits supported Python versions to 3.9-3.13. Ray is required here because WannierBerri 1.7.0's `wannierise/wannierizer.py` imports it even for this serial workflow.

These conditions were checked against official release metadata and source on 2026-10-01. Executing only Numba's upstream version guard with a simulated Python 3.14.6 version reproduced its rejection; this was not a native Python 3.14 installation test. The existing pins and upper bound remain unchanged. Use a Python 3.13 environment for the current release. Supporting Python 3.14 requires a separately checked dependency set, real installation and import tests under that interpreter, and numerical regression before changing the advertised support range. Removing version guards or using `--ignore-requires-python` would not provide that compatibility.

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

On allocated compute resources, run directly from the interface directory:

```bash
cd /path/to/interface
sawf-extract --output /path/to/results/symmetry
```

The defaults are `--seed wannier90`, `--wavecar WAVECAR`, and `--outcar OUTCAR`. Only the output path is required; keep it outside the protected input directories. Use explicit input options when file names differ.

Required inputs are one consistent set of `.win/.mmn/.amn/.eig`, the OUTCAR from that interface calculation, and the WAVECAR saved at its end. IBZ symmetry reduction is supported; full-BZ WAVECAR is not required. An earlier SCF input WAVECAR alone is outside the validated entry point. The algorithm retains its complete Γ-centered mesh, Cartesian spin-axis, and closed `NB=NW` spinor-subspace requirements. The `6×6×1` tSnS mesh must not be reduced.

The only outputs are `bloch.npz` and `report.json`. Download the complete `symmetry` directory and retain identical original interface files locally. No wavefunctions or UNK files are transferred.

### One exclusive large-memory node

[extract_ada.sbatch](extract_ada.sbatch) uses:

| Purpose | Setting |
| --- | --- |
| Source checkout | `$HOME/.src/vasp_sawf` |
| Read-only interface inputs | `/ptmp/tSnS/scdm661` |
| Job directory, logs, and caches | `/ptmp/tSnS/sawf661` |
| New result directory | `/ptmp/tSnS/sawf661/symmetry` |
| Allocation | `p.large`, one node, one task, 72 physical CPU cores |
| Node and memory | `--exclusive --mem=0` |
| Time limit | `24:00:00` |

ADA's large-memory nodes have 72 physical CPU cores and 2 TB of RAM. The template requests one exclusive node and all scheduler-available memory. `--exclusive` alone does not request all memory; `--mem=0` does. `--hint=nomultithread` selects one hardware thread per physical core. The 24-hour limit is the partition maximum, not a runtime estimate. These settings follow the [official ADA documentation](https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/MPSD_PKS_ADA.html) and [Slurm allocation semantics](https://slurm.schedmd.com/sbatch.html).

Submit with the installed Python 3.13 environment active. The script inherits it, sets single-threaded numerical kernels, and makes one `srun` call. That call changes into the input directory so default file names work, while logs, caches, and output remain in the separate job directory. It invokes `python -m vasp_sawf.extract_symmetry`, the same entry point as `sawf-extract`. No module selection, environment activation, file copies, or script-level preflight are needed.

Create the job directory before `sbatch`, because Slurm opens its logs before executing the script. Logs append to fixed `extract.out` and `extract.err` files; extraction refuses to overwrite an existing `symmetry` directory. For another material, change the two directory paths in the template. `#SBATCH` directives do not expand shell variables.

### Parallel execution and bounded memory

The program first reads the WAVECAR headers and one energy/count record per stored k point. It then knows the actual number of G vectors and selected bands without reading coefficients. A worker reads one complete selected-band k point, checks its transformations, and returns only small matrices, residuals, timing, and read records. Wavefunction arrays are not sent between processes or retained across the entire IBZ.

The Γ-point anchor and small MMN transport run first. Independent remaining IBZ checks use Python processes. The worker count is limited by the requested count, available CPU allocation, number of remaining k-point tasks, and estimated memory. The default requested count is `SLURM_CPUS_PER_TASK`, or one outside Slurm. `--workers N` sets an explicit upper limit. Numerical-library threads are fixed to one per worker to avoid multiplying CPU use.

An exclusive 72-core allocation does not guarantee 72 concurrent workers: for 24 stored k points, at most 23 remaining k-point tasks can run concurrently after Γ. Python G-vector enumeration, process startup, uneven k-point work, and shared-filesystem I/O also limit scaling. The program does not launch MPI ranks, use GPUs, or start a Ray service. Ray remains an upstream WannierBerri dependency.

The memory budget uses available host memory, cgroup v1/v2 limits including ancestors, and the Slurm per-node or per-CPU allocation. An unreadable declared memory controller requires an explicit `--memory-gb GIB` budget instead of assuming the whole host is available. This option does not request more memory from Slurm. Only 75% of the detected budget is assigned to estimated worker storage. If even one worker does not fit, extraction stops before reading coefficients. After Γ, the program refreshes available memory and raises its per-worker estimate to at least the measured Γ-process peak before selecting parallel concurrency. That high-water mark includes the parent process; other k points can still have different peaks.

Let `C_k = 2 NG_k` be the coefficient count in the WAVECAR k record, including both spinor components, and let `NB` be the number of selected bands. Selected coefficient I/O is `8 NB Σ_k C_k` bytes for RTAG45200. One worker's raw `complex64` array needs `8 NB C_max` bytes; one `complex128` copy needs `16 NB C_max` bytes. The estimate includes six such working copies, G arrays and indices, a Python-object allowance, and 1 GiB for imported libraries. All original G vectors, both spinor components, and the original precision are preserved.

This model is a heuristic with headroom, not a guaranteed RSS bound. IrRep's G enumeration builds Python lists and sorting arrays; LAPACK and allocation behavior add transients. The parent process and filesystem cache also consume memory. The 1 GiB library allowance exceeds a measured local small-fixture metadata process peak of approximately 588 MiB. It is not an ADA measurement.

A 1.1 TB WAVECAR does not imply a 1.1 TB resident working set. Unselected band coefficient records are never read, and working coefficient storage now scales with concurrent k points rather than all stored k points. Full-node reservation follows the requested deployment policy; it is not a claim that the calculation requires 2 TB. Actual tSnS resource needs and parallel speedup have not been measured.

`report.json` records the memory estimate and chosen worker count, stage times, logical read ledgers, and per-k resource observations. `total_logical_read_bytes` includes the initial 128-byte header check, metadata inspection, and worker reads. `peak_rss_kib` is the parent-process maximum; worker observations are lifetime process maxima and cannot be summed to obtain simultaneous node RSS. Check Slurm accounting after the job:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,AllocCPUS
```

Inspect the extraction step as well as the job row. Logical record bytes differ from physical filesystem I/O. No ADA job has been submitted during development.

### Symmetry scope and tSnS

The code handles general space-group operations `{S|t}`, including fractional translations, integer lattice shifts from group composition and reciprocal folding, and antiunitary conjugation. This applies to screws, glides, and origin-dependent translations; it is not a tSnS-specific exception. Multiple symmetry-related Wannier centers and independent target orbits are supported.

The local tSnS structure has candidate grey group `P2_11'`. Its nontrivial unitary operation has `S = diag(-1, 1, -1)` and `t = (0, 0.5, 0.3371566930075367)`; applying it twice gives the lattice translation `(0, 1, 0)`. Constructing a grey group from structure alone does not establish the magnetic state. Input, subspace closure, group composition, and target-compatibility checks remain mandatory.

SrVO₃ provides the real-data regression. Generic analytical tests do not certify tSnS. Its intended Wannier centers and local representations are still not specified, and no tSnS SAWF result is claimed. The remote WAVECAR/OUTCAR contents have not been inspected; they must meet the same documented interface-output requirements. The historical local tSnS OUTCAR has `LWAVE=F`, so it cannot establish that a final interface WAVECAR was saved remotely.

## 4. Run SAWF

Use a machine with the same dependencies. The center and orbital below apply only to the validated SrVO₃ example:

```bash
sawf-run \
  --seed /path/to/interface/wannier90 \
  --symmetry /path/to/results/symmetry \
  --center 0.5 0.5 0.5 --orbital t2g \
  --output /path/to/results/model
```

Both computational entry points refuse to overwrite existing output directories. Keep outputs separate from original inputs. For each independent target orbit, give one `--center X Y Z --orbital NAME` pair; repeat the pair for additional independent orbits. WannierBerri generates the symmetry-related centers automatically. Do not copy the SrVO₃ `t2g` target to another material without justification.

See README for the optional DFT path, three band plots, and replotting commands. Keep data outside the repository so that updating code does not change existing results.

## Update the code

Record `git rev-parse HEAD` when installing. Later, run `git pull --ff-only` only with a clean working tree and no calculation currently using that source checkout. If package metadata or dependencies change, rerun `python -m pip install -e .` in the chosen environment and repeat the checks above. Editable source changes are otherwise visible immediately, so do not update a checkout used by a running job. Use `git switch --detach <full-commit>` to fix the source version for reproduction. To modify source, create a separate worktree from the relevant commit and keep original calculation inputs read-only.

## Validation status

The final local suite for this update passed **247 tests, with 13 external-data cases skipped**, in 156.38 seconds. It ran from an editable install in a temporary Python 3.13 environment using the pinned scientific dependencies read-only. `SAWF_SRVO3_WANNIER` pointed to the small original interface calculation, and `SAWF_SRVO3_SYMMETRY` to its accepted symmetry bundle; other external fixtures were not supplied. Tests covered serial/process extraction, independent reader references, and preservation of completed Gamma I/O and resource evidence after an injected transport failure. The only warning was the optional pyFFTW import falling back to NumPy. Both installed computational commands passed `--help` outside the checkout; `bash -n extract_ada.sbatch` and `git diff --check` passed. No production environment was changed and no ADA job was submitted.

The general-translation implementation was checked on 2026-10-01 against analytical screw, glide, centering, shifted-origin, and spinor/TR examples. The complete SAWF tests include a four-point mesh where TR exchanges two distinct k points, multiple target centers, and all bands retained. These tests exercise translation phases and full-grid gauge reconstruction rather than only checking operations at Gamma.

The real SrVO3 selected-band reader agrees with independent pymatgen and raw-record references. Serial and two-process extraction reproduce the accepted Bloch matrices within `1e-12`, with each of the 120 selected band/k records read once. The new multi-center localization path converged in 21 iterations on the original 216-point, six-band dataset. Its eight model arrays were elementwise identical to the existing accepted model; this was an observed regression result, not a required gauge-equivalence criterion. Total spread remained `11.140780230059036` square angstroms. Spectra were unchanged at all 120 DFT path points and 37 additional off-mesh points. The original interface matrices remained unchanged. The localization run took 29.84 seconds with a 617100 KiB process peak locally; these are not ADA performance estimates.

The actual local tSnS WIN and compact interface matrices also passed structural and full-neighbor preparation on the unchanged `6x6x1` mesh: 36 k points, eight bands, and four candidate grey-group operations. The screw square yields the lattice translation `(0,1,0)` with the spinor factor `-1`. No tSnS wavefunction coefficients were read. This establishes that the former zero-translation restriction has been addressed at preparation level; actual subspace closure, target compatibility, and SAWF remain unverified for tSnS.

The 2026-09-29 package installation was tested in a fresh temporary local Python 3.13 environment, with runtime dependencies resolved from official PyPI. Editable installation and `pip check` passed. From a directory outside the checkout, all four commands, actual computational imports, the packaged acceptance record, and the full regression suite passed: 190 tests passed and 16 external-data cases were skipped. Replacing the editable install with a built wheel produced the same result, with imports resolving to `site-packages`. The wheel contains only the package and its distribution metadata; its acceptance JSON is byte-identical to the source record. The then-current batch script passed shell and embedded-Python syntax checks. No production environment was modified, no original WAVECAR was read, and no ADA job was submitted.

Before the packaging change, local Linux/Python 3.13 checks on 2026-09-28 passed `pip check`, actual computational-module imports, both computational entry points' `--help`, and 184 portable tests; 16 external-data cases were deselected. The final English source was tested from a temporary copy containing only the release files. Four additional tests passed against the existing SrVO₃ symmetry bundle, checking legacy compatibility and rejection of altered evidence without reading WAVECAR. The temporary copy was removed after testing. The optional pyFFTW package is not installed, so WannierBerri uses its official NumPy FFT fallback. The user subsequently reported a completed ADA installation with Python 3.13. ADA calculation reproduction remains unverified; no cluster job was submitted during these local checks.
