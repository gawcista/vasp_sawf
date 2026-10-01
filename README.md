# vasp_sawf: SCDM → SAWF

Reuse VASP subspace-SCDM matrices in two steps: extract symmetry information where WAVECAR is stored, then run WannierBerri localization constrained by space-group symmetry and time reversal. Optional plotting commands read existing results.

Repository: [gawcista/vasp_sawf](https://github.com/gawcista/vasp_sawf). No calculation data are included. Historical local paths and commit IDs record provenance; the old development history is not part of the public repository.

## Install in your current environment

The current release requires Python 3.13 (`>=3.13,<3.14`). The upper bound reflects actual incompatibilities in the pinned Numba and Ray versions, not only the range tested locally; see the [Python 3.14 explanation](DEPLOY_ADA.md#why-python-314-is-rejected). With your chosen compatible environment active:

```bash
git clone https://github.com/gawcista/vasp_sawf.git
cd vasp_sawf
python -m pip install -e .
python -m pip check
```

Pip installs the package and its dependencies into the environment selected by `python`. No project-specific virtual environment or fixed ADA module is required. Editable installation uses this checkout directly: keep it in place and do not change its source while a calculation uses it. The Python import name is `vasp_sawf`; the distribution name is `vasp-sawf`.

The installed commands are available from any working directory in that environment:

| Command | Purpose |
| --- | --- |
| `sawf-extract` | Extract the Bloch symmetry bundle |
| `sawf-run` | Run SAWF from the original SCDM initial guess |
| `sawf-plot-bands` | Plot saved DFT, ordinary Wannier, and SAWF bands |
| `sawf-plot-symmetry` | Plot saved symmetry-comparison residuals |

Critical library versions are pinned in `pyproject.toml`. [DEPLOY_ADA.md](DEPLOY_ADA.md) covers ADA execution, parallel extraction, and the optional full artifact lock. Source files are in `vasp_sawf/`; tests are in `tests/`.

The current SrVO₃ coefficient-closure residual has been accepted: see [ACCEPTANCE.md](ACCEPTANCE.md) for the reasoning and dataset scope. No trial option or repeated approval is required for that dataset. [SYMMETRY.md](SYMMETRY.md) explains the constraints and comparison with ordinary Wannierization.

## Step 1: extract on ADA

For a single batch job followed by downloading the result directory, use the [ADA submission instructions](DEPLOY_ADA.md#submit-once-download-the-result-continue-locally). The batch script calls the same extraction entry point shown below; its input and symmetry checks are part of the program.

```bash
cd /path/to/interface
sawf-extract --output /path/to/results/symmetry
```

The input defaults are `--seed wannier90`, `--wavecar WAVECAR`, and `--outcar OUTCAR`, relative to the current directory. Override them only when needed. `--output` remains explicit and must be outside the input directories. `--seed` is the shared prefix of the original `.win/.mmn/.amn/.eig` files. WIN must contain the complete lattice, atomic structure, and k-point list. OUTCAR must retain the IBZKPT/IBZKPT_HF and t-inv provenance tables and explicitly specify zero MAGMOM; the current entry point requires its reciprocal folding to match WIN. WAVECAR must have been saved at the end of the same calculation that generated the interface files. An earlier SCF input WAVECAR alone does not meet the requirements of the validated entry point. An IBZ calculation does not require recomputing full-BZ wavefunctions.

After checking structural symmetry and the full mesh, the program reads selected-band records with all G vectors and both spinor components. It obtains Γ-point anchors from IrRep, transports them across the full mesh using native VASP PAW MMN matrices, and checks independent IBZ transformations, group composition, TR², and matrix covariance. Original band indices are determined from WIN and the original band count in WAVECAR; excluded bands are not removed a second time from compact interface data. NNKP and UNK files are not required. The program neither exports complete compact wavefunctions nor reads, copies, or hashes the entire WAVECAR.

Each worker holds one stored k point and returns small matrices and resource records. The program reads k-record coefficient counts before choosing concurrency; it does not retain all selected wavefunctions at once. On Slurm, the default worker limit follows `SLURM_CPUS_PER_TASK`; elsewhere it is one. Available CPUs, independent IBZ tasks, and estimated memory can reduce that limit. Optional `--workers N` and `--memory-gb GIB` set explicit upper limits. Γ is evaluated first, and its measured process peak can raise the per-worker memory estimate before parallel work begins. A 1.1 TB WAVECAR does not require 1.1 TB resident memory because only the selected band records are read. No MPI launcher or GPU is required.

The outputs are `bloch.npz` and `report.json`. Together they form a small symmetry bundle; download the whole `symmetry` directory. Step 2 also needs the original WIN/MMN/AMN/EIG files. Their paths may change, but their contents must remain identical.

## Step 2: run SAWF locally

The following target has been validated for SrVO₃ only; do not reuse it for another material without justification:

```bash
sawf-run \
  --seed /path/to/local/interface/wannier90 \
  --symmetry /path/to/local/symmetry \
  --center 0.5 0.5 0.5 --orbital t2g \
  --output /path/to/results/model
```

The script first performs ordinary localization from the original AMN to determine a reversible cell and column-basis alignment. It then runs SAWF from the aligned original SCDM guess. There is no need to run ordinary Wannierization separately or provide its gauge. `Projection` specifies only the target representation; it does not recompute projection AMN. The Hamiltonian is not averaged in post-processing.

The target center and orbital specify how the final basis should transform. SCDM specifies where the numerical iteration starts. These inputs serve different purposes. The SrVO₃ target is a V-centered t2g spinor basis in WannierBerri order: `dxz↑, dxz↓, dyz↑, dyz↓, dxy↑, dxy↓`.

For multiple independent target orbits, repeat `--center X Y Z --orbital NAME` once per orbit, in matching order. Each center is one representative: WannierBerri generates its symmetry-related centers, so do not list those centers again as separate orbits. Group operations may permute centers and shift them into neighboring cells. Free positional coordinates may relax while the final centers must preserve these affine symmetry relations. The expanded spinor representation must match every target band; the program checks compatibility without dropping bands. A failed initial-guess alignment is not by itself a proof of a physical obstruction.

Step 2 produces only:

- `model.npz`: full-mesh U, k points, original EIG, centers and spreads, lattice, and real-space Hamiltonian.
- `summary.json`: convergence information, the applied acceptance decision, and key residuals.
- `bands.npz`: generated only when the optional DFT path input below is supplied.

The model uses `H(k)=Σ_R exp(+2πi k·R) H_R`. Here k is in reciprocal fractional coordinates, R is an integer lattice vector, H_R is in eV, lattice vectors and centers are in Å, and spreads are in Å². H_R already includes the official orbital-pair Wigner–Seitz weights; do not divide by degeneracies again. Reconstructing the Hamiltonian requires only the model. Checking its specific space-group representation also requires the symmetry bundle from step 1 and the target information in the summary.

## Three band plots

Add the following options to the step 2 command:

```bash
  --dft-eigenval /path/to/dft_path/EIGENVAL --energy-reference-ev 5.14729
```

This reference energy applies only to the current SrVO₃ example. The path must match `kpoint_path` in WIN. The program evaluates ordinary Wannier and SAWF interpolation along that same path and stores the unshifted arrays in eV. The DFT path file is optional for SAWF itself.

```bash
sawf-plot-bands /path/to/results/model/bands.npz --output /path/to/results/figures
```

Plotting depends only on NumPy and Matplotlib. It does not read WAVECAR, call IrRep or WannierBerri, or validate the model. Each of the three plots is saved as PNG and PDF. Replotting overwrites the same figure names. The reference energy is subtracted only during plotting.

The band style follows the existing `vaspsrc` example: 4×4 inches, Arial 18 pt, a 24 pt y-axis label, 3 pt axes borders, 0.5 pt curves, gray dashed lines at interior high-symmetry points, and transparent output at 600 dpi. Edit these parameters, energy limits, and colors in [vasp_sawf/plot_bands.py](vasp_sawf/plot_bands.py); layout is controlled in `plot_bands()`. Figure text and repository documentation are in English.

The SrVO₃ path is denser than the original example. `XTICK_FONT_SIZE=14` separately controls path labels so that `X|R` does not overlap the neighboring `M`; other text remains 18 pt. These settings can all be edited at the top of the script.

`bands.npz` contains `distance`, `segment_slices` (start-inclusive, end-exclusive), `tick_positions`, `tick_labels`, `dft/wannier/sawf`, and `energy_reference_ev`. Disconnected path segments are not joined. The legacy `sawf_note` field is not used for plotting, and acceptance notes are not added to figures.

## Fixed results directory

The current local setup retains one SrVO₃ result set, without date or sequence-number versions:

```text
DATA_ROOT/.sawf-bridge/runs/srvo3/
├── symmetry/    bloch.npz and the current report.json
├── model/       model.npz, bands.npz, and summary.json
├── figures/     Three band plots and a symmetry-residual plot, each as PNG/PDF
└── reference/   Ordinary comparison, comparison.json, and legacy-bundle reports
```

To replot existing results from the current development worktree:

```bash
sawf-plot-bands ../../runs/srvo3/model/bands.npz --output ../../runs/srvo3/figures
sawf-plot-symmetry ../../runs/srvo3/reference/comparison.json --output ../../runs/srvo3/figures
```

`sawf-plot-symmetry` plots saved residuals in English; it does not recompute symmetry. `reference` retains the ordinary gauge, H_R, and the provenance reports needed to check the existing comparison. It is not an additional computational entry point. Legacy provenance reports retain their original bytes and may therefore contain paths from before the directory cleanup. The layout above gives the current locations. Legacy-bundle regression uses `reference/report.json` and a relative link to `../symmetry/bloch.npz`, without storing another copy of the Bloch bundle.

Fixed directories do not mean that the computational scripts overwrite existing results. Before recomputing, remove only the generated results of the selected stage within the authorized scope, then reuse the same `--output` path. Do not remove original DFT inputs. Layout changes require only replotting, without rerunning calculations or creating dated directories.

## Other materials and current limits

For another material, provide its interface files, WAVECAR, OUTCAR, and independently established target center and orbitals. Band counts, mesh, elements, and symmetry-operation counts are read from the files; there are no SrVO₃ constants to edit for those quantities. Inputs are read-only, and outputs must go to a new directory outside the input directories.

The implementation supports nonmagnetic SOC grey groups, Cartesian SAXIS, a closed subspace with positive even `NB=NW`, and a complete Γ-centered mesh with an actual Γ-point wavefunction. Spatial transformations use the full rotation and translation, including screws, glides, origin-dependent translations, and integer lattice shifts; antiunitary transformations include complex conjugation. Targets may contain multiple symmetry-related centers and independent orbits. WAVECAR must have RTAG45200, with energy data contained in one record per k point. Arbitrary spin axes and arbitrary WAVECAR formats remain unsupported. Unsupported inputs or inconsistent representations stop; changing paths alone does not make the workflow valid for every material.

SrVO₃ is the only material covered by real-data regression so far. Analytical tests of dimension and general space-group transformations do not establish tSnS validity. The tSnS target centers and local representation remain to be established; no successful tSnS extraction or SAWF result is claimed. See the resource policy and input requirements in [DEPLOY_ADA.md](DEPLOY_ADA.md). Its 6×6×1 mesh must not be reduced.

The SrVO₃ coefficient-closure residual `2.589629272055618e-6` has been accepted for the current single-particle model. The [acceptance decision](ACCEPTANCE.md) is bound to the contents of the four original interface files and applies automatically within the approved `2.59e-6` bound. Other matrix checks and convergence requirements remain unchanged. The previously reviewed trial bundle can be used directly in step 2: the program applies the new decision in memory, without modifying that bundle or requiring another WAVECAR extraction. New results have status `ready` and `physical_acceptance_status=accepted_for_single_particle_model`; figures no longer carry a pending-acceptance note. Historical reports are unchanged and do not describe the current acceptance status. Other inputs do not inherit this decision, and residuals below the numerical reference do not by themselves establish physical acceptance for another dataset.

## Environment and validation

The 2026-10-01 general space-group update passed analytical screw/glide/centering and multi-center SAWF regressions, including antiunitary operations that exchange distinct k points. Serial and two-process extraction reproduced the accepted real SrVO3 Bloch matrices. The updated localization reproduced the existing model, spreads, DFT-path spectra, and additional off-mesh spectra without changing the input matrices. Actual tSnS structure and MMN-neighbor preparation passed on its original `6x6x1` mesh; tSnS wavefunction extraction and SAWF have not yet been run. See [validation details](DEPLOY_ADA.md#validation-status).

On 2026-09-29, editable installation with dependency resolution succeeded in a fresh temporary Python 3.13 environment using official PyPI. `pip check` passed. The complete test suite ran outside the checkout with 190 passed and 16 external-data tests skipped. A normal wheel was then built and installed in place of the editable package; imports resolved to `site-packages`, and the same suite again passed 190 tests with 16 skipped. Both modes checked all four commands, real computational imports, and the packaged acceptance record without relying on `PYTHONPATH`. The optional pyFFTW warning used WannierBerri's NumPy fallback. Core computation files and the acceptance JSON remained byte-identical during the namespace move. These checks did not read an original WAVECAR or execute an ADA job, and they do not extend the supported physical cases.

The measured local environment is Linux/Python 3.13 with WannierBerri 1.7.0, IrRep 2.6.3, NumPy 2.3.5, and SciPy 1.17.0. Exact dependencies are in `requirements.lock`. This lock was validated locally. The user has since reported completing installation on ADA with Python 3.13; ADA calculations have not yet been verified. Install this release into the compatible environment with `python -m pip install -e .` and use the installed commands.

The historical local validation interpreter is `DATA_ROOT/.sawf-bridge/envs/smoke-py313/bin/python`; it is not a required installation path. Extraction uses bounded per-k Python worker processes, with one numerical-library thread per worker. Γ anchoring and MMN transport precede the parallel IBZ checks; local SAWF retains its existing execution mode. MPI is not implemented, and large-system parallel scaling has not been validated. The memory estimate uses actual coefficient counts and leaves 25% of the detected budget as headroom, but is not a measured peak or guarantee against exhaustion. Jobs are not submitted automatically.

For development, install the test extra with `python -m pip install -e ".[test]"`, then run `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider` from the checkout. Real-data tests require explicit read-only input paths through `SAWF_SRVO3_ROOT`, `SAWF_SRVO3_SEED`, `SAWF_SRVO3_WANNIER`, `SAWF_SRVO3_SOC`, and `SAWF_SRVO3_SYMMETRY`; otherwise they are skipped. The first four still refer to the original SrVO₃ calculations. Legacy-bundle regression uses `SAWF_SRVO3_SYMMETRY=DATA_ROOT/.sawf-bridge/runs/srvo3/reference`. Tests include independent pymatgen and raw-record references, analytical Hamiltonians, antiunitary conjugation, and rejection of automatic band removal or false convergence.

The 2026-09-24 two-script refactoring regression passed 187 tests; 3 tSnS tests were not run. The new SrVO₃ symmetry arrays and all eight model arrays were elementwise identical to the earlier converged results. SAWF without a DFT path also converged to the identical model. Measured extraction took 52.1 seconds, with 3,015,936 logical bytes read plus a 128-byte header precheck. Local SAWF including the path took 20.6 seconds. These are local small-system measurements, not ADA or tSnS performance estimates.

The acceptance-status update on the same day passed 197 tests; 3 tSnS tests were not run. Default extraction and default SAWF reusing the legacy bundle both returned `ready`. All eight model arrays and the three path-energy arrays remained elementwise unchanged; only the pending-acceptance annotation was removed. Current results are in `DATA_ROOT/.sawf-bridge/runs/srvo3/`.

On 2026-09-28, results were consolidated and the three band plots were redrawn in the style of the local `/home/gawcista/scripts/vaspsrc/tools/wanplot.py`. The symmetry-comparison plot retained its layout and was translated into English. All 18 checks in `tests/test_workflow.py` passed, including relocated legacy-bundle loading and rejection of tampering. Model, band, symmetry-bundle, and binding-report bytes were unchanged by migration. All four figures were saved as PNG/PDF and checked for dimensions, legibility, and English labels. That work did not rerun localization or read WAVECAR.

The 2026-09-28 deployment preflight passed 184 tests with `-m 'not real_data'`; 16 real-data cases were deselected. The final English source passed the same checks from a temporary copy of the release files, plus four tests of existing SrVO₃ legacy-bundle compatibility and rejection of altered evidence. These tests did not read WAVECAR. The only warning reported the optional pyFFTW fallback. This checks the local code and environment, not an ADA installation or a new material.
