# vasp_sawf: SCDM → SAWF

Two computational scripts extract symmetry information on the machine holding the WAVECAR, then use the original SCDM initial guess for Wannierization constrained by space-group symmetry and time reversal. Plotting scripts read existing numerical results; they do not compute or validate a model.

Repository: [gawcista/vasp_sawf](https://github.com/gawcista/vasp_sawf). See [DEPLOY_ADA.md](DEPLOY_ADA.md) for cloning, creating an isolated environment, and checking the installation on ADA. Run the scripts directly; `pip install .` is not required. The repository contains no calculation data. `DATA_ROOT` and local paths below record the existing validation setup, not required installation locations. Historical local commit IDs record provenance; the old development history and raw calculation data are not part of the public repository.

The user has accepted the coefficient-closure residual for the current SrVO₃ dataset. Both scripts run directly, without a trial option or repeated approval. [ACCEPTANCE.md](ACCEPTANCE.md) records the physical reasoning, evidence, and scope of that decision.

See [SYMMETRY.md](SYMMETRY.md) for the constraints imposed by SAWF, the comparison with ordinary Wannierization, and measured SrVO₃ residuals.

```text
extract_symmetry.py   Step 1: extract symmetry information
run_sawf.py           Step 2: run SAWF
plot_bands.py         Optional: plot existing band arrays
plot_symmetry.py      Optional: plot existing symmetry residuals
core/                Internal implementation shared by the two scripts
tests/               Mathematical, file-reading, and real-data regression tests
requirements.lock    Exact dependencies of the validated local environment
DEPLOY_ADA.md        ADA deployment and execution instructions
ACCEPTANCE.md        Acceptance basis for the current SrVO₃ coefficient closure
```

## Step 1: extract on ADA

```bash
python extract_symmetry.py \
  --seed /path/to/interface/wannier90 \
  --wavecar /path/to/interface/WAVECAR \
  --outcar /path/to/interface/OUTCAR \
  --output /path/to/results/symmetry
```

`--seed` is the shared prefix of the original `.win/.mmn/.amn/.eig` files. WIN must contain the complete lattice, atomic structure, and k-point list. OUTCAR must retain the IBZKPT/IBZKPT_HF and t-inv provenance tables and explicitly specify zero MAGMOM; the current entry point requires its reciprocal folding to match WIN. WAVECAR must have been saved at the end of the same calculation that generated the interface files. An earlier SCF input WAVECAR alone does not meet the requirements of the validated entry point. An IBZ calculation does not require recomputing full-BZ wavefunctions.

The program reads selected-band records with all G vectors and both spinor components. It obtains Γ-point anchors from IrRep, transports them across the full mesh using native VASP PAW MMN matrices, and checks independent IBZ transformations, group composition, TR², and matrix covariance. Original band indices are determined from WIN and the original band count in WAVECAR; excluded bands are not removed a second time from compact interface data. NNKP and UNK files are not required. The program neither exports complete compact wavefunctions nor reads, copies, or hashes the entire WAVECAR.

The outputs are `bloch.npz` and `report.json`. Together they form a small symmetry bundle; download the whole `symmetry` directory. Step 2 also needs the original WIN/MMN/AMN/EIG files. Their paths may change, but their contents must remain identical.

## Step 2: run SAWF locally

The following target has been validated for SrVO₃ only; do not reuse it for another material without justification:

```bash
python run_sawf.py \
  --seed /path/to/local/interface/wannier90 \
  --symmetry /path/to/local/symmetry \
  --center 0.5 0.5 0.5 --orbital t2g \
  --output /path/to/results/model
```

The script first performs ordinary localization from the original AMN to determine a reversible cell and column-basis alignment. It then runs SAWF from the aligned original SCDM guess. There is no need to run ordinary Wannierization separately or provide its gauge. `Projection` specifies only the target representation; it does not recompute projection AMN. The Hamiltonian is not averaged in post-processing.

The target center and orbital specify how the final basis should transform. SCDM specifies where the numerical iteration starts. These inputs serve different purposes. The SrVO₃ target is a V-centered t2g spinor basis in WannierBerri order: `dxz↑, dxz↓, dyz↑, dyz↓, dxy↑, dxy↓`.

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
python plot_bands.py /path/to/results/model/bands.npz --output /path/to/results/figures
```

Plotting depends only on NumPy and Matplotlib. It does not read WAVECAR, call IrRep or WannierBerri, or validate the model. Each of the three plots is saved as PNG and PDF. Replotting overwrites the same figure names. The reference energy is subtracted only during plotting.

The band style follows the existing `vaspsrc` example: 4×4 inches, Arial 18 pt, a 24 pt y-axis label, 3 pt axes borders, 0.5 pt curves, gray dashed lines at interior high-symmetry points, and transparent output at 600 dpi. Edit these parameters, energy limits, and colors at the top of the script; layout is controlled in `plot_bands()`. Figure text and repository documentation are in English.

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
python plot_bands.py ../../runs/srvo3/model/bands.npz --output ../../runs/srvo3/figures
python plot_symmetry.py ../../runs/srvo3/reference/comparison.json --output ../../runs/srvo3/figures
```

`plot_symmetry.py` plots saved residuals in English; it does not recompute symmetry. `reference` retains the ordinary gauge, H_R, and the provenance reports needed to check the existing comparison. It is not an additional computational entry point. Legacy provenance reports retain their original bytes and may therefore contain paths from before the directory cleanup. The layout above gives the current locations. Legacy-bundle regression uses `reference/report.json` and a relative link to `../symmetry/bloch.npz`, without storing another copy of the Bloch bundle.

Fixed directories do not mean that the computational scripts overwrite existing results. Before recomputing, remove only the generated results of the selected stage within the authorized scope, then reuse the same `--output` path. Do not remove original DFT inputs. Layout changes require only replotting, without rerunning calculations or creating dated directories.

## Other materials and current limits

For another material, provide its interface files, WAVECAR, OUTCAR, and independently established target center and orbitals. Band counts, mesh, elements, and symmetry-operation counts are read from the files; there are no SrVO₃ constants to edit for those quantities. Inputs are read-only, and outputs must go to a new directory outside the input directories.

The implementation currently supports nonmagnetic SOC grey groups, Cartesian SAXIS, a closed subspace with positive even `NB=NW`, a complete Γ-centered mesh with an actual Γ-point wavefunction, and zero space-group translations. WAVECAR must have RTAG45200, with energy data contained in one record per k point. Initial-guess alignment requires all columns to share one center uniquely fixed by the group. Multiple target centers, nonzero space-group translations, arbitrary spin axes, and arbitrary WAVECAR formats are not supported. Unsupported cases stop; changing paths alone does not make the workflow valid for every material.

SrVO₃ is the only material covered by real-data regression so far. Eight-dimensional analytical tests check dimension handling, not tSnS validity. The tSnS target representation and possible need for multiple centers remain to be established. Its 6×6×1 mesh must not be reduced.

The SrVO₃ coefficient-closure residual `2.589629272055618e-6` has been accepted for the current single-particle model. The [acceptance decision](ACCEPTANCE.md) is bound to the contents of the four original interface files and applies automatically within the approved `2.59e-6` bound. Other matrix checks and convergence requirements remain unchanged. The previously reviewed trial bundle can be used directly in step 2: the program applies the new decision in memory, without modifying that bundle or requiring another WAVECAR extraction. New results have status `ready` and `physical_acceptance_status=accepted_for_single_particle_model`; figures no longer carry a pending-acceptance note. Historical reports are unchanged and do not describe the current acceptance status. Other inputs do not inherit this decision, and residuals below the numerical reference do not by themselves establish physical acceptance for another dataset.

## Environment and validation

The measured local environment is Linux/Python 3.13 with WannierBerri 1.7.0, IrRep 2.6.3, NumPy 2.3.5, and SciPy 1.17.0. Exact dependencies are in `requirements.lock`. This is a local lock, not an ADA installation that has already been replayed. Run the two scripts directly; installing the project as a Python package is unnecessary.

The existing local interpreter is `DATA_ROOT/.sawf-bridge/envs/smoke-py313/bin/python`. Both stages currently use serial Python with NumPy/SciPy and the official libraries. Small-system tests set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. MPI is not implemented, and large-system parallel scaling has not been validated. Memory must not be inferred from total WAVECAR size. Jobs are not submitted automatically.

Run development tests with `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider`. Real-data tests require explicit read-only input paths through `SAWF_SRVO3_ROOT`, `SAWF_SRVO3_SEED`, `SAWF_SRVO3_WANNIER`, `SAWF_SRVO3_SOC`, and `SAWF_SRVO3_SYMMETRY`; otherwise they are skipped. The first four still refer to the original SrVO₃ calculations. Legacy-bundle regression uses `SAWF_SRVO3_SYMMETRY=DATA_ROOT/.sawf-bridge/runs/srvo3/reference`. Tests include independent pymatgen and raw-record references, analytical Hamiltonians, antiunitary conjugation, and rejection of automatic band removal or false convergence.

The 2026-09-24 two-script refactoring regression passed 187 tests; 3 tSnS tests were not run. The new SrVO₃ symmetry arrays and all eight model arrays were elementwise identical to the earlier converged results. SAWF without a DFT path also converged to the identical model. Measured extraction took 52.1 seconds, with 3,015,936 logical bytes read plus a 128-byte header precheck. Local SAWF including the path took 20.6 seconds. These are local small-system measurements, not ADA or tSnS performance estimates.

The acceptance-status update on the same day passed 197 tests; 3 tSnS tests were not run. Default extraction and default SAWF reusing the legacy bundle both returned `ready`. All eight model arrays and the three path-energy arrays remained elementwise unchanged; only the pending-acceptance annotation was removed. Current results are in `DATA_ROOT/.sawf-bridge/runs/srvo3/`.

On 2026-09-28, results were consolidated and the three band plots were redrawn in the style of the local `/home/gawcista/scripts/vaspsrc/tools/wanplot.py`. The symmetry-comparison plot retained its layout and was translated into English. All 18 checks in `tests/test_workflow.py` passed, including relocated legacy-bundle loading and rejection of tampering. Model, band, symmetry-bundle, and binding-report bytes were unchanged by migration. All four figures were saved as PNG/PDF and checked for dimensions, legibility, and English labels. That work did not rerun localization or read WAVECAR.

The 2026-09-28 deployment preflight passed 184 tests with `-m 'not real_data'`; 16 real-data cases were deselected. The final English source passed the same checks from a temporary copy of the release files, plus four tests of existing SrVO₃ legacy-bundle compatibility and rejection of altered evidence. These tests did not read WAVECAR. The only warning reported the optional pyFFTW fallback. This checks the local code and environment, not an ADA installation or a new material.
