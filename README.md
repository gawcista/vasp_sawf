# vasp_sawf: VASP SCDM → SAWF → WANPROJ

Extract symmetry information for an existing VASP SCDM subspace, run symmetry-adapted Wannierization with WannierBerri, and export the final coefficients for VASP bare Coulomb integrals. Calculation data and licensed VASP source are not included.

## Installation

Python 3.13 is required (`>=3.13,<3.14`). Critical dependencies are pinned in `pyproject.toml`.

```bash
git clone https://github.com/gawcista/vasp_sawf.git
cd vasp_sawf
python -m pip install .
python -m pip check
```

For development, use `python -m pip install -e ".[test]"`. `requirements.lock` is an optional Linux/Python 3.13 artifact snapshot; it is not a portable environment specification. Keep an editable checkout unchanged while a calculation uses it.

| Command | Purpose |
| --- | --- |
| `sawf-extract` | Extract and validate a full-mesh Bloch symmetry bundle |
| `sawf-run` | Localize the original SCDM guess with symmetry constraints; export WANPROJ |
| `sawf-export-wanproj` | Export an existing validated model with its bound symmetry bundle |
| `sawf-plot-bands` | Plot saved DFT, ordinary Wannier, and SAWF band data |
| `sawf-plot-symmetry` | Plot saved symmetry residuals |

## Inputs and extraction

The original `.win/.amn/.mmn/.eig` files share a seed prefix. WIN must contain the full lattice, atomic structure, and complete k-point mesh. Provide WAVECAR saved at the end of the same interface calculation and its OUTCAR, including reciprocal folding tables and explicit zero MAGMOM. Matching band energies alone does not identify a shared Bloch gauge. An inherited SCF WAVECAR requires evidence that it matches the generated interfaces; an `ALGO=None` run alone does not establish that relationship.

Run on the machine holding WAVECAR:

```bash
cd /path/to/interface
sawf-extract --output /path/to/results/symmetry
```

Defaults are `--seed wannier90 --wavecar WAVECAR --outcar OUTCAR`. Override paths as needed. All computational outputs require a new directory outside the input directories; existing outputs and symlinks are rejected.

Extraction reads selected-band records with complete G vectors and both spinor components. It anchors the representation using IrRep, transports it with native PAW MMN, and checks independent IBZ transformations, group composition, time reversal, and covariance. It does not read, copy, or hash the entire WAVECAR. NNKP and UNK are unnecessary.

The outputs `bloch.npz` and `report.json` form one bound bundle. Download both along with the unchanged original WIN/AMN/MMN/EIG for localization.

Independent stored k points can be processed in parallel. `--workers N` and `--memory-gb GIB` set upper limits; CPU allocation, coefficient counts, and available memory can reduce concurrency. Under Slurm request one task with multiple CPUs, e.g. `sbatch --cpus-per-task=8 extract_ada.sbatch INPUT_DIR OUTPUT_DIR`; specify your cluster's partition, memory, and time separately. The helper uses the installed package. No jobs are submitted automatically.

## Symmetry-adapted localization

Supply independently established target centers and orbital representations. This example is the validated six-band, V-centered SrVO3 spinor target:

```bash
sawf-run \
  --seed /path/to/interface/wannier90 \
  --symmetry /path/to/results/symmetry \
  --center 0.5 0.5 0.5 --orbital t2g \
  --output /path/to/results/model
```

For independent target orbits, repeat `--center X Y Z --orbital NAME` in matching order. Each center is one orbit representative; symmetry-related centers are generated automatically. The expanded target must span every selected band. `Projection` defines the target representation and does not replace the original SCDM AMN.

The program first localizes the original guess to determine a reversible column/cell alignment, then runs SAWF from the aligned SCDM guess. Exhausting the iteration budget is an error. It does not drop bands or average the Hamiltonian in postprocessing.

A successful run writes:

- `model.npz`: final `U`, full-mesh k points, original EIG, centers, spreads, lattice, `R`, and `H_R`.
- `summary.json`: input and model hashes, target representation, convergence, numerical residuals, and WANPROJ readback evidence.
- `WANPROJ`: final SAWF coefficients with original VASP band numbers.
- `bands.npz`: optional, with `--dft-eigenval /path/to/EIGENVAL --energy-reference-ev VALUE`.

The model convention is `Psi_W(k) = Psi_Bloch(k) U(k)` and `H(k) = sum_R exp(+2*pi*i*k.R) H_R`. K points use fractional reciprocal coordinates; R uses integer lattice coordinates; H is in eV, centers/lattice in angstrom, spreads in angstrom squared. `H_R` already includes orbital-pair Wigner–Seitz weights.

## Exporting an existing model

No localization or WAVECAR access is needed:

```bash
sawf-export-wanproj \
  --model /path/to/results/model \
  --symmetry /path/to/results/symmetry \
  --output /path/to/results/wanproj
```

The command verifies readiness/convergence, the saved model hash, the bound symmetry package and source hashes, original band mapping, k points and eigenvalues, complete mesh, finiteness and unitarity. It writes `WANPROJ` and `report.json` into a new directory. The writer preserves the final complex coefficients and all k-dependent cell phases; it does not conjugate, transpose, rotate, or recenter them.

The formatted [VASP WANPROJ](https://vasp.at/wiki/WANPROJ) header contains `ISPIN NKPTS NB_TOT NW`. For SOC here `ISPIN=1`; both spinor components belong to each Bloch state. Compact selected-band rows retain their original one-based VASP indices. `NB_TOT` is the total band count of the bound source calculation, not the compact dimension and not a band count borrowed from another run.

## VASP bare Coulomb integrals

Use the exported WANPROJ with the **same Bloch gauge, band numbering, lattice, and full k mesh** that generated its source interfaces. Copy it into a separate calculation directory with the matching VASP inputs and WAVECAR. An old 72-band model cannot be applied unchanged to an unrelated 80-band WAVECAR.

VASP reads an existing WANPROJ and skips its Wannierization. A build with the VASP–Wannier90 interface is still required. For a compatible calculation, the relevant INCAR settings are:

```text
ALGO = 2E4WA
LOCALIZED_BASIS = MLWF
PRECFOCK = Accurate
LVPOT = .TRUE.
LWPOT = .FALSE.
NTARGET_STATES = 1 2 3 4 5 6
```

The target list above is specific to the six-column example. Retain the matching source calculation's other physical settings. [VIJKL](https://vasp.at/wiki/VIJKL) stores the R=0 bare tensor. [VRijkl](https://vasp.at/wiki/VRijkl), available from VASP 6.6.0 with `2E4WA`, also stores off-center interactions on the finite mesh. These are bare Coulomb integrals, not screened cRPA U. Verify tensor index conventions for the VASP version in use before interpreting complex exchange elements.

Orbital rotations and column-dependent lattice translations change tensor entries and their site interpretation. If Wannier columns use different equivalent-cell representatives, R=0 intercolumn interactions need not describe orbitals on one atom. Export preserves that choice; it does not silently convert it into an atom-centered tensor. A validated WANPROJ transfer is not a Coulomb cutoff, k-mesh, PAW, or screened-interaction convergence test.

## Numerical scope

Supported extraction/localization is a nonmagnetic SOC grey group with Cartesian SAXIS, a closed positive even `NB=NW` subspace, a complete Gamma-centered mesh and an actual Gamma wavefunction, and WAVECAR RTAG 45200 with one energy record per k point. Arbitrary spin axes, collinear multi-channel export, and disentanglement remain unsupported. Unsupported or inconsistent inputs fail.

Spatial operations retain full Seitz rotations/translations, including screws, glides and integer composition shifts; antiunitary operations include complex conjugation. Targets may contain multiple symmetry-related centers. Analytic tests cover these operations; SrVO3 is the only material with end-to-end real-data regression. This does not establish a completed tSnS workflow.

The packaged `accepted_closure.json` binds a previously assessed SrVO3 coefficient-closure residual `2.589629272055618e-6` to four exact source interface hashes, with maximum `2.59e-6`. Its scope is the single-particle SAWF/band model. Other matrix checks and convergence requirements remain unchanged. Other datasets do not inherit this decision; it does not certify bare Coulomb accuracy. A reviewed legacy bundle can be loaded without rewriting its report.

## Tests and plots

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider
sawf-plot-bands /path/to/model/bands.npz --output /path/to/figures
sawf-plot-symmetry /path/to/comparison.json --output /path/to/figures
```

Plotting reads saved numerical data and does not validate the model. Figure outputs can be overwritten. External-data tests are opt-in through the `SAWF_SRVO3_*` variables described in the tests; otherwise they are skipped. The unit suite covers complex transformations, original-band mapping, mesh checks, provenance binding, invalid inputs and output protection. Install the package before running the installation tests; they exercise commands and imports outside the checkout.
