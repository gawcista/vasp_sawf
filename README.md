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

The original `.win/.amn/.mmn/.eig` files share a seed prefix. WIN must contain the full lattice, atomic structure, and complete k-point mesh. Provide the matching WAVECAR and the interface calculation's OUTCAR, including reciprocal folding tables and effective spin settings. Inherited WAVECAR inputs and symbolic links are supported: [LWAVE](https://vasp.at/wiki/LWAVE) controls writing wavefunctions at the end of a run, so `LWAVE=F` does not reject extraction. `LWAVE`, `ISTART` and `ISYM` are optional provenance metadata. Initial [MAGMOM](https://vasp.at/wiki/MAGMOM) is recorded when present; it need not be zero and does not establish the final magnetic state.

Selected-band energies, lattice, source k points and the numerical PAW MMN/sewing checks must still match. Matching energies or an `ALGO=None` run alone does not identify a shared Bloch gauge; the reported MMN anchor-transport status describes numerical consistency, not an independent cross-run gauge certificate. No repeated manual source approval is required.

VASP 6.6.1 writes WIN lattice components with seven decimal places (`3F14.7`), whereas WAVECAR retains binary double precision. The lattice comparison uses an absolute tolerance of `1e-5` angstrom per component by default, configurable with `--tol=1e-5`; no relative tolerance is applied. This option affects only the WAVECAR/WIN lattice comparison, not any MMN, group, time-reversal, closure or localization check. The original WIN is unchanged, and coefficient/G-vector reads use the validated binary lattice. `report.json` records both cells, their differences and the chosen tolerance. A mismatch exceeding the tolerance still fails and prints these diagnostics.

The selected-band energy comparison uses a separate absolute tolerance of `1e-8` eV, configurable with `--energy-tol=1e-8`. It compares WAVECAR energies with the original EIG after band/k-point mapping, without shifting either energy reference. A restart interface calculation can update energies after reading WAVECAR, so EIG decimal precision is recorded as a serialization diagnostic, not used as the comparison threshold. The report records the chosen energy tolerance, maximum difference and the worst-matching band/k point. This option does not change energy covariance, MMN, group, antiunitary, coefficient-closure or localization checks. Passing it does not establish a shared Bloch gauge. A difference of `5.58e-5` eV still exceeds the default `1e-8` eV threshold.

Run on allocated compute resources on the machine holding WAVECAR:

```bash
cd /path/to/interface
sawf-extract
```

Defaults are `--seed wannier90 --wavecar WAVECAR --outcar OUTCAR --output symmetry`. The program creates `./symmetry` automatically or reuses it if it exists; a separate job directory is unnecessary. Use `--output PATH` to choose another directory. Only the generated `bloch.npz` and `report.json` are replaced, with a warning naming the existing files. An empty directory or one containing only unrelated files needs no warning. Other files remain untouched. A failed rerun leaves a `not_ready` report and no stale Bloch package. Output symlinks, nonregular generated targets, and paths or hard links that collide with input files are rejected. Original inputs remain unchanged.

Extraction caches default to `${XDG_CACHE_HOME}/vasp_sawf` when `XDG_CACHE_HOME` is absolute, otherwise `~/.cache/vasp_sawf`, with separate `numba` and `matplotlib` subdirectories. Explicit `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` settings take precedence.

Extraction reads selected-band records with complete G vectors and all components of the selected state: two for SOC, one for a scalar channel. It anchors the representation using IrRep, transports it with native PAW MMN, and checks independent IBZ transformations, group composition, the applicable antiunitary constraint, and covariance. It does not read, copy, or hash the entire WAVECAR. NNKP and UNK are unnecessary.

Before MMN transport, `report.json` records the raw Gamma symmetry matrices (real and imaginary parts), operation ordering and full Seitz/spin rotations, coefficient closure, independent least-squares difference and anchor unitarity. These diagnostics remain available if transport fails and allow offline replay with the original small interface files, without reading WAVECAR again. `Gamma evaluated` means the anchor calculation returned; it does not mean all symmetry checks passed. A failed extraction remains `not_ready` and cannot produce a usable Bloch package. An order-one MMN covariance residual requires diagnosis of the anchor basis, subspace and symmetry conventions; increasing lattice or energy tolerances does not resolve it.

The normal extraction outputs `bloch.npz` and `report.json` form one bound bundle. Download both along with the unchanged original WIN/AMN/MMN/EIG for localization.

### Explicitly accepted approximate inputs

Some reviewed datasets have small native symmetry inconsistencies that exceed
the strict matrix checks but fit a specified model energy budget. They may
use a separate, explicitly accepted approximate route. This does not change
the strict defaults or claim exact input symmetry.

For the reviewed tSnS ALGO=None data, reuse the completed all-24-point
diagnostic in the original calculation directory:

```bash
sawf-extract --from-diagnostic symmetry_full --accept-approximate

sawf-run --seed wannier90 --symmetry symmetry \
  --center 0.741850452829675 0.7360933822227451 0.1679410695036629 --orbital s \
  --center 0.24985348606314414 0.6996231291026112 0.1680181513884988 --orbital s \
  --output sawf
```

The first command creates or updates `symmetry/{bloch.npz,report.json}`.
It reads the completed diagnostic and the small original interface files;
it does not open WAVECAR or require OUTCAR to remain available. The recorded
approval binds the exact original WIN/AMN/MMN/EIG and diagnostic evidence.
It is not a general permission to accept arbitrary residuals or other inputs.
The local `s` label specifies the approved local spinor representation;
it does not assign an atomic s-orbital character to these Wannier functions.
Integer-cell equivalents of the approved centers retain periodic phases.

The second command automatically recognizes the approximate package and
localizes from the original SCDM AMN with the approved target. It writes
`model.npz`, `bands.npz`, and `summary.json`; the model includes the final
gauge and Fourier Hamiltonian. Download that result directory together with
the two symmetry-package files for local analysis. No new `--trial` flag is
needed. Existing trial files keep their historical status.

The reports explicitly record `accepted_approximate`, the approval scope,
raw and reconstructed action residuals, and failure of the strict reference
checks. Algebraic group relations, antiunitary conjugation, target
compatibility, gauge unitarity/covariance, convergence and unchanged inputs
remain checked. The final model is accepted only if its continuous Fourier
symmetry bound meets the approved **1e-4 eV (0.1 meV)** budget. There is no
Hamiltonian averaging or change to original energies or overlaps.

Acceptance concerns the approximate single-particle model. It does not
certify full source-wavefunction closure in the PAW metric, exact DFT
symmetry, or interpolation accuracy against a matched-settings DFT path.
The experimental multi-anchor fit is not used by this route.

On ADA, these two commands need only the small-matrix stage. A single task
and CPU with 8 GiB memory is a conservative allocation for the reviewed
eight-band case; the large WAVECAR-extraction worker/memory settings do not
apply. Use an available compute allocation, for example a batch script with
`--ntasks=1 --cpus-per-task=1 --mem=8G`, and set BLAS/OpenMP threads to one.
The second command requires a new output directory; preserve a prior model
or deliberately remove only its generated outputs before repeating it.

For focused diagnosis of failed covariance checks, use the same seed, WAVECAR and OUTCAR. For example, run these selected tSnS points within an existing Slurm allocation:

```bash
srun sawf-extract --diagnose-kpoints 1 4 20 21 23 24
```

These are **one-based stored WAVECAR k-point indices**, not full-mesh WIN indices. Supply distinct indices. The program reads each requested point once and automatically includes Gamma if absent; it retains every selected band and G component, and leaves the complete MMN mesh unchanged. Other systems need their own stored-point indices. The only result written is `symmetry/diagnostic.json` (`--output` changes its directory); existing `report.json` and `bloch.npz` remain untouched. This mode never creates a SAWF-ready package or relaxes normal extraction checks.

The diagnostic records raw local coefficient closure, unitarity, energy intertwining and full Seitz group relations. Comparisons with MMN-transported matrices and direct native-MMN covariance assume that source and interface Bloch gauges agree; they cannot independently prove that assumption. Coefficient residuals use the stored pseudo-wavefunction coefficients and do not replace PAW overlaps or certify PAW orthogonality. No coefficient approximation or projection is applied.

Independent stored k points can be processed in parallel. `--workers N` and `--memory-gb GIB` set upper limits; CPU allocation, coefficient counts, and available memory can reduce concurrency. Under Slurm request one task with multiple CPUs. For an exclusive ADA large-memory node, save this script in the interface directory and submit it with the installed environment active:

```bash
#!/bin/bash
#SBATCH --job-name=SAWF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --partition=p.large
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail
srun sawf-extract
```

Submit from the interface directory so the default input names and `./symmetry` resolve there. The program selects workers automatically from the allocation and available memory. The optional [extract_ada.sbatch](extract_ada.sbatch) helper instead takes `INPUT_DIR OUTPUT_DIR` arguments and resource options supplied to `sbatch`. No jobs are submitted automatically.

### Non-SOC calculations

The effective OUTCAR flags select the mode; there is no manual SOC switch.

| VASP mode | Model and antiunitary constraint |
| --- | --- |
| SOC (`LSORBIT=T`, `LNONCOLLINEAR=T`) | Two-component spinors, physical time reversal with square `-I`, even `NB=NW`, Kramers checks |
| Scalar `ISPIN=1` | Orbital model with complex conjugation `K`, square `+I`, any positive `NB=NW`; no explicit spin degeneracy or Kramers check |
| Scalar `ISPIN=2` | One selected up/down channel, channel-preserving spatial subgroup and orbital `K`; physical spin-flipping time reversal is **not** imposed |

VASP 6.6.1 writes separate `wannier90.1.*` (up) and `wannier90.2.*` (down) interfaces. Extract each channel separately:

```bash
sawf-extract --seed wannier90.1 --output symmetry-up
sawf-extract --seed wannier90.2 --output symmetry-down
```

The `.1`/`.2` suffix or WIN `spin=up/down` selects the WAVECAR channel. For renamed interfaces without a WIN spin field, specify `--spin-channel 1` or `2`; conflicting declarations are rejected. The source channel is recorded in the bundle and checked again during localization. Each `sawf-run` uses its matching seed and symmetry directory. A scalar `t2g` target contains three functions; the SOC version contains six.

Magnetic `ISPIN=2` needs the final per-site `magnetization (x)` table in OUTCAR, with atomic positions that can be matched to WIN in the same order. Projected moments and their printed precision identify candidate spatial operations using the pinned IrRep/spglib implementation. The rounding allowance is not a physical acceptance threshold: all retained operations still undergo the unchanged wavefunction, energy, PAW MMN and group checks. Initial MAGMOM or zero total magnetization cannot replace this information, particularly for antiferromagnets. Missing final site information produces an explicit diagnostic rather than assuming the nonmagnetic crystal group.

These separate-channel models do not enforce operations that exchange the up/down sectors. Their internal conjugation `K` must not be interpreted as a proof of physical time-reversal symmetry of a magnetic material. Joint constraints between the two sectors require a further extension.

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

Localization still requires a new output directory outside its input directories and never overwrites an existing result.

The program first localizes the original guess to determine a reversible column/cell alignment, then runs SAWF from the aligned SCDM guess. The constrained objective includes every edge of the original full-mesh PAW MMN, including both ends of each edge in its gradient. A fresh constrained gradient, not the optimizer's success message or a small change in spread, establishes numerical stationarity. Exhausting the iteration budget is an error. Numerical stationarity does not establish a global minimum. No bands are dropped and the Hamiltonian is not averaged in postprocessing.

Before localization, the normal command checks the raw Bloch action against the original MMN, EIG, group law and antiunitary constraints. Only after these checks pass does it reconstruct an algebraically consistent unitary/antiunitary action, using the full Seitz and spinor composition factors. The correction must fit the unchanged numerical matrix bound, and the same checks are repeated against the original inputs. A failed raw check cannot be repaired into production acceptance. Source arrays are preserved; the summary records raw residuals, the correction and reconstructed residuals.

The full-grid kernel is intended for compact subspaces. It refuses a dense tangent construction before allocation if the estimated per-chart workspace exceeds 256 MiB; this is not a total-process memory limit and excludes input arrays. Newton refinement, when needed, is limited to 1024 real parameters. The summary records the workspace estimate, optimization stops and final gradient. Nominally zero spreads can have a negative sign from floating-point cancellation: the unmodified values and an explicit arithmetic roundoff guard are recorded. Larger negative values still fail.

A successful run writes:

- `model.npz`: final `U`, full-mesh k points, original EIG, centers, spreads, lattice, `R`, and `H_R`.
- `summary.json`: input and model hashes, target representation, convergence, numerical residuals, and WANPROJ readback evidence.
- `WANPROJ`: final SAWF coefficients with original VASP band numbers for single-channel SOC or scalar `ISPIN=1`. A standalone `ISPIN=2` channel does not contain both VASP channels, so WANPROJ export is explicitly skipped and the reason is recorded.
- `bands.npz`: optional, with `--dft-eigenval /path/to/EIGENVAL --energy-reference-ev VALUE`.

The model convention is `Psi_W(k) = Psi_Bloch(k) U(k)` and `H(k) = sum_R exp(+2*pi*i*k.R) H_R`. K points use fractional reciprocal coordinates; R uses integer lattice coordinates; H is in eV, centers/lattice in angstrom, spreads in angstrom squared. `H_R` already includes orbital-pair Wigner–Seitz weights.

### Numerical trials from existing diagnostics

When extraction diagnostics exist but the physical residuals have not been accepted, an explicit trial can test representation compatibility and constrained localization without another WAVECAR read:

```bash
sawf-run --trial \
  --seed /path/to/interface/wannier90 \
  --symmetry /path/to/symmetry \
  --center X Y Z --orbital ORBITAL \
  --output /path/to/trial
```

Here `--symmetry` must contain a completed `diagnostic.json` from `sawf-extract --diagnose-kpoints`. The command binds its hashes, spin semantics, operations, band mapping and selected k-point evidence to the original WIN/AMN/MMN/EIG. It transports the raw anchor with native MMN, reconstructs an exact group action, checks the target representation, and optimizes on the complete original mesh. For multiple target orbits, repeat the center/orbital pair as above. Fractional cell representatives and their integer lattice shifts remain part of the Bloch phases, including complex conjugation for antiunitary operations.

Only `trial.npz` and `summary.json` are written. The arrays include the full gauge, original eigenvalues, centers/spreads, raw and reconstructed Bloch actions, target matrices and k mappings. The summary retains the original diagnostic report, raw and reconstructed residuals, action changes, convergence and Hamiltonian covariance. A selected-point diagnostic does not certify closure at every stored k point. Even a converged trial always has `accepted=false` and `sawf_ready=false`; it cannot be used by the production model exporter and does not write `model.npz`, `hr.dat` or `WANPROJ`. This mode does not accept a DFT path argument or establish interpolation accuracy. Physical error-budget assessment remains separate from numerical convergence.

### Analyzing a saved trial

A converged trial can be analyzed without repeating SAWF or reading WAVECAR:

```bash
sawf-run --analyze-trial /path/to/trial --output /path/to/analysis
sawf-plot-bands /path/to/analysis/bands.npz --output /path/to/figures --ylim -0.12 -0.04
```

`--seed` overrides the saved source location when moving between machines; the original interface hashes must still match. The target and constrained gauge come from the saved trial, so do not supply new center/orbital arguments. An ordinary-Wannier comparison is recomputed from the original SCDM AMN. The default path has 101 points per WIN segment (`--points-per-segment` changes this); `--energy-reference-ev` applies one common display reference and does not change any Hamiltonian.

The new analysis directory contains:

- `analysis.npz`: the saved full gauge and original EIG, lattice, centers/spreads, weighted `R/H_R`, ordinary-Wannier comparison arrays, and sampled symmetry/locality diagnostics.
- `bands.npz`: dense WIN-path energies for ordinary Wannier and SAWF, ready for the existing plotting command.
- `summary.json`: bound input/output hashes, Fourier roundtrip, symmetry residuals, continuous Fourier error bounds, antiunitary degeneracy checks and locality measurements. Numerical bounds describe the finite interpolated Hamiltonian, not its error relative to DFT.
- `dft_bands.npz`, optionally: DFT and both interpolations at the actual supplied DFT path coordinates. Supply one file or ordered split files with `--dft-eigenval EIGENVAL.0 EIGENVAL.1 ...`. This is a comparison reference; the command does not establish matching DFT settings. Plot it with the same `sawf-plot-bands` command.

The analysis retains `diagnostic_only=true`, `accepted=false` and `sawf_ready=false`; it does not export a production model, WANPROJ or symmetry-averaged hoppings. These flags do not prevent inspection of the saved approximate Hamiltonian. They distinguish that use from a certified source-wavefunction or exact-symmetry claim.

For direct numerical work, the Fourier arrays are self-contained:

```python
import numpy as np

with np.load("analysis.npz", allow_pickle=False) as result:
    R, H_R = result["R"], result["H_R"]
k = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
H_k = np.einsum("kr,rij->kij", np.exp(2j*np.pi*k@R.T), H_R)
energies_eV = np.linalg.eigvalsh(H_k)
```

Do not divide `H_R` by degeneracies again. Energies remain on the original EIG reference; a plotting reference is subtracted only when drawing figures.

### Interpreting symmetry errors

Let `E(k)` be the original diagonal eigenvalue matrix, `B_g(k)` the unitary Bloch sewing matrix, and `D_g(k)` the target Wannier representation. Sewing matrices describe how symmetry maps the Bloch basis between k points. If `B_g C_g(U)=U(gk) D_g`, then for `H=U† E U`,

```text
||H(gk) - D_g C_g(H(k)) D_g†||_2 = ||E(gk) B_g - B_g E(k)||_2.
```

Here `C_g` conjugates antiunitary inputs, and `||.||_2` is the largest singular value. Thus exact gauge covariance cannot remove an incompatibility between the input energies and the imposed sewing matrices. Equal sorted energies at symmetry-related k points do not ensure that the sewing matrices map the correct eigenspaces. Raw and reconstructed actions remain distinct evidence.

The continuous interpolation bound is calculated from the finite Fourier coefficients of the residual, retaining all site/cell phases. It covers the axes sampled by the original mesh, with other reduced coordinates fixed at zero (for a 6x6x1 slab, the full `kz=0` plane). It combines a triangle bound with a periodic-grid covering bound based on derivatives. Floating-point allowance is reported separately; the calculation does not use interval arithmetic. This bounds the symmetry defect of the interpolated model, not its total error relative to DFT.

An antiunitary operation that fixes k modulo reciprocal translations and squares to `-I` requires paired eigenvalues. This can hold on a boundary for screw times time reversal as well as at ordinary TR-invariant points. A measured adjacent-pair splitting `d` requires at least `d/2` maximum eigenvalue change for exact pairing. Keeping those energies also forces a covariance defect of at least `d`. These are mathematical lower bounds; the code does not average the Hamiltonian to enforce them.

Compare an energy budget with the observable's scale: a small fraction of the bandwidth may exceed a narrow avoided crossing. Individual eigenvectors and symmetry labels near such crossings need more care than band dispersion. Matching ordinary-Wannier and SAWF interpolations does not independently validate either against DFT.

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

All modes require a closed square `NB=NW` subspace, a complete Gamma-centered mesh and an actual Gamma wavefunction, and WAVECAR RTAG 45200 with one energy record per k point. SOC additionally requires a nonmagnetic grey-group candidate, Cartesian SAXIS and an even dimension. These restrictions reflect implemented algorithms or file formats; removing their checks would not implement the missing behavior. Arbitrary spin axes, noncollinear non-SOC calculations, joint collinear-channel constraints/export, and disentanglement remain unsupported.

Spatial operations retain full Seitz rotations/translations, including screws, glides and integer composition shifts; antiunitary operations include complex conjugation. Targets may contain multiple symmetry-related centers. Analytic tests cover these operations and scalar extraction/localization, including both collinear channels. A small real non-SOC Si WAVECAR is checked against independent pymatgen and IrRep readers. SrVO3 has accepted end-to-end real-data regression. tSnS has a converged eight-band diagnostic trial and an analysis route that retains its measured input errors; this does not promote it to an accepted production model. A real magnetic `ISPIN=2` calculation has not yet been tested.

The packaged `accepted_closure.json` records dataset-specific user decisions, each bound to four exact interface hashes. The original SrVO3 residual is `2.589629272055618e-6`, with maximum `2.59e-6`. On 2026-10-07 the user also accepted `1.5343413566685595e-6` for the new SrVO3 restart calculation (`ISTART=1`, `LWAVE=F`, `IALGO=2`); its bound `1.534345e-6` is the upper rounding boundary of the reviewed `1.53434e-6`. Acceptance concerns the single-particle SAWF/band model and does not modify other matrix checks or convergence requirements. Other inputs do not inherit either decision, and neither certifies bare Coulomb accuracy. A reviewed legacy bundle can be loaded without rewriting its report.

## Tests and plots

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider
sawf-plot-bands /path/to/model/bands.npz --output /path/to/figures
sawf-plot-symmetry /path/to/comparison.json --output /path/to/figures
```

Plotting reads saved numerical data and does not validate the model. Figure outputs can be overwritten. External-data tests are opt-in through the `SAWF_SRVO3_*` variables described in the tests and `SAWF_SCALAR_FIXTURE` for a small scalar WAVECAR directory; otherwise they are skipped. The unit suite covers complex transformations, original-band mapping, mesh checks, provenance binding, invalid inputs and output protection. Install the package before running the installation tests; they exercise commands and imports outside the checkout.
