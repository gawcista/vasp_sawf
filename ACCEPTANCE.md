# Acceptance rationale for the SrVO₃ coefficient-closure residual

Decision date: 2026-09-24. The review used the two-script version `85002e8515c8fd031446ae71cd6c62e7fdc9113e`, WannierBerri 1.7.0, and IrRep 2.6.3.

**Decision: accept the coefficient-closure residual of `2.589629272055618e-6` for the current six-band SrVO₃ dataset, for the checked SAWF single-particle model and its comparison with ordinary Wannier and DFT bands. This residual no longer blocks further use of this dataset.** This is a scoped judgment based on provenance, independent coefficient checks, native PAW MMN, group relations, the final model, and band errors. It is not a general increase in tolerances for all systems.

The user had previously allowed numerical trials to continue with this item pending acceptance. After reviewing the full physical argument, the user explicitly requested the following (English translation): "Very good. Please record these discussions in the documentation; they may be useful later to explain the reasoning. Next, completely remove the related blockers." This decision supersedes the previous pending status of this dataset; the same item does not require renewed manual confirmation. The historical fields `numerical_trial`, `pending_coefficient_closure`, and `sawf_ready=false` describe the status at the time those reports were written. The original files remain unchanged, and those fields must not be treated as current blockers.

## Dataset and scope of acceptance

- Original interface files: `/mnt/d/Working/SrVO3/results/scdm_test/wannier/wannier90.{win,amn,eig,mmn}`. The wavefunctions come from the final WAVECAR retained by that interface calculation.
- Original VASP bands 33–38; interface dimensions `NK=216, NB=NW=6`; a complete Gamma-centered `6×6×6` mesh. WAVECAR stores 20 IBZ points and 72 bands.
- Extraction retains every original G component and both SOC spinor components. Original complex64 coefficients are copied exactly into complex128 for computation. No G filtering, band removal, mesh reduction, or replacement of native MMN by pseudo-wavefunction overlaps is performed.
- The checked target is the `t2g` spinor representation at the V site, fractional coordinates `(0.5,0.5,0.5)`. Its six basis functions follow WannierBerri's orbital and spin conventions. The SCDM AMN is unchanged; ordinary localization is used only to determine a reversible cell and column-basis transformation of the initial guess.
- Acceptance concerns the numerical accuracy of the current single-particle model. It is not an analytic proof throughout the continuous BZ or an accuracy guarantee for fRG, interaction matrix elements, or arbitrarily small energy scales. It does not directly establish physical acceptance for tSnS or another dataset.

The program binds this decision to the actual interface contents, rather than matching folder names or user-supplied status fields:

| Input | SHA-256 |
|---|---|
| WIN | `8470fcee1d8bf6b5c1009fe6ae3111192b66c43f551a8803ed39e6e1f9c0f2ba` |
| AMN | `15ae2190883bfd963b083c5a31eee8a371e469174b5aad2755d4a8f570c767b6` |
| EIG | `e5369db82649a277a4819c0cdb9e7f5cf85ee9f3875b4bcceeeee567c138837d` |
| MMN | `9c055588d9f0ae702a5188e8c8241d0f024141019a88b24ec27e3f78fc553d62` |

The reviewed `bloch.npz` has SHA-256 `da002b0436a2593c5b0bc77fa65aff8bf11e3991bc141e5c58fc0b475e80153c`. These hashes associate the acceptance record with the actual data. They do not replace the numerical checks below, and they do not require hashing the large WAVECAR in full.

## What the residual measures

The current extractor performs **408 checks**, covering all applicable little-group operations at the 20 IBZ points. The residual is defined as

\[
\delta=\max_{k,\,g\in G_k}
\frac{\left\|d_g^{\mathrm{loc}}(k)^T W(k)-\mathcal T_gW(k)\right\|_F}
{\left\|\mathcal T_gW(k)\right\|_F}
=2.589629272055618\times10^{-6}.
\]

Each row of `W` contains all plane-wave and spinor coefficients of one target band. The local representation `d_loc` is computed directly from wavefunctions at that IBZ point. The operation `T_g W` applies spatial and spinor transformations using an independent integer-G dictionary mapping; antiunitary operations include complex conjugation. Coefficient reconstruction already allows linear mixing within the six bands, so an overall phase or a rotation of a degenerate basis cannot by itself explain the nonzero residual. These `d_loc` matrices are also compared independently with the full-mesh representation propagated from the Gamma anchor through native MMN.

This quantity is a relative reconstruction error in the Frobenius norm of all six bands together. It is not an energy error, a relative error for every coefficient, or a worst-single-state error. The worst check occurs at IBZ point 8, `k=(1/3,1/3,0)`, for operation index 31 in zero-based numbering, which includes time reversal. Its per-band relative residuals are:

| VASP band | Relative coefficient residual |
|---|---:|
| 33 | `3.791649859040997e-6` |
| 34 | `2.261834987799100e-6` |
| 35 | `3.316617080457584e-6` |
| 36 | `3.045252349000387e-6` |
| 37 | `1.002064050044293e-7` |
| 38 | `1.162013020597969e-7` |

These values rule out a large error in one band being hidden by the other five in this worst check. They are not a worst-state error bound over all linear combinations.

The Euclidean norm of PAW pseudo-wavefunction coefficients differs from the all-electron physical inner product. Thus, `δ²≈6.7e-12` must not be called a probability of leakage into omitted bands. Section 3.3 of the IrRep paper states that PAW pseudo-wavefunctions and all-electron wavefunctions transform under the same symmetry representations, supporting the use of pseudo-wavefunctions to determine representations. This does not imply equal norms. The production workflow continues to use native VASP PAW MMN. [IrRep paper](https://arxiv.org/pdf/2009.01764), [VASP PAW documentation](https://vasp.at/wiki/Projector-augmented-wave_formalism)

## Numerical and physical basis for acceptance

### 1. The sampled target subspace does not visibly cut through an external degeneracy

Using all 72 bands in the 20 energy tables of the current OUTCAR, the gaps `E33−E32` and `E39−E38` are:

| Scope | Gap to bands below | Gap to bands above |
|---|---:|---:|
| Separate minima over all 20 IBZ points | 1.2871 eV | 0.2330 eV |
| Point 8, where the coefficient residual is largest | 2.8569 eV | 1.1536 eV |

These values have the printed precision of OUTCAR. Little-group operations commuting with the Hamiltonian should preserve an isolated spectral subspace. The gaps support treating the selected six bands as a complete target subspace; in particular, the worst point is not close to a crossing with an external band. This observation covers only the sampled k points and does not prove isolation throughout the continuous BZ.

### 2. Independent routes give compatible symmetry representations

The full-mesh representation is propagated from a full-coefficient Gamma anchor through native MMN. Checks cover the reverse propagation tree, every neighbor edge, direct IBZ wavefunction results, group composition, and the square of time reversal. No polar decomposition of the sewing matrices is used to conceal nonunitarity. Results are listed below as maximum absolute matrix elements unless stated otherwise:

| Check | Measured residual |
|---|---:|
| Symmetry covariance of all native MMN | `3.253168480286036e-8` |
| Unitarity of the raw full-mesh Bloch representation | `3.105030791249429e-8` |
| Reverse-tree versus original-tree propagation | `2.943892576556475e-8` |
| Direct IBZ versus propagated representations | `5.469744088827509e-7` |
| Independent least-squares versus IrRep representation at Gamma | `4.827316226963777e-15` |
| Bloch energy covariance | `2.877022730028368e-8` eV |
| Group composition | `2.854775990566161e-8` |
| Spinful time reversal, `T²+I` | `8.725076491853150e-9` |
| Final full-mesh Wannier-gauge covariance | `2.172535532439475e-8` |

These quantities are not all statistically independent. Together, however, they support numerical consistency of the band ordering, G-to-coefficient association, spinor conventions, antiunitary conventions, and interface gauge. The checks go beyond matching band counts, k-point counts, or energies.

### 3. Symmetry errors in the final model are much smaller than path interpolation errors

| Model quantity | Measured result |
|---|---:|
| Full-mesh gauge unitarity residual | `1.970594043498863e-8` |
| Hamiltonian symmetry-covariance residual on the full sampled mesh | `2.182738816358665e-7` eV |
| Hamiltonian symmetry-covariance residual at 13 off-mesh test points | `1.855456739363603e-7` eV |
| Maximum Kramers splitting at TRIM | `1.308319159676330e-7` eV |
| Hamiltonian Fourier round-trip error | `1.154702780629605e-14` eV |
| Eigenvalue error on the sampled mesh | `1.570245240500867e-7` eV |
| Maximum coordinate difference from target centers | `9.935008371542153e-11` Å |
| Total SAWF spread | `11.140780230059036` Å² |
| Total ordinary Wannier spread | `11.200953437483777` Å² |

The Hamiltonian covariance residuals in the table are maximum matrix-element magnitudes. They are not directly operator norms or rigorous error bounds for arbitrary observables. For the `6×6` difference matrices here, the conservative inequality `||ΔH||₂ ≤ ||ΔH||F ≤ 6 max|ΔH_ij|` gives a corresponding sampled-mesh scale of about 1.31 μeV. Kramers splitting is a directly computed energy difference, about 0.131 μeV.

The 13 off-mesh points use fixed random seed 7. They provide a finite sampling check, not a uniform error bound on the continuous BZ. A separate comparison uses six sorted eigenvalues at 120 DFT points along `Γ–X–M–Γ–R–X | R–M`, with a common energy reference of 5.14729 eV:

| Difference from DFT path bands | Maximum absolute error | RMS error |
|---|---:|---:|
| Ordinary Wannier | `0.011858588681851856` eV | `0.003417665027782235` eV |
| SAWF | `0.011858592528570355` eV | `0.003417660044966340` eV |

The conservative scale of the current Hamiltonian symmetry residual is nearly four orders of magnitude below the path interpolation error of about 11.9 meV. SAWF and ordinary Wannier have comparable path-fitting accuracy. This supports accepting the small coefficient nonclosure for the present single-particle model; it does not establish a general relation that converts a coefficient residual into a particular energy error.

SAWF performed 21 iterations. The final global convergence measure was `7.522823623729765e-10`, satisfying the existing `1e-9` convergence setting. The original AMN/MMN/EIG were unchanged, and the final Hamiltonian was not averaged in post-processing.

### 4. The cause cannot be reduced to complex64 rounding

An independent pymatgen reader obtained exactly the same raw coefficients and the same `2.589629272055618e-6` closure residual. The independent G-dictionary transformation agreed elementwise with the IrRep transformation. Historical diagnostics gave a relative storage half-ULP bound of `4.305877892832910e-8` and a closure reference bound from storage rounding alone of `8.611758304708385e-8`. The measured residual is about 30.07 times the latter, so complex64 storage cannot explain it in full.

The actual setting in the current OUTCAR is `LREAL=Auto`, with the effective output confirming real-space projection, and `EDIFF=1e-8`. Real-space projection introduces numerical approximations. EDIFF controls energy changes in electronic iterations, not the relative accuracy of wavefunction coefficients. Small numerical deviations from the DFT solution procedure and discretization are therefore a plausible explanation. No controlled comparison isolates each contribution, however, so LREAL or another solver parameter cannot be identified as a confirmed cause. [VASP LREAL](https://vasp.at/wiki/LREAL), [VASP EDIFF](https://vasp.at/wiki/EDIFF)

## Removing the blocker while retaining the other checks

The decision record is [vasp_sawf/accepted_closure.json](vasp_sawf/accepted_closure.json). For the four interface hashes above, the reported precision used in this discussion, `2.59e-6`, is the upper limit of the accepted numerical range. It is slightly above the full measured value and is not a general closure threshold for other sources.

- A new extraction from the same source adopts this decision automatically if its residual is within this range and all other numerical checks pass. Neither `numerical_trial` nor another manual confirmation is required.
- Reuse of the historical small package requires both the four interface hashes and the reviewed `bloch.npz` hash above, together with the original numerical-check evidence. The program applies the new decision in memory only; it does not rewrite historical reports or packages.
- New reports mark this item `accepted_for_single_particle_model` and report readiness after all remaining checks complete. Historical pending fields no longer propagate to plot annotations or result status.
- A source mismatch, a residual outside the accepted range, missing checks, or failure of any other physical or numerical check prevents this decision from being inherited.
- The existing `1e-6` matrix checks, `1e-9` global localization convergence setting, provenance checks, target-representation checks, subspace dimensions, and antiunitary checks all remain in place. Bands must not be removed, other tolerances relaxed, nonconvergence ignored, inputs rewritten, or the Hamiltonian averaged.

Future work on μeV-scale splittings, sensitive wavefunction matrix elements, or interactions should reassess the error budget for the new observable. This does not add another manual gate for the already accepted single-particle model.

## Evidence locations

- The [historical symmetry-package report](../../runs/srvo3/reference/report.json) retains the original bytes from the numerical trial. The [current model report](../../runs/srvo3/model/summary.json) records readiness after applying this decision. The historical `numerical_trial` describes the qualification at that time; it does not mean the current model remains unaccepted. The model's `coefficient_closure_evidence.source_report_sha256` binds that historical report, so it is retained during directory cleanup. The relative link `reference/bloch.npz` points to the current `symmetry/bloch.npz`; both refer to the same package hash, without duplicate storage.
- The independent coefficient reader, per-band errors, and storage-rounding bounds remain in Git history under `coefficient_closure_independent` in `git show e654ac8:VALIDATION.json`. The cleaned historical report tree need not be restored.
- Energy gaps and effective settings come from `/mnt/d/Working/SrVO3/results/scdm_test/wannier/OUTCAR`. The path reference comes from `/mnt/d/Working/SrVO3/results/scdm_test/bandsoc/EIGENVAL`.
- Residual evaluation and full-mesh propagation are implemented in [vasp_sawf/symmetry.py](vasp_sawf/symmetry.py). Final-model, 13-point off-mesh, TRIM, and path checks are in [vasp_sawf/localize.py](vasp_sawf/localize.py).

## Validation after implementing the decision

On 2026-09-24, both scripts completed successfully without the trial option. The new extraction report was `ready`. SAWF directly reused the original trial package above and produced `ready`, `sawf_ready=true`, and `physical_acceptance_status=accepted_for_single_particle_model`. The latter read 0 WAVECAR bytes; the original trial package and its historical report were unchanged.

The full regression run passed 197 tests, with 3 tSnS tests not run. Coverage included direct adoption of this decision by the real SrVO₃ historical package and rejection of changed interfaces, increased residuals, forged passing states, or substituted historical-package contents. The environment still emitted the existing missing-pyfftw notice; computations used NumPy FFT.

Eight groups of arrays in the new model—U, k points, EIG, centers, spreads, lattice, R, and H_R—were elementwise identical to the previously converged model. All three sets of path bands and their path data were also elementwise identical. The new plotting inputs only removed the annotation that coefficient closure was pending acceptance; neither the gauge nor the Hamiltonian was modified to remove the residual.

Current usable results have fixed locations under `DATA_ROOT/.sawf-bridge/runs/srvo3/` in `symmetry/`, `model/`, and `figures/`. The [current symmetry report](../../runs/srvo3/symmetry/report.json) and [current model report](../../runs/srvo3/model/summary.json) record readiness. The `reference/` directory retains only the small evidence files needed to review the ordinary reference and regress reuse of the historical package. The real historical-package test points `SAWF_SRVO3_SYMMETRY` to this `reference` directory; original SrVO₃ input paths are unchanged. Migration does not rewrite existing provenance reports or numerical arrays and does not change the hash bindings above.
