# Symmetry comparison of SAWF and ordinary Wannier functions

This document explains the Wannier gauge in the current six-band SrVO₃ model and the space-group and time-reversal constraints that SAWF imposes on the final basis. The numerical results come from calculations performed on 2026-09-24, using code baseline `3f099dc`, WannierBerri 1.7.0, and IrRep 2.6.3. The conclusions apply only to this SrVO₃ dataset; they do not extend to tSnS.

Three objects must be distinguished: the Bloch states supplied by DFT, the Bloch states obtained by mixing bands, and the real-space Wannier functions. An earlier version denoted the second object by $W(k)$ without explaining its relationship to the third. Here, the second object is consistently written as $\Psi^{\mathrm W}(k)$, while lowercase $w_{nR}$ denotes a real-space Wannier function.

## From Bloch states to Wannier functions

DFT (density functional theory) provides energy eigenstates $|\psi_{mk}\rangle$, where $m$ labels the band and $k$ labels the crystal momentum. Bloch states obey lattice translation conditions and generally extend throughout the periodic crystal. Wannier functions are instead labeled by an orbital index $n$ and a cell index $R$, and describe spatially localized orbitals.

We use the coordinate conventions of the code: $k$ is expressed in fractional reciprocal coordinates, and $R=(R_1,R_2,R_3)$ is an integer cell coordinate. Both are dimensionless. For real-space basis vectors $\mathbf a_i$ and reciprocal basis vectors $\mathbf b_i$ satisfying $\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$, the physical lattice displacement and wavevector are $\mathbf L_R=\sum_iR_i\mathbf a_i$ and $\mathbf K(k)=\sum_i k_i\mathbf b_i$, respectively. Fourier phases therefore take the form $\exp(\pm2\pi i k\cdot R)$. In contrast, $\mathbf r$ is a continuous real-space position with units of length.

At each $k$, we fix the subspace spanned by the six target Bloch states: the set of all their linear combinations. Write the states side by side as columns:

$$
\Psi(k)=\bigl(|\psi_{1k}\rangle,\ldots,|\psi_{6k}\rangle\bigr).
$$

Each column of $\Psi(k)$ is an entire wavefunction, not a scalar; $\Psi(k)$ itself is not a $6\times6$ numerical matrix. The matrix $U(k)$ that mixes these six columns is $6\times6$.

**The Wannier gauge is the choice of this Bloch basis at every $k$.** Relative to the original DFT basis, the choice is represented by $U(k)$:

$$
|\psi^{\mathrm W}_{nk}\rangle
=\sum_{m=1}^{6}|\psi_{mk}\rangle U_{mn}(k),
\qquad
\Psi^{\mathrm W}(k)=\Psi(k)U(k).
\tag{1}
$$

Here the number of bands equals the number of Wannier functions, so $U(k)$ is a square unitary matrix satisfying $U^\dagger U=UU^\dagger=I$. A unitary change of basis preserves inner products and orthonormality; $\dagger$ denotes the complex-conjugate transpose. In Eq. (1), $m$ labels an original band and $n$ labels a Wannier orbital to be constructed.

For a single band, this freedom is a phase choice: $U(k)=e^{i\theta(k)}$. With multiple bands, different bands at the same $k$ can also be mixed. The mixed states need not be degenerate, meaning that they need not have equal energies. A transformed state $|\psi^{\mathrm W}_{nk}\rangle$ is generally no longer an energy eigenstate of a single band, but it still obeys the Bloch translation condition and spans the same subspace with the other transformed states.

Thus, **the earlier $W(k)$ denotes the transformed set of Bloch states, not real-space Wannier functions.** It is still labeled by $k$ because only bands within each $k$ point have been mixed; states at different $k$ points have not yet been combined.

Real-space Wannier functions are obtained by a subsequent Fourier transform. To make the normalization explicit, consider a finite crystal with $N_k$ cells and periodic boundary conditions, and normalize the Bloch states over that entire finite crystal. On the corresponding complete $N_k$-point mesh,

$$
|w_{nR}\rangle
=\frac{1}{\sqrt{N_k}}\sum_k
 e^{-2\pi i k\cdot R}|\psi^{\mathrm W}_{nk}\rangle
=\frac{1}{\sqrt{N_k}}\sum_k\sum_m
 e^{-2\pi i k\cdot R}|\psi_{mk}\rangle U_{mn}(k).
\tag{2}
$$

The position-space wavefunction is $w_{nR}(\mathbf r)=\langle\mathbf r|w_{nR}\rangle$. Because the system includes SOC (spin-orbit coupling), each wavefunction has two spin components and is a spinor. The spin-component index is omitted for brevity.

The inverse of Eq. (2) is

$$
|\psi^{\mathrm W}_{nk}\rangle
=\frac{1}{\sqrt{N_k}}\sum_R
 e^{+2\pi i k\cdot R}|w_{nR}\rangle.
\tag{3}
$$

Under this finite periodic convention, $R$ in Eq. (3) runs over the $N_k$ inequivalent cells within the supercell; periodic copies must not be counted again. Equation (3) shows that $|\psi^{\mathrm W}_{nk}\rangle$ is a Bloch sum of the same type of Wannier orbital over different cells. Both bases describe the same subspace: one is labeled by momentum $k$, and the other by cell $R$.

Functions constructed on a finite mesh have the periodicity of the corresponding supercell. For an infinite crystal, the $k$ sum becomes an integral over the Brillouin zone, while $R$ runs over the entire lattice. The Brillouin zone (BZ) is a nonredundant fundamental region of reciprocal space. Normalization prefactors depend on the chosen convention and must be interpreted together with the normalization of the Bloch states.

The full Bloch wavefunction must also be distinguished from its cell-periodic part:

$$
\psi_{mk}(\mathbf r)=e^{i\mathbf K(k)\cdot\mathbf r}u_{mk}(\mathbf r).
$$

The function $u_{mk}$ is unchanged by a lattice translation. It transforms with the same $U(k)$, but the Bloch phase must be restored when constructing Eq. (2) from $u^{\mathrm W}_{nk}$. Lowercase $u_{mk}$ is the periodic part of a wavefunction; uppercase $U(k)$ is the matrix that mixes bands.

Different gauges change the relative phases in the Fourier sum, and hence the shape and localization of the Wannier functions. For example, for an integer cell displacement $\Delta R\in\mathbb Z^3$, the single-band transformation $\psi^{\mathrm W}_{k}\to e^{+2\pi i k\cdot\Delta R}\psi^{\mathrm W}_{k}$ gives $w_R\to w_{R-\Delta R}$: it moves the orbital to an equivalent cell. More general $k$-dependent mixing changes the distribution of an orbital over several cells. Wannierization seeks a basis that makes these orbitals as localized as possible. Smoothness is required of the transformed Bloch basis; $U(k)$ cannot be judged smooth in isolation from the original DFT basis. [Wannier90 methodology](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/methodology/), [review of Wannier functions](https://arxiv.org/abs/1112.5411)

## What SAWF constrains

Ordinary localization chooses $U(k)$ to reduce the Wannier spread. The spread is the second central moment of the position distribution, $\langle r^2\rangle-|\langle\mathbf r\rangle|^2$, and measures the spatial extent of an orbital. SAWF (symmetry-adapted Wannier functions) additionally requires the orbitals to transform according to prescribed symmetry rules. [Sakuma's original paper](https://arxiv.org/abs/1306.0032)

A representation is a set of matrices describing how basis states transform under symmetry operations. A rotation, for example, can map one orbital to another or to a linear combination of orbitals. In a system with SOC, the spinor components must transform as well.

The target for the current SrVO₃ calculation is a $t_{2g}$ spinor basis on the V site at fractional position $c=(0.5,0.5,0.5)$. In a cubic environment, $t_{2g}$ is the orbital representation spanned by $d_{xz},d_{yz},d_{xy}$. The basis order used here is

$$
(d_{xz}\uparrow,d_{xz}\downarrow,
 d_{yz}\uparrow,d_{yz}\downarrow,
 d_{xy}\uparrow,d_{xy}\downarrow).
$$

These labels specify transformation properties; they do not require pure atomic d wavefunctions. Symmetry-compatible oxygen tails and hybridization are allowed. The arrows label the chosen spin basis. With SOC, they do not imply spin conservation or require the Hamiltonian, the operator determining the system's energy, to separate into uncoupled up-spin and down-spin blocks.

The calculation uses 48 representatives of the spatial operations of the cubic space group $Pm\bar3m$ (221), together with 48 antiunitary representatives obtained by combining them with time reversal. Including TR (time reversal) gives a grey magnetic group: both every spatial operation and its time-reversed partner belong to the group. IrRep records it as `Pm-3m1'`, `221.93`. These 96 representatives do not list the infinitely many lattice translations individually; translations enter through Bloch phases. The minus sign acquired by a spinor under a $2\pi$ rotation must also be retained, giving a double-valued representation.

The original SCDM AMN supplies the numerical initial guess. The target centers and orbital representation specify how the final basis should transform. The original AMN was retained in this calculation; projection matrices were not regenerated.

## Deriving the symmetry transformation equations

Let $\hat g$ denote a symmetry operator acting on wavefunctions. Its spatial part acts on fractional-coordinate column vectors as $x\mapsto S_gx+t_g$, where $S_g$ describes a rotation, inversion, mirror, or related operation, and $t_g$ is a translation. Set $a_g=0$ for a unitary operation and $a_g=1$ for an antiunitary operation containing time reversal. Then

$$
gk=(-1)^{a_g}S_g^{-T}k\pmod{\mathbb Z^3}.
\tag{4}
$$

Here $S_g^{-T}$ is the inverse transpose, $(S_g^{-1})^T=(S_g^T)^{-1}$. The superscript $T$ means transpose and $-1$ means inverse; this superscript does not denote time reversal. The inverse transpose follows from the phase pairing between real and reciprocal space: for the linear spatial transformation $x'=S_gx$, imposing $k'^Tx'=k^Tx$ gives $k'=S_g^{-T}k$. The translation $t_g$ contributes a separate wavefunction phase and does not change this wavevector mapping. For an orthogonal rotation matrix in Cartesian coordinates, $S_g^{-T}=S_g$. This simplification does not generally hold for the fractional lattice coordinates used here.

The factor $(-1)^{a_g}$ in Eq. (4) supplies the extra wavevector reversal due to time reversal. Taking the result modulo integer reciprocal vectors folds it into the chosen Brillouin zone; numerical comparisons must use the same $k$-point correspondence as the data. An antiunitary operation also conjugates the complex coefficients of a linear combination: $\hat g(z|\psi\rangle)=z^*\hat g|\psi\rangle$. To write both cases together, define

$$
\mathcal C_g[X]=
\begin{cases}
X,&a_g=0,\\
X^*,&a_g=1,
\end{cases}
$$

where $*$ denotes elementwise complex conjugation, without transposition.

The Bloch sewing matrix $d_g(k)$ describes the transformation of the original Bloch basis from $k$ to $gk$:

$$
\hat g\Psi(k)=\Psi(gk)d_g(k).
\tag{5}
$$

Here $d_g$ is a $6\times6$ matrix. Its $m$th column contains the expansion coefficients of the transformed $m$th original state in the six-state basis at the destination $k$ point. Such a square representation requires the target subspace to be closed under the operation: transformed states must remain expressible within that subspace. The existing subspace and matrix-residual checks assess numerical closure. The rationale for accepting the coefficient-closure residual is documented in [ACCEPTANCE.md](ACCEPTANCE.md).

The matrix $D_g(k)$ specifies the transformation rule of the **prescribed Wannier basis**; it was denoted $D_{\rm target}(g,k)$ in the earlier text. SAWF requires

$$
\hat g\Psi^{\mathrm W}(k)=\Psi^{\mathrm W}(gk)D_g(k).
\tag{6}
$$

Substituting Eq. (1) into Eq. (6), the left-hand side applies the operation to the original Bloch states and to their coefficients, giving $\Psi(gk)d_g(k)\mathcal C_g[U(k)]$. The right-hand side is $\Psi(gk)U(gk)D_g(k)$. Comparing coefficients in the same basis $\Psi(gk)$ gives

$$
d_g(k)\mathcal C_g[U(k)]=U(gk)D_g(k),
\qquad
U(gk)=d_g(k)\mathcal C_g[U(k)]D_g(k)^\dagger.
\tag{7}
$$

This is the gauge-covariance relation checked by the code. Covariance means that changing basis and applying the symmetry operation give the same result in either order. It does not require every matrix element to remain unchanged; the elements must transform according to the prescribed matrices.

Pure time reversal $\Theta$ sends $k$ to $-k$, so Eq. (7) becomes

$$
U(-k)=d_\Theta(k)U(k)^*D_\Theta(k)^\dagger.
\tag{8}
$$

The complex conjugation follows from the antilinearity of time reversal and cannot be omitted. In the current real-orbital, interleaved-spin convention, $D_\Theta=J=I_3\otimes i\sigma_y$. Here $I_3$ acts on the three orbitals, $\sigma_y$ is the Pauli matrix acting on the two spin components, and $\otimes$ denotes the tensor product of orbital and spin spaces. The full time-reversal operator also includes complex conjugation, so $\Theta^2=JJ^*=-I$. The alternative convention $-i\sigma_y$, also common in the literature, differs by an overall phase and leaves this square unchanged.

The matrix $D_g(k)$ also includes the phase needed to return an orbital center to an equivalent cell. All six orbitals share one center in this example. If

$$
\ell_g=S_gc+t_g-c\in\mathbb Z^3,
$$

then

$$
D_g(k)=p_g(k)O_g,\qquad
p_g(k)=e^{-2\pi i(gk)\cdot\ell_g}.
\tag{9}
$$

The matrix $O_g$ acts on the local orbital and spinor components. For $C_{4z}$, a $90^\circ$ rotation around the z axis, $\ell_g=(-1,0,0)$; the center phase cannot be omitted even though $t_g=0$. Multiple-center models also require center permutations and the associated cell translations. A common scalar phase cannot replace their full representation. The sign in Eq. (9) was checked against the `Dwann` implementation in WannierBerri 1.7.0.

For an arbitrary ordinary Wannier basis, the **actual** symmetry representation is

$$
D_{\rm eff}(g,k)=U(gk)^\dagger d_g(k)\mathcal C_g[U(k)].
\tag{10}
$$

SAWF requires $D_{\rm eff}=D_g$. Ordinary localization does not explicitly impose this condition. Its actual representation may therefore have additional $k$ dependence, corresponding to orbital mixing across cells in real space. Within the same complete, closed subspace, a change of basis does not remove physical symmetry. A mismatch with the target representation must not be identified directly with spontaneous symmetry breaking.

## How the basis constraints appear in the Hamiltonian

The Hamiltonian, denoted $\hat H$, determines energies and time evolution. Its matrix in the original Bloch eigenbasis is $E(k)=\operatorname{diag}(\varepsilon_{1k},\ldots,\varepsilon_{6k})$. In the Wannier gauge, it becomes

$$
H^{\mathrm W}(k)=U(k)^\dagger E(k)U(k).
\tag{11}
$$

This matrix is generally no longer diagonal, but its eigenvalues remain the original six energies. This equality also explains why matching bands on the original mesh does not by itself establish that the Wannier basis realizes the target representation.

If $\hat g$ is a symmetry of the system and Eq. (7) holds, then

$$
H^{\mathrm W}(gk)=D_g(k)\mathcal C_g[H^{\mathrm W}(k)]D_g(k)^\dagger.
\tag{12}
$$

Checking the Hamiltonian with an ordinary basis's own $D_{\rm eff}$ largely restates the original symmetry relation in a different basis. To test the orbital transformation rule required by SAWF, the prescribed $D_g$ must be used.

The theoretical real-space matrix elements can be defined as

$$
h_{mn}(R)=\langle w_{m0}|\hat H|w_{nR}\rangle,
\qquad
H^{\mathrm W}(k)=\sum_R e^{+2\pi i k\cdot R}h(R).
\tag{13}
$$

The hopping matrix element $h_{mn}(R)$ describes the coupling between orbital $m$ in the reference cell and orbital $n$ in cell $R$; both indices here label Wannier orbitals. The $R=0$ term also includes energies and orbital mixing within the same cell. For a finite periodic crystal, Eq. (13) again sums only over inequivalent cells in the supercell. Only in the infinite crystal does it sum over the entire lattice.

The stored `H_R` needs a separate explanation. A finite $k$ mesh determines Fourier data folded according to the periodicity of the sampling supercell. WannierBerri constructs a finite interpolation model using Wigner–Seitz equivalent images, equivalent lattice vectors chosen by spatial distance, with weights assigned to each orbital pair. These weights are already included in the output arrays. We denote the stored coefficients by $H_R$ and use them in

$$
H_{\rm interp}(k)=\sum_R e^{+2\pi i k\cdot R}H_R.
\tag{14}
$$

An exported coefficient must not be identified without qualification with the exact infinite-crystal $h(R)$, nor divided again by the number of equivalent images. On the original mesh, Eq. (14) should reproduce Eq. (11). Off-mesh bands require an independent DFT-path comparison. The hopping residuals and full-BZ error bound below refer to the exported $H_R$.

For the common-center basis used here, the common phase in Eq. (9) cancels from Hamiltonian covariance. The target real-space relation simplifies to

$$
H_{S_gR}=O_g\mathcal C_g[H_R]O_g^\dagger.
\tag{15}
$$

For example, $C_{4z}$ maps the x-direction neighbor to the y-direction neighbor, requiring $H_{(010)}=O_{C_{4z}}H_{(100)}O_{C_{4z}}^\dagger$. Orbital and spinor components must both transform; the two matrices need not be elementwise equal. Pure TR requires $H_R=JH_R^*J^\dagger$, with no additional $R\to-R$. The d orbitals here have even inversion parity, giving $H_{-R}=H_R$. Combined with Hermiticity, $H_{-R}=H_R^\dagger$, this makes each $H_R$ Hermitian. A Hermitian matrix equals its complex-conjugate transpose, a basic requirement for real energy eigenvalues.

## Comparing ordinary Wannier functions with SAWF

The ordinary reference uses unconstrained WannierBerri localization with the same VASP SCDM AMN/MMN/EIG files as SAWF. It is not the result of a separate run of the Wannier90 executable. The comparison reads existing gauges and real-space matrices; it does not rerun localization or read WAVECAR.

The ordinary orbital centers already occupy equivalent cells of the V site, but their cell labels and orbital linear combinations may differ from the prescribed target basis. The earlier label "ordinary Wannier (aligned)" means the ordinary model after these two conventions have been made consistent. This prevents convention differences from contributing to residuals when comparing the same orbital representation. The transformation is

$$
A(k)=\operatorname{diag}(e^{+2\pi i k\cdot n_j})Q,
\qquad
U_{\rm aligned}(k)=U_{\rm ordinary}(k)A(k).
\tag{16}
$$

Here $n_j$ is the integer cell displacement of the $j$th ordinary orbital center relative to the target center. The positive-sign phase returns that center to the target cell. The unitary matrix $Q$ is determined by matching the representations of the full space group and TR at $\Gamma$, where $k=0$. It allows general linear combinations of orbital and spin components, rather than only column permutations or individual column phases. The same $Q$ is used at every $k$; the $k$ dependence of $A(k)$ comes solely from the known integer cell displacements. No arbitrary $Q(k)$ is selected independently at each point, and the ordinary gauge is not required to equal the SAWF gauge point by point.

The Hamiltonian is transformed at the same time as $A(k)^\dagger H_{\rm ordinary}(k)A(k)$, preserving its eigenvalues at every $k$. This step makes cell and basis conventions consistent; it does not shift the energy zero or fit one band curve to another.

The original $H_R$ undergoes the exact transformation equivalent to Eq. (16), without selecting equivalent images again, rerunning localization, or averaging hoppings. In spot checks, the transformed real-space model differs from the directly transformed $H(k)$ by `1.07e-14 eV`; on the original mesh, its difference from $U^\dagger EU$ is `1.42e-12 eV`. The chosen $Q$ is not claimed to minimize the maximum-element residual over all constant matrices.

A residual is the difference between the two sides of an equality. For example, the gauge residual is

$$
\epsilon_U=\max_{g,k,i,j}\left|
\left[U(gk)-d_g(k)\mathcal C_g[U(k)]D_g(k)^\dagger\right]_{ij}
\right|.
\tag{17}
$$

Rows labeled "all spatial operations" or "pure TR" restrict the set of operations included in this maximum. Hamiltonian residuals likewise use the maximum absolute matrix-element difference between the sides of Eq. (12), with $H_{\rm interp}$ used off mesh. Gauge residuals are dimensionless; Hamiltonian and hopping residuals are in eV. They measure matrix-element deviations and are not directly errors in arbitrary physical observables.

| Check | Ordinary Wannier (common cell and basis conventions) | SAWF |
| --- | ---: | ---: |
| Gauge covariance: 48 unitary operations, 216 k points | 8.84771e-5 | 2.17254e-8 |
| Gauge covariance: 48 antiunitary operations, 216 k points | 8.84791e-5 | 2.00014e-8 |
| Pure-TR gauge covariance | 1.74123e-8 | 1.56043e-8 |
| Inversion gauge covariance | 2.64774e-7 | 8.30739e-9 |
| H covariance: all operations, original mesh / eV | 8.78060e-5 | 2.18274e-7 |
| H covariance: all operations, 13 off-mesh points / eV | 3.56526e-5 | 1.85546e-7 |
| Hopping covariance over all exported R / eV | 5.61086e-6 | 1.22745e-7 |
| C4z: relation between H(100) and H(010) / eV | 3.96338e-6 | 8.13689e-9 |
| Pure TR: relation for H(100) / eV | 1.32069e-9 | 6.55532e-9 |
| Maximum Kramers splitting at 8 TRIM / eV | 7.95500e-9 | 1.30832e-7 |

TRIM are time-reversal-invariant momenta, where $-k$ differs from $k$ by an integer reciprocal vector. For a system with $\Theta^2=-1$, TR requires Kramers degeneracy at these points: two linearly independent states have the same energy. The splitting is the difference between their computed energies. SrVO₃ has both inversion $P$ and TR, so their combination $P\Theta$ also requires twofold degeneracy at every $k$. At $\Gamma$, where the full cubic point group is preserved, SOC allows the $t_{2g}$ spinor states to split into a quartet and a doublet. Generic $k$ points have less symmetry; these multiplicities do not apply everywhere, and sixfold degeneracy must not be required throughout the BZ.

In this example, the space-group gauge residual is reduced by about four thousand times, and the $C_{4z}$ nearest-neighbor hopping residual by about five hundred times. Pure TR is already well preserved by the ordinary result. The SAWF TR hopping residual and Kramers splitting are slightly larger, so the results do not support an improvement in every metric. These values are comparisons; no additional acceptance threshold is introduced.

The maximum eigenvalue difference between the two models on the same 120-point DFT path is only `2.87490e-7 eV`. The maximum errors relative to DFT are `0.01185859 / 0.01185859 eV` for ordinary Wannier/SAWF, both about 11.86 meV. Their band plots should nearly overlap, so basis and matrix checks are needed to distinguish their symmetry representations.

After integer lattice displacements are removed, the ordinary centers are about `1.65e-9 Å` from the V site, compared with about `9.94e-11 Å` for SAWF. The reported total spreads are `11.20095 / 11.14078 Å²`, respectively. The slightly smaller SAWF spread indicates that the two optimizations reached different local solutions. It does not mean that adding constraints necessarily improves localization: for the same objective functional, the unconstrained global minimum cannot exceed the constrained global minimum.

## What the two additional checks establish

The first check tests whether another constant basis rotation could remove the discrepancy. For the unitary operation $C_{4z}$, first remove the known center phase from Eq. (9):

$$
\overline D_g(k)=D_{\rm eff}(g,k)/p_g(k),
\qquad
\eta=\max_k\|\overline D_g(k)-\overline D_g(\Gamma)\|_{\rm F}.
\tag{18}
$$

The Frobenius norm $\|M\|_{\rm F}=\sqrt{\sum_{ij}|M_{ij}|^2}$ combines the deviations of all matrix elements. Under any constant unitary $Q$, the representation becomes $Q^\dagger\overline D_gQ$, leaving this norm unchanged. The measured values are `eta=1.432949e-4` for ordinary Wannier and `eta=3.112163e-8` for SAWF. The ordinary value was verified to be unchanged by multiplication by $Q$. Thus, for the current choice of equivalent cells, the ordinary representation retains $k$ dependence that another constant rotation cannot completely remove. This expression applies to the unitary $C_{4z}$; an antiunitary representation instead transforms with $Q^*$ on the right.

The second check covers all exported $R$ and supplements the finite set of off-mesh spot checks. Define the difference in Eq. (15) as

$$
\Delta C_g(R)=H_{S_gR}-O_g\mathcal C_g[H_R]O_g^\dagger.
\tag{19}
$$

The Fourier expansion and triangle inequality bound the target-covariance error throughout the BZ:

$$
\sup_k\max_{ij}|\Delta H_g(k)_{ij}|
\leq\max_{ij}\sum_R|\Delta C_g(R)_{ij}|,
\tag{20}
$$

where $\Delta H_g(k)=H_{\rm interp}(gk)-D_g(k)\mathcal C_g[H_{\rm interp}(k)]D_g(k)^\dagger$. The sum includes the original set of $R$ vectors and its inverse-rotated image; missing coefficients are treated as zero. Maximizing over all 96 operations gives an upper bound of `1.07847e-4 eV` for ordinary Wannier and `4.41998e-7 eV` for SAWF.

This bound concerns the exported finite Fourier model and is evaluated in floating-point arithmetic. It is neither rigorous interval-arithmetic certification nor an error bound relative to the continuous DFT solution. The simplification in Eq. (19) also relies on the common center in this example; multiple-center models must retain the complete center mapping.

These checks support the conclusion that the current SAWF model realizes the prescribed local-orbital representation more accurately. Defined centers, orbital labels, and transformation rules make it possible to impose consistent constraints on hoppings and interaction tensors in subsequent local-orbital models. An ordinary basis can also be used for physical calculations, but interactions and symmetry representations must transform consistently with it; atomic-orbital parameters from another basis cannot be carried over automatically.

## Evidence, implementation, and scope

Inner products in this document are physical-state inner products. VASP's PAW (projector-augmented-wave) method uses reconstruction near atoms and the corresponding metric. The Euclidean inner product of pseudo-wavefunction plane-wave coefficients must not be treated directly as the physical inner product. These notation clarifications do not change the workflow: native VASP MMN files remain in use, together with the existing basis-consistency and numerical checks.

- DATA_ROOT: `/mnt/d/Working/tSnS_abacus/vasp/SCDM/661`.
- Current results are in `DATA_ROOT/.sawf-bridge/runs/srvo3/`. The [comparison data](../../runs/srvo3/reference/comparison.json) retain provenance, Q, cell translations, and per-operation residuals. English figures are in `figures/symmetry_comparison.png/pdf`.
- SAWF data come from `model/model.npz` and `model/summary.json`; Bloch sewing matrices come from `symmetry/bloch.npz`.
- Ordinary data come from `reference/gauge_centers_spreads.npz` and `reference/bands_wannier.npz`. Provenance reports are `ordinary_report.json`, `archive_readback.json`, and `interpolation_report.json`. Their original bytes and hashes are preserved; old absolute paths record their locations at generation time. Ordinary path energies exactly match the ordinary arrays in the current model.
- The existing comparison covers all 216 original-mesh k points, 96 operations, and all exported R. The off-mesh spot check uses 13 points from `default_rng(7)`. These 13 points alone are not a full-BZ proof; Eq. (20) supplies a separate model error bound.
- An earlier independent check recomputed gauge and nearest-neighbor hopping relations using manually constructed C4z orbital/spinor matrices and J, without calling the project's validation functions or WannierBerri Dwann. The results agreed; the constant-Q invariant was also checked independently.
- `vasp_sawf/localize.py` checks Eqs. (7) and (12) and exports the model with `symmetrize=False`; `vasp_sawf/bands.py` evaluates Eq. (14). WannierBerri 1.7.0's `Dwann.get_on_points` supplies the target representation, while `Rvectors` includes equivalent-image weights in the output matrices. See the [pinned SymmetrizerSAWF source](https://github.com/wannier-berri/wannier-berri/blob/v1.7.0/wannierberri/symmetry/sawf.py) for the target interface.

This documentation revision clarifies the distinction between basis states and real-space functions, normalization, coordinates, complex conjugation, and theoretical matrix elements versus interpolation coefficients. Existing numerical results and physical conclusions are unchanged. No WAVECAR was read, UNK exported, localization rerun, threshold changed, or Hamiltonian averaged in post-processing.

To redraw the existing English comparison figure from the development worktree:

```bash
sawf-plot-symmetry ../../runs/srvo3/reference/comparison.json --output ../../runs/srvo3/figures
```

The plotting script only reads saved values; it does not rerun physical checks.
