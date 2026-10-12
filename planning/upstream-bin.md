# First-bin kick integral: Bmad's approximation when the source is upstream of the kick element

**Component:** Bmad 1D CSR (`csr_method = one_dim`)
**File:** `bmad/space_charge/csr_and_space_charge_mod.f90`, subroutine `I_csr`, lines 1414–1426
**Version checked:** `bmad-ecosystem` at `ee81f2a9e` (release 20261001-0)
**Status:** accuracy limitation of an approximation, not an arithmetic error. Measured in CSR.jl's port
(phase 1, branch `phase1-reference`). Not confirmed by Bmad maintainers. Not compared against Bmad output
(no Bmad build available).

---

## Summary

The kick from bin `i` uses `I_int(i) = ∫ I_csr(u) du` over that bin's range of source-to-kick separations `u`.
For the first bin, `u ∈ [0, dz]`, Bmad splits the integral at `zz`, the separation reached when the source
sits at the kick element's entrance:

- `[0, zz]`: a closed form that handles the near-field log singularity. Exact (verified below).
- `[zz, dz]`: approximated as `I_csr(dz) · (dz − zz)`, i.e. `I_csr` held at its value at the bin edge.

The second part is accurate only if `I_csr` is nearly constant over `[zz, dz]`. It isn't:

- **Steady state**, `I ∝ u^(−1/3)`: the approximation gives `I(dz)·dz`, the integral is `(3/2) I(dz)·dz`.
  The first bin is **2/3 of its correct value**.
- **Entrance transient**, source on the upstream drift: `I_csr` falls by orders of magnitude across the bin.
  The first bin is **3–4× too small** in the case below.

The near-field first bin carries a large share of the kick (`∝ (dz/σ_z)^(2/3)` in steady state), so this
dominates the kick error for coarse binning. The error vanishes once `dz < zz` (fine enough bins, or a kick point
far into a long element), which is when the source of the first bin lies inside the kick element.

---

## Current code

```fortran
if (k%ix_ele_source == ix_ele_kick) then
  Ls = k%L + k%dL
  k%I_int_csr = -csr%kick_factor * ((g_bend * Ls/2)**2 - log(2 * csr%gamma2 * z / Ls) / csr%gamma2)
else
  ! Since source pt is in another element, split integral into pieces.
  ! First integrate over element containing the kick point.
  L = s_chord_kick
  eleinfo => csr%eleinfo(ix_ele_kick)
  dtheta_L = eleinfo%spline%coef(1) + eleinfo%spline%coef(2) * L + eleinfo%spline%coef(3) * L**2
  dL = dspline_len(0.0_rp, L, eleinfo%spline, dtheta_L) ! = Ls - L
  Ls = L + dL
  zz = L / (2 * csr%gamma2) + dL
  k%I_int_csr = -csr%kick_factor * ((g_bend * Ls/2)**2 - log(2 * csr%gamma2 * zz / Ls) / csr%gamma2)
  ! Now add on rest of integral
  k%I_int_csr = k%I_int_csr + k%I_csr * (z - zz)
endif
```

The last line is the approximation in question.

---

## Evidence

All numbers are from CSR.jl's port of the Bmad algorithm. "Quadrature" means `∫ I_csr(u) du` evaluated with a
source-point solve at each node, composite Simpson in `t = u^(1/3)` (smooth for the `u^(−1/3)` near field).
Scripts: `planning/diagnostics/*.jl` (run from the CSR.jl root with `julia --project=.`). The key checks are also CSR.jl tests (listed at the end).

### 1. Closed-form part is exact

Steady-state circle (ρ = 10 m, γ = 10⁴, 80 bends of 0.1 m, kick at mid-element):

| dz [m] | First-bin source in kick element? | Bmad `I_int(1)` / quadrature |
|---|---|---|
| 2e-8 | yes | 1.0000 |
| 1e-6 | no | 0.7715 |
| 6e-5 | no | 0.6753 |
| 1e-3 | no | 0.6683 |

The ratio tends to 2/3 as `zz/dz → 0`, as predicted for `I ∝ u^(−1/3)`.

### 2. Steady-state kick vs the analytic wake

Same lattice, Gaussian line bunch σ_z = 1 mm, 1 nC. Error = max |kick − analytic| over |z| ≤ 2σ, divided by the
analytic peak. The analytic wake is `−(2/3^(1/3)) N r_e/γ ρ^(−2/3) ∫ λ′(z′)(z−z′)^(−1/3) dz′`.

| n_bin | dz/σ | Bmad first bin | First bin by quadrature |
|---|---|---|---|
| 50 | 0.27 | 11.3% | 1.3% |
| 100 | 0.13 | 6.7% | 0.49% |
| 200 | 0.062 | 4.0% | 0.28% |
| 400 | 0.030 | 2.5% | 0.18% |
| 800 | 0.015 | 1.6% | 0.13% |

Bmad's error decays roughly as n_bin^(−2/3).

### 3. Entrance transient (Bmad regression-test parameters)

Parameters from `regression_tests/csr_and_space_charge_test`: 42 MeV electrons (γ = 82.19), one sbend
L = 0.1 m with B = 0.17 T (ρ = 0.824 m), 77 pC, σ_z = 0.899 mm, `particle_bin_span = 2`, `ds_track_step = 0.01`.
Deterministic Gaussian bunch, centroid on axis (the regression test's 1000-particle random bunch and its
`b1_gradient` are not reproduced).

At kick point s = 0.05 m with n_bin = 40, all first-bin sources lie on the upstream drift. Across the first bin
(`zz = 1.1e-5 m`, `dz = 3.1e-4 m`) the integrand falls from −61 m⁻¹ to −0.14 m⁻¹. Bmad's `I_int(1)` is
−7.95e-4; with quadrature it is −2.72e-3 (quadrature converged: n = 16 and n = 1024 agree to 4e-4).

Kick error vs a converged reference (every bin by quadrature, n_bin = 1600), max over |z| ≤ 2σ / reference peak:

| s [m] | n_bin | Every bin by quadrature | First bin by quadrature, trapezoid elsewhere | Bmad |
|---|---|---|---|---|
| 0.05 | 40 | 23% | 23% | 73% |
| 0.05 | 100 | 5.7% | 5.9% | 63% |
| 0.05 | 400 | 0.26% | 5.0% | 0.66% |
| 0.05 | 1600 | 0 (reference) | 0.37% | 0.37% |
| 0.10 | 40 | 9.9% | 9.8% | 46% |
| 0.10 | 100 | 1.8% | 2.2% | 2.2% |
| 0.10 | 400 | 0.19% | 0.20% | 0.20% |
| 0.10 | 1600 | 0 (reference) | 0.062% | 0.062% |

Observations:
- At n_bin = 40 (the regression test's setting) Bmad's maximum kick error is 46–73% of the peak kick.
  Integrating only the first bin by quadrature reduces this to 10–23%; the remainder is binning error that all
  three methods share at this resolution.
- At n_bin = 1600 the first-bin source is inside the kick element, Bmad's branch is not used, and Bmad equals the
  first-bin quadrature.
- **Second, smaller issue:** bins k ≥ 2 use the trapezoid `(I(k) + I(k−1))·dz/2`. In the transient, where `I_csr`
  varies steeply over the first few bins, this is also inaccurate (5% at s = 0.05, n_bin = 400). There Bmad's
  too-small first bin partly cancels the trapezoid's overestimate, which is why Bmad (0.66%) beats
  first-bin-only quadrature (5.0%) in that row. Integrating every bin by quadrature converges monotonically.

---

## Resolution options

| Option | Change | Cost | Accuracy |
|---|---|---|---|
| A. First bin by quadrature | Replace `I_csr * (z − zz)` with `∫_zz^z I_csr du` (n = 16 Simpson in `u^(1/3)`) | +17 source-point solves per step | Fixes the dominant error; leaves trapezoid error in steep transients |
| B. Every bin by quadrature | Also replace the trapezoid for k ≥ 2 (n = 8 per bin was enough here) | ~9× the source-point solves per step | Monotone convergence in all cases tested |
| C. Adaptive | Quadrature only where `I_csr` changes by more than a tolerance across a bin | Between A and B | As B, but needs a tolerance and a data-dependent branch (bad for GPU kernels) |

**Current state of CSR.jl:** option A is implemented as the default (`I_csr_integral` in `src/wake.jl`).
`bmad_compat = true` restores Bmad's expression.

**Recommendation:** option B. Each (step, lag) is an independent root solve, so the extra solves parallelize
on the GPU (phase 4). Needs a decision because it multiplies the wake-table cost.

For Bmad (if reported upstream): option A is a local change to `I_csr` (a few extra `s_source_calc` calls for
`i_bin = 1`). Expect `regression_tests/csr_and_space_charge_test/output.correct` to change, since its n_bin = 40
bend is in the regime where the approximation matters.

### Caveats

- The comparison integrates Bmad's own kernel `I_csr`. Whether that kernel is right is a separate question
  (Sagan & Mayes, IPAC2017 THPAB076; see `PLAN.md` Section 7).
- The transient numbers use CSR.jl's port, which differs from Bmad in the `dspline_len` s⁴ term
  (`dspline_coeff.md`). For an on-axis centroid that term is zero, so it does not affect these results.
- Not checked against Bmad output.

---

## Handling in CSR.jl

- Default: option A. `bmad_compat = true`: Bmad's expression.
- Tests (`test/wake_test.jl`, "First-bin integral"; `test/kick_test.jl`, "Steady state in a long bend"):
  - same-element first bin matches the steady-state integral;
  - cross-element first bin matches the steady-state integral to 0.3%;
  - `bmad_compat` first bin is 2/3 of it;
  - steady-state kick within 2% of the analytic wake at n_bin = 200, and `bmad_compat` is worse.
- Pending decision: option B.
