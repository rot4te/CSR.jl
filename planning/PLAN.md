# CSR.jl implementation plan

1D (line-charge) coherent synchrotron radiation for the SciBmad ecosystem, ported from the Bmad
`csr_method = one_dim` algorithm. Physics background and derivations: `csr_gpu_ad_design.md`.
Numerical checks of the analytic relations: `verify_analytic.jl` (Section 7).

Status: phase 0 done (branch `phase0-skeleton`, `25b8232`). Phase 1 done on branch `phase1-reference`,
except the open decision on first-bin vs all-bin quadrature (`upstream-bin.md`). Phase 2 next.

---

## 1. Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | How CSR hooks into tracking | **Route (a):** CSR.jl slices each element and calls `BeamTracking.track!` per slice, applying the CSR kick between slices. No upstream changes. |
| 2 | Macroparticle charge | Same as IBS: `weight` = real particles per macroparticle. `q_p = w_p * chargeof(species; C=true)`; `weight === nothing` means 1. |
| 3 | Bmad reference build | conda env `bmad` (conda-forge bmad 20260918.0, `tao` executable; added by the user 2026-10-06). |
| 4 | Backends | KernelAbstractions **CPU backend only** for now. Kernels stay backend-generic; no GPU testing. |

---

## 2. Facts established from source

Checked against local checkouts on 2026-10-01: BeamTracking.jl `4456fb9`, Beamlines.jl `725565b`,
SciBmad.jl `45fef3a`, SpaceCharge.jl `e316d10`, bmad-ecosystem `ee81f2a9e`.

### BeamTracking.jl
- **Coordinates:** `bunch.coords.v` is an `N×6` matrix, columns `XI, PXI, YI, PYI, ZI, PZI`
  (`src/types.jl`). `pz = ΔP/P0`. `z > 0` is ahead of the reference (`t = -z/v + t_ref`,
  `src/utils/z_to_time.jl`). Both match Bmad.
- **Particle state:** `coords.state::Vector{UInt8}`, alive is `STATE_ALIVE = 0x1`.
- **Weights:** `coords.weight` is `nothing`, a scalar, or a vector. IBS uses
  `log_N = log(weight) + log(N)` (or `log(sum(weight))`), i.e. weight = real particles per macroparticle.
- **Tracking:** one element = one fused per-particle kernel (`KernelChain`, `launch!` in `src/kernel.jl`);
  all integrator steps run inside it. Callbacks are also per particle, inside the kernel.
  **A whole-bunch operation (histogram) cannot run between integrator steps**; it must run between launches.
- **IBS precedent** (`ext/BeamTrackingBeamlinesExt/general_bl.jl`, `ibs_kick`): host-side bunch moments,
  one thick kick per element. Not sufficient for CSR, which needs a new histogram per step.
- **Stale branch** `origin/space_charge` (Aug 2025) interleaved space-charge solves with integrator sub-steps.
  Uses the old API (`R_ref`, `ExactTracking`); reference only.
- **Steps:** `find_steps(tm, L)` in `src/utils/find_stuff.jl`; `Fringe` enum has
  `NoEnd, BothEnds, EntranceEnd, ExitEnd`.

### Beamlines.jl
- `BendParams`: `g_ref`, `tilt_ref`, `e1`, `e2`, `edge1_int`, `edge2_int` (`src/bend.jl`).
- **No floor/survey coordinates** in Beamlines, BeamTracking, or SciBmad. CSR.jl builds its own planar geometry.
- **No element-slicing API** (`src/superimpose.jl` is empty).

### SpaceCharge.jl
- Standalone; not hooked into BeamTracking on `main`. `Mesh3D{T<:AbstractFloat}` (no Duals/TPS), trilinear CIC deposit.
- **Not a dependency.** Revisit only if a shared deposit becomes useful.

### AtomicAndPhysicalConstants.jl
- `chargeof(species)` in units of e (`C=true` for coulombs); `massof(species)` in eV/c².
  Exports `E_CHARGE`, `R_ELECTRON`, `C_LIGHT`, `EPS_0`, `CLASSICAL_RADIUS_FACTOR`.

### Bmad (`bmad/space_charge/csr_and_space_charge_mod.f90`, 2027 lines)
Routines to port: `track1_bunch_csr`, `csr_bin_particles`, `csr_bin_kicks`, `s_source_calc`,
`ddz_calc_csr`, `I_csr`, `image_charge_kick_calc`, `csr_and_sc_apply_kicks` (CSR-only branch),
`dspline_len`, `s_ref_to_s_chord`.

Differences from the older copy described in `csr_gpu_ad_design.md`:
- Convolution is now FFT-based (zero-padded to a power of two ≥ `2*n_bin`).
- With images (`y_source ≠ 0`), lags run over `-n_bin:n_bin` and the sum convolves **bin charge**, not its gradient.
- The CSR-only kick updates `vec(6)` only. **No `z` rescaling by β** (that is done only in the space-charge branch).
- Kick coefficient still `ds * r_c / (rel_mass * e * |q| * γ)` with `r_c = classical_radius(species) = r_e q²/rel_mass`
  (`bmad/code/classical_radius.f90`). See Section 7, item V2: the derivation disagrees for non-electrons.
- `dspline_len` s⁴ term appears to be missing a factor 1/5 (Section 7, item V1).
- `eleinfo` is built for elements `0 .. min(ix_ele+10, n_ele_track)`.
- Grid: `n_bin_eff = n_bin - 2 - (particle_bin_span + 1)`, `dz = 1.0000001 * (z_max - z_min) / n_bin_eff`, centered.

### Reference cases in bmad-ecosystem
- `regression_tests/csr_and_space_charge_test`: sbend `L = 0.1`, `b_field = 0.17`, `b1_gradient = -10`,
  42 MeV electrons, 77 pC, `σ_z = 0.899377 mm`, 1000 particles, `n_bin = 40`, `particle_bin_span = 2`,
  `ds_track_step = 0.01`; shielding case `n_shield_images = 4`, `beam_chamber_height = 0.01`.
  `output.correct` holds particle-10 deltas for Bmad's own random bunch, so it is **not reproducible without Bmad**.
- `bmad-doc/tao_examples/csr_beam_tracking`: multi-element lattice (pipes, combined-function bends, optional patch).

### Not verified
- Enzyme support for KernelAbstractions kernels.
- Whether GTPSA TPS can be used in KA CPU-backend kernels (TPS is not `isbits`).
- The steady-state coefficient `2/3^{1/3}` and sign (check against a primary source before using as a test tolerance).
- Overlap/double counting between 1D CSR and LSC/space charge.

---

## 3. Package layout

The core operates on plain arrays (`z`, `pz`, `q`, `state`) and has no BeamTracking dependency.
Integration and AD live in extensions.

```
CSR.jl/
  Project.toml
  PLAN.md
  src/
    CSR.jl            module, includes, exports
    params.jl         CSRParams: n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height
    geometry.jl       planar centroid survey; per-element chord, theta_chord, L_chord, cubic spline, dL_s;
                      dspline_len, s_ref_to_s_chord; struct-of-arrays CSRGeometry
    binning.jl        grid setup, triangle-overlap deposit, edge gradient
    source_point.jl   ddz residual, element bracketing, root solve, upstream infinite-drift quadratic
    wake.jl           I_csr, first-bin I_int_csr, image_charge_kick
    convolve.jl       direct-sum convolution (all element types); FFT path for plain floats
    kick.jl           linear interpolation between bin centers, apply to pz
    step.jl           csr_step!: bin -> wake table -> convolve -> kick
    analytic.jl       steady-state wake for tests
  ext/
    CSRBeamTrackingBeamlinesExt/   track_csr!, centroid tracking, element slicing, weight -> charge
    CSRAbstractFFTsExt/            FFT convolution
    CSRChainRulesCoreExt/          rrule for csr_step!, implicit differentiation of the root solve
    CSRGTPSAExt/                   serial TPS deposit path
  test/
    runtests.jl
    geometry_test.jl, binning_test.jl, source_point_test.jl, wake_test.jl,
    analytic_test.jl, slicing_test.jl, tracking_test.jl, ad_test.jl
```

Proposed user API (new names, not existing SciBmad API):

```julia
params = CSRParams(n_bin = 40, particle_bin_span = 2, ds_step = 0.01)
track_csr!(bunch, beamline; params)          # extension; uses BeamTracking.track! per slice
csr_step!(pz, z, q, state, geom, ix_ele, s_kick, gamma, params; kick_factor)  # core
```

---

## 4. Phases

Each phase ends with passing tests. Work happens on feature branches; no pushes.

### Phase 0: package skeleton
- [x] `Project.toml` with deps and weakdeps.
- [x] `julia` compat entry (set to `"1.9"`; note the test target needs BeamTracking, which requires 1.10).
- [x] `[extensions]` (4 stub modules in `ext/`), `[extras]`, `[targets]`.
- [x] `test/runtests.jl` checks all four extensions load; `Pkg.test()` passes on Julia 1.12.7.
- ForwardDiff is added to the test target in phase 6, when it is first used.

### Phase 1: serial CPU reference port (Float64)
A line-by-line port of the Bmad CSR-only path, plain loops, no kernels. This is the ground truth for later phases.
- `geometry.jl`: centroid floor positions, chord, spline `create_a_spline` equivalent with end slopes
  `theta0`, `theta1`; `dspline_len`; `s_ref_to_s_chord`.
- `binning.jl`: Bmad grid and triangle-overlap deposit; `edge_dcharge_density_dz`.
- `source_point.jl`: element walk + bracketed root solve (Brent-equivalent), upstream-drift quadratic.
- `wake.jl`: `I_csr`, `I_int_csr` (first-bin log formula, trapezoid otherwise).
- `convolve.jl`: direct sum `k_i = coef * Σ_{j≤i} I_int[i-j+1] * g[j]`.
- `kick.jl`: interpolation and kick, half weight at `i_step ∈ {0, n_step}`.

Tests:
- `dspline_len` against numerical quadrature of `x'^2/2`.
- Spline geometry: a circular arc reproduces `ρφ³/24` path excess (small φ).
- Straight drift: kick identically zero.
- Long bend, Gaussian bunch, far from the entrance: wake matches the steady-state formula within noise.
  Fix the coefficient against a primary source first (Section 2, "Not verified").
- Charge conservation in deposit; `particle_bin_span` overlap weights sum to 1.

### Phase 2: BeamTracking integration (route a)
- Centroid: track a single particle at the bunch centroid through the beamline with `BeamTracking.track!`,
  recording coordinates at each element exit (`centroid(0:)` equivalent).
- Slicing: build per-step copies of each `LineElement` with length `L/n_step`, edge angles and fringe only on
  the end slices (`fringe_at = EntranceEnd / NoEnd / ExitEnd`), integrated strengths rescaled if needed.
- `weight` → `q_p` as in Section 1; alive mask from `coords.state`.
- `γ` per step from the reference energy (linear interpolation across RF elements, as Bmad).
- Reject elements CSR v1 cannot handle (see Section 5).

Tests:
- **Gate test, first:** with CSR off, `n` slices reproduce one unsliced `track!` for drift, sbend
  (with `e1`, `e2`, fringe), quadrupole and combined-function bend, to round-off.
- CSR on, single sbend from `csr_and_space_charge_test` parameters: energy loss sign and magnitude
  consistent with phase 1 analytic tests.

### Phase 3: Bmad cross-check (deferred)
Blocked on a Bmad build. When available:
- Export a fixed initial bunch from Bmad (or import one into Bmad) so both codes see the same particles.
- Compare per-step wake (`space_charge_com%diagnostic_output_file`) and final phase space for
  `csr_and_space_charge_test` and `tao_examples/csr_beam_tracking`.

Until then, phase 1 analytic tests and phase 2 consistency tests are the validation.
**Done 2026-10-06** with conda-forge Bmad 20260918.0 (branch `phase3-bmad-crosscheck`, Section 14).

### Phase 4: KernelAbstractions kernels (CPU backend)
Replace loops with KA kernels: deposit (Atomix atomics), gradient, wake table (one work item per lag),
convolution, kick. Generic element type (`zero(x)`-style constructors). Backend taken from `get_backend(z)`.

Tests: KA CPU backend equals the phase 1 reference to round-off (allow atomic-ordering differences in the deposit).

### Phase 5: shielding (image charges)
Port `image_charge_kick_calc` and the `-n_bin:n_bin` charge convolution with `(-1)^ns`, factor 2, half weight at ends.

Tests: free-space limit as `beam_chamber_height → ∞`; strong shielding reduces the wake.
**Done 2026-10-06** (branch `phase5-shielding`, Section 13).

### Phase 6: automatic differentiation
- ForwardDiff Duals through the generic kernels (direct sum, no FFT).
- `rrule` for `csr_step!` with explicit adjoint kernels (design doc §3.8); implicit differentiation of the
  source-point root. Grid bounds (`z_min`, `dz`) treated as constants.

Tests: reverse-mode gradient = ForwardDiff = finite difference on a small bunch.

Steps (D16): [x] 0 baseline; [x] 1 ForwardDiff on particle coordinates; [x] 2 `rrule` with explicit adjoint kernels
(**done 2026-10-09**, branch `phase6-ad`, Section 17); [ ] 3 derivatives w.r.t. geometry/γ (implicit differentiation
of the source root), after the Section 2 open questions (D19).

### Phase 7: TPS (CPU, serial)
Serial deposit path for GTPSA TPS; optional higher-order shape function and matching interpolation.

Tests: first-order TPS coefficients = Jacobian; second order = ForwardDiff Hessian.

### Phase 8: performance and upstream
- FFT convolution for plain floats (`CSRAbstractFFTsExt`) when `n_bin` is large.
- Benchmarks vs particle count, `n_bin`, steps.
- Draft an issue for BeamTracking proposing a collective-step hook (route b), using CSR.jl as the motivating case.

---

## 5. v1 limitations

- Planar geometry only: bends with `tilt_ref ∈ {0, π}`. Vertical bends are rejected.
- Patches with out-of-plane offsets rejected (Bmad rejects `floor_shift`).
- No LSC or transverse space charge.
- No wigglers (Bmad warns that the centroid spline is inaccurate there).
- Requires live particles ≥ `n_bin` (Bmad behavior).
- CPU backend only tested.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Element slicing does not reproduce unsliced tracking (edges, fringe, integrated strengths) | Phase 2 gate test before any CSR work in the extension |
| Per-slice `track!` overhead (element unpacking is type-unstable) | Measure in phase 8; route (b) upstream if dominant |
| No Bmad reference numbers | Analytic tests now; phase 3 when a build exists |
| Unverified steady-state coefficient | Check a primary source before using it as a tolerance |
| `|q|` placement for ions | Electrons only in tests until resolved |

---

## 7. Verification of analytic relations

Script: `verify_analytic.jl` (Base Julia only). Result on 2026-10-01: 33/33 checks pass, using the corrected
`dspline_len` (V1).

### Verified numerically

| Check | Relation | Result |
|---|---|---|
| C1 | Retardation `z = Ls − βL`; Bmad's `z ≈ L/(2γ²) + dL` has error `L/(8γ⁴)` | matches to < 1% |
| C2 | Circle: `L = ρφ(1 − φ²/24)`, `dL = ρφ³/24 − ρφ⁵/1920` | rel. err < 1e-7 |
| C3 | Overtaking length `Ls = (24ρ²z)^{1/3}` (γ → ∞) | O(φ²) |
| C4 | Bmad `I_csr` on a circle → `−2·3^{−1/3} ρ^{−2/3} z^{−1/3}` for γφ ≫ 1 | rel. err 4e-6 |
| C5 | Bmad first-bin `I_int` (γ → ∞) `= ∫₀ᶻ I = −3^{2/3} ρ^{−2/3} z^{2/3}` | O(φ²) |
| C6 | Bmad first-bin `I_int` with the `ln(2γ²z/Ls)/γ²` term vs quadrature of `I_csr`, γ = 30, 100, γφ = 0.3–20 | rel. diff ≤ 4e-4 |
| C7 | Bmad scheme (edge gradients, first-bin + trapezoid, causal sum) converges to the doc §2.4 integral | 6e-3 → 6e-4 as n_bin 100 → 1600 |
| C8 | `dspline_len` closed form vs quadrature of `(x' − dθ)²/2` | exact with the V1 correction |
| C9 | Upstream infinite-drift quadratic satisfies `Ls − βL = dz` exactly, root has `ds < 0` | 1e-12 |
| C10 | Triangle overlap weights sum to 1 | 1e-15 |
| C11 | Reverse-mode adjoints of doc §3.8 (q̄, z̄, K̄) vs finite differences | 1e-10 |

Algebraic (by hand, no code):
- C4 + Bmad prefactor ⇒ the doc §2.4 steady-state formula, including the coefficient `2/3^{1/3}` and sign,
  for electrons. This shows the doc formula and Bmad's kernel are mutually consistent; it is not
  independent of Bmad. An independent check needs a primary source (below).
- Bmad `image_charge_kick_calc` has the Liénard–Wiechert structure: `coulomb1 = B·(N−β')/(γ²L²κ³)`,
  `radiate = B·(N×((N−β')×G))/(Lκ³)`, `G = β² g n̂ ⟂ β'` (= β̇/c on a circle), with `e/(4πε₀)` factored out.
- Implicit-function derivative `ds/dp = −F_p/F_s`: standard.

### Findings

**V1. `dspline_len` (Bmad `csr_and_space_charge_mod.f90:1796`).** The s⁴ term is `… * 9*c(3)**2`.
Integrating `9c₃²s⁴` over the interval gives `9c₃²/5 · (s₁⁵ − s₀⁵)`, so the factor 1/5 appears to be missing.
Exact when `c₃ = 0` (centroid symmetric about its chord, e.g. on-axis sector bend).
For an asymmetric chord spline (end slopes 0.010, −0.012) Bmad's dL is 1.71× the quadrature value.
Affects `dL`, hence the retardation root and `I_csr`, for off-axis or asymmetric centroid orbits.
**Action:** implement the corrected form; keep a `bmad_compat` switch for phase 3 cross-checks; report to Bmad maintainers.

**V2. Kick prefactor for non-electrons.** Derivation: `δ = ΔE/(β²E₀)`, `ΔE = q_w e · Q · K · ds/(4πε₀)`,
`1/(4πε₀) = r_e m_e c²/e²` ⇒ `δ = q_w r_e Q K ds/(e γ rel_mass)`.
Bmad: `classical_radius(species)/(rel_mass · e · |q| · γ) = |q| r_e/(e γ rel_mass²)`.
With `Q` = macroparticle charge in C (`bmad_struct.f90:558`, set from `charge_tot/n`), the two differ by a factor
`rel_mass` (agree for electrons and positrons). Not resolved: needs the definition of `particle%charge` for ions
and the Sagan & Mayes paper. **Action:** electrons/positrons only in tests until resolved.

### Not verified, and where to check

Citations below are from memory unless marked; check the reference before relying on it.

| Item | Where to verify |
|---|---|
| `I_csr` formula from first principles (`−2(dL/z + γ²θ_slθ_lk/(1+γ²θ_sl²))/L`) | D. Sagan & C. Mayes, "Coherent Synchrotron Radiation Simulations for Off-Axis Beams Using the Bmad Toolkit", IPAC2017, THPAB076. URL from the Bmad source header: https://accelconf.web.cern.ch/ipac2017/papers/thpab076.pdf |
| First-bin `I_int` log term, as an analytic result (C6 checks it only numerically, to ~4e-4) | Same paper; also the Bmad manual CSR section |
| Steady-state wake coefficient `2/3^{1/3}` from an independent source | Derbenev, Rossbach, Saldin, Shiltsev, DESY report TESLA-FEL 95-05 (1995); Saldin, Schneidmiller, Yurkov, Nucl. Instrum. Methods A 398, 373 (1997); Murphy, Krinsky, Gluckstern, Part. Accel. 57, 9 (1997) |
| CSR impedance from retarded potentials | Stupakov & Zhou, Phys. Rev. Accel. Beams 19, 044402 (2016) |
| Validity criterion `σ_x ≪ (ρσ_z²)^{1/3}` | Derbenev et al. (above) |
| Liénard–Wiechert field (structure used in image kick) | J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Ch. 14 (eq. 14.14 in Gaussian units; check the equation number) |
| Shielding image geometry (`(−1)^n`, factor 2, beam on midplane) | Bmad manual CSR/space-charge chapter; parameters `n_shield_images`, `beam_chamber_height` |
| B-spline smoothness (degree d is C^{d−1} with simple knots) | C. de Boor, *A Practical Guide to Splines* |
| Newton iteration on power series doubles the correct order per step | Brent & Kung, "Fast algorithms for manipulating formal power series", J. ACM 25 (1978) |
| Kick prefactor for ions (V2) | Sagan & Mayes paper; Bmad manual definition of macroparticle `charge`; Bmad maintainers |
| `dspline_len` (V1) | Bmad maintainers (`bmad-sim/bmad-ecosystem` issues) |
| Enzyme + KernelAbstractions support | Enzyme.jl and KernelAbstractions.jl current docs |
| CSR vs LSC overlap | Bmad manual CSR/space-charge chapter |

**V3. `s_ref_to_s_chord`, bend branch (lines 1817–1842). Open question, ported literally.**
- The offset `x = dr(1)*cos(t) + dr(3)*sin(t)` with `t = θ_ref + π/2` projects `dr` onto (−sin θ, cos θ), the
  direction of motion, not the local x-axis (cos θ, −sin θ). Since `dr` (reference minus centroid at the
  entrance) is transverse, `x ≈ 0` always.
- For an on-axis bend the formula gives `ρ·atan(s/ρ)`, which at `s = L` is not `L_chord = 2ρ sin(L/2ρ)`
  (e.g. 0.09967 vs 0.09996 for L/ρ = 0.1). The kick point at the last step is then slightly short of the exit.
- Intent unclear without the Sagan & Mayes paper or the maintainers. Effect is O((L/ρ)²) on the kick-point position.

**V4. `s_source_calc` round-off branches (lines 951–959). Ported literally.**
`s_source = ddz_calc_csr(einfo_s%L_chord, status)` assigns the residual, not the position `L_chord`
(likewise for 0). Free-space CSR only uses `s_chord_source` for diagnostics, but the image-charge kick
(phase 5) uses it to evaluate `g_bend`. Phase 5 decision (user, 2026-10-06): keep Bmad's behaviour, in default
mode too (bends are typically symmetric: `g_bend` is then the same at both ends, so the effect vanishes).

**V5. First-bin integral when the source is upstream of the kick element.** Bmad holds `I_csr` constant over
the rest of the bin: 2/3 of the integral in steady state, 3–4× too small in the entrance transient.
Full report: `upstream-bin.md`. CSR.jl integrates the first bin by quadrature (`bmad_compat` restores Bmad).
**Decision pending:** integrate every bin by quadrature (~9× the source-point solves).

**V6. Energy interpolation in `track1_bunch_csr` (lines 330–331). Confirmed (2026-10-01).**
`e_tot = f1 * E(ix_ele−1) + (1 − f1) * E(ix_ele)` with `f1 = s/L`. `ele%value(E_tot$)` is the exit energy
(`low_level/compute_reference_energy.f90:583–588, 685`: `E_tot_start = ele0%value(E_tot)`, `E_tot = E_tot_start + delta_e_ref`),
so Bmad uses the exit energy at s = 0 (weights reversed). Only matters for elements whose reference energy changes (lcavity).
**Status:** `csr_ref_energy` fixes it; `bmad_compat` reproduces it. In BeamTracking the reference energy is constant inside
every element (shift applied at the next element's entrance), so the ramp is currently always constant (Bmad rfcavity behaviour).

**V7. Source geometry after `super_zbrent` (s_source_calc, line 997). Found 2026-10-01.**
`ddz_calc_csr` stores the source geometry in `kick1` as a side effect; after `super_zbrent` it holds the last evaluation,
not the returned root (super_zbrent ends with a linear interpolation, and returns at once when `f(a) == 0` after evaluating
`f(b)`). Normally within the 1e-8 m tolerance. When `ddz(0) == 0` exactly, the fields describe the far end of the element
(source ahead of the kick point) and `I_csr` is garbage (here 9.6e6 instead of −40). CSR.jl's first-bin quadrature (V5 option A)
evaluates exactly there (lower limit `zz`), which made the kick depend on 1e-18 centroid noise (×1e4). Bmad's own first-bin
code does not evaluate at that point, so the catastrophic case is unlikely in Bmad; the within-tolerance inconsistency remains.
**Status:** CSR.jl re-evaluates at the root unless `bmad_compat`. Regression test in `wake_test.jl`.

**V8. Source direction for a retarded source ahead of the kick point (ddz_calc_csr, line 1077). Found 2026-10-06.**
`floor_s%theta = theta_sl + theta_L`. In the same-element branch `theta_sl = x'(s0) − dtheta_L`, with `dtheta_L` the angle
of the L *line* in the chord frame, while `theta_L = atan2(L_vec)` is the angle of the *vector* kick − source. For a source
ahead (`ds < 0`) `L_vec` points backwards, so `theta = true direction + π`. Only `image_charge_kick_calc` reads it (line 1469):
source velocity and acceleration are reversed. Drift check (γ = 100, y = 0.01, kick at s = 5 of a 10 m drift; script
`planning/diagnostics/v8_drift.jl`, CSR.jl's literal port + literal image kick, coulomb term): behind sources match the analytic
uniform-motion field `γΔz/(γ²Δz² + y²)^{3/2}` with either angle; ahead sources (Δz = −0.011, −0.03) give 1.21, 8.99 with
Bmad's angle vs −0.826, −0.111 analytic; the true direction reproduces the analytic values to 5 digits.
History: line from `ff57a4dda` (2016-04-23, "More csr devel for off-axis beams"); the image path has looped over
negative lags since at least then. Only reached when |lag| > β·L ≥ ns·beam_chamber_height, i.e. bunch extent longer than
the chamber height; `csr_and_space_charge_test` (h = 1 cm, max |lag| ≈ 7 mm, my estimate) never reaches it.
Size: the analytic field peaks at `0.385/y² ≈ 3850` (γΔz = y/√2), so the ahead-source errors are ≤ 0.2% of the peak
here, but the field there is wrong by 80× and of the wrong sign. Effect on the bunch kick: Section 13.
Side observation: at lag 0 the image kick is 0.125 vs analytic 0 (3e-5 of the peak): Bmad's approximate retardation (C1),
not V8. (An earlier note said "peak ≈ 4": wrong, from a coarse sample.)

**Constants.** AtomicAndPhysicalConstants uses CODATA 2022 `r_e = 2.8179403205e-15 m`; Bmad hard-codes
`2.8179403227e-15 m` (`sim_utils/interfaces/physical_constants.f90:35`). Relative difference 8e-10; matters only
for bit-level comparison in phase 3.

---

## 8. Phase 1 results

Branch `phase1-reference`. Files: `src/{params,geometry,binning,source_point,wake,convolve,kick,step}.jl`;
tests in `test/{geometry,binning,wake,kick}_test.jl` with helpers in `test/utils.jl`.

Ported from Bmad (CSR only, free space, no shielding, no match/floor_shift elements): centroid chord and spline
geometry, `dspline_len`, `s_ref_to_s_chord`, triangle binning, `s_source_calc` (element walk + `super_zbrent`
equivalent + upstream infinite drift), `I_csr`, causal convolution (direct sum instead of Bmad's FFT;
same result), linear-interpolation kick, kick coefficient. Added: `planar_survey` (Bmad floor conventions:
g > 0 bends to −x, θ decreases).

Differences from Bmad (all restored by `bmad_compat = true`): V1 (`dspline_len` 9c₃²/5), V5 (first-bin quadrature).

Test results: all pass on Julia 1.12.7. Key checks:
- Source point on a circle: retardation residual < 1e-10; `dL`, `θ_sl`, `θ_lk`, `I_csr` match the circle within 0.5%.
- Steady-state kick (ρ = 10 m, σ_z = 1 mm, n_bin = 200) within 2% of the analytic wake (measured 0.28%).
- Straight drift: kick < 1e-6 of the bend kick scale.
- Upstream infinite drift: exact retardation `Ls − βL = dz` to 1e-8.

---

## 9. Phase 2 results (2026-10-01/02)

Branch `phase2-beamtracking` (from `phase1-reference`): `f225d96`, `625bdfc`, `2b8628b`, `12bc98b`. 290/290 tests pass
(BeamTracking 0.8.4, Beamlines 0.10.4 as resolved by `Pkg.test`; the local checkouts differ).

API (extension `CSRBeamTrackingBeamlinesExt`):
- `track_csr!(bunch, bl; params, csr_on = ele -> true, centroid_start = nothing, diagnostics = nothing)`.
- `slice_element(ele, s0, L_slice, first, last; context)`, `csr_sliceable(ele)`.
- Core: `advance_floor`, `advance_floor_patch`, `CSRGeometry(L, g, ref_exit, centroid)`, `csr_ref_energy`.

Decisions (user, 2026-10-01): option A kept; Bmad equivalence at the CSR level (slicing, geometry, γ, kicks on top of
BeamTracking's tracking), not particle level; RF γ from Bmad's ramp with V6 fixed; per-element opt-in (`csr_on`).

Slicing (Bmad `makeup_super_slave1` rules), each with a gate test (sliced == unsliced ≤ 1e-15, CSR off) and a control:
fringe/edges only at ends (`fringe_at`; `SciBmadStandard` → the `Symplectic` order BeamTracking picks), integrated
multipoles scaled, apertures at ends, alignment recomputed per slice (straight: offsets follow the slice centre; bend:
element transform at slice centre inverted with BeamTracking's `coord_alignment_bend_mid/at_s`), patch transform on the first
slice with exit face moved back by L − L_slice, RF voltage ∝ length and rate → frequency (slice in a one-element Beamline),
four-potential evaluated at s + s0, t + s0/(β₀c). Map elements, zero-length elements, and `csr_on = false` elements are
tracked whole without CSR. `SaganCavity` unsupported. Vertical bends (tilt_ref ∉ {0, π}) rejected.

Geometry: bend and patch floor conventions verified by tracking a field-free particle through drifts, bends (tilt_ref 0, π)
and patches: it stays on a straight floor line to 1e-14.

Physics: `track_csr!` through 80 bends (ρ = 10 m, γ = 1e4) reproduces the analytic steady-state wake within 2%.

Notes:
- BeamTracking `R_to_gamma` omits the charge state (wrong for |q| ≠ 1); CSR.jl uses `R_to_E / massof`.
- Depends on BeamTracking 0.8.4 internals: `check_bl_bunch!`, quaternion helpers, `coord_alignment_bend_*`, the
  `SciBmadStandard` → `Symplectic` order mapping. Gate tests catch changes.
- Bmad's `csr_method` per element defaults to off; `track_csr!` defaults to all elements on (`csr_on`).
- RF γ: BeamTracking keeps the reference energy constant inside an element, so γ is constant (= Bmad rfcavity). An lcavity-like
  ramp would need a reference-energy change inside the element (not representable in BeamTracking 0.8.4).
- Test suite takes ~2.5 min (mostly compilation).

## 10. Phase 4 results (2026-10-02)

Branch `phase4-kernels` (from `rungekutta-slicing` ← `v1-tracking`): `9ad2cfb`. 420/420 tests pass on 1 and 4 threads,
run against the local dev BeamTracking.jl and Beamlines.jl (v0.10.5, needed for `RungeKutta`/`EMFieldParams`).

Kernels (KernelAbstractions 0.9.43, Atomix 1.2.1, CPU backend), backend from `get_backend` of the particle arrays:
- Deposit `csr_deposit_kernel!`: one item per particle, Atomix atomic adds for Float32/Float64; other element types
  (BigFloat, Duals, TPS) use a serial CPU loop over the same per-particle function (atomics on Duals fail: checked).
- Edge gradient, convolution (one item per bin), kick interpolation (one item per particle).
- Wake table `csr_wake_table`: blocks of lags (default 4 per thread, workgroup size 1); inside a block the source search
  warm-starts from the previous lag, as Bmad. Identical results for any blocking (tested). CPU only: mutable `CSRKick1`
  and non-isbits `CSRGeometry`. A GPU version needs an isbits geometry (Adapt) and an immutable work struct.
- Out-of-grid particles are counted in the kernels; the host raises the same errors as before.
- `@index(Global)` must be a standalone assignment on the KA CPU backend (`@index(Global) + 1` fails).

Reference: phase 1 loops kept as non-exported `csr_*_reference` in `src/reference.jl`. Gate (`test/kernels_test.jl`):
grid, wake table and source points equal exactly; charges/kicks to ≤ 1e-12 relative (atomic order); 5 geometries ×
`bmad_compat`; Float32 and BigFloat particles.

Generic types: `CSRBins`, `CSRStepResult` parametric; particle-side kernels use the particle element types. The wake table
and geometry stay Float64 (phase 6 decides how derivatives enter: implicit differentiation per the design notes).
`nint` has an exact generic fallback (BigFloat lacks `RoundNearestTiesAway`).

Timing (20001 particles, n_bin 200, steady-state bend, M-series Mac): kernels 1.1 ms (1 thread) / 0.5 ms (4 threads),
reference 1.0 ms. The wake table dominates (0.75 ms / 0.27 ms).

## 11. lcavity mode (2026-10-02)

Branch `lcavity-mode` (from `phase4-kernels`): `ace9a8c`. 522/522 tests on 1 and 4 threads (dev BeamTracking/Beamlines).

Decision (user, 2026-10-02): lcavity mode needed; implement CSR-side from BeamTracking's internal Sagan kernels (no upstream
change; D1 kept).

Facts (BeamTracking dev after v0.8.4): an lcavity is an RF element with `tracking_method = SaganCavity` in its own Beamline
with `dE_ref` (Branch puts it there; `dE_ref` is a design input, only the first element of a Beamline sees it, and only
SaganCavity reads it). Inside, the reference momentum steps by `dE_ref/n_cells` at each cell kick (half at the end kicks);
fringe kicks always at both ends of the active region (no `fringe_at`). No BeamTracking `track!` for a Branch.

Implementation (`ext/CSRBeamTrackingBeamlinesExt/sagan.jl`): `SaganSlice` = window [s0, s1] of the whole cavity's kernel
sequence; events at p belong to the slice with s0 ≤ p < s1 (≤ for the last). Kernel chain built with BeamTracking's
`KernelChain`/`RefState`/`launch!`, ext `alignment`/`aperture`/`rf_omega_calc`/`rf_phi0_calc`/`rf_step_calc`/`get_strengths`.
`track_csr!(bunch, ::Branch)`; γ linear between E_ref − dE_ref and E_ref; applied Δpz × pc_lin(s)/P0c_step(s)
(`pz_scale`). Slice boundaries must be shared exactly (track_csr! uses `L_slice = i·ds − s0`).

Results: one slice == whole cavity exactly; n slices ≤ 1e-18 (traveling wave, no multipoles), ≤ 1e-7 standing wave
(ponderomotive half-kick split, ∝ V²) and with multipoles. CSR in a cavity doubling the energy: 2 vs 8 cells agree to 5e-4
with the rescale, 4% without.

BeamTracking issues found (not changed, report upstream?):
- `universal!` for SaganCavity sets `bunch.t_ref = dt_ref` (drops the entrance `t_ref`); CSR.jl reproduces it.
- Its error messages interpolate an undefined `ele_name`; a stray `\` ends the FourPotentialParams check line.
- RFParams with `voltage = 0` count as inactive, so a SaganCavity with zero voltage errors ("without RFParams").
- `bunch.p_over_q_ref` is not updated after a SaganCavity.

## 12. Type stability and latency (2026-10-06)

Branch `type-opt` (from `lcavity-mode`): `80f05d0`. 522/522 tests on 1 and 4 threads; results bit-identical to
`lcavity-mode` (1 thread, vector/scalar/no weights).

Analysis (JET `@report_opt`, SnoopCompile `@snoop_inference`/`@snoop_invalidations`):
- Core (`src/`): type-stable except the wake kernel launch with a static workgroup size (fixed: size passed at launch).
- `track_csr_line!`: `p_over_q_ref`, energies, γ, `pz_scale` inferred as `Any` (Beamlines `getproperty`/`deval` return
  `Any`), so every `csr_kick!` was a runtime dispatch. Fixed with the barrier `track_csr_element!`.
- Centroid bunch had `weight = nothing`; for weighted bunches `Coords` differs, so BeamTracking compiled all its kernels
  twice. Fixed: same weight type, value 1 (only IBS reads it, via log N; identical results).
- `slice_element`: ~50 runtime dispatches per call, all from Beamlines parameter access (by design). Left as is: a
  barrier there adds specializations (compile time) for a µs-scale per-step cost.
- No invalidations from loading CSR. `Adapt` is an unused dep but KernelAbstractions loads it anyway (no saving).

Timings (M-series Mac, 1 thread), before → after: precompile CSR ~1.0 → 0.9 s, ext ~2.0–2.5 → 1.8 s (noise level);
first `track_csr!` with CSR off after a `track!` 3.6 → 0.37 s; fresh first `track_csr!` (4 elements, CSR on)
17–28 → 15 s; runtime unchanged (`csr_kick!` 20001 particles/200 bins 1.06 ms; `track_csr!` example 5.9 ms).
Remaining 15 s: 7 s BeamTracking's own first `track!`, ~8 s BeamTracking kernels specialized per slice
(`fringe_at` Entrance/No/Exit are `Val` type parameters). Not reducible from CSR.jl without upstream changes.
No PrecompileTools workload added (would raise precompile time; user asked for minimum).

## 13. Phase 5: shielding by image charges (2026-10-06)

Branch `phase5-shielding` (from `type-opt`): `7ad2bdc`. 602/602 tests on 1 and 4 threads.

Decisions (user, 2026-10-06): V4 kept as Bmad in default mode; sources ahead of the kick point beyond the kick element
are searched for downstream (option c: the code should generalize to space-charge-dominated beams with long bunches);
V8 fixed by default, `bmad_compat` reproduces (after checking that no good reason exists: Section 7, V8).

Implementation:
- `CSRParams`: `n_shield_images` (≥ 0), `beam_chamber_height` (> 0 when images are on; full chamber height: Bmad
  manual "height of the beam chamber", regression test comment "1 inch full height").
- `image_charge_kick` (literal port of `image_charge_kick_calc`; `g_bend = 0` on the infinite drifts),
  `csr_image_table` (kernel, lags `−n_bin:n_bin`, blocks of lags warm-started as `csr_wake_table`),
  `csr_image_convolve!` (kernel: `kick[i] += coef Σ_j image(i−j) charge[j]`), loop over layers in `csr_kick!`
  (`y_source = ns·h`, `kick_factor = (−1)^ns·2·kick_factor`) and in `csr_kick_reference!`.
- Source search, default mode (not in Bmad): for `dz < 0`, ddz > 0 at both ends of the kick element walks downstream.
  `ddz_calc_csr!` downstream branch: spline excess from the kick point to the source (same `modulo2(·, π/2)` line-angle
  folding as upstream), `dL = −(excess + 2 L_plane)`, then the `y_source` correction; source direction from the spline.
  Past the lattice end: infinite drift along the centroid exit direction with exact retardation, `t²/γ² − 2bt + c = 0`,
  `u = −(Lke + dz)`, `b = u − β² Lz`, `c = u² − β² L0²`, smaller root `t = c/(b + √(b² − c/γ²))`
  (`ix_ele_source = n + 1`, `source_s` gives `s_end + t`). A warm-started search past the end walks back if `t < 0`.
- Seam: Bmad's same-element ahead formula uses `2|ds|` (chord) where the cross-element one uses `2 L_plane`; they differ
  by O(ds·θ²). A root in that gap is caught by Bmad's round-off branch (V4), not an error.

Results:
- Free-space results bit-identical to `type-opt` (8 cases: 4 geometries × compat, `csr_kick_reference!`).
- Drift: image kick = uniform-motion field `γΔz/(γ²Δz² + y²)^{3/2}` within 5e-8 of the peak (sources behind/ahead,
  downstream elements, past the end), 3e-5 of the peak where Bmad's approximate retardation applies inside elements.
- Kernels == reference to ≤ 1e-12 (kicks, 3 geometries × compat); image table identical for any lag blocking.
- Steady-state bend (ρ = 10 m, σ = 1 mm, γ = 1e4), energy loss / free space, converged (32 layers, ≥ 480 bins):
  h = 10, 1 m: 1.000; 0.1 m: 0.93; 3 cm: 0.040; 1 cm: 0.0018; 3 mm: 0.042 (needs ≥ 1920 bins).
  **Correction (2026-10-09): the 3 mm value is not converged in the layer count** (partial sums alternate with layer
  parity; 32 vs 64 layers differ ~10×). See PLAN_continued.md top section. Not compared with a
  shielding theory (no confirmed formula at hand); the non-monotonic 3 mm value (h < bunch length) is unexplained.
- **Resolution (Bmad's algorithm):** the image sum samples the image kernel at bin centres times bin charge. Its structure
  in dz is roughly `Ls³/(24ρ²)` with `Ls = (12ρ²y²)^{1/4}` (my γ → ∞ estimate), ~1e-5 m for y = 3 mm, ρ = 10 m. With
  coarser bins the result is wrong, not just noisy: h = 3 mm gives 2.4 → 3.1 → 4.6 × free space at 60/120/240 bins,
  0.042 when converged. Bmad's default n_bin = 40 is far from enough there. **Done (D20, Section 18):** the default
  mode integrates the image kernel against the bin hat function; `bmad_compat` keeps point sampling.
- V8 effect, same bend, 16 layers, 480 bins, max |kick(fixed) − kick(compat)| / max |kick|: σ = 1 mm, h = 1 cm: 0.33;
  5 mm, 1 cm: 0.056; 10 mm, 1 cm: 0.049; 10 mm, 3 mm: 4.8. Net energy loss changes by ≤ 6%. (The 10 mm cases are not
  resolved in n_bin; the comparison is at equal discretization.)
- Cost: per layer 2n_bin + 1 source solves (free space: n_bin). 2001 particles, 120 bins: ~0.5 ms per layer (4 threads).

## 14. Phase 3: Bmad cross-check (2026-10-06)

Branch `phase3-bmad-crosscheck` (from `phase5-shielding`, `f8b8302`), directory `crosscheck/` (README there). Bmad: conda env `bmad`,
conda-forge bmad 20260918.0; its `csr_and_space_charge_mod.f90` is identical to the local 20261001 source. Tao only (no
Fortran compilation: MacPorts gfortran 14 vs the env's module version 16, and the Xcode license is not accepted).

Method: Tao generates the initial bunch (1000 particles) and writes it at BEGINNING (ASCII, 16 digits); both codes track
it. CSR n_bin 40, span 2, ds 0.01; shielded n = 4, h = 0.01; `bmad_compat = true`. Tao passes its single-particle design
orbit (from `particle_start`) as the CSR centroid (`tao_lattice_calc_mod.f90:570`), so `centroid_start = particle_start`.
Compared: (1) tracking with CSR off, (2) per-step wake (Bmad diagnostic file, 7 digits), (3) CSR-induced change of the
final coordinates, (4) `geom_check.jl`: CSR.jl geometry and I_csr built from Bmad's orbit (CSR alone).

Results (max |CSR.jl − Bmad| / max |Bmad|):

| Case | wake kick/m (free, shielded) | Δpz final (free, shielded) | I_csr from Bmad's orbit | floor |
|---|---|---|---|---|
| pure bend, on axis | 1.5e-6, 2.1e-6 | 3.2e-6, 2.4e-6 | — | — |
| pure bend, centroid offset | 1.6e-6, 2.1e-6 | 3.2e-6, 2.4e-6 | — | — |
| regression bend (b1_gradient −10, offset) | 9.9e-5, 1.1e-4 | 4.8e-5, 6.0e-5 | 2.8e-7 | 1e-17 |
| Tao example (drifts + 2 combined bends) | 2.3e-5, 9.8e-5 | 1.8e-5, 1.0e-5 | 2.0e-6 | 4e-16 |
| Tao example with patch | not meaningful (tracking) | — | 6.1e-7 | 5e-16 |

- CSR.jl reproduces Bmad's CSR (free space and image charges) to the precision of Bmad's diagnostic output when both
  use the same centroid orbit and particles. Patch floor geometry matches to 5e-16.
- Remaining end-to-end differences are BeamTracking vs Bmad tracking, not CSR:
  - Combined-function bend: BeamTracking's SBend with Bn1 is a bend–kick integrator with straight multipoles (it warns:
    "Straight multipoles are being used in a curved reference system"), default `n_steps = 1`. The centroid exit
    angle differs from Bmad by 2.3e-6 rad; with Bmad's orbit, I_csr agrees to 1.3e-6 instead of 1.2e-4.
  - Particle tracking without CSR (regression bend): px 1.8e-6 (unsliced) / 1.3e-7 (sliced) of 5e-4.
  - Patch case: the beam leaves the patch 17 mm / 93 mrad off the new reference, so the combined bends act at large
    amplitude; px differs by 1.6e-3 of 0.15 with CSR off.
- **Slicing changes BeamTracking's integration** (finding): default Symplectic takes `n_steps = 1` per element, and each
  slice keeps 1 step (`max(1, round(n_steps·frac))`), so `track_csr!` integrates with n_step steps where `track!` uses 1.
  Regression bend, zero charge: sliced − unsliced px = 1.7e-6. The phase 2 gate (sliced == unsliced ≤ 1e-15) holds only
  when the step counts match. The CSR centroid is tracked unsliced (`track!`), the bunch sliced. Possible change:
  track the centroid with the same slices (sliced tracking is closer to Bmad here). Not done; user decision.
- Default mode (fixes on) vs Bmad: final Δpz differs by 53% (single bend) and 6% (Tao example); the kick by up to 3.3×
  in the first steps of an element. By elimination this is V5 (first-bin integral; I_csr itself is unchanged; V1 needs
  c₃ ≠ 0, V6 an energy change, V8 images; V7 is ~1e-7). Short bends are entrance-transient dominated: the D6 question
  (first bin only vs all bins) matters here.
- Bmad diagnostic file quirks: with images, kick and I_csr are divided by the last layer's `kick_factor`
  (`(−1)^n·2` × free space) and `S_Source` is the last layer's; S-position has 6 decimals (I_csr sensitivity
  ≈ 1e-13 per 1e-15 m at short lags made this a 3e-5 effect until `i_step·ds` was used).

## 15. Sliced centroid (2026-10-06)

Branch `sliced-centroid` (from `phase3-bmad-crosscheck`): `a04f3ee`. 606/606 tests on 1 and 4 threads.
Decision (user): `track_csr!(...; sliced_centroid = true)` default; `false` = whole-element centroid tracking.
The centroid goes through CSR elements in the bunch's slices (shared `csr_slice` helper), so it is the orbit of a bunch
particle (test: geometry rebuilt from a zero-charge particle started on the centroid gives I_csr to < 1e-10 of the max).
- Fidelity vs Bmad (crosscheck, bmad_compat): regression bend wake 9.9e-5 → 5.9e-6, Δpz 4.8e-5 → 3.1e-6 (shielded
  1.1e-4 → 4.8e-6, 6.0e-5 → 3.4e-6); pure bends and the on-axis Tao example unchanged; patch case (tracking-model
  limited) Δpz 1.4% → 3.9%, not meaningful either way.
- Timing (Tao example, 1000 particles, 46 steps): warm 15.7 → 15.8–17.2 ms; fresh first call 12.5 → 11.0–11.7 s.
- Reproducibility: unchanged where BeamTracking's maps are exact (pure bends, drifts, round-off differences only);
  changed at ~1e-4 relative for combined-function bends/multipoles; the centroid now depends on `ds_step` and `csr_on`.

## 16. D6 option B: every bin by quadrature (2026-10-06)

Branch `all-bin-quadrature` (from `phase3-bmad-crosscheck`): `c256b87`. 613/613 tests on 1 and 4 threads.
Decision (user): option B "for now". `CSRParams(all_bin_quadrature = true)` default; `false` = option A (bit-identical
to the previous default, checked on 3 geometries, 1 thread); ignored with `bmad_compat` (crosscheck compat sections
unchanged). Bins ≥ 2: Simpson in u^(1/3), 8 nodes per bin (`I_csr_quadrature!`), kernel over blocks of bins with
warm-started searches (results independent of blocking: tested).

Fidelity (kick error, max over |z| ≤ 2σ / peak):
| Case | A | B |
|---|---|---|
| steady state vs analytic, n_bin 50 / 100 / 200 / 400 | 1.29e-2 / 4.89e-3 / 2.76e-3 / 1.82e-3 | 1.29e-2 / 4.37e-3 / 1.96e-3 / 1.16e-3 |
| transient s = 0.05, n_bin 40 / 100 / 400 (ref: B, 1600) | 0.234 / 0.059 / 0.050 | 0.233 / 0.057 / 0.0026 |
| transient s = 0.10, n_bin 40 / 100 / 400 (ref: B, 1600) | 0.098 / 0.022 / 0.0020 | 0.099 / 0.018 / 0.0019 |
The transient reference is B itself (convergence, not independent accuracy). At coarse binning both are dominated by
binning error. Default mode vs Bmad (crosscheck Δpz): regression bend 54.8% → 49.0%, pure bend 53.4% → 47.7%, Tao
example 6.4% → 4.7% (the first-bin fix, V5, still dominates).

Timing: `csr_kick!` (steady state, 20001 particles, n_bin 200): 1.08 → 7.09 ms (1 thread), 0.81 → 3.24 ms (4 threads).
`track_csr!` Tao example (1000 particles, n_bin 40, 46 steps): 16.8 → 18.3 ms (1 thread), 20.2 → 23.0 ms (4 threads).
Image layers unaffected (they do not use I_int). Phase 6: ~9× the source solves to differentiate.

Reproducibility: every default-mode free-space result changes (not bmad_compat, not images, not the I_csr diagnostic);
`all_bin_quadrature = false` restores option A exactly.

Branches `sliced-centroid` and `all-bin-quadrature` are independent (both from `phase3-bmad-crosscheck`); merging both
conflicts only in `crosscheck/*/result.txt` (rerun the crosscheck after merging). The combined effect is not measured.

## 17. Phase 6 steps 0–2: AD with respect to the particles (2026-10-09)

Branch `phase6-ad` (from `merge-recent`): `bb1dc13`. **837/837 tests on 1 and 4 threads** (617 existing + 219 in
`ad_test.jl` + 1 extension check). The existing 617 passed before any AD code was added, so the `csr_kick!` split is
behavior-neutral.

API: `csr_kick_pz(pz, z, q, alive, geom, ix, s_kick, gamma, species, params; kick_factor, pz_scale) -> pz_new`
(exported, non-mutating). ForwardDiff Duals pass through it (`CSRForwardDiffExt`: `CSR.primal(::Dual)`); with
ChainRulesCore an `rrule` gives `(p̄z, z̄, q̄)` by explicit adjoint kernels (`src/adjoint.jl`, `csr_kick_adjoint`).
Tangents: `NoTangent` for alive/ix/species/params, `@not_implemented` for geom, s_kick, gamma; keywords none.
`AbstractZero` and thunks pass through.

Constants under differentiation: the bin grid (`dz`, `z_min`; from the extreme live particles, via `primal`),
`n_particle`, and the kick tables (`I_int`, image tables). Consequence: `∂/∂z` of the min/max-z particle omits the
grid motion (finite differences disagree by O(1)–O(1e5) relative there); `∂/∂q` is exact for every particle.

Accuracy (6 cases: kick in bend / at element entrance; images on/off; `bmad_compat`; `all_bin_quadrature`;
`kick_factor = 0.5`, `pz_scale = 1.3`; 400 particles, n_bin 40):
- rrule vs ForwardDiff `J'ȳ`: ~1e-15 relative (test bound 1e-12).
- ForwardDiff vs central FD: `z` ≲ 1.5e-9 at h = 1e-8 (bound 1e-7), `q` ≲ 1e-12 (exact step h = q; bound 1e-9).
- FD is checked on core particles only: tail charges (~1e-19 C vs ~1e-11 C core) are below the rounding and
  atomic-order noise of the kick.

Implementation notes: the adjoint's kick-cotangent scatter uses atomic adds (serial CPU loop for non-hardware
floats); the deposit adjoint is a per-particle gather (no atomics). Forward with Duals uses the serial deposit (Atomix
cannot add Duals). Test env needs `Pkg.resolve()` after the weakdep is added, or the extension silently does not load.

## 18. Stage B: Section 2 open questions (2026-10-09/10)

Branch `phase6-open-questions` (from `bb1dc13`). **857/857 tests on 1 and 4 threads** (~2 min).

- D21 (V6 fixed upstream, Bmad PR #2199): `csr_ref_energy(E_start, E_end, s, L)` has no `bmad_compat`; both modes
  follow current Bmad.
- D23: `[compat] Beamlines = "0.10.6"`. BeamTracking compat waits for a release with `RungeKutta`.
- D20, image kernel integrated over bins (default mode; `bmad_compat` keeps Bmad's point sampling):
  `csr_image_table(...; bin_integrate = true, rtol = 1e-6)`. Per lag interval `[m dz, (m+1) dz]`,
  `m = −n_bin−1:n_bin`, adaptive Simpson of `(∫K, ∫K t)` gives the hat moments; `image(l) = (rise[l−1] + fall[l])/dz`.
  The two intervals touching u = 0 are split geometrically down to `u_min = 0.01 y/γ` (`image_u_min`); Simpson does
  not split below `u_min`. Tolerance per interval `rtol·dz·max|K|` over the lags and the geometric nodes
  (`image_hat_tol`).
  - Bug found while wiring it (fixed): `s_source_calc!` warm-started on the upstream infinite drift
    (`ix_ele_source = 0`) returned the closed-form root even when the source is inside the lattice. Bmad's monotone
    lag loops never hit this; the reference's second pass and Simpson's out-of-order nodes did. Now walks back
    (`ds_source > 0 && last_step == 0` → element 1), as the downstream drift already did.
  - Tolerance from the point table alone was ~1e4× too tight when dz ≫ y/γ (lags miss the peak), below the kernel's
    root-finder noise (~2e-7 relative at γ = 1e4): Simpson recursed to depth 40 (tests ran 22 min). Fixed by the
    geometric nodes in the scale and the `u_min` floor.
  - Tests: drift table vs analytic hat integral of the uniform-motion field (second difference of
    `−asinh(γu/y)/γ²`) < 1e-4 (γ = 100, y = 1 cm, dz = 2e-4; γ = 1e4, y = 3 mm, dz = 1e-4, where point sampling is
    > 50% off); kernels == reference; hat table independent of the lag blocking; upstream warm-start walk-back;
    steady-state bend h = 3 mm, 16 layers: 60 vs 480 bins within 3% (hat), not within 50% (compat).
  - Steady-state bend (ρ = 10 m, σ = 1 mm, γ = 1e4, 16 layers), energy loss / free space, n_bin 60/120/240/480/960:
    h = 3 mm point 2.68/3.41/4.88/0.365/0.355, hat 0.345/0.342/0.341/0.340/0.340; h = 1 cm point
    1.29/−0.061/0.0054/0.0050/0.0049, hat 0.0157/0.0078/0.0057/0.0051/0.0049. At 1 cm the hat result still depends on
    n_bin below 240 bins; likely charge-density resolution (~5 bins per σ at 60) acting on a ~0.5% residual, not
    verified.
  - Cost: ~5× the source solves of point sampling (0.05–0.6 s per kick, 16 layers, 60–960 bins, serial).
- D24 (user, 2026-10-10): layer sum by a closed-form/transformed sum; in progress (PLAN_continued.md top).
  256-layer brute force: 1 + Σa_n = 0.00284 (3 mm), 0.00265 (1 cm), so the 16-layer values above are not converged.
- Open: the image-layer sum is not converged at h = 3 mm (partial sums alternate with layer parity; Section 13
  correction). D22 (SaganCavity exact split) not started.
