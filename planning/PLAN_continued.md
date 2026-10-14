# CSR.jl: handoff for a new session

## ★ Session 2026-10-10: D20/D21/D23 committed (read this first; supersedes older notes below)

**Paths.** Two machines (`$HOME` = `/Users/alexcoxe` or `/Users/coxea3`): write paths with `~`. Package
`~/.julia/dev/CSR.jl`; dev checkouts beside it: `BeamTracking.jl`, `Beamlines.jl`, `AtomicAndPhysicalConstants.jl`,
`GTPSA.jl`. **Bmad source at `~/bmad-ecosystem`**. Conda env `bmad` at `~/miniforge3/envs/bmad`. Google Drive paths
in Sections 0 and 8 are stale.

**Stage C (2026-10-14), branch `stage-c` (from `468fc83`).** Step 0: CSR.jl compat `Beamlines = "0.10.6, 0.11"`,
`BeamTracking = "0.8.7"` (registered 0.8.5–0.8.7 contain `RungeKutta`; only 0.8.7 tested). 857/857 on 1 and 4 threads
against dev BeamTracking 0.8.7 (`65e343f`) and Beamlines 0.11.0 (`35fe016`). `R_to_gamma`/`R_to_v` ion bug still in
0.8.7. Plan: D24 → D22 → Phase 6 step 3.

**Git.**
- `phase6-ad`: `bb1dc13` Phase 6 steps 0–2 (ForwardDiff + rrule), 837/837 on 1 and 4 threads. Not pushed.
- **`phase6-open-questions`** (from `bb1dc13`): `9030f17` D20 + D21 + D23, **857/857 on 1 and 4 threads**. Not pushed.

**Test env** (scratchpad, wiped between sessions): devs CSR.jl, BeamTracking.jl, Beamlines.jl; adds Test, StaticArrays,
AbstractFFTs, ChainRulesCore, GTPSA, AtomicAndPhysicalConstants, KernelAbstractions, Atomix, ForwardDiff. Run
`Pkg.resolve()` after changing CSR's weakdeps, or new extensions silently do not load. From `test/`:
`julia --project=<env> -t 4 -e 'include("runtests.jl")'` (~2 min).

**Decisions (D16–D19 from 2026-10-08, D20–D23 from 2026-10-09):**
| # | Decision |
|---|---|
| D16 | Phase 6 = steps 0–2 now (done); step 3 (geometry/γ derivatives via implicit differentiation of the source root) later. |
| D17 | Non-mutating function: `csr_kick_pz`. |
| D18 | No ChainRulesTestUtils; test pullbacks by hand vs ForwardDiff `J'ȳ` and finite differences. |
| D19 | All Section 2 open questions handled before step 3. |
| D20 | Image kernel: **hat-weighted bin integral** (adaptive quadrature) in default mode; `bmad_compat` keeps Bmad's point sampling. |
| D21 | V6 is fixed in Bmad (PR #2199, 2026-10-06): **`bmad_compat` follows current Bmad**; V6 compat branch removed. |
| D22 | SaganCavity standing-wave ~1e-7 sliced-vs-unsliced difference: **investigate an exact split** (not accepted as is). |
| D23 | Compat: **`Beamlines = "0.10.6"` now** (registered, has `EMFieldParams`); BeamTracking when a release has `RungeKutta`. |
| D24 | Image-layer sum (2026-10-10): **closed-form / transformed sum** (not half-weight last layer, not leave-as-Bmad). Research + derive + validate first; default mode only, `bmad_compat` keeps Bmad's sum. |

**Stage B status:**
- Upstream reports: **drafted, not filed** in `planning/upstream_reports.md` (Bmad V1, V5, V7, V8; BeamTracking 6b items
  + new R_to_gamma/R_to_v ion bug). The user reviews and files. No existing upstream issue covers them (GitHub search
  via WebFetch, summarized: moderately reliable). Open Bmad PR #2085 rewrites `s_source_calc`; V7/V8 lines may move.
- **New BeamTracking finding (verified numerically):** `R_to_gamma` (`src/utils/energy.jl:6`) and `R_to_v` (`:9`) are
  wrong for |Z| ≠ 1 (He²⁺, R = 1: γ 1.00323 vs 1.01286; v/c 0.16034 vs 0.15882). `R_to_v` feeds `rf_omega_calc`
  (harmonic-number RF frequency), which CSR.jl's lcavity code also calls.
- D23 done: `Project.toml` `[compat] Beamlines = "0.10.6"`.
- D21 done: `csr_ref_energy(E_start, E_end, s, L)` has no `bmad_compat` keyword; caller in
  `ext/.../track.jl` and test in `track_csr_test.jl` updated; `crosscheck/run_case.jl` comment drops V6. Crosscheck cases
  are bends only (unaffected).
- **D20 done (`9030f17`, PLAN.md §18).** Two bugs found and fixed: `s_source_calc!` warm-started on the upstream
  infinite drift did not walk back into the lattice (the cause of the reference mismatch); the tolerance taken from the
  point table was far too tight when dz ≫ y/γ, so Simpson recursed to depth 40 (fixed: `image_hat_tol`, `u_min` floor).
  Unverified: at h = 1 cm the hat result still depends on n_bin below 240 (likely charge-density resolution).
- Prototype results (steady-state bend ρ = 10 m, σ = 1 mm, γ = 1e4, 16 layers; energy loss / free space):
  h = 3 mm, n_bin 60/120/240/480/960: point 2.68/3.41/4.88/0.365/0.355, hat 0.345/0.342/0.341/0.340/0.340.
  Hat costs ~5× the source solves of point sampling (0.05–0.6 s per kick here).
- **New finding: the image-layer sum is not converged at h = 3 mm.** Partial sums alternate with layer parity
  (hat, n_bin 960: 15/16/31/32/63/64 layers → −0.377/0.340/−0.043/0.042/0.0005/0.0032). PLAN.md §13's "converged
  0.042 at 3 mm" is a 32-layer truncation artifact (likely also its "unexplained non-monotonic" value). At h = 1 cm,
  31–64 layers agree within ~7%. Not acted on: ask the user (e.g. averaging consecutive partial sums / half weight on
  the last layer, or a closed-form layer sum). Script: `planning/diagnostics/image_layers.jl`. PLAN.md §13 not yet
  corrected.
- D22 not started: SaganCavity exact split (`ext/CSRBeamTrackingBeamlinesExt/sagan.jl`, test bounds in
  `lcavity_test.jl`: ≤ 2e-7 standing wave, ≤ 1e-7 multipoles).

**D24 started (layer-sum transform), nothing implemented.** Diagnostic `planning/diagnostics/layer_terms.jl [n_layers]`
writes per-layer energy-loss terms a_n / free-space loss (hat tables, n_bin 240) to `layer_terms_<h>_240.txt`
(gitignored dir; run with the scratch env + `DelimitedFiles`). ~1.5 s for 256 layers. First run, 256 layers:
1 + Σa_n = 0.00284 (h = 3 mm), 0.00265 (h = 1 cm). Note the h = 1 cm value differs from 16 layers (0.0057 at 240
bins), so 16 layers is not converged there either; PLAN.md §18's 1 cm numbers are 16-layer values. Not yet looked
at: the shape of a_n (decay law, smoothness in n), which decides the method. Candidates to evaluate against
brute-force sums (thousands of layers are cheap): linear per-lag weights from an Euler/repeated-averaging
transform (keeps AD linear in the tables), Wynn ε or Levin on scalars (nonlinear: bad for per-lag tables/AD),
or an analytic tail (Poisson/Euler–Maclaurin-type for alternating sums, needs the large-y asymptotics of the
kernel). Caveat: the circle test lattice is 8 m long, so large-y layers source from the upstream infinite drift
(not a true ring). Shielding parameter Π = σ_z ρ^{1/2}/h^{3/2} (from memory, verify): 19 at 3 mm, 3.2 at 1 cm, i.e.
strong shielding, consistent with ratios ~0.003, not 0.34.

**Next steps:** D24 (inspect a_n, pick/validate a transform, then propose to the user before implementing); D22;
then Phase 6 step 3. PLAN.md §13 note and §18 written.

---

Written 2026-10-01 (end of phase 1); rewritten 2026-10-02 after phase 2, RungeKutta slicing, phase 4 (kernels) and the
lcavity mode; updated 2026-10-06 after phase 5 (shielding). **Continue from `phase3-bmad-crosscheck` + feature branches `sliced-centroid`, `all-bin-quadrature` (to be merged by the user).** Read this first, then `PLAN.md` as needed
(Section 4 = phases, 7 = Bmad findings, 8–13 = results of phase 1, 2, 4, lcavity, type stability, phase 5).

---

## 0. Orientation

**Goal:** a 1D (line-charge) coherent synchrotron radiation (CSR) module for the SciBmad Julia ecosystem, ported from
Bmad's Fortran `csr_method = one_dim`, eventually GPU-capable (KernelAbstractions) and differentiable (ForwardDiff,
ChainRules, GTPSA TPS).

**Workspace root:** `/Users/coxea3/Library/CloudStorage/GoogleDrive-rot4te@gmail.com/My Drive/code/Bmad+/`
(path has spaces: quote it in shell commands). Paths below are relative to it unless absolute.

**Package:** `CSR.jl/` (git repo with remote `origin`).

**User preferences (also in the user's CLAUDE.md, loaded automatically):**
- Accuracy over helpfulness; flag uncertainty ("verify this", "from memory"); never invent APIs or sources.
- Fewest words; lead with the answer. Paragraphs ≤ 5 sentences.
- Code changes: report (1) what was wrong, (2) what changed, (3) tested/verified?, (4) what the user must do next.
- Git: never commit to or push existing branches; create a new branch and commit there; never push (the user pushes).
- Show a detailed plan and get approval before launching any subagent.
- Match existing style: 2-space indent, `@kwdef`, docstrings with `## Properties` (BeamTracking style), sparse comments.
- `planning/` is gitignored (user's choice): planning docs live there, uncommitted.
- When a design choice is the user's (physics model, upstream changes), ask with options and a recommendation.

---

## 1. Decisions already made (do not re-ask)

| # | Decision |
|---|---|
| D1 | Integration route (a): CSR.jl slices each element and tracks each slice with BeamTracking, applying the CSR kick between slices. No upstream BeamTracking changes (reaffirmed for the lcavity, D10). |
| D2 | Macroparticle charge from `bunch.coords.weight`: weight = real particles per macroparticle; `q_p = w_p * chargeof(species; C=true)`; `weight === nothing` means 1. |
| D3 | Bmad available since 2026-10-06: conda env `bmad` (conda-forge 20260918.0, `tao`). Phase 3 done (PLAN.md §14). |
| D4 | KernelAbstractions CPU backend only for now. |
| D5 | Default behavior fixes proven Bmad errors; `bmad_compat = true` reproduces Bmad. Ambiguous Bmad code (V3, V4) is ported literally and logged. |
| D6 | First-bin quadrature: option A (first bin only) for now (2026-10-01). |
| D7 | "Same as Bmad" means CSR-level equivalence (slicing rules, geometry, γ, kicks) on top of BeamTracking's own tracking, not particle-level. |
| D8 | RF γ: Bmad's linear reference-energy ramp, V6 fixed by default, compat reproduces. Δpz rescaled to BeamTracking's reference. |
| D9 | Per-element CSR opt-in (`csr_on`), default all elements. |
| D10 | lcavity mode is needed (2026-10-02). Implemented CSR-side from BeamTracking's internal Sagan kernels (user chose this over an upstream `fringe_at` for SaganCavity, and over Symplectic RF + reference shifts). |
| D11 | V4 (s_source_calc round-off branches) kept as Bmad in default mode too (2026-10-06; bends are typically symmetric). |
| D12 | Image charges: sources ahead of the kick point beyond the kick element are searched for downstream, and past the lattice end on an infinite drift (2026-10-06; code should generalize to space-charge-dominated, long bunches). Not in Bmad; compat restores Bmad. |
| D13 | V8 (source direction reversed for sources ahead) fixed by default, compat reproduces (2026-10-06, after an analytic check ruled out a reason for Bmad's choice). |
| D14 | `track_csr!(...; sliced_centroid = true)`: centroid in the bunch's slices (2026-10-06, branch `sliced-centroid`). |
| D15 | D6 → option B for now: `CSRParams(all_bin_quadrature = true)` (2026-10-06, branch `all-bin-quadrature`). |

## 2. Open questions for the user

- Is the ≤ 1e-7 sliced-vs-unsliced difference for standing-wave SaganCavity (ponderomotive half-kick split) acceptable?
- Report the BeamTracking SaganCavity issues (Section 6b) upstream?
- Report Bmad findings V1, V5, V6, V7 to the Bmad maintainers?
- Image-kernel resolution (PLAN.md §13): Bmad's point-sampled image sum needs bins much finer than the kernel structure
  (~1e-5 m for h = 3 mm, ρ = 10 m); coarse bins give wrong (not noisy) results. Integrate the image kernel over each bin?
- Phase 3 (PLAN.md §14): track the CSR centroid with the same slicing as the bunch? (BeamTracking's integration depends
  on slicing.) Revisit D6 (V5 changes the regression bend's CSR energy change by 53% vs Bmad).
- Report V8 to the Bmad maintainers (with V1, V5, V6, V7)?
- Project.toml has no compat entries for BeamTracking or Beamlines (user's change). Registration will need them; the code
  needs BeamTracking with `RungeKutta` and Beamlines with `EMFieldParams` (both unreleased as of 2026-10-02).

---

## 3. Git and environment

```
CSR.jl branches (all pushed by the user to origin, local == origin):
  main                474ac1b  user's commits (deps, gitignore planning/); origin/main is ae0ae77 (1 behind)
  phase0-skeleton     25b8232  Phase 0: package skeleton
  phase1-reference    b0944dc  Phase 1: serial CPU reference port            (from phase0-skeleton)
  phase2-beamtracking 625bdfc  Phase 2 (early part)                           (from phase1-reference)
  v1-tracking         51a7d79  user's: phase 2 complete + Project.toml weakdeps for dev BeamTracking/Beamlines
  rungekutta-slicing  a5e0120  RungeKutta + EMFieldParams slicing             (from v1-tracking)
  phase4-kernels      9ad2cfb  Phase 4: KA kernels                            (from rungekutta-slicing)
  lcavity-mode        ace9a8c  lcavity mode                                   (from phase4-kernels)
  type-opt            80f05d0  type stability, centroid weight type           (from lcavity-mode; PLAN.md §12)
  phase5-shielding    7ad2bdc  Phase 5: image charges                   (from type-opt; PLAN.md §13; not pushed)
  phase3-bmad-crosscheck f8b8302  Phase 3: Bmad cross-check      (from phase5-shielding; PLAN.md §14; not pushed)
  sliced-centroid     a04f3ee  sliced centroid (D14)                         (from phase3-bmad-crosscheck; PLAN.md §15; not pushed)
  all-bin-quadrature  c256b87  D6 option B (D15)                             (from phase3-bmad-crosscheck; PLAN.md §16; not pushed)
```

Next work: create a new branch from `phase3-bmad-crosscheck`. Commit messages end with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Dependencies.** CSR.jl's code uses the local dev checkouts: `BeamTracking.jl/` (BeamTracking 0.8.4 + 138 commits,
HEAD `7d79938`: `RungeKutta`) and `Beamlines.jl/` (v0.10.5, unregistered: `EMFieldParams`). Registered BeamTracking
0.8.4 / Beamlines 0.10.4 lack these, so `Pkg.test()` in CSR.jl's own environment fails the RungeKutta tests unless that
environment `Pkg.develop`s both.

**Run tests (as done in this session).** A scratch environment that devs CSR.jl, BeamTracking.jl, Beamlines.jl and adds
Test, StaticArrays, AbstractFFTs, ChainRulesCore, GTPSA, AtomicAndPhysicalConstants, KernelAbstractions, Atomix,
ForwardDiff. Then, from `CSR.jl/test/`: `julia --project=<env> -t 4 -e 'include("runtests.jl")'`.
Result on `phase5-shielding`: **602/602 pass on 1 and 4 threads**, ~1.6 min (mostly BeamTracking compilation). A single
file: `julia --project=<env> -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore,
GTPSA; include("utils.jl"); include("<file>_test.jl")'`. BeamTracking prints a harmless "Straight multipoles … curved
reference system" warning.

Julia 1.12.7 via juliaup. Project.toml: `julia = "1.10"` (the earlier 1.9 issue is resolved).

---

## 4. Source map (`lcavity-mode`, 2924 lines incl. tests)

### Core (`src/`), no BeamTracking dependency

| File | Contents |
|---|---|
| `CSR.jl` | Module. `using StaticArrays, AtomicAndPhysicalConstants, KernelAbstractions; import Atomix`. Exports `CSRParams, FloorPosition, CSRGeometry, CSRElementInfo, CSRBins, CSRKickPoint, CSRStepResult, planar_survey, csr_bin_particles, csr_kick!, csr_kick_coef, csr_n_step, dspline_len, track_csr!`. |
| `params.jl` | `CSRParams(; n_bin=40, particle_bin_span=2, ds_step=0.0, n_shield_images=0, beam_chamber_height=0.0, bmad_compat=false)`; validates `n_bin − 2 − (span+1) ≥ 1`, `h > 0` when images are on. |
| `geometry.jl` | `FloorPosition(r::SVector{3}, theta)`; `planar_survey(L, g; start)` (g>0 bends to −x); `advance_floor`, `advance_floor_patch`; `centroid_floor`; `modulo2`; spline helpers; `chord_spline` (= Bmad `create_a_spline`); `dspline_len(s0,s1,c,dθ; bmad_compat)` (V1); `CSRElementInfo`; `CSRGeometry(L, g, centroid; ref_start, bmad_compat)` and `CSRGeometry(L, g, ref_exit, centroid)` (centroid row 1 = start, row i+1 = exit of element i); `s_ref_to_s_chord` (V3, literal). |
| `binning.jl` | **Kernels.** `CSRBins{T,V,W}` (parametric); `triangle_overlap` (generic); `nint` (Fortran nint; exact generic fallback, BigFloat lacks `RoundNearestTiesAway`); `csr_deposit_kernel!` (one item per particle, Atomix atomic add for Float32/Float64; other types use a serial CPU loop over the same `deposit_particle!`); `csr_edge_gradient_kernel!`; `csr_bin_particles(z, q, alive, params)` (grid on host via `mapreduce`; out-of-grid count checked on host). |
| `source_point.jl` | `CSRKickPoint`; `mutable struct CSRKick1`; `ddz_calc_csr!`; `zbrent` (`super_zbrent` port); `s_source_calc!` (element walk + Brent + upstream infinite-drift quadratic; V4 literal; V7 re-evaluation at the root unless compat; V8 fix; downstream walk and downstream infinite drift `ix_ele_source = n+1` for dz < 0 unless compat); `source_s`. |
| `wake.jl` | `I_csr_value`; `I_csr!` (first bin closed form + `I_csr_integral` unless compat; else trapezoid); **kernels** `csr_wake_kernel!` (one work item per block of lags, warm-started search within a block as Bmad) and `csr_trapezoid_kernel!`; `csr_wake_table(geom, kp, dz, n_bin, backend; n_block = min(n_bin, 4·nthreads))` (CPU only: mutable `CSRKick1`, non-isbits geometry); `I_csr_integral` (Simpson in u^(1/3)). Image charges: `image_charge_kick`, kernel `csr_image_kernel!`, `csr_image_table` (lags −n_bin:n_bin). |
| `convolve.jl` | **Kernel** `csr_convolve_kernel!`; `csr_convolve!(kick, I_int, edge_grad, coef)` (direct causal sum, one item per bin); `csr_image_convolve!` (image sum with bin charge). |
| `kick.jl` | **Kernel** `csr_apply_kicks_kernel!`; `csr_apply_kicks!(pz, z, alive, bins, kick; pz_scale = 1)` (linear interpolation between bin centers; no z rescaling, as Bmad's CSR-only branch). |
| `step.jl` | `csr_kick_coef` (V2 unresolved for ions); `CSRStepResult{...}` (parametric: z_center, charge_per_meter, kick_per_meter, I_csr_per_meter, s_source); `csr_kick!(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0, pz_scale = 1.0)` (kernels on `get_backend(pz)`; `pz_scale` scales applied Δpz only, not diagnostics); `csr_ref_energy` (V6); `csr_n_step`. |
| `reference.jl` | Phase 1 serial reference (non-exported): `csr_bin_particles_reference`, `csr_convolve_reference!`, `csr_apply_kicks_reference!`, `csr_kick_reference!`. Ground truth for the kernels. |
| `interface.jl` | Stubs + docstrings for `slice_element`, `csr_sliceable`, `track_csr!` (implemented in the extension). |

### Extension `ext/CSRBeamTrackingBeamlinesExt/` (BeamTracking + Beamlines)

| File | Contents |
|---|---|
| `CSRBeamTrackingBeamlinesExt.jl` | Imports BeamTracking internals: `Fringe, AbstractSymplectic, R_to_beta_gamma, R_to_E, quat_rotate, quat_mul, rot_quaternion, inv_rot_quaternion, one_cos_norm, coord_alignment_bend_mid, coord_alignment_bend_at_s`; Beamlines `deval, isactive, SciBmadStandard, ApertureAt, RateMeaning`; `gyromagnetic_anomaly`; `StaticArrays.SA`. |
| `slicing.jl` | `slice_element(ele, s0, L_slice, first, last; context)` following Bmad `makeup_super_slave1`: fringe only at ends (`fringe_at`), SciBmadStandard → the Symplectic order BeamTracking picks (`sbs_order`), RungeKutta (n_steps scaled, ds_step kept, IBS flags), integrated multipoles scaled, apertures at ends, alignment per slice, patch on first slice, RF (voltage ∝ length, rate → frequency, one-element Beamline), four-potential and `EMFieldParams` shifted by s0, t0 = s0/(β₀c) (`ShiftedPotential`); SaganCavity → `slice_sagan` (sagan.jl); maps whole only. `csr_sliceable`. |
| `sagan.jl` | lcavity. `SaganSlice` = window [s0, s1] of the whole cavity's kernel sequence (outer drifts, entrance/exit fringe, cell kicks with reference-momentum steps, cell drifts); events at p belong to the slice with s0 ≤ p < s1 (≤ for the last). `sagan_slice_kernel!` (plain function, scalar branches only: BeamTracking calls kernels with SIMD lanes); `sagan_P0c_at`, `sagan_exit_P0c`, `sagan_dt_ref`; `track_slice!(bunch, ::LineElement, p_over_q_ref)` and `track_slice!(bunch, ::SaganSlice)` (builds `KernelChain`/`RefState`, ext `alignment`/`aperture`, `launch!`). Uses BeamTracking internals `sagan_cavity_outside_drift!`, `sagan_cavity_inside_drift!`, `sagan_cavity_kick!`, `sagan_cavity_fringe!`, `reference_momentum_shift!`, `dpc_given_dE`, `c_light`, `E_to_R`, `E_to_v`, and ext `rf_omega_calc`, `rf_phi0_calc`, `rf_step_calc`, `get_strengths`, `check_bl_bunch!`. |
| `track.jl` | `track_csr!(bunch, ::Beamline; ...)` and `track_csr!(bunch, ::Branch; ...)` → `track_csr_line!`: reference survey (patches), centroid orbit tracked without CSR, `CSRGeometry`, step loop with half kicks at ends. Slice boundaries `L_slice = i·ds − s0` (exact shared boundaries, Sterbenz). `csr_ref_energies`: γ ramps from `E_ref − dE_ref` to `E_ref` only for SaganCavity; `pz_scale = pc_lin(s)/P0c_step(s)` there. Keywords: `params`, `csr_on`, `centroid_start`, `diagnostics(ix_ele, i_step, s_kick, result)`. |

Other extensions (`CSRAbstractFFTsExt`, `CSRChainRulesCoreExt`, `CSRGTPSAExt`) are empty stubs.

### Tests (`test/`)

| File | Covers |
|---|---|
| `runtests.jl` | Loads weakdeps, checks the 4 extensions load, includes the files below. |
| `utils.jl` | `simpson`, `gaussian`, `dgaussian`, `steady_state_dpz_ds`, `circle_lattice`, `gaussian_line_bunch`. |
| `geometry_test.jl` | Survey conventions, modulo2, chord spline, dspline_len vs quadrature, bend geometry. |
| `binning_test.jl` | Overlap sums, charge conservation, grid formula, CSRParams validation. |
| `wake_test.jl` | zbrent, source point on a circle, first-bin integral, upstream infinite drift, V7. |
| `kick_test.jl` | Kick coefficient, csr_n_step, steady-state long bend vs analytic (< 2%), compat, half kick, dead particles, drift ≈ 0, errors. |
| `shielding_test.jl` | Params; drift image kick = uniform-motion field (behind/ahead, downstream elements, past the end) and compat; V8; downstream infinite drift exact retardation and warm-start walk-back; kernels == reference with images; image table vs blocking; steady-state bend free-space limit and shielding. Uses `kernel_test_bunch`/`maxrel` from kernels_test.jl. |
| `kernels_test.jl` | Phase 4 gate vs `reference.jl`: grid, wake table, source points exactly; charges/kicks ≤ 1e-12 (atomic order); 5 geometries × compat; wake table identical for any lag blocking; Fortran nint; Float32 and BigFloat particles. |
| `slicing_test.jl` | Phase 2 gate: sliced == unsliced (CSR off) for all element kinds incl. RungeKutta and EMFieldParams; controls; fringe/aperture/n_steps rules; maps. |
| `track_csr_test.jl` | Floor conventions, CSR off == track!, steady state via track_csr!, per-element selection, weights, patches/misalignment, errors, csr_ref_energy. |
| `lcavity_test.jl` | SaganCavity slices vs whole cavity (exact for 1 slice; ≤ 1e-17 traveling wave; ≤ 2e-7 standing wave; ≤ 1e-7 multipoles); reference momentum steps; Branch: CSR off == track!, 2 vs 8 cells agree < 5e-3 (rescale check), kick positions, Branch == Beamline; `pz_scale`. |

---

## 5. Physics and implementation notes worth knowing

- Conventions (BeamTracking = Bmad): `pz = ΔP/P0`, `z > 0` ahead, coords N×6 `XI..PZI = 1..6`, `STATE_ALIVE = 0x1`.
- γ for the kick uses `R_to_E / massof` (BeamTracking `R_to_gamma` omits the charge state).
- Deposit atomics make multithreaded results differ in the last bits between runs; compare CSR-on runs with a tolerance.
- KA CPU backend: `@index(Global)` must be a standalone assignment (`@index(Global) + 1` fails); default workgroup on CPU
  covers the whole ndrange, so small ndranges need an explicit workgroup size for threading (wake table uses 1).
- Atomix atomic add fails on ForwardDiff Duals (checked); phase 6 must keep the serial deposit path or write adjoints.
- lcavity in SciBmad: RF element with `tracking_method = SaganCavity`, in its own Beamline with `dE_ref` (a Branch
  splits it out). `dE_ref` is a design input, seen only by the first element of a Beamline and only used by SaganCavity.
  RFParams with `voltage = 0` count as inactive. A SaganCavity first in a Beamline built with `p_over_q_ref` cannot
  read `dE_ref` (Beamlines error), in BeamTracking and in CSR.jl alike.
- Splitting a SaganCavity cell drift is exact for traveling waves without multipoles; the standing-wave ponderomotive
  half kicks and multipole half kicks give integrator-order differences (∝ V² for the ponderomotive part).
- Timing (20001 particles, n_bin 200): kernels 1.1 ms (1 thread) / 0.5 ms (4 threads); reference 1.0 ms. Wake table dominates.

---

## 6. Findings

### 6a. Bmad (details in PLAN.md Section 7; file `bmad-ecosystem/bmad/space_charge/csr_and_space_charge_mod.f90`)

| ID | Where | Status in CSR.jl |
|---|---|---|
| V1 | `dspline_len` line 1796: `9*c(3)**2` should be `/5`. | Fixed; compat flag. Write-up: `dspline_coeff.md`. |
| V2 | Kick coefficient `rel_mass` factor for non-electrons. | Unresolved; Bmad formula kept; electrons only in tests. |
| V3 | `s_ref_to_s_chord` (1817–1842) projection. | Ported literally. Open. |
| V4 | `s_source_calc` round-off branches (951–959) return `ddz`, not a position. | Ported literally (D11). |
| V5 | `I_csr` first bin (1414–1426) holds `I_csr` constant. | Quadrature (option A); compat flag. Write-up: `upstream-bin.md`. |
| V6 | `track1_bunch_csr` 330–331: energy interpolation weights reversed. | Fixed in `csr_ref_energy`; compat flag. Now active in lcavities. |
| V7 | `s_source_calc` after `super_zbrent` (997): geometry from the last evaluation, not the root. | Re-evaluated unless compat. |
| V8 | `ddz_calc_csr` 1077: `floor_s%theta = theta_sl + theta_L` is the source direction + π for sources ahead (image kick). | Fixed; compat flag. Drift check: `planning/diagnostics/v8_drift.jl`. |

Constants: CODATA 2022 `R_ELECTRON = 2.8179403205e-15` vs Bmad's `2.8179403227e-15`; matters only for bit-level phase 3.

### 6b. BeamTracking (dev `7d79938`; not changed, PLAN.md Section 11)

- `universal!` for SaganCavity (`ext/BeamTrackingBeamlinesExt/unpack_bl.jl` ~352–460) sets `bunch.t_ref = dt_ref`,
  dropping the entrance `t_ref`. CSR.jl reproduces it so CSR-off results match.
- Its error messages interpolate an undefined `ele_name`; a stray `\` ends the FourPotentialParams check line.
- `bunch.p_over_q_ref` is not updated after a SaganCavity.
- `R_to_gamma` omits the charge state.

---

## 7. Planning files (`CSR.jl/planning/`, gitignored)

| File | Contents |
|---|---|
| `PLAN.md` | Full plan: decisions, source facts, layout, phases 0–8 with tests, limitations, risks; Section 7 verification and findings; Sections 8–11 results (phase 1, 2, 4, lcavity). |
| `PLAN_continued.md` | This handoff. |
| `csr_gpu_ad_design.md` | User's original design notes (physics, numerics, KA sketches, AD adjoints). Partly superseded by PLAN.md Section 2; its KA sketches were untested (the `@index(...) + 1` form fails on CPU). |
| `verify_analytic.jl` | 33 numerical checks of analytic relations (Base Julia only); all pass. |
| `dspline_coeff.md`, `upstream-bin.md` | Write-ups for V1 and V5. |
| `diagnostics/*.jl` | Scripts reproducing `upstream-bin.md` numbers (run from `CSR.jl/` with `--project=.`). |

---

## 8. Reference sources (local checkouts in the workspace root)

- **Bmad Fortran** `bmad-ecosystem/`: `bmad/space_charge/csr_and_space_charge_mod.f90` (`track1_bunch_csr` 137–404,
  `csr_bin_particles` 434–678, `csr_bin_kicks` 701–835 incl. image charges 813–826, `s_source_calc` 857–1003,
  `ddz_calc_csr` 1023–1086, `I_csr` 1368–1432, `image_charge_kick_calc` 1451–1494, `csr_and_sc_apply_kicks` 1513–1652,
  `dspline_len` 1770–1799, `s_ref_to_s_chord` 1817–1842); slicing `bmad/code/create_element_slice.f90`,
  `bmad/modules/bookkeeper_mod.f90` (`makeup_super_slave1` 1172–1509; lcavity voltage ∝ length 1287–1290).
  Reference cases: `regression_tests/csr_and_space_charge_test/`, `bmad-doc/tao_examples/csr_beam_tracking/`.
- **BeamTracking.jl**: `src/types.jl`, `src/kernel.jl` (`KernelChain`, `RefState`, `launch!`; SIMD lanes on CPU),
  `src/tracking_methods.jl` (`Symplectic`…, `Exact`, `SaganCavity`, `RungeKutta`), `src/kernels/sagan_cavity.jl`,
  `src/kernels/runge_kutta.jl`, `src/fields.jl`, `ext/BeamTrackingBeamlinesExt/` (`unpack_bl.jl`, `general_bl.jl`
  alignment/aperture, `utils_bl.jl` RF helpers, `sagan_cavity_bl.jl`, `rungekutta.jl`, `scibmadstandard_bl.jl`).
- **Beamlines.jl**: `src/beamline.jl` (reference energy, `dE_ref`, `BeamlineParams`), `src/branch.jl` (`Branch`,
  `Lattice`), `src/misc.jl` (`EMFieldParams`), `src/element.jl`.
- **Others**: AtomicAndPhysicalConstants.jl, SpaceCharge.jl (3D solver, not a dependency), SciBmad.jl.

---

## 9. Next steps

Per PLAN.md Section 4 (phase 3 stays blocked on D3):

1. ~~Phase 5~~ done (PLAN.md §13). Was: **Phase 5, shielding (image charges).** Port `image_charge_kick_calc` (1451–1494) and the image part of
   `csr_bin_kicks` (813–826): `-n_bin:n_bin` charge convolution with `(-1)^ns`, factor 2, half weight at ends; params
   `n_shield_images`, `beam_chamber_height` (Bmad `space_charge_com`). Revisit V4 (y_source ≠ 0 exercises it). Extend
   `csr_wake_table` and `kernels_test.jl` (kernel == reference). Tests: free-space limit as height → ∞; strong shielding
   reduces the wake; compare with `csr_and_space_charge_test` parameters qualitatively (no Bmad numbers).
2. Phase 6: ForwardDiff through the particle-side kernels (serial deposit for Duals), `rrule` for `csr_kick!` with
   explicit adjoints (design notes §3.8), implicit differentiation of the source root.
3. Phase 7: TPS (serial deposit, smooth shape function).
4. Phase 8: FFT convolution ext, benchmarks, GPU wake table (isbits geometry via Adapt, immutable work struct),
   upstream collective-step hook proposal.

Follow-ups not requested: per-element `ds_step` (Bmad `csr_ds_step`); caching slices per element (rebuilt every step).
