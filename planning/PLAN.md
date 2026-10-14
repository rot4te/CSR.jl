# CSR.jl plan and handoff

One document for planning and session handoff (merged from `PLAN.md` and `PLAN_continued.md` on 2026-10-14).

**How to use it.** A new session reads Parts 1–3 (status, how to work, decisions), then whatever later part the task
needs. Part 9 is the dated results log; older entries there are history, and where they conflict with Parts 1–8, the
earlier parts win.

Contents
1. Status and next steps
2. How to work on this project (preferences, git, paths, test environment)
3. Decisions (D1–D24)
4. Open questions and pending upstream reports
5. Code map
6. Physics and implementation notes
7. Findings in Bmad and BeamTracking
8. Validation: analytic checks and unverified items
9. Results log, by phase
10. Roadmap, limitations, risks
11. Reference sources and planning files

---

## 1. Status and next steps

**Goal.** A 1D (line-charge) coherent synchrotron radiation (CSR) module for the SciBmad Julia ecosystem, ported from
Bmad's Fortran `csr_method = one_dim`; GPU-capable (KernelAbstractions) and differentiable (ForwardDiff, ChainRules,
GTPSA TPS).

**Branches (2026-10-14, none pushed by Claude):**
- `stage-c` (from `phase6-open-questions` `468fc83`): the physics work of 2026-10-14 (commits below).
- **`reorg`** (from `stage-c` `f3c69bf`), head `5ee107c` + this handoff edit: source layout as BeamTracking.jl
  (`src/kernels/`, `src/utils/`), a docstring for every function, README.md, Documenter docs (`docs/`, GitHub
  Actions workflow). 1035/1035 tests on 1 and 4 threads; docs build clean (`checkdocs = :all`).
- **`phase6-geom_speed-deltas`** (from `reorg` `08da04e`): Phase 6 step 3 (forward mode and rrules w.r.t. geometry,
  `s_kick`, γ), ForwardDiff hard dependency, immutable `CSRKick1`. 1355/1355 on 1 and 4 threads. **Latest state.**

| Commit | Branch | Content |
|---|---|---|
| `59e86fa` | stage-c | Step 0: compat `Beamlines = "0.10.6, 0.11"`, `BeamTracking = "0.8.7"`; `~` paths in planning |
| `1b92650` | stage-c | Particle-side kernels run on Metal (Float32 grid, Int32 counters, typed literals) |
| `6d1b8ea` | stage-c | D24 diagnostics |
| `9cbc8bf` | stage-c | D24 averaged image-layer sum (`image_sum_order = 8`); GPU float-type fallback; large-y diagnosis |
| `ffb4004` | stage-c | Exact path-length excess in default mode (`path_excess`, V9); D24 layer-term data tracked |
| `a908ef9` | stage-c | Merged PLAN.md |
| `bf3d3c9` | stage-c | D22: exact SaganCavity split |
| `f3c69bf` | stage-c | Cross-check rerun (Section 9, Phase 3); V9 upstream report draft |
| `5ee107c` | reorg | Reorganized src, docstrings, README, docs |

**Done:** phases 0–5, phase 3 (Bmad cross-check, rerun 2026-10-14), phase 6 steps 0–2 (AD w.r.t. particles),
lcavity mode, Stage B (D20, D21, D23), D24, GPU number types, exact path excess (V9), D22, source reorganization,
README, Documenter docs.

**Phase 6 step 3 done** (3a forward mode, D28/D29, 3b rrules) on branch **`phase6-geom_speed-deltas`** (from
`reorg`); see Section 9. **Next (user to choose):** Phase 8 GPU table work (Adapt-based `CSRGeometry` with the element
vector as a type parameter; errors in the source search as return codes; iterative adaptive Simpson), Phase 7 (TPS),
or a cheaper geometry pullback (per-lag sparsity, or hand-written adjoints of the tables).

Original notes for step 3 (from the reorg handoff, 2026-10-14): Goal: derivatives of a kick w.r.t. the geometry and the reference speed/energy (γ), and `s_kick`.
Starting points (suggestions, not decided):
- Today the `rrule` (`ext/CSRChainRulesCoreExt/CSRChainRulesCoreExt.jl`) returns `@not_implemented` for `geom`,
  `s_kick`, `gamma`; ForwardDiff passes only through the particle side (tables are `Float64`, built on the CPU in
  `src/tables.jl` with the `CSRKick1` work struct).
- The tables depend on the parameters through the retarded source point: implicit differentiation of the root of
  `ddz(s; p) = L/(2γ²) + ΔL − dz = 0` (`kernels/source_point.jl`), `ds/dp = −∂ddz/∂p / ∂ddz/∂s`, then the chain rule
  through `I_csr_value` / `I_csr_calc` / quadratures (`kernels/wake.jl`) and `image_charge_kick` + hat integration
  (`kernels/image_charges.jl`). Also `csr_kick_coef` (∝ 1/γ, `utils/kick_setup.jl`).
- Decide with the user first: which parameters count as "geometry" (centroid orbit rows? element L, g? `s_kick`),
  forward (Duals through the table kernels: needs the work struct and tables generic in the number type) vs reverse
  (adjoint of the table w.r.t. parameters), and test strategy (D18: by hand vs ForwardDiff and finite differences).
- Docs: update `docs/src/usage.md` ("Derivatives") and Section 3 / "Physics decisions" if any physics changes.

Later: Phase 7 (TPS), Phase 8 (FFT convolution, benchmarks, GPU wake table, upstream hook proposal).

**For the user:** file the upstream reports in `planning/upstream_reports.md` (Section 4, incl. V9); verify the
action versions in `.github/workflows/Documentation.yml` (from memory) and enable GitHub Pages (`gh-pages`) after
pushing.

---

## 2. How to work on this project

### User preferences (also in the user's global CLAUDE.md)
- Accuracy over helpfulness; flag uncertainty ("verify this", "from memory"); never invent APIs or sources.
- Fewest words; lead with the answer; paragraphs ≤ 5 sentences.
- Code changes: report (1) what was wrong, (2) what changed, (3) tested/verified?, (4) what the user must do next.
- Git: never commit to or push existing branches; create a new branch and commit there; never push.
- Show a detailed plan and get approval before launching any subagent.
- Physics-model and upstream choices are the user's: ask with options and a recommendation.
- Style: 2-space indent, `@kwdef`, docstrings with `## Properties` (BeamTracking style), sparse comments.
- `planning/` is tracked in git (since `468fc83`).

### Paths (two machines: `$HOME` is `/Users/alexcoxe` or `/Users/coxea3`; always write `~`)
- Package `~/.julia/dev/CSR.jl`; dev checkouts beside it: `BeamTracking.jl` (v0.8.7, `65e343f`), `Beamlines.jl`
  (v0.11.0, `35fe016`), `AtomicAndPhysicalConstants.jl`, `GTPSA.jl`, `SciBmad.jl`, `SpaceCharge.jl`.
- Bmad source `~/bmad-ecosystem` (`7c9df4059`, REL 20261014-0 on one machine); conda env `~/miniforge3/envs/bmad`.
- Julia 1.12.7 via juliaup. `Project.toml`: `julia = "1.10"`.

### Git
- Remote `origin` (`github.com/rot4te/CSR.jl`). Branches: `main` (`ae0ae77`), `phase6-open-questions` (`468fc83`, pushed
  by the user), `stage-c`, `reorg` (unpushed). Next: `phase6-geom_speed-deltas` from `reorg`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Test environment
A scratch environment (wiped between sessions) that `Pkg.develop`s CSR.jl, BeamTracking.jl, Beamlines.jl (from `~`)
and adds Test, StaticArrays, AbstractFFTs, ChainRulesCore, GTPSA, AtomicAndPhysicalConstants, KernelAbstractions,
Atomix, ForwardDiff (+ DelimitedFiles for diagnostics). Run `Pkg.resolve()` after changing CSR's weakdeps, or new
extensions silently do not load.
- Full suite, from `test/`: `julia --project=<env> -t 4 -e 'include("runtests.jl")'` (~1.8 min; also run `-t 1`).
- One file: `julia --project=<env> -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines,
  ChainRulesCore, GTPSA; include("utils.jl"); include("<file>_test.jl")'`.
- BeamTracking prints a harmless "Straight multipoles … curved reference system" warning.
- Metal checks (not in the suite): a separate env with Metal.jl + CSR (dev) + AtomicAndPhysicalConstants, StaticArrays.

---

## 3. Decisions

| # | Date | Decision |
|---|---|---|
| D1 | 10-01 | Integration route (a): CSR.jl slices each element and tracks each slice with BeamTracking, applying the CSR kick between slices. No upstream BeamTracking changes (reaffirmed for the lcavity, D10). |
| D2 | 10-01 | Macroparticle charge from `bunch.coords.weight` (real particles per macroparticle): `q_p = w_p * chargeof(species; C=true)`; `weight === nothing` means 1. |
| D3 | 10-06 | Bmad reference: conda-forge bmad (Tao only, no Fortran compilation). |
| D4 | 10-01 | KernelAbstractions CPU backend only "for now"; relaxed 2026-10-14: particle-side kernels and `csr_kick!` run on GPU backends with the tables computed on the CPU (Section 9, Stage C). |
| D5 | 10-01 | Default behavior fixes proven Bmad errors; `bmad_compat = true` reproduces Bmad. Ambiguous Bmad code (V3, V4) is ported literally and logged. |
| D6 | 10-01 | First-bin quadrature option A (superseded by D15). |
| D7 | 10-01 | "Same as Bmad" means CSR-level equivalence (slicing rules, geometry, γ, kicks) on top of BeamTracking's own tracking, not particle-level. |
| D8 | 10-01 | RF γ: Bmad's linear reference-energy ramp; Δpz rescaled to BeamTracking's reference (V6 part superseded by D21). |
| D9 | 10-01 | Per-element CSR opt-in (`csr_on`), default all elements. |
| D10 | 10-02 | lcavity mode needed; implemented CSR-side from BeamTracking's internal Sagan kernels (not an upstream `fringe_at`, not Symplectic RF + reference shifts). |
| D11 | 10-06 | V4 (s_source_calc round-off branches) kept as Bmad in default mode too (bends are typically symmetric). |
| D12 | 10-06 | Image charges: sources ahead of the kick point beyond the kick element are searched downstream, and past the lattice end on an infinite drift (generalizes to long bunches). Not in Bmad; compat restores Bmad. |
| D13 | 10-06 | V8 (source direction reversed for sources ahead) fixed by default, compat reproduces. |
| D14 | 10-06 | `track_csr!(...; sliced_centroid = true)`: centroid tracked in the bunch's slices. |
| D15 | 10-06 | `CSRParams(all_bin_quadrature = true)`: every bin by quadrature (D6 option B) "for now". |
| D16 | 10-08 | Phase 6 = steps 0–2 first (done); step 3 (geometry/γ derivatives) later. |
| D17 | 10-08 | Non-mutating function `csr_kick_pz`. |
| D18 | 10-08 | No ChainRulesTestUtils; pullbacks tested by hand vs ForwardDiff `J'ȳ` and finite differences. |
| D19 | 10-08 | All open questions handled before phase 6 step 3. |
| D20 | 10-09 | Image kernel integrated against the bin hat function (adaptive quadrature) in default mode; compat keeps point sampling. |
| D21 | 10-09 | V6 fixed in Bmad (PR #2199, 2026-10-06): `bmad_compat` follows current Bmad; V6 compat branch removed. |
| D22 | 10-09 | SaganCavity standing-wave ~1e-7 sliced-vs-unsliced difference: investigate an exact split (not accepted as is). Done 2026-10-14: half kicks as point events at drift-segment ends; slices == whole cavity to round-off. |
| D23 | 10-09 | Compat entries: Beamlines when registered with `EMFieldParams`, BeamTracking when a release has `RungeKutta`. Done 2026-10-14: `Beamlines = "0.10.6, 0.11"`, `BeamTracking = "0.8.7"`. |
| D24 | 10-10 | Image-layer sum by a transformed sum, default mode only. Done 2026-10-14: `image_sum_order = 8` (binomial averaging of the last partial sums). |
| — | 10-14 | GPU: fall back to `Float32` where `KernelAbstractions.supports_float64(backend)` is false. |
| — | 10-14 | Exact path-length excess (`path_excess`) in default mode, accepted because it costs < 10% (measured +1%). |
| D25 | 10-14 | Phase 6 step 3 parameters: all geometry builder inputs (centroid rows, element L and g), `s_kick`, γ. |
| D26 | 10-14 | Phase 6 step 3 route: forward first (number-type-generic tables, primal root + implicit tangent), then a hybrid `rrule` (convolution adjoint × table Jacobian by chunked Duals) plus an `rrule` for `CSRGeometry(L, g, centroid)`. |
| D27 | 10-14 | Tests may use ChainRulesTestUtils (reverses D18 for new rrule tests; existing hand tests stay). |
| D28 | 10-14 | ForwardDiff is a hard dependency (already one of BeamTracking; CSR.jl is rarely used alone). `CSRForwardDiffExt` removed; `primal(::Dual)`, `root_tangent`, `RootTag` in core. Compat `ForwardDiff = "1.3.3"` (BeamTracking's; only 1.4.6 tested). |
| D29 | 10-14 | `CSRKick1` immutable (user: prefer immutable types); the source-point functions return updated copies. |

---

## 4. Open questions and pending upstream reports

**Open (user decisions or unresolved):**
- V2: kick prefactor for non-electrons (`rel_mass` placement). Electrons/positrons only in tests until resolved.
- V3: `s_ref_to_s_chord` projection (ported literally).
- Seam/V4: at y = 3 mm, lag −3 mm (retarded image source at the kick point) the image kernel is −1.3e-4 vs exact
  −1.11e-3 (~1e-7 of the peak). Lag 0 at small y: 0.94 of exact (Bmad's approximate retardation where the kernel
  crosses zero). Both negligible so far; not investigated further.
- At h = 1 cm the hat-integrated shielded result depends on n_bin below ~240 bins (likely charge-density resolution
  on a small residual; not verified).
- Float32 on GPUs limits strongly shielded kicks to ~3e-4 relative (cancellation); a mixed-precision table/convolve
  path would be needed for better.

**Upstream reports — drafted in `planning/upstream_reports.md`, not filed (the user files):** Bmad V1, V5, V7, V8;
BeamTracking Section 7b items and the `R_to_gamma`/`R_to_v` ion bug. A GitHub search (via WebFetch, summarized:
moderately reliable) found no existing issue. Open Bmad PR #2085 rewrites `s_source_calc`; V7/V8 line numbers may
move. V9 (small-angle path excess) drafted as Bmad item 5 (2026-10-14).

---

## 5. Code map

### Core (`src/`), no BeamTracking dependency — layout as BeamTracking.jl (since branch `reorg`, 2026-10-14)

Top level: types and host-side functions (launch kernels, allocate, check). `kernels/`: KernelAbstractions
`@kernel`s, `@inline` device functions, and the per-work-item physics the kernels call. `utils/`: helpers.

| File | Contents |
|---|---|
| `CSR.jl` | Module, exports (`CSRParams, FloorPosition, CSRGeometry, CSRElementInfo, CSRBins, CSRKickPoint, CSRStepResult, planar_survey, csr_bin_particles, csr_kick!, csr_kick_pz, csr_kick_coef, csr_n_step, dspline_len, track_csr!`), includes (types first: method signatures annotate them). |
| `types.jl` | `FloorPosition`, `CSRElementInfo`, `CSRGeometry`, `CSRKickPoint`, `CSRKick1`, `CSRBins`, `CSRStepResult`, `CSRParams` (with validation; docstring lists every compat difference). |
| `geometry.jl` | `CSRGeometry(L, g, centroid; ref_start, bmad_compat)`, `CSRGeometry(L, g, ref_exit, centroid)`, `CSRKickPoint(geom, ix, s_kick, γ; y_source, kick_factor)`. |
| `binning.jl` | `csr_bin_particles` (grid in the particles' float type; Int32 counter; serial path for non-atomic types). |
| `tables.jl` | `csr_wake_table`, `csr_image_table` (CPU only). |
| `kick.jl` | `csr_convolve!`, `csr_image_convolve!`, `csr_apply_kicks!`, `csr_kick!`, `csr_kick_pz`, `csr_binned_kick` (tables on `table_backend`, `to_backend`, layer weights), `csr_step_result`. |
| `adjoint.jl` | `csr_kick_adjoint`. |
| `reference.jl` | Serial reference (`csr_*_reference`), ground truth for the kernels. |
| `interface.jl` | Stubs + docstrings for `slice_element`, `csr_sliceable`, `track_csr!`. |
| `kernels/binning.jl` | `triangle_overlap`, `deposit_ix0`, `deposit_particle!`, `csr_deposit_kernel!`, `csr_edge_gradient_kernel!`. |
| `kernels/source_point.jl` | `ddz_calc_csr`, `s_source_calc` (return `(k, value)`; V4 literal, V7, V8, downstream search, infinite drifts with walk-back), `csr_kick1_like`. |
| `kernels/wake.jl` | `I_csr_value`, `I_csr_calc` (Bmad `I_csr`), `I_csr_integral`, `I_csr_quadrature`, `csr_wake_kernel!`, `csr_trapezoid_kernel!`, `csr_bin_quadrature_kernel!`. |
| `kernels/image_charges.jl` | `image_charge_kick`, `csr_image_kernel!`, hat integration (`image_kernel_at`, `image_simpson`, `image_segment`, `image_interval_moments`, each returning `(k, value)`, `csr_image_hat_kernel!`, `csr_image_hat_combine_kernel!`). |
| `kernels/convolve.jl`, `kernels/kick.jl` | `csr_convolve_kernel!`, `csr_image_convolve_kernel!`; `csr_apply_kicks_kernel!`. |
| `kernels/adjoint.jl` | `triangle_overlap_dzp`, `kick_cell`, `kick_adjoint_particle!`, five adjoint kernels. |
| `utils/math.jl` | `nint`, `modulo2`, `cross3`, `dot3`. |
| `utils/backend.jl` | `csr_float_type`, `device_scalar`, `table_backend`, `to_backend`, `AtomicAdd`, `PlainAdd`, `supports_atomic_add`, `primal`. |
| `utils/spline.jl` | `spline_val/d1/d2`, `chord_spline`, `dspline_len` (Bmad, V1), `GL4_NODES/WEIGHTS`, `path_excess` (exact, V9). |
| `utils/floor.jl` | `x_axis`, `z_axis`, `advance_floor`, `advance_floor_patch`, `planar_survey`, `centroid_floor`. |
| `utils/chord_coordinates.jl` | `s_ref_to_s_chord` (V3 literal), `source_s`. |
| `utils/root_finding.jl` | `zbrent` (`super_zbrent`). |
| `utils/image_layers.jl` | `image_layer_weight` (D24), `image_u_min`, `image_hat_tol` (D20). |
| `utils/kick_setup.jl` | `csr_kick_coef` (V2), `csr_ref_energy`, `csr_n_step`. |

Every function has a docstring (the docs build enforces it with `checkdocs = :all`).

### Extensions (`ext/`)

| Extension | Contents |
|---|---|
| `CSRBeamTrackingBeamlinesExt` | `slicing.jl`: `slice_element` following Bmad `makeup_super_slave1` (fringe at ends, SciBmadStandard → Symplectic order, RungeKutta n_steps scaled, multipoles scaled, apertures at ends, alignment per slice, patch on first slice, RF voltage ∝ length, four-potential/`EMFieldParams` shifted, SaganCavity → `slice_sagan`; maps whole); `csr_sliceable`. `sagan.jl`: `SaganSlice` windows of the cavity's kernel sequence, `track_slice!`. `track.jl`: `track_csr!(bunch, ::Beamline / ::Branch)` → `track_csr_line!` (survey, centroid, `CSRGeometry`, step loop with half kicks at ends, `L_slice = i·ds − s0`, `csr_ref_energies`, `pz_scale` in SaganCavity; keywords `params`, `csr_on`, `centroid_start`, `sliced_centroid`, `diagnostics`). Uses BeamTracking internals listed in the files' imports. |
| `CSRChainRulesCoreExt` | `rrule` for `csr_kick_pz`. |
| `CSRAbstractFFTsExt`, `CSRGTPSAExt` | Empty stubs. |

### Tests (`test/`)

| File | Covers |
|---|---|
| `runtests.jl` | Loads weakdeps, checks the extensions load, includes the files below. |
| `utils.jl` | `simpson`, `gaussian`, `dgaussian`, `steady_state_dpz_ds`, `circle_lattice`, `gaussian_line_bunch`. |
| `geometry_test.jl` | Survey conventions, modulo2, chord spline, `dspline_len` and `path_excess` vs quadrature, bend geometry. |
| `binning_test.jl` | Overlap sums, charge conservation, grid formula, Float32 grid, backend number types (mock no-Float64 backend), CSRParams validation. |
| `wake_test.jl` | zbrent, source point on a circle, exact vs small-angle path excess on a circle, first-bin integral, upstream infinite drift, V7. |
| `kick_test.jl` | Kick coefficient, csr_n_step, steady-state long bend vs analytic (< 2%), compat, half kick, dead particles, drift ≈ 0, errors. |
| `kernels_test.jl` | Kernels vs `reference.jl`: grid, wake table, source points exactly; charges/kicks ≤ 1e-12; 5 geometries × compat; blocking independence; Fortran nint; Float32 and BigFloat particles. |
| `shielding_test.jl` | Params; drift image kick = uniform-motion field (behind/ahead, downstream, past the end) and compat; V8; infinite drifts; hat table vs analytic; kernels == reference with images; blocking; steady-state bend free-space limit and shielding; averaged layer sum (weights, convergence). |
| `ad_test.jl` | rrule vs ForwardDiff vs finite differences (6 cases). |
| `slicing_test.jl` | Phase 2 gate: sliced == unsliced (CSR off) for all element kinds; controls. |
| `track_csr_test.jl` | Floor conventions, CSR off == track!, steady state via track_csr!, per-element selection, weights, patches/misalignment, errors, csr_ref_energy. |
| `lcavity_test.jl` | SaganCavity slices vs whole cavity; reference momentum steps; Branch; `pz_scale`. |

---

## 6. Physics and implementation notes

- Conventions (BeamTracking = Bmad): `pz = ΔP/P0`, `z > 0` ahead, coords N×6 `XI..PZI = 1..6`, `STATE_ALIVE = 0x1`.
- γ for the kick uses `R_to_E / massof` (BeamTracking's `R_to_gamma` is wrong for |Z| ≠ 1).
- Deposit atomics make multithreaded results differ in the last bits between runs; compare CSR-on runs with a tolerance.
- KA CPU backend: `@index(Global)` must be a standalone assignment (`@index(Global) + 1` fails); default workgroup on
  CPU covers the whole ndrange, so small ndranges need an explicit workgroup size (wake table uses 1).
- Atomix atomic add fails on ForwardDiff Duals; Duals use the serial deposit path.
- ForwardDiff ≥ 1.0 (checked 1.4.6): `==` compares partials, and `<`, `>`, `<=`, `>=`, `max`, `min` order Duals with
  equal values by their partials (`Dual(0, 1) > 0` is true). Every branch on a possibly-Dual value must compare
  `primal` values (else the branch depends on the seeded directions, and at ties differs from the Float64 run);
  `primal_max`/`primal_min` for value-producing max/min. `mod(Dual(0, 0), π)` has NaN partials: `modulo2` has a Dual
  method.
- GPU (Metal): no Float64 and no 64-bit atomics. Float32 ranges carry Float64 internals (build grids elementwise).
  Float literals in kernels must take the argument's type. The table kernels need the CPU (`CSRKick1` immutable since D29, but
  non-isbits geometry).
- lcavity in SciBmad: RF element with `tracking_method = SaganCavity` in its own Beamline with `dE_ref` (a Branch
  splits it out). `dE_ref` is seen only by the first element of a Beamline, only used by SaganCavity. RFParams with
  `voltage = 0` count as inactive.
- SaganCavity slices (D22): only drifts/solenoids are split; the half ponderomotive and multipole kicks of each drift
  segment are point events at the segment ends (window rule s0 ≤ p < s1, ≤ for the last slice), so slices reproduce
  the whole cavity to round-off. Segment ends use the same expressions as the neighbouring events (`Lo + i·lc`).
- Image-layer series (D24): terms alternate, amplitude smooth with one sign change, tail ∝ 1/n (exact LW field at lag
  0 also ∝ 1/y). Plain truncation error ~|a_M|/2. Image source arc distance Ls ≈ (12ρ²y²)^{1/4} (γ → ∞ estimate);
  test lattices must be long enough that high layers' sources stay on the arc (use `circle_lattice(10.0, 0.1, 400)`).
- Shielding parameter Π = σ_z ρ^{1/2}/h^{3/2} (from memory, verify): 19 at 3 mm, 3.2 at 1 cm (strong shielding).

---

## 7. Findings in Bmad and BeamTracking

### 7a. Bmad (`bmad/space_charge/csr_and_space_charge_mod.f90`)

| ID | Where | Finding | Status in CSR.jl |
|---|---|---|---|
| V1 | `dspline_len` line 1796 | s⁴ term `9*c(3)**2` should be `/5`. Bmad's dL 1.71× quadrature for an asymmetric spline (end slopes 0.010, −0.012); exact when c₃ = 0. Write-up: `dspline_coeff.md`. | Superseded in default mode by `path_excess`; compat keeps Bmad. |
| V2 | Kick coefficient | `rel_mass` factor for non-electrons: derivation gives `q_w r_e Q K ds/(e γ rel_mass)`, Bmad `|q| r_e/(e γ rel_mass²)`. | Unresolved; Bmad formula; electrons only in tests. |
| V3 | `s_ref_to_s_chord` 1817–1842 | Projects `dr` onto the direction of motion (x ≈ 0 always); on-axis bend gives `ρ·atan(s/ρ)` ≠ `L_chord` at s = L (0.09967 vs 0.09996 for L/ρ = 0.1). | Ported literally. Open. |
| V4 | `s_source_calc` 951–959 | Round-off branches assign `ddz_calc_csr(...)` (a residual) to `s_source`. Matters for the image kick's `g_bend`. | Ported literally (D11). |
| V5 | `I_csr` first bin 1414–1426 | Holds `I_csr` constant over the rest of the bin: 2/3 of the integral in steady state, 3–4× too small in the entrance transient. Write-up: `upstream-bin.md`. | Quadrature (D15); compat flag. |
| V6 | `track1_bunch_csr` 330–331 | Energy interpolation weights reversed. | Fixed upstream (PR #2199); both modes follow it (D21). |
| V7 | `s_source_calc` 997 | After `super_zbrent`, source geometry from the last evaluation, not the root. Catastrophic only when `ddz(0) == 0` exactly with CSR.jl's first-bin quadrature (I_csr 9.6e6 instead of −40). | Re-evaluated unless compat; regression test. |
| V8 | `ddz_calc_csr` 1077 | `floor_s%theta` = source direction + π for sources ahead (image kick only). Drift check: ahead sources 1.21, 8.99 (Bmad) vs −0.826, −0.111 analytic; true direction matches to 5 digits. Line from `ff57a4dda` (2016). | Fixed (D13); compat flag. Script `diagnostics/v8_drift.jl`. |
| V9 | `dspline_len` (all callers) | Small-angle path excess `½∫(x′ − dθ)²` drops the dθ⁴ terms: on a circle too large by Ls⁵/(1920ρ⁴) (1.8e-5, 1.9e-4, 1.9e-3 m at Ls = 3.2, 5.2, 8.2 m; ρ = 10 m). Acts as a lag shift; the image kernel near lag 0 at large y was ~14% low. | Exact `path_excess` in default mode; compat keeps Bmad. Drafted in `upstream_reports.md` (item 5). |

Other Bmad notes:
- At lag 0 the image kick on a drift is 0.125 vs analytic 0 (3e-5 of the peak): Bmad's approximate retardation
  `z ≈ L/(2γ²) + dL` (error `L/(8γ⁴)`), not V8.
- Seam: the same-element ahead formula uses `2|ds|` (chord) where the cross-element one uses `2 L_plane`; they differ
  by O(ds·θ²). A root in that gap is caught by the V4 round-off branch.
- Constants: CODATA 2022 `r_e = 2.8179403205e-15 m` vs Bmad `2.8179403227e-15 m` (8e-10 relative).
- Bmad diagnostic file quirks: with images, kick and I_csr are divided by the last layer's `kick_factor` and
  `S_Source` is the last layer's; S-position has 6 decimals.

### 7b. BeamTracking (checked at `7d79938`; ion bug still in 0.8.7)
- `universal!` for SaganCavity (`ext/BeamTrackingBeamlinesExt/unpack_bl.jl`) sets `bunch.t_ref = dt_ref`, dropping the
  entrance `t_ref`. CSR.jl reproduces it so CSR-off results match.
- Error messages interpolate an undefined `ele_name`; a stray `\` ends the FourPotentialParams check line.
- RFParams with `voltage = 0` count as inactive, so a zero-voltage SaganCavity errors ("without RFParams").
- `bunch.p_over_q_ref` is not updated after a SaganCavity.
- `R_to_gamma` (`src/utils/energy.jl:6`) and `R_to_v` (`:9`) are wrong for |Z| ≠ 1 (He²⁺, R = 1: γ 1.00323 vs
  1.01286; v/c 0.16034 vs 0.15882; verified numerically). `R_to_v` feeds `rf_omega_calc`, which CSR.jl's lcavity code
  calls.
- Combined-function SBend with Bn1: bend–kick integrator with straight multipoles (warns), default `n_steps = 1`.
- Slicing changes BeamTracking's integration: default Symplectic takes `n_steps = 1` per element and each slice keeps 1
  step, so sliced and unsliced tracking differ (regression bend px 1.7e-6). Motivated D14.

---

## 8. Validation: analytic checks and unverified items

Script `verify_analytic.jl` (Base Julia only): 33/33 checks pass (2026-10-01).

| Check | Relation | Result |
|---|---|---|
| C1 | Retardation `z = Ls − βL`; Bmad's `z ≈ L/(2γ²) + dL` has error `L/(8γ⁴)` | < 1% |
| C2 | Circle: `L = ρφ(1 − φ²/24)`, `dL = ρφ³/24 − ρφ⁵/1920` | < 1e-7 |
| C3 | Overtaking length `Ls = (24ρ²z)^{1/3}` (γ → ∞) | O(φ²) |
| C4 | Bmad `I_csr` on a circle → `−2·3^{−1/3} ρ^{−2/3} z^{−1/3}` for γφ ≫ 1 | 4e-6 |
| C5 | Bmad first-bin `I_int` (γ → ∞) `= −3^{2/3} ρ^{−2/3} z^{2/3}` | O(φ²) |
| C6 | First-bin `I_int` with the `ln(2γ²z/Ls)/γ²` term vs quadrature, γ = 30, 100 | ≤ 4e-4 |
| C7 | Bmad scheme converges to the design-doc §2.4 integral | 6e-3 → 6e-4, n_bin 100 → 1600 |
| C8 | `dspline_len` closed form vs quadrature of `(x′ − dθ)²/2` | exact with V1 |
| C9 | Upstream infinite-drift quadratic satisfies `Ls − βL = dz` | 1e-12 |
| C10 | Triangle overlap weights sum to 1 | 1e-15 |
| C11 | Reverse-mode adjoints (q̄, z̄, K̄) vs finite differences | 1e-10 |

Algebraic (by hand): C4 + Bmad prefactor ⇒ the steady-state formula with `2/3^{1/3}` and sign for electrons
(consistent with Bmad, not independent of it). `image_charge_kick_calc` has the Liénard–Wiechert structure
`coulomb1 = B·(N−β′)/(γ²L²κ³)`, `radiate = B·(N×((N−β′)×G))/(Lκ³)`, with `e/(4πε₀)` factored out — confirmed
numerically on 2026-10-14 against an independent exact-retardation LW field (`diagnostics/image_kernel_exact.jl`,
`image_large_y.jl`): CSR.jl's kernel equals the exact field to ≤ 3e-5 at y = 3 cm–1.9 m after V9.

### Not verified, and where to check (citations from memory unless marked; check before relying on them)

| Item | Where to verify |
|---|---|
| `I_csr` formula from first principles | D. Sagan & C. Mayes, "Coherent Synchrotron Radiation Simulations for Off-Axis Beams Using the Bmad Toolkit", IPAC2017, THPAB076 (URL in the Bmad source header: https://accelconf.web.cern.ch/ipac2017/papers/thpab076.pdf) |
| First-bin `I_int` log term analytically | Same paper; Bmad manual CSR section |
| Steady-state coefficient `2/3^{1/3}` independently | Derbenev, Rossbach, Saldin, Shiltsev, TESLA-FEL 95-05 (1995); Saldin, Schneidmiller, Yurkov, NIM A 398, 373 (1997); Murphy, Krinsky, Gluckstern, Part. Accel. 57, 9 (1997) |
| CSR impedance from retarded potentials | Stupakov & Zhou, PRAB 19, 044402 (2016) |
| Validity `σ_x ≪ (ρσ_z²)^{1/3}` | Derbenev et al. |
| Liénard–Wiechert field | Jackson, *Classical Electrodynamics*, 3rd ed., Ch. 14 |
| Shielding image geometry (`(−1)^n`, factor 2) and a shielding theory to compare against | Bmad manual CSR/space-charge chapter |
| Kick prefactor for ions (V2) | Sagan & Mayes; Bmad manual definition of macroparticle `charge`; maintainers |
| Enzyme + KernelAbstractions | Current docs |
| CSR vs LSC overlap | Bmad manual |

---

## 9. Results log, by phase

### Phase 0 — skeleton (10-01)
`Project.toml` with deps, weakdeps, 4 extension stubs, test target; `Pkg.test()` passes on Julia 1.12.7.

### Phase 1 — serial CPU reference port (10-01)
Line-by-line port of the Bmad CSR-only path (centroid chord/spline geometry, `dspline_len`, `s_ref_to_s_chord`,
triangle binning, `s_source_calc` with element walk + `super_zbrent` + upstream infinite drift, `I_csr`, causal
convolution as a direct sum, interpolated kick, kick coefficient) plus `planar_survey`. Results: circle source point
residual < 1e-10, geometry within 0.5%; steady-state kick within 2% of analytic (measured 0.28%, ρ = 10 m, σ_z = 1 mm,
n_bin 200); drift kick < 1e-6 of the bend scale.

### Phase 2 — BeamTracking integration (10-01/02)
`track_csr!`, `slice_element`, `csr_sliceable`; Bmad `makeup_super_slave1` slicing rules, each with a gate test
(sliced == unsliced ≤ 1e-15, CSR off) and a control. Floor conventions verified (field-free particle stays on a
straight floor line to 1e-14). 80 bends: steady state within 2%. Depends on BeamTracking internals (gate tests catch
changes). RungeKutta + `EMFieldParams` slicing added afterwards (`a5e0120`).

### Phase 4 — KernelAbstractions kernels (10-02)
Deposit (atomics), gradient, wake table (blocks of lags warm-started as Bmad; identical for any blocking), convolution,
kick. Kernels == reference: grid/wake/source points exactly, charges/kicks ≤ 1e-12. Timing (20001 particles, n_bin 200):
kernels 1.1 ms (1 thread) / 0.5 ms (4 threads); wake table dominates.

### lcavity mode (10-02)
`SaganSlice` windows of the cavity's kernel sequence; `track_csr!(bunch, ::Branch)`; γ linear between
E_ref − dE_ref and E_ref; Δpz × pc_lin(s)/P0c_step(s). One slice == whole cavity exactly; n slices ≤ 1e-18 traveling
wave, ≤ 1e-7 standing wave and multipoles (fixed by D22, Stage C). Cavity doubling the energy: 2 vs 8 cells agree to 5e-4 with the
rescale, 4% without.

### Type stability and latency (10-06)
JET/SnoopCompile: core type-stable; function barrier `track_csr_element!`; centroid bunch weight type matched. First
`track_csr!` 17–28 → 15 s (7 s BeamTracking's own first `track!`, ~8 s kernels specialized per slice; not reducible
from CSR.jl). No invalidations. No PrecompileTools workload (user: minimum precompile time).

### Phase 5 — shielding by image charges (10-06)
`image_charge_kick` (literal port), `csr_image_table`, `csr_image_convolve!`, layer loop; downstream source search and
downstream infinite drift (D12). Drift image kick = uniform-motion field within 5e-8 of the peak. V8 effect (σ = 1 mm,
h = 1 cm): max kick change 0.33 of max kick, net energy ≤ 6%. Cost per layer 2n_bin + 1 source solves.
(The "converged" 32-layer values reported then were truncation artifacts at small h; see D24 below.)

### Phase 3 — Bmad cross-check (10-06; directory `crosscheck/`, README there)
Tao generates the bunch; both codes track it; `bmad_compat = true`. Max |CSR.jl − Bmad| / max |Bmad|:

| Case | wake kick/m (free, shielded) | Δpz final (free, shielded) | I_csr from Bmad's orbit |
|---|---|---|---|
| pure bend, on axis | 1.5e-6, 2.1e-6 | 3.2e-6, 2.4e-6 | — |
| pure bend, centroid offset | 1.6e-6, 2.1e-6 | 3.2e-6, 2.4e-6 | — |
| regression bend (b1_gradient −10) | 9.9e-5, 1.1e-4 | 4.8e-5, 6.0e-5 | 2.8e-7 |
| Tao example (drifts + 2 combined bends) | 2.3e-5, 9.8e-5 | 1.8e-5, 1.0e-5 | 2.0e-6 |
| Tao example with patch | not meaningful (tracking) | — | 6.1e-7 |

CSR.jl reproduces Bmad's CSR to the precision of Bmad's diagnostic output; remaining end-to-end differences are
BeamTracking vs Bmad tracking. Default mode vs Bmad: final Δpz differs by 53% (single bend), 6% (Tao example), by
elimination V5.

**Rerun 2026-10-14** (`stage-c` after D22; BeamTracking 0.8.7, Beamlines 0.11.0, Bmad conda 20260918.0; result files
in `crosscheck/<case>/result.txt`, `geom_check.txt`). New section `shielded_default` (default mode with images: D20
hat kernel, D24 averaging, V8, V9). Max |CSR.jl − Bmad| / max |Bmad|:

| Case | CSR off, sliced (x, px) | wake kick/m compat (free, shielded) | Δpz compat (free, shielded) | Δpz default (free, shielded) | I_csr from Bmad's orbit |
|---|---|---|---|---|---|
| pure bend, on axis | 8.0e-15, 2.4e-15 | 3.6e-7, 4.9e-7 | 4.1e-8, 7.8e-8 | 0.48, 0.42 | — |
| pure bend, centroid offset | (as on axis) | 4.1e-7, 4.1e-7 | 4.1e-8, 7.8e-8 | 0.48, 0.42 | — |
| regression bend | 1.2e-8, 1.3e-7 | 1.1e-5, 9.3e-6 | 2.8e-6, 6.2e-6 | 0.49, 0.43 | 2.8e-7 |
| Tao example | 4.3e-8, 2.0e-7 | 9.6e-6, 2.1e-4 | 3.6e-6, 5.1e-6 | 0.046, 0.85 | 2.0e-6 |
| Tao example with patch | not meaningful (tracking) | — | — | — | 6.1e-7 |

- **Compat sections changed, but not because of CSR.jl:** the pre-session code (`468fc83`, run in a worktree against
  today's dependencies) gives the same compat numbers as `stage-c`. The change comes from BeamTracking: pure-bend
  tracking without CSR now agrees with Bmad to 8e-15 (was 1e-8), so the pure-bend Δpz improved 80× (3.2e-6 → 4.1e-8);
  regression bend wake 5.9e-6 → 1.1e-5 and Tao-example shielded wake 9.8e-5 → 2.1e-4 moved the other way, with
  Δpz still ≤ 6e-6. Reference floor geometry unchanged (≤ 5e-16).
- **Default mode, free space:** I_csr now differs from Bmad by up to 1.8% (pure bend, step 1; 1e-4 by step 10; Tao
  example 5.8%), was 1e-5. Cause: V9 at the bend entrance. With all sources on the upstream infinite drift,
  `dL = Lsz0 − Lz` is a small difference (2.5e-7 m) of lengths of order 1 cm, and Bmad's small-angle arc length to the
  kick point (`Lsz0`, spline slope ~0.06 in this 0.12 rad bend) is too long by 1.1e-8 m: far-source dL 4.6% too large
  in compat vs 1.9e-4 (finite-distance residual) in default, checked against the quadrature arc length. Δpz default vs
  Bmad unchanged (V5 dominates: 0.48–0.49 single bends, 0.046 Tao example).
- **Default mode, shielded:** 0.42–0.43 (single bends), 0.85 (Tao example) vs Bmad: D20, D24 (4 layers averaged),
  V8 and V9 together; for information only.

### Sliced centroid, D14 (10-06)
Regression bend vs Bmad: wake 9.9e-5 → 5.9e-6, Δpz 4.8e-5 → 3.1e-6 (shielded 1.1e-4 → 4.8e-6, 6.0e-5 → 3.4e-6).

### All-bin quadrature, D15 (10-06)
Kick error (max over |z| ≤ 2σ / peak), option A / B: steady state n_bin 50/100/200/400: 1.29e-2/4.89e-3/2.76e-3/1.82e-3
vs 1.29e-2/4.37e-3/1.96e-3/1.16e-3; transient s = 0.05 at n_bin 400: 0.050 vs 0.0026. Default vs Bmad Δpz: regression
bend 54.8% → 49.0%. `csr_kick!` 1.08 → 7.09 ms (1 thread).

### Phase 6 steps 0–2 — AD w.r.t. particles (10-09, `bb1dc13`)
`csr_kick_pz`; ForwardDiff through the kernels (serial deposit); `rrule` with explicit adjoint kernels. Constants under
differentiation: bin grid, `n_particle`, kick tables (∂/∂z of the min/max-z particle omits grid motion). rrule vs
ForwardDiff ~1e-15; ForwardDiff vs central FD: z ≲ 1.5e-9, q ≲ 1e-12.

### Stage B — open questions (10-09/10, `9030f17`)
D21 (V6 follows Bmad), D23 (Beamlines compat), D20 hat-integrated image kernel: per lag interval adaptive Simpson of
`(∫K, ∫K t)`, geometric split near u = 0 down to `0.01 y/γ`, tolerance `rtol·dz·max|K|`. Bugs found and fixed:
warm-started search on the upstream infinite drift did not walk back into the lattice; the point-table tolerance was
far too tight when dz ≫ y/γ (Simpson recursed to depth 40). Steady-state bend, 16 layers, h = 3 mm, n_bin 60…960:
point 2.68/3.41/4.88/0.365/0.355, hat 0.345/0.342/0.341/0.340/0.340 (16-layer values: not converged in layers).
Hat costs ~5× the source solves of point sampling.

### Phase 6 step 3a — forward-mode derivatives w.r.t. geometry, s_kick, γ (10-14, branch `phase6-geom_speed-deltas`)
- `FloorPosition{T}`, `CSRElementInfo{T}`, `CSRGeometry{T}`, `CSRKickPoint{T}`, `CSRKick1{T}` (immutable except
  `CSRKick1`, as before); promoting constructors; `number_type`, `table_type`; tables allocated in `T`;
  `csr_kick_pz` promotes over geometry, `s_kick`, γ, `kick_factor`, `pz_scale`.
- Source root: `zbrent` on primal values (Float64 results unchanged), then `root_tangent` (identity in core;
  `CSRForwardDiffExt`: `ds/dp = −∂ₚddz/∂ₛddz` with ∂ₛddz from a nested Dual under a private tag `RootTag` ordered
  inside any tag, so untagged `Dual{Nothing}` works). `k` is re-evaluated at the Dual root (default) or at zbrent's last
  point with the root's tangent (compat, V7). `ddz_calc_csr!` now stores `s_chord_source`.
- Bugs found on the way: ForwardDiff ≥ 1.0 `==` compares partials (`Dual(0, 1) == 0` is false). `advance_floor`'s
  `angle == 0` gave NaN for a Dual g at 0 (now `primal(angle) == 0` + second-order expansion); same fix in
  `I_csr!` (`s_chord_kick == 0`) and `y_source != 0`. First version kept `k` at the primal root in default mode:
  0.4% derivative error, found by FD (FD converged over h = 0.1…1e-5).
- Primal choices: `csr_n_step` rounds `primal(L)`; `image_u_min`, `image_hat_tol` primal.
- Checks (`test/ad_geometry_test.jl`, 106 tests): AD vs best-of-steps central FD. Free space ≤ 1e-5 (typ. 1e-8);
  images compat ≤ 1e-4; default images at kick level ≤ 1e-3 because the shielded kick's derivative is 7–25× smaller
  than its free-space part (cancellation amplifies FD noise of the hat quadrature); hat image table alone (rtol 1e-10)
  ≤ 2e-5 (measured ≤ 5e-6). Exact identity J[:,pz] = J[:,px]·(−px/(1+pz)) to ≤ 3e-10; residual of ddz at the Dual
  root has zero first-order partials; `advance_floor` at g = 0 vs analytic. FD outliers (centroid pz columns, L₁ at
  ~1e-4) traced to zbrent `abs_tol = 1e-8` noise in FD, not AD.
- Float64 path: 1141/1141 on 1 and 4 threads; `csr_kick!` 6.81 ms (free, was 6.94) and 70.5 ms (16 layers, was
  71.0), `diagnostics/bench_excess.jl`, 1 thread. Docs: `usage.md` "Derivatives" updated with an example; build clean.
- Not done: nested Duals (Hessians) in `root_tangent` (errors); GTPSA/TPS (Phase 7: needs Newton iterations per
  order); Duals on GPU backends.

### D28, D29 — ForwardDiff hard dependency, immutable `CSRKick1` (10-14, branch `phase6-geom_speed-deltas`)
- `CSRKick1{T}` immutable and isbits (Float64, Float32, Dual; tested). `with_fields(k; ...)` copies with fields
  replaced; `CSRKick1(geom, kp; ix_ele_source, dz_particles)`. Renamed (now non-mutating, returning `(k, value)`):
  `ddz_calc_csr`, `s_source_calc`, `I_csr_calc` (was `I_csr!`), `I_csr_quadrature`, `image_kernel_at`,
  `image_simpson`, `image_segment`, `image_interval_moments`. `zbrent` returns `(x, ok, x_last)`; compat (V7)
  re-evaluates `k` at `x_last` instead of relying on mutation. `ddz_calc_csr` is `@inline` (else ~3% slower).
- Diagnostics scripts updated (prototypes `image_proto.jl`, `image_layers.jl` hold `k` in a `Ref`); all rerun and
  reproduce their logged numbers.
- 1150/1150 on 1 and 4 threads (+9: isbits/`with_fields`). Timing, interleaved runs vs `8042ef1`, 1 thread:
  free space 7.29/7.19 vs 7.15/7.23 ms, 16 layers 74.7/76.6 vs 73.8/74.3 ms (noise ~3%; machine slower than the
  6.81 ms run above). No allocations in `s_source_calc`.

### Phase 6 step 3b — rrules w.r.t. geometry, s_kick, γ (10-14, branch `phase6-geom_speed-deltas`)
- `csr_kick_pz` rrule (`CSRChainRulesCoreExt`): particle part as before (`csr_kick_adjoint` now also returns the kick
  cotangent per bin `kbar`); geometry/`s_kick`/γ by `csr_kick_param_adjoint`: ForwardDiff gradient of `kbar ⋅ kick`
  through `csr_bins_kick` (split out of `csr_binned_kick`) with the bins fixed, on the CPU. Geometry cotangent is a
  `Tangent{CSRGeometry}` (no `bmad_compat` entry), built from `geometry_to_vector` order (4 + 26 n_ele numbers;
  `geometry_from_vector` the inverse). rrules for `CSRGeometry(L, g, centroid)` and `(L, g, ref_exit, centroid)`:
  transposed ForwardDiff Jacobian; keywords (`ref_start`, `kick_factor`, `pz_scale`) get no cotangents.
- Sparse seeding (`geometry_active_indices`): kick element (26), walked elements (11: floor0, theta_chord, L_chord,
  spline, dL_s), drift sums and `start`/last `floor1` when sources leave the lattice; range from source searches at
  the extreme lags. Assumes the source element is monotone in the lag. Pullback (n_bin 200, ρ = 10 m arc, 1 thread):
  dense 0.05 s (10 elements), 17 s (80); sparse 0.02 s (10), 2.6 s (80 and 400; 367 active numbers), kick 7 ms.
- Bugs found: (1) `mod` NaN partials at 0 → NaN derivatives for a kick at an element entrance (source on the boundary)
  and on-axis straight elements; (2) Dual comparisons at ties use partials → sparse and dense gradients differed by
  2.5% at a boundary root; single-direction AD and FD confirmed the sparse value. Fixed by primal comparisons in
  `source_point.jl`, `wake.jl`, `image_charges.jl`, `geometry.jl`, `chord_coordinates.jl`, the deposit
  (`primal_max`/`primal_min`) and the kick kernel. Float64 results unchanged (suite exact comparisons pass).
- Tests (`test/ad_rrule_test.jl`): rrule vs ForwardDiff ≤ 1e-12 (geometry vector, s_kick, γ; free space, images,
  compat); composed with the CSRGeometry rrules vs ForwardDiff over L (≤ 1e-6, cancellation), g, centroid, ref_exit
  (≤ 1e-9); sparse == dense in 5 cases (mid-lattice, entrance, upstream drift, downstream drift with images, compat);
  ChainRulesTestUtils `test_rrule` (z fixed via `rrule_f`; FD steps capped at 1e-6 because of curvature, rtol 1e-2
  because of the zbrent 1e-8 noise at those steps; `to_vec`/`test_approx` for `CSRGeometry` in the test file).
  ChainRulesTestUtils in `[extras]` (compat 1.13). 1355/1355 on 1 and 4 threads; docs updated, build clean.

### Stage C (10-14, branch `stage-c`)
- **Step 0.** Compat `Beamlines = "0.10.6, 0.11"`, `BeamTracking = "0.8.7"` (registered 0.8.5–0.8.7 contain
  `RungeKutta`; only 0.8.7 tested). 857/857 against dev BeamTracking 0.8.7, Beamlines 0.11.0.
- **GPU (Metal, M3, Metal.jl 1.11.1).** Float32 grid, Int32 counters, typed literals; then `backend.jl` (tables on the
  CPU in Float64, copied in `csr_float_type`). Full `csr_kick!` on `MtlArray{Float32}` vs CPU Float32: 2.7e-6 (free
  space), 3e-4 (32 layers, h = 3 mm; Float32 cancellation, the result is ~3e-3 of its terms). Metal 17 ms vs CPU 9 ms per
  free-space kick (tables on the CPU). Particle-side alone (bin + convolve + kick, 1M particles): Metal 6.9 ms vs CPU
  125 ms (4 threads; CPU Float32 atomics not investigated).
- **D24 investigation.** 8 m arc artifact: 1 cm terms constant (−7.78e-11) once sources leave the arc (n ≳ 185).
  40 m arc, 1024 layers, reference avg(8) (before V9): 1 + Σa = 2.8869e-3 (3 mm), 2.6444e-3 (1 cm), 4.1094e-2 (3 cm).
  Methods from the first M layers (relative error vs reference):

  | M | h | plain | avg(4) | avg(8) | Wynn ε |
  |---|---|---|---|---|---|
  | 32 | 3 mm | 13.8 | 3.7e-4 | 7.7e-7 | 1.1e-6 |
  | 64 | 3 mm | 0.45 | 4.0e-6 | 5.6e-9 | 5.5e-8 |
  | 32 | 1 cm | 3.2e-3 | 7.0e-6 | 7.2e-8 | 8.8e-8 |

  Per-bin kick (max over |z| ≤ 3σ): avg(8) at M = 32: 6.6e-7 (3 mm), 1.4e-6 (1 cm); plain at 16 layers 170× (3 mm),
  6× (1 cm). Data: `diagnostics/layer_terms_<h>_240_400.txt`.
- **D24 implementation.** `image_sum_order = 8`; layer weights `2^-k Σ_{j ≥ n−M+k} C(k, j)` (k = min(order, M),
  Σw = M − k/2) on `kick_factor` in `csr_binned_kick` and `csr_kick_reference!`; AD unchanged. The h = 3 mm
  coarse-vs-fine-bin test pins `image_sum_order = 0`.
- **V9, exact path excess.** Diagnosis (`image_large_y.jl`): CSR.jl's kernel = exact LW field at its own source
  position; source misplaced by the dropped Ls⁵/(1920ρ⁴). `path_excess = ∫ x′²/(1+√(1+x′²)) ds + 2 sin²(dθ/2) Δs −
  sin(dθ) Δx` (from L = ∫cos(θ − θ_L) ds_arc), Gauss–Legendre 4 points. After: kernel vs exact ≤ 3e-5 (y = 3 cm–1.9 m),
  circle excess ≤ 1e-8 relative (compat 4e-5–6.5e-4 = Ls²/(80ρ²)), image kernel at y = 3 mm, lag 6 mm
  1.0010 → 0.99999 of exact.
  Cost: `csr_kick!` 6.68 → 6.79 ms (free space), 70.4 → 70.6 ms (16 layers), 1 thread (`diagnostics/bench_excess.jl`).
  Default free-space I_csr changes up to ~7e-4 relative at long lags.
- **D22, exact SaganCavity split.** `sagan_segment!` in `sagan.jl`: a drift segment [a, b] applies its half kicks
  (ponderomotive for standing waves, multipoles with radiation) at a and b as point events with the full segment
  strength, and only the drift/solenoid is split. Sliced vs whole (max |Δv|, 3/4/7/20 slices): standing wave 2e-7 →
  ≤ 7e-19; n_cells = 0 and multipoles with radiation damping ≤ 9e-19; with a solenoid ≤ 2e-16 (BeamTracking's own
  Solenoid split once: 6e-17, the exact solenoid map's round-off). `lcavity_test.jl` bounds 1e-17 (1e-15 with a
  solenoid), new cases added. 1035/1035.

---

## 10. Roadmap, limitations, risks

### Phases
| Phase | Content | Status |
|---|---|---|
| 0 | Package skeleton | done |
| 1 | Serial CPU reference port (Float64) | done |
| 2 | BeamTracking integration (route a) | done |
| 3 | Bmad cross-check | done |
| 4 | KernelAbstractions kernels | done (GPU: particle side + CPU tables, Stage C) |
| 5 | Shielding (image charges) | done (+ D20, D24, V9) |
| 6 | AD: steps 0–2 particles (done); step 3 geometry/γ via implicit differentiation of the source root | done (3a forward, 3b rrules) |
| 7 | TPS (serial deposit, optional smooth shape function); tests: first order = Jacobian, second = ForwardDiff Hessian | not started |
| 8 | FFT convolution (`CSRAbstractFFTsExt`) for large n_bin; benchmarks; GPU wake table (isbits geometry via Adapt; work struct immutable since D29; errors/recursion in the source search); issue for a BeamTracking collective-step hook (route b) | not started |

Follow-ups not requested: per-element `ds_step` (Bmad `csr_ds_step`); caching slices per element (rebuilt every step);
Metal tests in the suite (conditional on Metal.jl).

### v1 limitations
- Planar geometry only: bends with `tilt_ref ∈ {0, π}`; vertical bends rejected.
- Patches with out-of-plane offsets rejected (Bmad rejects `floor_shift`).
- No LSC or transverse space charge; no wigglers.
- Requires live particles ≥ `n_bin` (Bmad behavior).
- Wake and image tables on the CPU only; GPU kernels in Float32 on Metal.

### Risks
| Risk | Mitigation |
|---|---|
| Slicing does not reproduce unsliced tracking | Phase 2 gate tests |
| Per-slice `track!` overhead | Measure in phase 8; route (b) upstream if dominant |
| Dependence on BeamTracking internals | Gate tests; compat pinned at 0.8.7 |
| Steady-state coefficient not independently verified | Check a primary source (Section 8) |
| `|q|` placement for ions (V2) | Electrons only in tests |
| Float32 on GPUs in strong shielding | Documented ~3e-4; mixed precision if needed |

---

## 11. Reference sources and planning files

### Sources (local checkouts under `~`)
- **Bmad** `~/bmad-ecosystem`: `bmad/space_charge/csr_and_space_charge_mod.f90` (line numbers as of 2026-10-01:
  `track1_bunch_csr` 137–404, `csr_bin_particles` 434–678, `csr_bin_kicks` 701–835 incl. images 813–826,
  `s_source_calc` 857–1003, `ddz_calc_csr` 1023–1086, `I_csr` 1368–1432, `image_charge_kick_calc` 1451–1494,
  `csr_and_sc_apply_kicks` 1513–1652, `dspline_len` 1770–1799, `s_ref_to_s_chord` 1817–1842); slicing
  `bmad/code/create_element_slice.f90`, `bmad/modules/bookkeeper_mod.f90` (`makeup_super_slave1` 1172–1509);
  reference cases `regression_tests/csr_and_space_charge_test/` (sbend L = 0.1, 42 MeV, 77 pC, σ_z = 0.899377 mm,
  n_bin 40; shielded n = 4, h = 0.01), `bmad-doc/tao_examples/csr_beam_tracking/`.
- **BeamTracking.jl**: `src/types.jl`, `src/kernel.jl` (`KernelChain`, `RefState`, `launch!`), `src/tracking_methods.jl`,
  `src/kernels/sagan_cavity.jl`, `src/kernels/runge_kutta.jl`, `src/utils/energy.jl`, `ext/BeamTrackingBeamlinesExt/`.
- **Beamlines.jl**: `src/beamline.jl` (reference energy, `dE_ref`), `src/branch.jl`, `src/misc.jl` (`EMFieldParams`),
  `src/bend.jl`.
- **Others**: AtomicAndPhysicalConstants.jl (`chargeof`, `massof`; CODATA 2022), SpaceCharge.jl (3D solver, not a
  dependency), SciBmad.jl. KernelAbstractions: `supports_float64`, `supports_atomics` (checked in 0.9.35–0.9.45).

### Planning files (`planning/`)
| File | Contents |
|---|---|
| `PLAN.md` | This document. |
| `csr_gpu_ad_design.md` | User's original design notes (physics, numerics, KA sketches, AD adjoints); partly superseded; its KA sketches were untested. |
| `verify_analytic.jl` | Section 8 checks (Base Julia). |
| `dspline_coeff.md`, `upstream-bin.md` | Write-ups for V1 and V5. |
| `upstream_reports.md` | Drafted upstream reports (Section 4). |
| `diagnostics/` | Scripts (run with the test env; they include `~/.julia/dev/CSR.jl/test/utils.jl`): `v8_drift.jl`, `first_bin_vs_quadrature.jl`, `transient_*.jl`, `steady_state_convergence.jl`, `image_proto.jl`, `image_layers.jl`, `layer_terms.jl [n_layers] [n_ele] [h,...]`, `layer_shape.jl`, `layer_sum_methods.jl`, `layer_kick_profile.jl`, `image_kernel_exact.jl`, `image_large_y.jl`, `bench_excess.jl`; data `layer_terms_*.txt` (the `_400` files predate V9). |
| `../crosscheck/` | Bmad/Tao cross-check (README there). |
| `../docs/` | Documenter docs (`julia --project=docs docs/make.jl`; `docs/README.md`); pages Home, Usage, Physics model, Physics decisions, Validation, API, Internals. Deployed by `.github/workflows/Documentation.yml` from `main`. Keep "Physics decisions" in sync with Section 3 when physics changes. |
