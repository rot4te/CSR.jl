# Stage 04: fixes from the whole-codebase review of 2026-10-18

Spec date: 2026-10-24. Base: branch `stage/csr/04-review-fixes` (`06fb544`, same as `stage/csr/03-bt-pr369-check`).
Findings: `planning/PLAN.md` Section 9, "Whole-codebase review 2026-10-18" (labels W1–W6, S1–S10 below refer to that
list only; the Stage 2c review in `specs/stage2c.md` reused the same labels for different findings). User decisions
of 2026-10-24 in Section 13.

## 1. Overview

Fix the findings of the whole-codebase review, in four sub-stages (Section 11), each on its own branch:

- **A. Type stability** (W1, W2, W6): `path_excess` and `image_charge_kick` type-stable for every number type; a new
  `@inferred` testset over the core source-point functions.
- **B. Correctness** (W3, W4, W5): `shifted_to_slice` with mixed number types; `csr_kick_pz` and its `rrule` forward
  pass on backends without `Float64` (Metal); a clear error for γ ≤ 1.
- **C. Docs and entry checks** (S1, S2, S8): stale docstring and `docs/src/validation.md`; `track_csr!` checks
  `params.ds_step > 0` and the length of `centroid_start`; `track_csr!` documented as CPU-only.
- **D. Cleanup** (S3, S4, S5, S6, S7, S9, S10): rename, docstring corrections, GTPSA compat, remove `Adapt` and the
  tracked snapshots, `@kwdef` `SaganSlice`, no boxed closure state, last SaganCavity slice end, range test order.

## 2. Out of scope

- No change to physics, kick formula, tables, deposit, root finding, or the source search.
- W4: the `rrule` **pullback** on backends without `Float64` (it stays CPU/`Float64`-only; documented, Phase 8).
  `Dual`/TPS particles on GPU backends.
- The CSR.jl `SaganSlice` follow-up for BeamTracking PR #369 (PLAN.md Section 1, item 14): not part of this stage.
- Deferred findings in PLAN.md Section 4 and Phase 8 items.
- No upstream changes (BeamTracking, Beamlines, GTPSA).
- Do not edit `planning/PLAN.md`. Do not run git (no add, rm, commit, branch, stash, checkout). File deletions
  required by D.4 are done with the shell (`rm -r`); the coordinator stages them.

## 3. Language and toolchain

- Julia 1.13.2 (juliaup `release`); `Project.toml` keeps `julia = "1.10"`.
- Style: 2-space indent, sparse comments, a docstring for every new function (docs build with `checkdocs = :all`,
  including the extensions). Match the surrounding code. Error messages start with the public function name and a
  colon, as existing ones (`"track_csr!: ..."`).

## 4. Dependencies (existing; use these, do not reimplement)

Registry checked 2026-10-24 (General registry `Versions.toml` on GitHub): GTPSA 1.5.6, BeamTracking 0.8.7, Beamlines
0.11.0 are the latest registered versions.

| Package | Version | Used for |
|---|---|---|
| KernelAbstractions | 0.9.x (compat `"0.9.42"`) | `KernelAbstractions.supports_float64(::Backend)` (default `true`), `abstract type Backend`, `get_backend`. CSR does `using KernelAbstractions` (`src/CSR.jl:13`). |
| ForwardDiff | 1.4.6 (test env) | `ForwardDiff.Dual{T}(value, partials...)`, `ForwardDiff.Tag(f, V)` (`src/config.jl:16`) for tests. |
| GTPSA | 1.5.6 | Tests: `Descriptor(nv, mo)`, `@vars(d)` (static descriptor). |
| BeamTracking / Beamlines | 0.8.7 / 0.11.0 (registered, not the dev checkouts; Section 10) | Tests of the extension. |
| CSR core | this repo | `CSR.primal`, `CSR.number_type`, `CSR.csr_float_type`, `CSR.device_scalar` (`src/utils/backend.jl`). |

## 5. Project layout (files to change or add)

| Sub-stage | Files |
|---|---|
| A | `src/utils/spline.jl` (W1), `src/kernels/image_charges.jl` (W2), `src/utils/root_finding.jl` (A.4), new `test/type_stability_test.jl`, `test/runtests.jl` (one `include`). |
| B | `ext/CSRBeamTrackingBeamlinesExt/slicing.jl` (W3), `src/utils/backend.jl` (W4 helper), `src/kick.jl` (W4), `ext/CSRChainRulesCoreExt/CSRChainRulesCoreExt.jl` (W4), `src/geometry.jl` (W5), `src/utils/kick_setup.jl` (W5), `docs/src/usage.md` (W4, one sentence), new `test/review_fixes_test.jl`, `test/runtests.jl` (one `include`). |
| C | `src/interface.jl` (S1), `docs/src/validation.md` (S1), `ext/CSRBeamTrackingBeamlinesExt/track.jl` (S2, S8), `test/review_fixes_test.jl` (S2 testset). |
| D | `src/kernels/adjoint.jl` (S3), `src/types.jl`, `src/tables.jl`, `src/param_adjoint.jl` (S4, S9), `Project.toml` (S5, S6), delete `diagnostics/base/` and `diagnostics/head/` (S6), `ext/CSRBeamTrackingBeamlinesExt/sagan.jl` (S7, S10), `src/reference.jl` (S10), `test/review_fixes_test.jl` (S10 testsets), four `diagnostics/*.jl` include lines (D.9). |

## 6. Components

For plain `Float64` inputs every change except D.7 (S10, last SaganCavity slice) must give bitwise the same results as
before this stage.

### 6.A Type stability (sub-stage A)

**A.1 (W1) `path_excess`** (`src/utils/spline.jl`, `function path_excess(s0, s1, c, dtheta_ref = 0.0; ...)`).
Replace `acc = 0.0` by

```julia
acc = zero(float(promote_type(eltype(c), typeof(s0), typeof(s1))))
```

Nothing else in the function changes.

**A.2 (W2) `image_charge_kick`** (`src/kernels/image_charges.jl`). In the `else` branch replace `g_bend = 0.0` by

```julia
g_bend = zero(promote_type(number_type(geom), number_type(k)))
```

**A.3 (W6) New file `test/type_stability_test.jl`**, one top-level `@testset "Type stability (core)"`. Add
`include("type_stability_test.jl")` to `test/runtests.jl` directly after `include("tps_collective_test.jl")`.

Setup (inside the testset):

```julia
L3 = [0.2, 0.3, 0.2]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
tag_in = ForwardDiff.Tag(sin, Float64)
D1 = ForwardDiff.Dual{typeof(tag_in),Float64,1}
tag_out = ForwardDiff.Tag(cos, D1)
d_tps = Descriptor(1, 2)
x_tps = @vars(d_tps)
```

Cases, as `(name, g2, s_kick, gamma, lag)` where `g2` is the curvature of element 2:

| name | `g2` | `s_kick` | `gamma` | `lag` |
|---|---|---|---|---|
| `"Float64"` | `1.5` | `0.1` | `200.0` | `1e-4` |
| `"Float32"` | `1.5` | `0.1f0` | `200.0f0` | `1f-4` |
| `"Dual"` | `ForwardDiff.Dual{typeof(tag_in)}(1.5, 1.0)` | `0.1` | `200.0` | `1e-4` |
| `"nested Dual"` | `ForwardDiff.Dual{typeof(tag_out)}(ForwardDiff.Dual{typeof(tag_in)}(1.5, 1.0), ForwardDiff.Dual{typeof(tag_in)}(1.0, 0.0))` | `0.1` | `200.0` | `1e-4` |
| `"TPS"` | `1.5 + x_tps[1]` | `0.1` | `200.0` | `1e-4` |

For each case, in a `@testset "$name"`:

```julia
geom = CSRGeometry(L3, [0.0, g2, 0.0], c3)
T = CSR.number_type(geom)
kp = CSRKickPoint(geom, 3, s_kick, gamma)
search = CSR.primal_search(geom, kp, T)
k0 = CSR.CSRKick1(geom, kp; dz_particles = lag)
```

and these assertions (each call wrapped in `@inferred`; `@test` on the stated property):

1. `@test (@inferred CSR.path_excess(zero(s_kick), s_kick, geom.ele[2].spline, 0.01)) isa promote_type(T,
   typeof(s_kick))`; the same with `bmad_compat = true`.
2. `k1, s1 = @inferred CSR.s_source_calc(k0, geom, kp, search)`; test `k1 isa CSR.CSRKick1`.
3. `@inferred CSR.ddz_calc_csr(k1, s1, geom, kp)`.
4. `@inferred CSR.I_csr_calc(k1, 1, lag, geom, kp, 0.0, search)` and `@inferred CSR.I_csr_calc(k1, 2, lag, geom, kp,
   0.0, search)` (first-bin and later-bin branches).
5. `@inferred CSR.with_fields(k0; dz_particles = 2 * lag)` and `@inferred CSR.with_fields(k0; s_chord_source = 0.0)`.
6. Image kick, source inside the lattice: `kp_i = CSRKickPoint(geom, 3, s_kick, gamma; y_source = 0.01)`,
   `ki, _ = CSR.s_source_calc(CSR.CSRKick1(geom, kp_i; dz_particles = lag), geom, kp_i, CSR.primal_search(geom, kp_i,
   T))`; `@test 1 <= ki.ix_ele_source <= 3`; `@inferred CSR.image_charge_kick(ki, geom, kp_i)`.
7. Image kick, source on the upstream infinite drift (W2's branch): `kp_u = CSRKickPoint(geom, 1, 0.05, gamma;
   y_source = 0.01)`, `ku, _ = CSR.s_source_calc(CSR.CSRKick1(geom, kp_u; dz_particles = 1e-3), geom, kp_u,
   CSR.primal_search(geom, kp_u, T))`; `@test ku.ix_ele_source == 0`; `@inferred CSR.image_charge_kick(ku, geom,
   kp_u)`.

If any assertion fails for a reason other than W1/W2/A.4 (e.g. another function is not inferred, or a call signature
in this section does not exist), **stop and report** the failing call, case, and the inferred type; do not change other
source code.

**A.4 (amendment 2026-10-24) `root_tangent`** (`src/utils/root_finding.jl`). For a TPS, `root_order(t::TPS) =
Int(t.mo)` is a runtime value, so the early return `n == 0 && return s0` (Float64 primal root) made the result
`Union{Float64, TPS}` (found by A.3 assertion 2, TPS case). Replace that line by

```julia
n == 0 && return convert(promote_type(typeof(s0), typeof(f0)), s0)
```

(the type the Newton steps return). Docstring: "Returns `s0` when `f0` carries no derivatives." → "Returns `s0`
(converted to the promoted type of `s0` and `f0`) when `f0` carries no derivatives." Nothing else changes. For
`Float64`/`Dual` the result is bitwise unchanged.

**A.5 (amendment 2026-10-24; code review of sub-stage A, findings W1–W4, all approved by the user).**

1. **`path_excess` with all-`Float32` arguments** (`src/utils/spline.jl`). The accumulator becomes
   `acc = zero(float(promote_type(eltype(GL4_WEIGHTS), eltype(c), typeof(s0), typeof(s1))))` (the Gauss–Legendre
   weights are `Float64`, so a `Float32` `acc` became `Union{Float32, Float64}`). Float64/Dual/TPS unchanged.
2. **Test file** `test/type_stability_test.jl`:
   a. Header comment (lines 1–2) replaced by:
      ```julia
      # Stage 04 sub-stage A (W1, W2, W6): return types (`@inferred`) of the core source-point, kick and image-charge
      # functions for Float64, Float32 scalars (the geometry is always at least Float64), a Float32 spline in
      # `path_excess`, Dual, nested Dual, static- and dynamic-descriptor TPS. Detects A.4 regressions; the W1/W2
      # local-variable unions leave the return types unchanged and are not detected here.
      ```
   b. A sixth case after `"TPS"`: `("TPS dynamic", 1.5 + x_dyn[1], 0.1, 200.0, 1e-4)`, with, after `x_tps = @vars(d_tps)`:
      `d_dyn = Descriptor(1, 2)` and `x_dyn = @vars(d_dyn, dynamic = true)` (so `GTPSA.desc_current` is `d_dyn` while
      the cases run; create no other `Descriptor` before the loop ends).
   c. Every `@inferred` counts as a test and does not stop the case on failure:
      - Line `k1, s1 = @inferred CSR.s_source_calc(k0, geom, kp, search)` becomes two lines:
        `@test (@inferred CSR.s_source_calc(k0, geom, kp, search)) isa Tuple` and
        `k1, s1 = CSR.s_source_calc(k0, geom, kp, search)`.
      - Each bare `@inferred X` line (`ddz_calc_csr`, both `I_csr_calc`, both `with_fields`, both `image_charge_kick`)
        becomes `@test (@inferred X) isa Any`.
   d. After the `for` loop (inside the top-level testset), `@testset "Float32 spline"`:
      `c32 = SVector{4,Float32}(1f-3, 2f-3, -1f-3, 5f-4)`;
      `@test (@inferred CSR.path_excess(0f0, 0.1f0, c32, 0.01f0)) isa Float64`;
      `@test (@inferred CSR.path_excess(0f0, 0.1f0, c32, 0.01f0; bmad_compat = true)) isa Float64` (after A.5.3).
      `SVector` comes from `StaticArrays`, which `runtests.jl` and the one-file command (Section 10) load; do not add
      a `using` line.
   e. Last, `@testset "root_tangent with an order-0 TPS"`: `d0 = Descriptor(1, 0)`; `f0 = zero(first(@vars(d0)))`;
      `@test CSR.root_order(f0) == 0`; `r = @inferred CSR.root_tangent(s -> s - 0.25, 0.25, f0, 1.0)`;
      `@test r isa typeof(f0)`; `@test GTPSA.scalar(r) == 0.25`. If `Descriptor(1, 0)` or `@vars(d0)` raises, or
      `CSR.root_order(f0) != 0`, omit this testset and report what happened (do not try other constructions).
      It must be the last testset in the file (it changes `GTPSA.desc_current`).
   **Outcome (spec-implementer, 2026-10-24):** `Descriptor(1, 0)` is created as `MO = 1` and `root_order(f0) == 1`,
   so 5.2e was omitted as specified; the executed `n == 0` branch stays untested (inference only).
3. **`path_excess` compat branch return type** (amendment 2026-10-24, user's choice). With all-`Float32` arguments
   the compat branch (`dspline_len`) returned `Float32` and the default branch `Float64`, so `path_excess` inferred
   `Union{Float32, Float64}` (`bmad_compat` is a runtime `Bool`). Replace the line
   `bmad_compat && return dspline_len(s0, s1, c, dtheta_ref; bmad_compat)` by

   ```julia
   R = float(promote_type(eltype(GL4_WEIGHTS), eltype(c), typeof(s0), typeof(s1), typeof(dtheta_ref)))
   bmad_compat && return convert(R, dspline_len(s0, s1, c, dtheta_ref; bmad_compat))
   ```

   (`R` is the default branch's return type; for Float64, `Dual` and TPS inputs it equals `dspline_len`'s type, so
   `convert` is the identity and results are bitwise unchanged). Add to the docstring: "The result type is the same
   in both modes (`Float64` for `Float32` arguments)." Nothing else changes.
4. Acceptance (criterion 1e): the file passes; report the new assertion count. Sensitivity (scratch `@eval`):
   (i) A.5.1 without `eltype(GL4_WEIGHTS)`: report only (not detectable by `@inferred`; `code_warntype` of the
   keyword body shows `acc::Union{Float32, Float64}` with the old line, a concrete `acc` with the fix; user
   2026-10-24);
   (ii) A.4 old line → TPS and "TPS dynamic" `s_source_calc` assertions fail **and** their later assertions still
   run (counted as pass/fail, not skipped); the order-0 testset's `r isa typeof(f0)` fails (if present);
   (iii) A.5.3 old line (no `convert`) → both "Float32 spline" assertions fail.
   **Observed 2026-10-24 (Julia 1.13.2, scratch `@eval`):** (ii) 78 pass, 2 Error (`@inferred` throws, "Error During
   Test") at the `s_source_calc` line of "TPS" and "TPS dynamic"; all later assertions ran; the order-0 testset is
   omitted (5.2e). (iii) 78 pass, 2 Error: both "Float32 spline" lines error, the default-mode one because the
   method's inferred return is `Union{Float32, Float64}` (the compat branch returns `Float32`).
   (i) Observed 80 pass, 0 fail (replacement text above approved by the user). With `acc = zero(float(promote_type(eltype(c),
   typeof(s0), typeof(s1))))` the inferred return of the default mode stays `Float64` (`Base.return_types`;
   `code_warntype` of the keyword body shows `acc::Union{Float32, Float64}`), so `@inferred` cannot detect A.5.1. Proposed
   replacement for (i): "report only (not detectable by `@inferred`; `code_warntype` of the keyword body shows
   `acc::Union{Float32, Float64}` with the old line, a concrete `acc` with the fix)". Alternative for the user: add an
   introspection test for `acc`'s concreteness (brittle; needs `code_warntype`/`code_typed` on internals).

### 6.B Correctness (sub-stage B)

**B.1 (W3) `shifted_to_slice`** (`ext/CSRBeamTrackingBeamlinesExt/slicing.jl`). Replace the return line by

```julia
s0p, t0 = promote(float(s0), s0 / (beta_0 * C_LIGHT))
return ShiftedPotential(f, s0p, t0)
```

and add to its docstring: "`s0` and `t0` are promoted to one type (e.g. a `Dual`/TPS `p_over_q_ref` with a `Float64`
`s0`)."

**B.2 (W4) Device number type.** In `src/utils/backend.jl`, after `device_scalar`, add

```julia
"""
    device_number_type(backend, T)

Element type for a particle array on `backend` that must hold numbers of type `T`: `csr_float_type(backend)` when
`T <: AbstractFloat` and the backend lacks `Float64` (e.g. Metal), else `T`. `Dual` and TPS types are returned
unchanged (not supported on such backends).
"""
device_number_type(backend, ::Type{T}) where {T} =
  (T <: AbstractFloat && !KernelAbstractions.supports_float64(backend)) ? csr_float_type(backend) : T
```

- `csr_kick_pz` (`src/kick.jl`): allocate with
  `similar(pz, device_number_type(get_backend(pz), promote_type(eltype(pz), eltype(z), eltype(q), number_type(geom),
  typeof(s_kick), typeof(gamma), typeof(kick_factor), typeof(pz_scale))))` (same promoted type as now, wrapped).
- `rrule` for `csr_kick_pz` (`ext/CSRChainRulesCoreExt/CSRChainRulesCoreExt.jl`): allocate `pz_new` the same way
  (`CSR.device_number_type(CSR.get_backend(pz), ...)`), and call
  `CSR.csr_apply_kicks!(pz_new, z, alive, st.bins, st.kick; pz_scale = CSR.device_scalar(CSR.get_backend(pz),
  pz_scale))`. The pullback is unchanged. If `CSR.get_backend` is not accessible from the extension, stop and report.
- `csr_kick_pz` docstring: add the paragraph "On a backend without `Float64` (e.g. Metal) the returned vector has
  element type `Float32` (`CSR.csr_float_type`). The `rrule`'s pullback needs a backend with `Float64` (the CPU)."
- `docs/src/usage.md`, section "GPU backends": after the code block add the sentence "`csr_kick_pz` works the same
  way; its `rrule` pullback runs on the CPU only."

**B.3 (W5) γ check.** Error message, exactly: `"<fname>: gamma must be > 1 (got $(primal(gamma)))"`.

- `CSRKickPoint(geom::CSRGeometry, ix_ele_kick, s_kick, gamma; ...)` (`src/geometry.jl`): directly after the
  `check_descriptors` line add `primal(gamma) > 1 || error("CSRKickPoint: gamma must be > 1 (got $(primal(gamma)))")`.
- `csr_kick_coef` (`src/utils/kick_setup.jl`): as the first statement, `primal(gamma) > 1 || error("csr_kick_coef:
  gamma must be > 1 (got $(primal(gamma)))")`.
- Docstrings of both: add "Errors unless `gamma > 1` (primal value; NaN fails)."

**B.4 Tests.** New file `test/review_fixes_test.jl` with a top-level `@testset "Review fixes 2026-10-18"`; add
`include("review_fixes_test.jl")` as the last `include` inside `@testset "CSR.jl"` in `test/runtests.jl`. The file uses
`TC_EXT`, `TC_E`, `tc_pq`, `tc_line_bunch` (from `track_csr_test.jl`) and `LC_EXT`, `lc_cavity`, `lc_in_branch`
(from `lcavity_test.jl`), which `runtests.jl` includes earlier. Testsets for sub-stage B:

1. `@testset "W3 shifted_to_slice"`: `f(x, y, s, t, pp = nothing) = (s, t)`;
   `ele = (; species_ref = TC_E, p_over_q_ref = ForwardDiff.Dual(tc_pq(200), 1.0))`;
   `sp = TC_EXT.shifted_to_slice(f, ele, 0.25)`; test `sp.s0 isa ForwardDiff.Dual`, `typeof(sp.s0) == typeof(sp.t0)`,
   `ForwardDiff.value(sp.s0) == 0.25`, `ForwardDiff.value(sp(0.0, 0.0, 0.1, 0.0)[1]) ≈ 0.35`.
   Float64: with `ele64 = (; species_ref = TC_E, p_over_q_ref = tc_pq(200))`, `sp64 = TC_EXT.shifted_to_slice(f, ele64,
   0.25)`: `sp64.s0 === 0.25` and `sp64.t0 isa Float64`. If `shifted_to_slice` cannot take a `NamedTuple` `ele`
   (it reads only `ele.species_ref` and `ele.p_over_q_ref`), stop and report.
2. `@testset "W4 device_number_type"`: define at file top level (outside testsets)
   `struct NoFloat64Backend <: KernelAbstractions.Backend end` and
   `KernelAbstractions.supports_float64(::NoFloat64Backend) = false`. Test:
   `CSR.device_number_type(NoFloat64Backend(), Float64) === Float32`,
   `CSR.device_number_type(NoFloat64Backend(), Float32) === Float32`,
   `CSR.device_number_type(NoFloat64Backend(), ForwardDiff.Dual{Nothing,Float64,1}) === ForwardDiff.Dual{Nothing,Float64,1}`,
   `CSR.device_number_type(KernelAbstractions.CPU(), Float64) === Float64`. Also on the CPU: `csr_kick_pz` with
   `Float32` `pz` and `Float64` everything else returns an array with `eltype == Float64` (unchanged behaviour); use
   the `kick_test.jl`-style setup: `geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)` with `L3`, `c3` as 6.A.3,
   `z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)` (from `kernels_test.jl`), `params = CSRParams(n_bin = 40,
   ds_step = 0.05)`, kick at element 3, `s_kick = 0.1`, `gamma = 200.0`, species `Species("electron")`.
   `KernelAbstractions` must be imported in the test file (`import KernelAbstractions`).
3. `@testset "W5 gamma check"`: with that `geom`: `@test_throws ErrorException CSRKickPoint(geom, 3, 0.1, g)` for
   `g` in `(1.0, 0.5, NaN)`; same for `csr_kick_coef(Species("electron"), g, 0.01)` and with `bmad_compat = true`;
   TPS: `d = Descriptor(1, 1); x = @vars(d)`, `@test_throws ErrorException CSRKickPoint(geom, 3, 0.1, 1.0 + x[1])`
   (must raise, not abort the process); and `CSRKickPoint(geom, 3, 0.1, 1.0001)` runs.

**B.5 Review fixes (sub-stage B review, approved by the user 2026-10-25: W1, W2, W3, suggestion 1).**

B.5.1 (suggestion 1) `ext/CSRChainRulesCoreExt/CSRChainRulesCoreExt.jl`, `csr_kick_pz_pullback`: as the first statement
of the `else` branch (before `CSR.csr_kick_adjoint`), add

```julia
CSR.csr_float_type(backend) === Float64 ||
  error("csr_kick_pz pullback: needs a backend with Float64 (the CPU); the forward pass ran on $(typeof(backend))")
```

(`backend` is the local already defined in the `rrule`). The `AbstractZero` branch is unchanged. If
`CSR.csr_float_type` does not exist or does not return `Float64` for `KernelAbstractions.CPU()`, stop and report.

B.5.2 Tests, `test/review_fixes_test.jl`.

0. Remove the file's own `struct NoFloat64Backend ...` and its `KernelAbstractions.supports_float64` method (keep
   `import KernelAbstractions`); use the `NoFloat64Backend` (`<: CSR.KernelAbstractions.GPU`) defined in
   `test/binning_test.jl`, which `runtests.jl` includes earlier. Update the file's header comment to say so. Reason: the
   second definition has a different supertype; Julia 1.13.2 accepts the redefinition, Julia 1.10.12 does not (checked 2026-10-25,
   spec-tester: `invalid redefinition of constant Main.S`; Julia 1.12 not checked). For the one-file command (Section 10), add `include("binning_test.jl")` before the file.

All TPS use moves to one new testset, the last inside
`@testset "Review fixes 2026-10-18"`, named `"TPS cases"`, preceded by the comment
`# Keep last: Descriptor(...) changes GTPSA.desc_current.` Testsets added by sub-stages C and D go before it.

a. `"W3 shifted_to_slice"` (W1 of the review): after the existing Dual assertions add
   `bg = TC_EXT.R_to_beta_gamma(TC_E, tc_pq(200)); beta0 = bg / sqrt(1 + bg^2)` and
   `@test ForwardDiff.value(sp.t0) ≈ 0.25 / (beta0 * TC_EXT.C_LIGHT)`,
   `@test ForwardDiff.partials(sp.t0)[1] != 0`, `@test ForwardDiff.partials(sp.s0)[1] == 0`; after the Float64
   assertions add `@test sp64.t0 ≈ 0.25 / (beta0 * TC_EXT.C_LIGHT)`.
b. `"W4 device_number_type"` (W2): replace the four `===` assertions by the same calls wrapped as
   `@test (@inferred CSR.device_number_type(...)) === <expected>` (same expected values). After the `csr_kick_pz`
   assertion add `y, _ = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, 3, 0.1, 200.0, Species("electron"),
   params)`, `@test eltype(y) == Float64`, and `@test maximum(abs, y .- pz_new) <= 1e-6 * maximum(abs, pz_new)`
   (tolerance, not `==`: the threaded CPU deposit may sum in a different order; `pz` is zero, so `pz_new` is the
   kick). Requires `ChainRulesCore` in the test namespace (loaded by `runtests.jl`); if it is not, stop and report.
c. `"W5 gamma check"` (W3): replace every `@test_throws ErrorException <call>` by `@test_throws "gamma must be > 1"
   <call>` (string form; Julia ≥ 1.8). Remove the TPS lines (`Descriptor`, `@vars`, the TPS `@test_throws`) from this
   testset. Add `@test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1, ForwardDiff.Dual(1.0, 1.0))`,
   `@test CSRKickPoint(geom, 3, 0.1, ForwardDiff.Dual(200.0, 1.0)) isa CSRKickPoint`, and
   `@test csr_kick_coef(electron, ForwardDiff.Dual(200.0, 1.0), 0.01) isa ForwardDiff.Dual`.
d. `"TPS cases"`: `d = Descriptor(1, 1); x = @vars(d)`; `geom` as in W5; then
   - W5: `@test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1, 1.0 + x[1])` (moved) and
     `@test CSRKickPoint(geom, 3, 0.1, 200.0 + x[1]) isa CSRKickPoint`;
   - W3: `f(x, y, s, t, pp = nothing) = (s, t)`, `spt = TC_EXT.shifted_to_slice(f, (; species_ref = TC_E,
     p_over_q_ref = tc_pq(200) + x[1]), 0.25)`, `@test typeof(spt.s0) == typeof(spt.t0)`, `@test scalar(spt.s0) ==
     0.25` (GTPSA `scalar`).
   If `shifted_to_slice` throws for the TPS `p_over_q_ref` (e.g. no `promote` rule), or a TPS γ > 1 is rejected,
   stop and report (it is a B.1/B.3 defect for the user).

B.5.3 Metal script `diagnostics/stage04_metal_kick_pz.jl`: add a check that calling the `rrule` pullback on Metal
with a nonzero cotangent (`pb(MtlArray(ones(Float32, length(pz))))`) raises an `ErrorException` whose message
contains `"needs a backend with Float64"`. Not part of the suite; the tester runs the script.

### 6.C Docs and entry checks (sub-stage C)

**C.1 (S1) `slice_element` docstring** (`src/interface.jl`). In the "RF:" bullet replace the sentence
"`SaganCavity` tracking is not supported." by "A `SaganCavity` element gives a CSR.jl `SaganSlice` instead, with the
cavity constants of the whole element; the slices compose to the whole cavity to round-off."

**C.2 (S1) `docs/src/validation.md`**: the sentence "`test/runtests.jl` (about 1000 checks, run on 1 and 4 threads)"
becomes "(about 2500 checks, run on 1 and 4 threads)". The test-file table gets one row per file that `runtests.jl`
includes, in that order (existing rows kept where the file exists; rows may group files as the existing
`track_csr_test.jl`, `lcavity_test.jl` row does). Add rows for `ad_geometry_test.jl`, `ad_rrule_test.jl`,
`tps_collective_test.jl`, `track_tps_collective_test.jl`, `type_stability_test.jl`, `review_fixes_test.jl`, each with a
one-sentence description taken from the file's header comment or testset names. Do not change other sections of the
page. **Clarified (user, 2026-10-25):** rows follow the `runtests.jl` include order strictly; a grouped row may only
join files that are adjacent in `runtests.jl` (e.g. `track_csr_test.jl`, `lcavity_test.jl`). Split the existing
`tps_test.jl, track_tps_test.jl` row into two rows (reuse its description, divided between them), and reorder all rows
to the `runtests.jl` order. Keep existing descriptions otherwise unchanged.

**C.3 (S2) Entry checks in `track_csr!`** (`ext/CSRBeamTrackingBeamlinesExt/track.jl`, `track_csr_line!`): as its first
statements (before `n = length(line)`):

```julia
params.ds_step > 0 || error("track_csr!: params.ds_step must be > 0 (got $(params.ds_step))")
isnothing(centroid_start) || length(centroid_start) == 6 ||
  error("track_csr!: centroid_start must have length 6 (got $(length(centroid_start)))")
```

In the `track_csr!` docstring change "`params::CSRParams` (`ds_step` must be set)." to "`params::CSRParams`
(`ds_step > 0` required; error otherwise)." and "`centroid_start`: 6-vector" stays (it is now checked).

**C.4 (S8) CPU only.** Add to the `track_csr!` docstring, after the "Macroparticle charge" paragraph: "Runs on the CPU:
the bunch's arrays must be CPU arrays (GPU bunches are not supported). Each kick uses `csr_kick!`, which also runs on
GPU backends when called directly."

**C.5 Tests** (`test/review_fixes_test.jl`, `@testset "S2 track_csr! entry checks"`): `pq = tc_pq(200)`,
`bl = Beamline([Drift(L = 0.3), SBend(L = 0.5, g_ref = 0.4, Kn0 = 0.4)]; p_over_q_ref = pq, species_ref = TC_E)`,
`b = tc_line_bunch(1e-3, 1e-9, 101, pq)`, `v0 = copy(b.coords.v)`:
- `@test_throws ErrorException track_csr!(b, bl; params = CSRParams(n_bin = 40))` (ds_step 0) and
  `b.coords.v == v0` afterwards.
- `@test_throws ErrorException track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.05), centroid_start =
  zeros(5))` and `b.coords.v == v0`.
- `track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.05), centroid_start = zeros(6))` runs.
- The error messages contain `"ds_step"` and `"centroid_start"` respectively (`@test_throws` with a check on
  `sprint(showerror, err)`, or `try`/`catch`). The `ds_step` message also contains `"must be > 0"` (the C.3 text):
  without the C.3 check, `csr_n_step` (`src/utils/kick_setup.jl`) raises "csr_n_step: params.ds_step must be set"
  later in the loop, which contains `"ds_step"` but not `"must be > 0"`. Sensitivity (scratch, `@eval`): with the C.3
  checks removed, the `"ds_step"` assertions pass (the later error matches) and, among the `ds_step` assertions, only the `"must be > 0"` check fails. The `centroid_start` assertions also fail (`DimensionMismatch` from `centroid_bunch`, observed 2026-10-25: the `@test_throws ErrorException` and the message check).

**C.6 Review fixes (sub-stage C review, approved by the user 2026-10-26: W1, W2, S1).**

1. (W1) `track_csr!` docstring (`ext/CSRBeamTrackingBeamlinesExt/track.jl`): "(`ds_step > 0` required; error
   otherwise)." → "(`ds_step > 0` required; error otherwise, for a non-empty line)." The checks stay in
   `track_csr_line!` (an empty `Beamline`/`Branch` returns before them).
2. (W2) Rewrap to ≤ 120 characters, text unchanged: the C.4 paragraph in the `track_csr!` docstring (break after
   "Each kick uses", so the second line starts with `` `csr_kick!` ``), and in `test/review_fixes_test.jl` the
   `centroid_start = zeros(5)` `@test_throws` line (break before `centroid_start = zeros(5))`, aligned under the
   first argument of `track_csr!`). Do not rewrap any other line.
3. (S1) `test/review_fixes_test.jl`, S2 testset: the positive case becomes
   `@test track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.05), centroid_start = zeros(6)) === b`
   (wrapped to ≤ 120 characters as in item 2).

### 6.D Cleanup (sub-stage D)

**D.1 (S3)** `src/kernels/adjoint.jl`: rename `kick_cell(zp, zc1, z_center, dz)` (definition, its docstring signature,
and the one call at the `kick_adjoint_particle!`-side line `i0, r1 = kick_cell(z[i], z_center[1], z_center, dz)`) to
`adjoint_kick_cell`. The 3-argument `kick_cell` in `src/kernels/kick.jl` is unchanged. Search `src/`, `ext/`,
`test/`, `docs/` for other uses of the 4-argument form and rename them too.

**D.2 (S4)** Docstrings only:
- `src/types.jl` (`CSRParams`, `all_bin_quadrature`): "Simpson in `u^(1/3)`, 8 nodes per bin" → "Simpson in
  `u^(1/3)`, 8 intervals (9 nodes) per bin".
- `src/tables.jl` (`csr_wake_table`): "Simpson in `u^(1/3)` with 8 nodes per bin" → "Simpson in `u^(1/3)` with 8
  intervals (9 nodes) per bin"; and "8 more source solves per bin" → "9 source solves per bin".
- `src/param_adjoint.jl` (`hat_nodes`): signature line `-> Vector{Tuple{Float64,Float64}}` →
  `-> Vector{Pair{Float64,Float64}}` (it returns `collect` of a `Dict{Float64,Float64}`).

**D.3 (S5)** `Project.toml` `[compat]`: `GTPSA = "1.5.6"` → `GTPSA = "~1.5.6"` (CSR.jl uses GTPSA internals
`getdesc`, `desc_current`; D46).

**D.4 (S6)**
- `Project.toml`: remove the `Adapt` line from `[deps]` and from `[compat]`. Verify with a search that `src/`, `ext/`,
  `test/` do not use `Adapt`.
- Delete the directories `diagnostics/base/` and `diagnostics/head/` (shell `rm -r`; they are snapshots of the
  package with CSR's UUID, not referenced by any script outside them).
- After the `Project.toml` change, re-resolve the test env and the docs env (Section 10).

**D.5 (S7)** `ext/CSRBeamTrackingBeamlinesExt/sagan.jl`: declare `SaganSlice` with `Base.@kwdef` (fields, order,
types and comments unchanged), and build it in `slice_sagan` with keyword arguments, one per field, with the same
values as today (e.g. `ele = s, tm, species, mass = float(mass), ...`). The positional default constructor remains
(used by `@kwdef`); no other code may construct `SaganSlice` positionally.

**D.6 (S9)** `geometry_active_indices` (`src/param_adjoint.jl`): remove the closure `visit`, which reassigns the
captured `lo`, `hi`, `up`, `down`. Replace with:

```julia
src_ix(kp, u) = s_source_calc(CSRKick1(geom, kp; dz_particles = u), geom, kp)[1].ix_ele_source
```

(local, captures only `geom`), and a new top-level helper in the same file, with a docstring:

```julia
"""
    widen_source_range(lo, hi, up, down, ix, n) -> (lo, hi, up, down)

Element range `lo:hi` and the upstream/downstream infinite-drift flags after a source search ended in element `ix`
(`0`: upstream of the lattice, `> n`: past its end), for `geometry_active_indices`.
"""
widen_source_range(lo, hi, up, down, ix, n) =
  (ix == 0 ? 1 : min(lo, ix), ix > n ? n : max(hi, ix), up || ix == 0, down || ix > n)
```

Each former `visit(kp_x, u)` becomes `lo, hi, up, down = widen_source_range(lo, hi, up, down, src_ix(kp_x, u), n)`,
in the same order. The rest of the function is unchanged.

**D.7 (S10)**
- `slice_sagan` (`sagan.jl`): the slice end passed to `SaganSlice` is `float(last ? L : s0 + L_slice)` instead of
  `float(s0 + L_slice)` (in the `vt` tuple). Docstring of `slice_sagan`: add "The last slice ends at `L` exactly."
- `csr_apply_kicks_reference!` (`src/reference.jl`): test the cell index before indexing. The loop body becomes

```julia
alive[i] || continue
zp = z[i]
i0 = trunc(Int, (zp - zc1) / dz) + 1   # Fortran int() truncates toward zero
(i0 < 1 || i0 >= n_bin) && error("csr_apply_kicks!: particle outside the bin range (internal error)")
r1 = (zp - bins.z_center[i0]) / dz
r0 = 1 - r1
(r1 < -0.01 || r1 > 1.01) && error("csr_apply_kicks!: particle outside the bin range (internal error)")
pz[i] += r0 * kick[i0] + r1 * kick[i0+1]
```

  If `trunc(Int, ...)` itself throws `InexactError` for a non-finite or huge `zp`, leave it (not in scope).

**D.9 Diagnostics include path** (user, 2026-10-25). The scripts moved from `planning/diagnostics/` to `diagnostics/`,
so `joinpath(@__DIR__, "..", "..", "test", "utils.jl")` points outside the repo. In each of these four files, the
`include` line for `test/utils.jl` becomes exactly `include(joinpath(@__DIR__, "..", "test", "utils.jl"))`:
`diagnostics/first_bin_vs_quadrature.jl`, `diagnostics/steady_state_convergence.jl`,
`diagnostics/transient_all_bin_quadrature.jl`, `diagnostics/transient_first_bin_integrand.jl`. In the first three
the line currently reads `include(joinpath("/Users/alexcoxe/.julia/dev/CSR.jl", "test", "utils.jl"))` (an uncommitted
working-tree edit); replace it the same way. Change nothing else in these files (the stale `planning/diagnostics/`
paths in run-instruction comments stay). **Do not run or test these scripts** (user's instruction); no test or
acceptance criterion covers D.9.

**D.10 (D.7 type, user 2026-10-26).** In `ext/CSRBeamTrackingBeamlinesExt/sagan.jl` add a top-level helper next to
`slice_sagan`:

```julia
"""
    slice_end(L, s0, L_slice, last)

End of a SaganCavity slice: `L` for the last slice, else `s0 + L_slice`, as one floating-point type (the promoted
type of `L` and `s0 + L_slice`) so the result is inferable for mixed `Float64`/`Dual`/TPS arguments.
"""
function slice_end(L, s0, L_slice, last::Bool)
  s1 = s0 + L_slice
  T = promote_type(typeof(L), typeof(s1))
  return float(last ? convert(T, L) : convert(T, s1))
end
```

and in `slice_sagan` replace `float(last ? L : s0 + L_slice)` by `slice_end(L, s0, L_slice, last)`. For `Float64`
arguments the result is bitwise the same as D.7. Tests (D.8, new testset `"D.10 slice_end types"`, before
`"TPS cases"`): `@test TC_EXT.slice_end(0.3, 0.2, 0.1, true) === 0.3`; `@test TC_EXT.slice_end(0.3, 0.2, 0.1, false)
=== 0.2 + 0.1`; `r = @inferred TC_EXT.slice_end(0.3, ForwardDiff.Dual(0.2, 1.0), 0.1, true)`, `@test r isa
ForwardDiff.Dual && ForwardDiff.value(r) == 0.3`; `r2 = @inferred TC_EXT.slice_end(ForwardDiff.Dual(0.3, 1.0), 0.2,
0.1, false)`, `@test r2 isa ForwardDiff.Dual && ForwardDiff.value(r2) == 0.2 + 0.1`. In `"TPS cases"` add
`@test (@inferred TC_EXT.slice_end(0.3, 0.2 + x[1], 0.1, true)) isa TPS` with `scalar(...) == 0.3`. If a name
`slice_end` already exists in the extension, stop and report.

**D.8 Tests** (`test/review_fixes_test.jl`):
1. `@testset "S10 last SaganCavity slice"`: `ele = lc_in_branch(lc_cavity(L = 0.3, tracking_method =
   SaganCavity()))`; `@test 0.2 + 0.1 != 0.3` (precondition); `sl = CSR.slice_element(ele, 0.2, 0.1, false, true)`:
   `@test sl.s1 == 0.3` and `sl.s1 == sl.L`; `sm = CSR.slice_element(ele, 0.2, 0.1, false, false)`: `@test sm.s1 ==
   0.2 + 0.1`. If `lc_cavity(L = 0.3, ...)` cannot be sliced (e.g. `L_active > L`), stop and report.
2. `@testset "S10 reference range test"`: `z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)`, `params =
   CSRParams(n_bin = 40)`, `bins = csr_bin_particles(z, q, alive, params)`, `kick = zeros(40)`; for
   `zf in (bins.z_center[end] + 10bins.dz, bins.z_center[1] - 10bins.dz)`:
   `@test_throws ErrorException CSR.csr_apply_kicks_reference!([0.0], [zf], [true], bins, kick)`.
3. `@testset "S7 SaganSlice keywords"`: `sl` from item 1 equals (`==` per field via
   `all(f -> getfield(sl, f) == getfield(sl2, f), fieldnames(typeof(sl)))`, skipping fields `ele` and `tm`) a
   `TC_EXT.SaganSlice(; (f => getfield(sl, f) for f in fieldnames(typeof(sl)))...)` rebuilt by keywords.
4. `@testset "S9 no boxed state"`: `geom` as 6.B.4 item 2, `params = CSRParams(n_bin = 40, ds_step = 0.05,
   n_shield_images = 2, beam_chamber_height = 0.02)`; `@inferred CSR.geometry_active_indices(geom, 3, 0.1, 200.0,
   params, 1e-4)` returns a `Vector{Int}`; and `CSR.widen_source_range(3, 3, false, false, 0, 4) == (1, 3, true,
   false)`, `CSR.widen_source_range(3, 3, false, false, 5, 4) == (3, 4, false, true)`,
   `CSR.widen_source_range(3, 3, false, false, 2, 4) == (2, 3, false, false)`.

## 7. Data formats

No change.

## 8. Numerical and physics requirements

No physics change: no equation, model, constant, approximation or numerical scheme is added or altered. W5 is input
validation (β = √(1 − 1/γ²) and the default 1/β² factor are already in the code; γ ≤ 1 had no defined result). S10
changes the last SaganCavity slice's end by at most one ulp of `L`. No `physics-design-sourcer` entry is needed.

Float64 results: bitwise unchanged except D.7 (lcavity tests keep their existing tolerances).

## 9. Acceptance criteria

Per sub-stage:

1. **A:** `type_stability_test.jl` passes, all five cases. Sensitivity (scratch, `@eval`, repo unchanged):
   a. A.4: with the old line `n == 0 && return s0` restored in `root_tangent`, TPS assertion 2 fails.
   b. W2: `code_warntype` (or `Base.code_typed`) of `image_charge_kick` for the Dual and TPS cases shows a
      `Union{Float64, …}` for `g_bend` with `g_bend = 0.0` restored, and none with the fix. Report only (`@inferred`
      cannot detect it: the return type is the same).
   c. W1: on Julia 1.13.2 restoring `acc = 0.0` does not change the return type (so `@inferred` cannot see it), but
      `code_warntype` of the keyword body of `path_excess` shows `acc::Union{Float64, …}` for Dual and TPS inputs with
      the old line and a concrete `acc` with the fix. Report only; no requirement.
   d. Julia 1.10 (user, 2026-10-24): repeat b and c (and report whether `acc` / `g_bend` / the return types are
      unions with the old lines) on Julia 1.10 in a separate env (Section 10). Report only. If Julia 1.10 is not
      installed (`julia +lts --version` does not report 1.10.x), report "blocked: no Julia 1.10" and do not install it.
      Also repeat A.5.1 (i) and run `type_stability_test.jl` on 1.10 (report only).
2. **B:** the B testsets pass. **Metal check** (outside the suite; separate scratch env with Metal.jl + CSR (dev) +
   AtomicAndPhysicalConstants + StaticArrays + ChainRulesCore, Section 10): with the 6.B.4 item 2 geometry and bunch on
   Metal (`Float32` `MtlArray`s for `pz`, `z`, `q`; `MtlArray(alive)`), `csr_kick_pz(...; pz_scale = 1.3)` returns an
   `MtlArray{Float32}` equal to `csr_kick!` on a copy of `pz` with the same arguments (`==` after `Array`), and
   `ChainRulesCore.rrule(csr_kick_pz, ...; pz_scale = 1.3)[1]` equals it too. Sensitivity (same script, `@eval`):
   with `CSR.device_number_type(backend, ::Type{T}) where {T} = T`, `csr_kick_pz` on Metal fails (report the error).
   If Metal is not available on this machine, report "blocked: no Metal" for this item only.
   **Amended (user, 2026-10-25):** "equal" and "equals" above mean, after `Array`,
   `maximum(abs, a .- b) <= 1e-6 * maximum(abs, pz_out .- pz_in)`, where `pz_out .- pz_in` is the `csr_kick!` kick. The
   exact `==` is not reproducible on Metal: Float32 atomic adds in the deposit make the summation order run-dependent
   (two identical `csr_kick!` runs differed by up to 7.3e-12 against a largest kick of 2.2e-5).
3. **C:** S2 testset passes; docs build (criterion 6).
4. **D:** D.8 testsets pass; `Project.toml` has no `Adapt`; `diagnostics/base`, `diagnostics/head` are gone;
   `@code_warntype` (scratch) of `geometry_active_indices` shows no `Core.Box`. Sensitivity: with the old
   `csr_apply_kicks_reference!` body, D.8 item 2 raises `BoundsError` (so the `@test_throws ErrorException` fails).
5. **Every sub-stage:** the full suite passes on `-t 4`, no failures or errors; report total and time. Previous total
   2535 (end of Stage 2c, after 6.M removed one assertion); report the new total and the number added by the
   sub-stage.
6. **Sub-stage D (final):** also the full suite on `-t 1`, and the docs build `julia --project=docs docs/make.jl`
   completes with no new warnings (existing: the `internals.md` size warning, "no deployment environment").
7. All pre-existing tests pass unmodified (only `runtests.jl` gains two `include`s).

## 10. Build and run

Test environment (scratch, rebuilt this session; create it if missing):
`ENV=/private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/0578c516-7599-4fb1-b993-8e86e5c6b452/scratchpad/env`.

```sh
julia --project=$ENV -e 'using Pkg;
  Pkg.develop(path = expanduser("~/.julia/dev/CSR.jl"));
  Pkg.add([PackageSpec(name = "BeamTracking", version = "0.8.7"), PackageSpec(name = "Beamlines", version = "0.11.0")]);
  Pkg.add(["Test", "StaticArrays", "AbstractFFTs", "ChainRulesCore", "ChainRulesTestUtils", "GTPSA",
           "AtomicAndPhysicalConstants", "KernelAbstractions", "Atomix", "ForwardDiff", "DelimitedFiles"]);
  Pkg.resolve()'
```

Use the **registered** BeamTracking 0.8.7 and Beamlines 0.11.0 (not `Pkg.develop`): the BeamTracking dev checkout is on
the PR #369 branch. Do not change the dev checkouts. After D.4: `julia --project=$ENV -e 'using Pkg; Pkg.resolve()'`
and `julia --project=docs -e 'using Pkg; Pkg.resolve()'` (docs `Manifest.toml` is untracked).

- Full suite, from `test/`: `julia --project=$ENV -t 4 -e 'include("runtests.jl")'` (~17–21 min); `-t 1` likewise.
- One file, from `test/`: `julia --project=$ENV -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking,
  Beamlines, ChainRulesCore, ForwardDiff, GTPSA; include("utils.jl"); include("kernels_test.jl");
  include("track_csr_test.jl"); include("lcavity_test.jl"); include("binning_test.jl"); include("<file>")'`.
- Docs, from the repo root: `julia --project=docs docs/make.jl`.
- Metal env (criterion 2): `$ENV_METAL=<scratchpad>/env_metal`, `Pkg.develop(path = ...CSR.jl)`, `Pkg.add(["Metal",
  "AtomicAndPhysicalConstants", "StaticArrays", "ChainRulesCore", "KernelAbstractions"])`. Script in the scratchpad;
  copy it to `diagnostics/stage04_metal_kick_pz.jl` when it passes.

- Julia 1.10 env (criterion 1d): `<scratchpad>/env110`, `julia +lts --project=<scratchpad>/env110 -e 'using Pkg;
  Pkg.develop(path = expanduser("~/.julia/dev/CSR.jl")); Pkg.add(["Test", "StaticArrays", "ForwardDiff", "GTPSA"])'`.
  If resolution fails on 1.10, report the error (do not change compat).
- Julia 1.10 env with the full test deps (for `review_fixes_test.jl`; user, 2026-10-25): `<scratchpad>/env110_full`,
  `julia +lts --project=<scratchpad>/env110_full -e 'using Pkg; Pkg.develop(path = expanduser("~/.julia/dev/CSR.jl"));
  Pkg.add([PackageSpec(name = "BeamTracking", version = "0.8.7"), PackageSpec(name = "Beamlines", version = "0.11.0")]);
  Pkg.add(["Test", "StaticArrays", "AbstractFFTs", "ChainRulesCore", "ChainRulesTestUtils", "GTPSA",
  "AtomicAndPhysicalConstants", "KernelAbstractions", "Atomix", "ForwardDiff", "DelimitedFiles"]); Pkg.resolve()'`.
  Known (pre-existing, not in this stage's scope): `lcavity_test.jl:171` fails on 1.10.12 (PLAN.md Section 4).

Scripts and logs in the scratchpad.

## 11. Implementation order

Four sub-stages, each handed to `spec-implementer` separately, tested, reviewed and committed on its own branch
(coordinator): A → `stage/csr/05-type-stability`, B → `stage/csr/06-correctness`, C → `stage/csr/07-docs-checks`,
D → `stage/csr/08-cleanup`, each branched from the previous. Within a sub-stage: code, docstrings/docs, then tests.

## 12. Open items

None affecting implementation.

## 13. Decisions (user, 2026-10-24)

| Item | Decision |
|---|---|
| Scope | All review groups (W1–W6, S1–S10), one spec in four sub-stages. |
| W4 | Fix now: `csr_kick_pz` and the `rrule` forward pass on backends without `Float64`; the pullback stays CPU-only (documented). |
| W6 types | Float64, Float32, `Dual`, nested `Dual`, TPS. |
| S6 | Remove `diagnostics/base/`, `diagnostics/head/` from git (kept in history); remove `Adapt` (re-add in Phase 8 if used). |
| Branches | New per-sub-stage branches 05–08 from `stage/csr/04-review-fixes`. |
| Coordinator's choices (not asked) | W5 check in `CSRKickPoint` and `csr_kick_coef` (every kick and `track_csr!` passes through them; `track_csr!` can still stop mid-line at an element where γ ≤ 1); S2 checks at the top of `track_csr_line!`; S3 name `adjoint_kick_cell`; S10 fix as D.7; tests against registered BeamTracking 0.8.7. |

## Spec changes

| Date | Change | Reason | Status |
|---|---|---|---|
| 2026-10-24 | 6.A.5.3 added (`path_excess` compat branch converted to the default branch's type); 6.A.5.2d compat assertion `isa AbstractFloat` → `isa Float64`; 5.2e outcome recorded (omitted: `Descriptor(1, 0)` has `MO = 1`); criterion 1e (iii). | spec-implementer stopped: both Float32-spline `@inferred` errored, `Union{Float32, Float64}` from the two return branches (`acc` itself concrete). | Approved by the user |
| 2026-10-24 | 6.A.5 added (Float32 `path_excess` accumulator; test file: header, dynamic-descriptor TPS case, counted `@inferred`, Float32 spline testset, order-0 TPS `root_tangent` testset); criterion 1e. | Code review of sub-stage A, warnings W1–W4. | Approved by the user |
| 2026-10-24 | 6.A.5.4 (criterion 1e (ii), (iii)): "fail" wording now also records the observed form, "Error" (`@inferred` throws). Section 6.A.5.4, observed-results paragraph added. | spec-tester sensitivity runs on Julia 1.13.2 (`type_stability_test.jl` under scratch `@eval`): (ii) 2 Error, 78 pass; (iii) 2 Error, 78 pass. Clarification only; no expectation changed. | Editorial clarification, applied (spec-tester) |
| 2026-10-24 | 6.A.5.4 (i): old expectation "the 'Float32 spline' default-mode assertion fails" replaced by "report only (not detectable by `@inferred`)". | spec-tester sensitivity (i) run: 80 pass, 0 fail. Default-mode return type stays `Float64` on Julia 1.13.2; the W1 union is internal to `acc` (`code_warntype`). The test file's own header already says W1 unions are not detected. This weakens a sensitivity expectation, so it is not applied. | Approved by the user (2026-10-24), applied |
| 2026-10-24 | Criterion 1d and Section 10: Julia 1.10 is `julia +lts` (1.10.12), not `julia +1.10`. | spec-tester: juliaup has 1.10.12 under channels `lts` and `1.10~aarch64`; `+1.10` is not a channel. | Applied (editorial; user asked to run 1d) |
| 2026-10-24 | 6.A.4 added (`root_tangent` early return converted to the promoted type); Section 5 row A. | spec-implementer stopped: TPS `s_source_calc` infers `Tuple{CSRKick1{TPS}, Union{Float64, TPS}}` (union from `root_source`); coordinator traced it to `root_order(::TPS)` being a runtime value (by reading, unverified). | Approved by the user |
| 2026-10-24 | 6.A.3 assertion 1: `isconcretetype(typeof(r))` → `isa promote_type(T, typeof(s_kick))`. Criterion 1: sensitivity replaced by 1a–1d (A.4 detection; W2 by `code_warntype`, report only; W1 not reproducible on 1.13.2; W1/W2 rerun on Julia 1.10). | spec-implementer: `isconcretetype` of a value's type is always true; restoring `acc = 0.0` or `g_bend = 0.0` does not change `@inferred` results on 1.13.2. | Approved by the user (1.10 check added by the user) |
| 2026-10-24 | 9.1c: "does not change inference" → "does not change the return type ... but `code_warntype` of the keyword body shows `acc::Union{...}` with the old line". | Criterion 1 sensitivity (sub-stage A test, coordinator): `code_warntype` of the `path_excess` keyword body (Julia 1.13.2) shows `acc::Union{Float64, Dual}` and `acc::Union{Float64, TPS64}` with `acc = 0.0`, concrete with the fix; return types are the same, so `@inferred` does not see it. The old wording was true only of the return type. | Applied (clarification; no requirement changed) |
| 2026-10-24 | Sub-stage B, criterion 2 (Metal check): proposed replacing the exact `==` comparison (csr_kick_pz and rrule output vs csr_kick! on a copy) by a tolerance comparison, `maximum(abs, a .- b) <= 1e-5 * maximum(abs, pz_out .- pz_in)`. | spec-tester Metal run (Apple M1 Pro, Metal 4, Julia 1.13.2): `==` fails by 1 ulp-scale noise on the last element. Two identical csr_kick! runs also differ (max 7.3e-12; max kick 2.2e-5), so `==` is not reproducible (atomic-add order). The sensitivity run (device_number_type = T) errors as expected. | Approved by the user (2026-10-25) with tolerance 1e-6 instead of 1e-5 |
| 2026-10-25 | 6.D.9 added (portable `include` path for `test/utils.jl` in four `diagnostics/*.jl` scripts; no testing); Section 5 row D. | Coordinator found the `@__DIR__/../..` paths broken after the move to `diagnostics/`, and uncommitted absolute-path edits in three of the files. | Requested by the user (not to be tested) |
| 2026-10-25 | 6.B.5 added: pullback error on backends without Float64 (B.5.1); stronger B tests (t0 values and partials, `@inferred device_number_type`, CPU rrule forward check, message-checking `@test_throws`, Dual/TPS γ > 1, TPS `shifted_to_slice`), TPS uses moved to a final `"TPS cases"` testset (B.5.2); duplicate `NoFloat64Backend` removed (B.5.2.0); Metal pullback-error check (B.5.3). | code-reviewer, sub-stage B review: W1, W2, W3, suggestion 1; B.5.2.0 found by the coordinator (conflicts with `binning_test.jl`'s definition). | Approved by the user. Declined (not selected): W4 (print Metal noise ratio), suggestion 2 (allocation helper), suggestion 3 (reject `Inf` γ). |
| 2026-10-25 | Section 10, one-file command: `include("binning_test.jl")` added before `include("<file>")` (old: `...include("lcavity_test.jl"); include("<file>")'`; new: `...include("lcavity_test.jl"); include("binning_test.jl"); include("<file>")'`). | 6.B.5.2.0 directs this; Section 10 was not updated. `NoFloat64Backend` is defined only in `binning_test.jl`. Used by the 1.13 and 1.10 one-file runs (34/34). | Applied (editorial, directed by 6.B.5.2.0) |
| 2026-10-25 | 6.B.5.2.0 reason: "Julia ≥ 1.12 accepts the redefinition, Julia 1.10 does not (coordinator, from memory; verify)" replaced by "Julia 1.13.2 accepts ..., Julia 1.10.12 does not (checked 2026-10-25 ...; Julia 1.12 not checked)". | Task 4a scratch script (`redef_S_110.jl`): Julia 1.10.12 `ERROR: invalid redefinition of constant Main.S`; Julia 1.13.2 accepts (supertype `B`). | Applied (clarification; verified; no requirement changed) |
| 2026-10-25 | PROPOSED — Section 10: add a Julia 1.10 env for the B tests. New line after the criterion 1d env: `julia +lts --project=<scratchpad>/env110_full -e 'using Pkg; Pkg.develop(path = expanduser("~/.julia/dev/CSR.jl")); Pkg.add([PackageSpec(name = "BeamTracking", version = "0.8.7"), PackageSpec(name = "Beamlines", version = "0.11.0")]); Pkg.add(["Test", "StaticArrays", "AbstractFFTs", "ChainRulesCore", "ChainRulesTestUtils", "GTPSA", "AtomicAndPhysicalConstants", "KernelAbstractions", "Atomix", "ForwardDiff", "DelimitedFiles"]); Pkg.resolve()'`. | The criterion 1d env (4 packages) cannot run `review_fixes_test.jl` (needs BeamTracking, Beamlines, the Section 10 deps). This env resolved and precompiled on 1.10.12 (2026-10-25). | Approved by the user (2026-10-25) |
| 2026-10-26 | 6.C.6 added: `track_csr!` docstring "for a non-empty line" (W1); rewrap of the C.4 paragraph and one test line (W2); positive S2 case tests `=== b` (S1). | code-reviewer, sub-stage C review. | Approved by the user. Declined (not selected): S2 (usage.md/CSRParams docstring wording), S3 (`centroid_start isa AbstractVector`), S4 (string `@test_throws`). |
| 2026-10-25 | 6.C.2 clarified: strict `runtests.jl` order, grouping only of adjacent files, `tps_test.jl`/`track_tps_test.jl` row split, rows reordered. "About 2500 checks" kept. | spec-implementer stopped: the existing grouped tps row joins non-adjacent files and the table was not in `runtests.jl` order. | Decided by the user |
| 2026-10-25 | 6.C.5: the `ds_step` error message must also contain `"must be > 0"` (the C.3 text); sensitivity expectation added to 6.C.5. | Sensitivity run (tester, scratch `@eval` removing the C.3 checks, Julia 1.13.2): the `"ds_step"` assertions still pass, because `csr_n_step` (`src/utils/kick_setup.jl`) raises "csr_n_step: params.ds_step must be set" later in `track_csr_line!`; the centroid_start assertions fail (DimensionMismatch). | Applied (added detail that follows from C.3's message text; the repo test does not check it yet, see the tester report) |
| 2026-10-25 | 6.C.5 sensitivity sentence: the `centroid_start` assertions are stated to fail too (observed `DimensionMismatch`); "only" is scoped to the `ds_step` assertions. | Sub-stage C tester run (scratch `@eval`, Julia 1.13.2, S2 testset as in `test/review_fixes_test.jl`): 4 pass, 3 fail (the `"must be > 0"` message; both `centroid_start` assertions). The `ds_step` assertions pass. The sentence as written implied only the `"must be > 0"` check fails. | Applied (clarification; observed result; no expectation weakened) |
| 2026-10-26 | PROPOSED — needs user approval: 6.D.7 `slice_sagan`: type the slice end as one type, e.g. `float(promote_type(typeof(L), typeof(s0 + L_slice))(last ? L : s0 + L_slice))`, instead of `float(last ? L : s0 + L_slice)`. The `vt` tuple last element is then concrete in every case. | Tester, sub-stage D (D.7 type check, scratch, Julia 1.13.2): with `L` a Dual and `s0`, `L_slice` Float64, `float(last ? L : s0 + L_slice)` infers `Union{Float64, Dual}` (`Base.return_types`); the proposed form infers `Dual`. Values and Float64 bits unchanged. In `slice_sagan` the union is hidden because `vt[8]` is already `Any` (`code_warntype`). The in-tree path (`csr_slice`) gives `s0`, `L_slice` the same type as `L`, so the mixed case needs a direct `slice_element` call with mismatched types. | Approved by the user (2026-10-26); written as 6.D.10 (helper `slice_end`, with tests) |
| 2026-10-26 | Section 6.D.4: `diagnostics/base` was a gitlink (nested repository at commit `9211f86`, not in CSR.jl's history), not tracked files; its deletion with `rm -r` removed that repository. | Found by the coordinator after D.4 ran. | User accepted the loss (2026-10-26); gitlink removal committed with Stage 08 |
