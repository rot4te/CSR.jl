# Stage 04: fixes from the whole-codebase review (2026-10-18)

Spec date: 2026-10-22. Base: branch `stage/csr/04-review-fixes` (from `stage/csr/03-bt-pr369-check` `06fb544`, plus
`ba000a8`). Findings: `planning/PLAN.md` Section 9, "Whole-codebase review 2026-10-18" (W1–W6, S1–S10). Decisions:
Section 13 below; D51 (slicing boundary rule).

## 1. Overview

Fixes for the review's findings, in four groups, plus the D51 boundary test:

- **Type stability (W1, W2, W6).** `path_excess` and `image_charge_kick` start from a `Float64` literal and then mix in
  the number type (`Union` results for `Dual`/TPS); a new `@inferred` testset covers the core hot-path functions in
  Float64, Float32, `Dual`, nested `Dual` and TPS.
- **Correctness (W3, W4, W5).** `shifted_to_slice` builds `ShiftedPotential` from two values of different types
  (`MethodError` with a TPS/`Dual` reference); `csr_kick_pz` and its `rrule` allocate in `Float64` and pass `pz_scale`
  without `device_scalar` (fail on backends without Float64, e.g. Metal); γ ≤ 1 gives NaN (Float64) or a GTPSA abort
  (TPS) instead of an error.
- **Docs and entry checks (S1, S2).** Stale `slice_element` docstring and `docs/src/validation.md`; `track_csr!`
  checks `params.ds_step` only when it reaches the first CSR element (after tracking the elements before it) and does
  not check `length(centroid_start)`.
- **Cleanup (S3, S4, S5, S6, S7, S8, S9, S10).** Duplicate function name, wrong counts in docstrings, GTPSA compat,
  unused `Adapt`, `SaganSlice` keyword constructor, CPU-only `track_csr!` undocumented, boxed closure variables,
  reference kick indexing before its range test, last SaganCavity slice end.
- **D51 boundary test.** `slicing.jl` and `sagan.jl` may use only `CSR.primal`, `CSR.primal_max`, `CSR.primal_min`
  (plus the method definitions `CSR.slice_element`, `CSR.csr_sliceable`) from CSR.

## 2. Out of scope

- No physics change: every Float64 result of the suite stays the same (exact comparisons in the suite must still pass
  bitwise; tolerances unchanged). The only intended Float64 behaviour changes are the new errors (6.E, 6.G) and S10b
  (6.O), which may change the last slice of each CSR element (all element types) at the round-off level; tests that
  compare such results exactly, if any, are reported, not loosened.
- S6 snapshots `diagnostics/base/`, `diagnostics/head/`: kept (user). Do not touch `diagnostics/`.
- The BeamTracking PR #369 follow-up (`track_slice!(::SaganSlice)` bookkeeping, PLAN.md Section 1 item 14).
- The GPU pullback (`csr_kick_param_adjoint` w.r.t. geometry, `s_kick`, γ on a GPU backend), `csr_apply_test_kicks!`
  backends (PLAN.md Section 4, Phase 8), GPU wake tables.
- Float32 precision preservation in the core (the tables are Float64 on the CPU by design); W6 checks inference only.
- Deferred review items from earlier reviews (PLAN.md Section 4) unless listed here.
- Do not edit `planning/`. Do not run git.

## 3. Language and toolchain

- Julia 1.13.x (juliaup `release`); `Project.toml` keeps `julia = "1.10"`.
- Style: 2-space indent, sparse comments, BeamTracking-style docstrings; a docstring for every new function
  (`checkdocs = :all`). Match the surrounding code. Type stability is a primary requirement (PLAN.md Section 2): new
  code must infer for Float64, Float32, `Dual` and TPS.

## 4. Dependencies (existing)

| Package | Use |
|---|---|
| KernelAbstractions | `supports_float64`, `get_backend` (W4). Existing helpers `csr_float_type`, `device_scalar` in `src/utils/backend.jl`. |
| ForwardDiff | `Dual` (W6 tests). |
| GTPSA 1.5.6 | TPS (W6 tests); `Descriptor`, `@vars` as in `test/tps_collective_test.jl`. |
| Test | `@inferred`, `@test_throws`. |
| Metal.jl | Manual check of W4 only (not in the suite; PLAN.md Section 2, "Metal checks"). |

## 5. Files to change

`Project.toml`; `src/utils/spline.jl`, `src/kernels/image_charges.jl`, `src/geometry.jl`, `src/kick.jl`,
`src/utils/backend.jl`, `src/adjoint.jl` (only if needed for W4), `src/kernels/adjoint.jl`, `src/kernels/kick.jl`
(docstring references only), `src/types.jl`, `src/tables.jl`, `src/param_adjoint.jl`, `src/reference.jl`,
`src/interface.jl`; `ext/CSRChainRulesCoreExt/CSRChainRulesCoreExt.jl`; `ext/CSRBeamTrackingBeamlinesExt/slicing.jl`,
`sagan.jl`, `track.jl`; `docs/src/validation.md`, `docs/src/usage.md` (S8, W4 sentence if needed); tests: one new
file `test/inference_test.jl` (W6) included from `test/runtests.jl`, one new testset for D51 (in
`test/slicing_test.jl`), and new testsets in existing files as listed per component.

## 6. Components

### 6.A W1 `path_excess` accumulator (`src/utils/spline.jl:88`)
`acc = 0.0` then `acc += …` in the number type of `s0`, `s1`, `c`. Start the accumulator from the first Gauss–Legendre
term (`i = 1`) and loop `i = 2:4`, so no `zero(T)` is built (avoids the dynamic-descriptor `zero(T)` hazard, D46). The
Float64 result must be bitwise unchanged (same summation order: `((w1 f1 + w2 f2) + w3 f3) + w4 f4` equals
`(((0.0 + w1 f1) + w2 f2) + …)` since `0.0 + x == x`). Must infer for every number type in 6.F.

### 6.B W2 `image_charge_kick` `g_bend` (`src/kernels/image_charges.jl:19`)
The infinite-drift branch sets `g_bend = 0.0` while the other branch is in the geometry/work-struct number type. Make
both branches the same type (e.g. `zero` of the other branch's type, computed from types, not by evaluating a spline
outside the lattice). Also the `0.0` literals in `t_vec` and `B_vec` `SVector`s: keep them only if the result infers
for every type in 6.F (an `SVector(sin(θ), 0.0, cos(θ))` promotes, which is stable; change only if `@inferred` fails).
Float64 results bitwise unchanged.

### 6.C W3 `shifted_to_slice` (`ext/CSRBeamTrackingBeamlinesExt/slicing.jl:179`)
`ShiftedPotential(f, float(s0), s0 / (beta_0 * C_LIGHT))` fails when the two values differ in type (a TPS/`Dual`
`p_over_q_ref` with a Float64 `s0`). Promote the two values to one type before constructing (`promote`). Test (in
`test/track_tps_collective_test.jl` or `test/slicing_test.jl`): `slice_element` of an element with
`FourPotentialParams` and of one with `EMFieldParams`, in a Beamline whose `p_over_q_ref` is a `Dual` (and one with a
TPS), at `s0 > 0`: no error, and the slice's potential evaluated at a point equals the element's potential at the
shifted point (s + s0, t + t0) in value. Float64 path unchanged.

### 6.D W4 `csr_kick_pz` and its `rrule` on backends without Float64
- New helper in `src/utils/backend.jl`: `device_eltype(backend, T)`: `T` if `supports_float64(backend)` or `T` is not
  an `AbstractFloat`; otherwise `csr_float_type(backend)` (Float32). Docstring.
- `csr_kick_pz` (`src/kick.jl:212`): allocate `pz_new` with `device_eltype(get_backend(pz), <promoted type as now>)`.
  CPU results unchanged (the helper is the identity there).
- `rrule` (`ext/CSRChainRulesCoreExt`): the same allocation; pass
  `pz_scale = CSR.device_scalar(get_backend(pz), pz_scale)` to `csr_apply_kicks!` and to `csr_kick_adjoint`. The
  geometry/`s_kick`/γ part of the pullback stays CPU-only (out of scope); document in the `csr_kick_pz` docstring that
  on GPU backends the `rrule` supports the particle cotangents (`pz`, `z`, `q`) only, if that is what the Metal check
  finds; if the particle pullback fails on Metal too, document "CPU only" for the `rrule` and report it.
- Tests (suite): `device_eltype` on the existing mock `NoFloat64Backend` (`test/binning_test.jl`) and on `CPU()`:
  `Float64 → Float32` on the mock, identity for `Float32`, `Dual`, `Int` on the mock and for everything on `CPU()`.
- Manual check (spec-tester, not in the suite): in a Metal env (Metal.jl + dev CSR + AtomicAndPhysicalConstants,
  StaticArrays), `csr_kick_pz` on `MtlArray{Float32}` inputs with a Float64 geometry and default
  `kick_factor`/`pz_scale` returns an `MtlArray{Float32}` equal to `csr_kick!` on copies (exact) and to the CPU Float32
  run within 1e-5 relative (free space; the Stage C measurement was 2.7e-6); and the rrule's forward value equals
  `csr_kick_pz`. If Metal is unavailable on the machine, report "not verified" (do not fake it).

### 6.E W5 γ ≤ 1
- `CSRKickPoint(geom, ix, s_kick, gamma; ...)` (`src/geometry.jl:85`): after the descriptor check,
  `primal(gamma) > 1 || error("CSRKickPoint: gamma must be > 1 (gamma = $gamma)")`. This covers `csr_kick!`,
  `csr_kick_pz`, the rrule and the reference path; the error comes before any `pz` is changed (check by test).
- `csr_kick_coef` (`src/utils/kick_setup.jl:13`): the same check (it is exported and divides by `1 − 1/γ²`).
- `track_csr!`: before anything is tracked (in `track_csr_line!` before the centroid and bunch loops, or in the entry
  checks), for every element with `csr_on(ele) && csr_sliceable(ele; context)`, compute `csr_ref_energies` and require
  `primal(E_start) > mass` and `primal(E_end) > mass` (`mass = massof(species)`), else
  `error("track_csr!: reference energy at or below the rest mass in element $i (lcavity from rest is not supported)")`
  (wording free, must name the element index). This must run before `sqrt(E_start^2 - mass^2)` in
  `track_csr_element!` can be reached with a TPS.
- Tests: `CSRKickPoint` and `csr_kick_coef` with γ = 1.0 and 0.5 raise `ErrorException`; `csr_kick!` with γ = 1 leaves
  `pz` unchanged; `track_csr!` through a Branch whose SaganCavity starts at E_ref − dE_ref = rest mass raises before
  tracking (bunch coordinates and `t_ref` unchanged). A TPS γ = 1 + 0·knob raises an error, not a process abort.

### 6.F W6 inference tests (`test/inference_test.jl`, new)
`@inferred` (no `Union`, no `Any` result) for: `path_excess`, `ddz_calc_csr`, `image_charge_kick`, `s_source_calc`
(both the method without and with a `PrimalSearch`), `I_csr_calc`, `with_fields`, `CSRKickPoint`. Number types: Float64,
Float32 (inputs in Float32 where the API accepts them; promotion to Float64 is fine), `ForwardDiff.Dual`, nested `Dual`,
static-descriptor TPS (`@vars(Descriptor(n, 2))`). Geometry and kick points built as in
`test/tps_collective_test.jl` / `kernels_test.jl` (reuse their helpers if loaded; the file must also run alone after
`include("utils.jl")` and `include("kernels_test.jl")`, document that in a comment). Cases: a source inside the lattice
and one on each infinite drift (so the W2 branch is exercised), default and `bmad_compat`.
If `@inferred` fails for a function not covered by 6.A/6.B: fix it if the cause is a literal or accumulator of the
wrong type in that function (as 6.A/6.B) and list it in the report; otherwise stop and report (do not restructure).
Mutation check (report only): revert 6.A or 6.B and confirm the testset fails.

### 6.G S2 `track_csr!` entry checks (`ext/CSRBeamTrackingBeamlinesExt/track.jl`)
Before anything is tracked: if any element has `csr_on(ele) && csr_sliceable(ele; context)`, require
`params.ds_step > 0` (error naming `params.ds_step`); if `centroid_start !== nothing`, require
`length(centroid_start) == 6`. With all elements CSR off, `ds_step = 0` stays allowed (existing tests rely on
`CSRParams()` with `csr_on = ele -> false`; check). Tests: both errors leave the bunch unchanged (coordinates, state,
`t_ref`), with a non-CSR element before the first CSR element in the line.

### 6.H S1 docs
- `slice_element` docstring (`src/interface.jl:25-26`): remove "`SaganCavity` tracking is not supported"; say that a
  SaganCavity element returns a SaganCavity slice object tracked by the extension (not a `LineElement`), per
  `slice_sagan`. Check the return-type line of the signature (`-> LineElement`) against `slice_sagan` and make it true.
- `docs/src/validation.md`: test list matches `test/` (add `ad_geometry_test.jl`, `ad_rrule_test.jl`,
  `tps_collective_test.jl`, `track_tps_collective_test.jl`, `inference_test.jl`, one line each, from the files'
  testsets); replace "about 1000 checks" with the count from the final suite run (rounded, "about N").

### 6.I S3 duplicate `kick_cell` (`src/kernels/adjoint.jl:22`)
Rename the adjoint's 4-argument `kick_cell(zp, zc1, z_center, dz)` to `adjoint_kick_cell` (docstring and call site at
line 34). The kick kernel's `kick_cell(zp, z_center, dz)` keeps its name.

### 6.J S4 docstring counts
`src/types.jl:208`, `src/tables.jl:21`: "8 nodes per bin" → "8 intervals (9 nodes) per bin" (`I_csr_quadrature` with
`n = 8` evaluates `j = 0:n`). `hat_nodes` docstring (`src/param_adjoint.jl:385`): return type
`Vector{Pair{Float64,Float64}}` (it returns `collect(::Dict)`). Grep `src/`, `docs/src/` for other "8 nodes".

### 6.K S5 GTPSA compat
`Project.toml` `[compat]`: `GTPSA = "~1.5.6"` (CSR.jl uses GTPSA internals `getdesc`, `desc_current`; D46).

### 6.L S6 `Adapt`
Remove `Adapt` from `[deps]` and `[compat]` (unused in `src/`, `ext/`, `test/`, `docs/`; checked by grep 2026-10-22).
Check that `docs/Project.toml` and the test env do not need it.

### 6.M S7 `SaganSlice` keyword constructor (`sagan.jl`)
`@kwdef` on `SaganSlice` (no defaults), and `slice_sagan` constructs it with keywords. Type parameters as now
(`SaganSlice{E,TM,M,K,TE,T}`), field types inferred as now; `@inferred slice_element(...)` on a SaganCavity in
Float64 and with a TPS knob as before (`sagan_exit_P0c` test still passes). No behaviour change.

### 6.N S8 `track_csr!` CPU only
`track_csr!` docstring (method docstring in `track.jl`) and `docs/src/usage.md`: one sentence that `track_csr!` runs
on the CPU (bunches with GPU arrays are not supported); `csr_kick!` itself runs on GPU backends. No code change.

### 6.O S9, S10
- S9 `geometry_active_indices` (`src/param_adjoint.jl:76`): `lo`, `hi`, `up`, `down` are captured and reassigned in
  the closure `visit` (boxed). Make `visit` return the source index and update the four variables in the caller (or
  pass and return them). Same results (the sparse == dense tests in `ad_rrule_test.jl` cover it).
- S10a `csr_apply_kicks_reference!` (`src/reference.jl:77-82`): test `1 <= i0 < n_bin` before reading
  `bins.z_center[i0]`, so an out-of-range particle gives the function's error message instead of a `BoundsError`.
  Test: a particle far outside the range raises the `ErrorException`.
- S10b last SaganCavity slice: `csr_slice` passes `L_slice = n_step*ds − s0`, so the last slice ends at `n_step*ds`,
  which may differ from `L` by an ulp, and `SaganSlice.s1` is that value. For the last slice (`i_step == n_step`) use
  `L_slice = L − s0` so `s1 == L` exactly (for every element type, in `csr_slice`). Check that the D22 tests
  (`lcavity_test.jl`, slices == whole cavity) and `slicing_test.jl` still pass with their bounds; report any change in
  the measured differences.

### 6.P D51 boundary test (`test/slicing_test.jl`, new testset "Slicing uses only primal helpers from CSR (D51)")
Parse `ext/CSRBeamTrackingBeamlinesExt/slicing.jl` and `sagan.jl` with `Meta.parseall`, walk the expressions, and
collect (a) every `CSR.<name>` reference and (b) every bare symbol that is an exported name of CSR (`names(CSR)`, minus
`:CSR`). Pass if (a) ⊆ {`primal`, `primal_max`, `primal_min`, `slice_element`, `csr_sliceable`} and (b) is empty.
Docstrings are strings and are not collected. Mutation check (report only): adding a `CSR.csr_n_step` call to a scratch
copy makes it fail.

## 7. Acceptance criteria

1. Full suite passes on `-t 1` and `-t 4` (record counts and times). Every pre-existing exact comparison passes.
2. Each component's tests exist and fail without its fix (mutation check per component where stated; report which
   were run).
3. `@inferred` testset (6.F) passes for all listed functions and types.
4. Docs build clean except the two existing warnings (`internals.md` size, no deployment environment);
   `checkdocs = :all` satisfied (renamed/new functions documented).
5. Metal check (6.D) run and reported, or reported as not possible.
6. `Adapt` removed and the package, tests and docs load without it.
7. No changes outside Section 5's files; no git commands.

## 8. Build and run

Test environment as PLAN.md Section 2 ("Test environment"): a scratch env in the session scratchpad that
`Pkg.develop`s CSR.jl, BeamTracking.jl, Beamlines.jl from `~/.julia/dev` and adds Test, StaticArrays, AbstractFFTs,
ChainRulesCore, ChainRulesTestUtils, GTPSA, AtomicAndPhysicalConstants, KernelAbstractions, Atomix, ForwardDiff; run
`Pkg.resolve()` after `Project.toml` changes. Full suite from `test/`:
`julia --project=<env> -t 4 -e 'include("runtests.jl")'`, then `-t 1`. Docs: `julia --project=docs docs/make.jl`.

## 9. Implementation order

6.K, 6.L (Project.toml) → 6.A, 6.B, 6.F (inference) → 6.C → 6.D → 6.E, 6.G → 6.I, 6.J, 6.O → 6.M → 6.H, 6.N → 6.P.

## 10. Open items

- Whether the rrule's particle pullback runs on Metal (6.D): found by the Metal check.

## 11. Decisions (user, 2026-10-22)

- All review groups in scope (2026-10-18).
- S6: keep `diagnostics/base/` and `diagnostics/head/`; remove `Adapt`.
- W4: fix now (not only document).
- W5, S2: clear errors up front; no other behaviour changes.
- D51: boundary test for the slicing files.

## Spec changes

| Date | Section | Change | Approved |
|---|---|---|---|
