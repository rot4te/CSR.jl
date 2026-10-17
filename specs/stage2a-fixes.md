# Stage 2a fixes: TPS descriptor check, per-block test tolerances, `csr_kick!` with TPS `pz`

Spec date: 2026-10-16. Source: review of Stage 2a (branch `stage2-tps`, `2ad0ca8`), findings 1, 2, 4.
Amended 2026-10-16 (see "Spec changes" at the end).

## 1. Overview

Four changes to CSR.jl after the Stage 2a review:

- **A. Descriptor check (finding 1).** CSR.jl creates TPS values from a *type* (`zero(T)`, `KernelAbstractions.zeros(backend, T, n)`,
  `T(x)` / `SVector{3,T}(r)` conversions, `oftype(zi, ±Inf)`). GTPSA.jl 1.5.6 has two TPS modes:
  - static-descriptor TPS `TPS{Float64,D}` (`D` a `Descriptor`), which `@vars(d)` returns by default
    (`src/ctors.jl:130-151`): the descriptor is in the type, so values created from the type are in the right
    descriptor. **Not affected; not checked.**
  - dynamic-descriptor TPS `TPS{Float64,GTPSA.Dynamic}`, from `@vars(d, dynamic = true)`: values created from the type
    are in `GTPSA.desc_current` (docstring of `TPS`, `src/tps.jl:145-162`), not in the descriptor of the user's
    inputs. If they differ, GTPSA's C library is handed TPSs of two descriptors (MAD-NG `mad_tpsa_ops.c`:
    `ensure(IS_COMPAT(...), "incompatibles GTPSA (descriptors differ)")`), whose outcome (Julia error or process
    abort) is unknown.
  Fix: every public entry point listed in Section 6.A checks its TPS inputs first and raises a clear `ErrorException`
  when any dynamic-descriptor TPS among them is not in `GTPSA.desc_current`.
- **B. Per-block tolerances (finding 2).** In `test/tps_collective_test.jl` the Hessian and order-3 checks are
  normalized by the largest entry overall, so a small block can be wrong unnoticed. Replace them with per-block checks.
- **C. `csr_kick!` with TPS `pz` (finding 4).** Documented but untested. Add a test that it equals `csr_kick_pz`.
- **E. `pz` element type in `csr_kick!`.** `csr_kick!` with a hardware-float `pz` (e.g. `Vector{Float64}`) and TPS or
  `Dual` kick inputs fails deep in the kick kernel (`src/kernels/kick.jl:21`, `pz[i] += ...` stores a TPS/`Dual` into
  the `Float64` array: `MethodError: no method matching Float64(::TPS64{...})`; diagnosed 2026-10-16). Raise a clear
  error up front instead.

## 2. Out of scope

- No change to any physics, kernel, table, deposit or root-finding code.
- No check in `csr_apply_test_kicks!` or for `test_particles` / `test_bunch` (the frozen-wake path creates no TPS from a
  type and works today with test particles in a non-current descriptor; collective + test particles together is
  Stage 2b).
- No check in `track_csr!` (Stage 2c), `FloorPosition(...)`, `CSRElementInfo(...)`, `CSRKick1(...)`, kernels, or any
  other function not listed in 6.A and 6.E.
- Static-descriptor TPS (`TPS{Float64,D}`) and `ComplexTPS64` are not checked by 6.A (the check returns "no mismatch").
- No change to the `root_order` docstring (finding 6) or to `thread_safe_number` (finding 5).
- Do not edit `PLAN.md` (the coordinator does). Do not commit, push, or change branches.

## 3. Language and toolchain

- Julia 1.13.2 (juliaup `release`); `Project.toml` keeps `julia = "1.10"`.
- No new dependencies, no `Project.toml` changes. GTPSA.jl 1.5.6 (latest registered, checked 2026-10-16 in the General
  registry `G/GTPSA/Versions.toml`; also the version of `~/.julia/dev/GTPSA.jl`).
- Style: 2-space indent, sparse comments, a docstring for every new function (the docs build uses
  `checkdocs = :all`; `docs/src/internals.md` already autodocs `utils/backend.jl`). Match the surrounding code.

## 4. Dependencies (existing; use these, do not reimplement)

| Package | Version | Used for |
|---|---|---|
| GTPSA.jl | 1.5.6 (compat `GTPSA = "1.5.6"`, weakdep, extension `CSRGTPSAExt`) | `GTPSA.getdesc(t)` (internal, unexported; `src/utils.jl:1`: `getdesc(t::TPS{T,Dynamic}) = Descriptor(t.d)`), `GTPSA.desc_current` (internal global, `src/global.jl:8`), `GTPSA.Dynamic` (`src/tps.jl:1`), `Descriptor`, `@vars` (with `dynamic = true` for dynamic TPS), `scalar`, `GTPSA.jacobian`, `GTPSA.hessian`, `GTPSA.deriv` (tests). `Descriptor` is an immutable struct with one `Ptr` field and no custom `==`, so `==` compares the pointer. A `Descriptor(nv, mo)` call sets `GTPSA.desc_current` to the new descriptor (`src/descriptor.jl:18-22`). |
| ForwardDiff | as in the test env | Reference Jacobians/Hessians (existing tests); `ForwardDiff.Dual` in the 6.E test. |

## 5. Project layout (files to change; no new source files)

- `src/utils/backend.jl`: add `descriptor_mismatch`, `check_descriptors` (6.A).
- `ext/CSRGTPSAExt/CSRGTPSAExt.jl`: add the TPS method of `descriptor_mismatch` (6.A).
- `src/utils/floor.jl` (`planar_survey`), `src/geometry.jl` (both `CSRGeometry` outer methods taking `L, g` vectors,
  `CSRKickPoint(geom, ...)`), `src/binning.jl` (`csr_bin_particles`), `src/kick.jl` (`csr_kick!`, `csr_kick_pz`): add
  the check as the first statement, and the docstring sentences (6.A). `src/kick.jl` `csr_kick!`: also 6.E.
- `test/tps_collective_test.jl`: modify the two checks (6.B), add three testsets (6.C, 6.E, then the 6.A test).
- `docs/src/usage.md`: one paragraph (6.D).

## 6. Components

### 6.A Descriptor check

**Core (`src/utils/backend.jl`, after `cpu_workgroupsize`).** `types.jl` is included before `utils/backend.jl`
(`src/CSR.jl:22-25`), so the struct types and `number_type` are available.

```julia
"""
    descriptor_mismatch(x) -> Bool

`true` if `x` (a number, an array, or a `FloorPosition`, `CSRElementInfo`, `CSRGeometry`, `CSRKickPoint`) holds a
GTPSA TPS whose Descriptor is not GTPSA's current one. CSR.jl creates TPS constants from types (e.g. `zero(T)`),
which GTPSA puts in the current Descriptor, so mixed Descriptors would reach GTPSA's C library. `false` for all
other number types; CSRGTPSAExt adds the method for dynamic-descriptor TPS.
"""
descriptor_mismatch(x) = false
descriptor_mismatch(A::AbstractArray) = eltype(A) <: Real ? false : any(descriptor_mismatch, A)
descriptor_mismatch(f::FloorPosition) = !(number_type(f) <: Real) && (descriptor_mismatch(f.theta) || descriptor_mismatch(f.r))
descriptor_mismatch(e::CSRElementInfo) =
  !(number_type(e) <: Real) && any(i -> descriptor_mismatch(getfield(e, i)), 1:fieldcount(CSRElementInfo))
descriptor_mismatch(g::CSRGeometry) = !(number_type(g) <: Real) && (descriptor_mismatch(g.start) || descriptor_mismatch(g.ele))
descriptor_mismatch(kp::CSRKickPoint) =
  !(number_type(kp) <: Real) && any(i -> descriptor_mismatch(getfield(kp, i)), 1:fieldcount(CSRKickPoint))
```

(`CSRKickPoint`'s `ix_ele_kick::Int` field goes to the fallback and returns `false`. `g.ele` is a
`Vector{CSRElementInfo}` and goes through the array method. The `<: Real` short-circuits keep Float64/Dual calls from
walking the structures.)

```julia
"""
    check_descriptors(fname; args...)

Raise an error naming function `fname` and the first keyword argument `name = value` for which
`descriptor_mismatch(value)` is `true`. No-op without GTPSA TPS inputs.
"""
function check_descriptors(fname; args...)
  for (name, x) in pairs(args)
    descriptor_mismatch(x) && error(...)   # message below
  end
  return nothing
end
```

Error message, exactly (with `$fname` and `$name` interpolated; `name` is the keyword's `Symbol`):

```
$fname: TPS in argument `$name` is not in GTPSA's current Descriptor (GTPSA.desc_current). CSR.jl creates TPS constants in the current Descriptor; use one Descriptor for all TPS inputs and make it current (the last Descriptor created is current).
```

**Extension (`ext/CSRGTPSAExt/CSRGTPSAExt.jl`).** Add, with a one-line comment in the style of the existing methods:

```julia
CSR.descriptor_mismatch(t::TPS{Float64,GTPSA.Dynamic}) = GTPSA.getdesc(t) != GTPSA.desc_current
```

No other TPS methods (static-descriptor and complex TPS fall back to `false`).

**Call sites.** Insert as the first statement of each function (before any other code, including the existing
argument checks), with the function name as written and the arguments in this order:

| Function (file) | Call |
|---|---|
| `planar_survey(L, g; start)` (`src/utils/floor.jl:56`) | `check_descriptors("planar_survey"; L, g, start)` |
| `CSRGeometry(L, g, centroid; ref_start, bmad_compat)` (`src/geometry.jl:11`) | `check_descriptors("CSRGeometry"; L, g, centroid, ref_start)` |
| `CSRGeometry(L, g, ref_exit, centroid; ref_start, bmad_compat)` (`src/geometry.jl:22`) | `check_descriptors("CSRGeometry"; L, g, ref_exit, centroid, ref_start)` |
| `CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source, kick_factor)` (`src/geometry.jl:74`) | `check_descriptors("CSRKickPoint"; geom, s_kick, gamma, y_source, kick_factor)` |
| `csr_bin_particles(z, q, alive, params)` (`src/binning.jl:17`) | `check_descriptors("csr_bin_particles"; z, q)` |
| `csr_kick!(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor, pz_scale, test_particles)` (`src/kick.jl:113`) | `check_descriptors("csr_kick!"; pz, z, q, geom, s_kick, gamma, kick_factor, pz_scale)` (not `test_particles`) |
| `csr_kick_pz(...)` (`src/kick.jl:138`) | `check_descriptors("csr_kick_pz"; pz, z, q, geom, s_kick, gamma, kick_factor, pz_scale)` |

The inner `CSRKickPoint(ix_ele_kick::Integer, ...)` constructor in `types.jl` is not changed.

**Docstrings.** Append to the docstrings of `csr_kick!` and `csr_kick_pz` (as the last paragraph) this sentence:
"Dynamic-descriptor TPS arguments (`@vars(d, dynamic = true)`) must all be in GTPSA's current Descriptor (error
otherwise; test particles are not checked); static-descriptor TPS (the `@vars` default) carry their Descriptor in the
type and are not checked." Append to the docstrings of the first `CSRGeometry` method and of `planar_survey`:
"Dynamic-descriptor TPS arguments must be in GTPSA's current Descriptor."

### 6.B Per-block tolerances (`test/tps_collective_test.jl`)

Order-2 testset (originally line 59, `@test maximum(abs, Ht .- Hf) <= 1e-11 * maximum(abs, Hf)`): replace that line
by, with `Hmax = maximum(abs, Hf)`, for every knob pair `1 ≤ a ≤ b ≤ 6` (21 blocks):

```
ref = maximum(abs, Hf[:, a, b]);  err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
@test err <= 1e-10 * ref + 1e-14 * Hmax
```

Order-3 testset (originally line 73, `@test maximum(abs, T3t .- T3f) <= 1e-10 * maximum(abs, T3f)`): replace it by,
with `Tmax = maximum(abs, T3f)`, for every `1 ≤ a ≤ b ≤ c ≤ 2` (4 blocks):

```
ref = maximum(abs, T3f[:, a, b, c]);  err = maximum(abs, T3t[:, a, b, c] .- T3f[:, a, b, c])
@test err <= 1e-9 * ref + 1e-13 * Tmax
```

Keep the `@test any(!iszero, ...)` lines and all other existing checks unchanged. Use a plain loop with `@test`
inside (no nested `@testset` per block). These testsets keep the static `@vars(d)`.

### 6.C `csr_kick!` with TPS `pz` (new testset in `test/tps_collective_test.jl`)

Add after the "Order 3" testset, inside the outer `@testset "Collective TPS"`:

```
@testset "csr_kick! with TPS pz"
```

- `p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 3e-3)`.
- `d = Descriptor(2, 2)`; `x = @vars(d)` (static, as in the other Stage 2a testsets). Knobs: element 2's `g` =
  `1.5 + x[1]`, charge scale `qx = q .* (1 + x[2])`. `geom = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)`; kick
  element 2, `s_kick = 0.17`, `gamma = 200.0`, species `electron`, params `p`; `z`, `alive`, `pz0`, `n` are the
  file's existing variables.
- `ref = csr_kick_pz(pz0, z, qx, alive, geom, 2, 0.17, 200.0, electron, p)`.
- `pz_t = [zero(x[1]) for _ in 1:n]`; `csr_kick!(pz_t, z, qx, alive, geom, 2, 0.17, 200.0, electron, p)`.
- Tests, all exact (`==`): `scalar.(pz_t) == scalar.(ref)`; `GTPSA.jacobian(pz_t) == GTPSA.jacobian(ref)`;
  `all(GTPSA.hessian(pz_t[i]) == GTPSA.hessian(ref[i]) for i in 1:n)`; plus `any(!iszero, GTPSA.jacobian(ref))`.
  If they are not exactly equal, report the measured difference; do not switch to a tolerance without approval.

### 6.E `pz` element type check in `csr_kick!` (`src/kick.jl`)

In `csr_kick!`, immediately after the 6.A `check_descriptors` line (so a descriptor mismatch is reported first) and
before `csr_binned_kick`:

```julia
Tk = promote_type(eltype(z), eltype(q), number_type(geom), typeof(s_kick), typeof(gamma), typeof(kick_factor),
                  typeof(pz_scale))
eltype(pz) <: AbstractFloat && !(Tk <: AbstractFloat) && error(...)   # message below
```

Error message, exactly (interpolating `$(eltype(pz))`, `$Tk` and `$(promote_type(eltype(pz), Tk))`):

```
csr_kick!: pz has element type $(eltype(pz)), which cannot hold the kick in number type $Tk; use csr_kick_pz, or a pz with element type $(promote_type(eltype(pz), Tk)).
```

Rationale: hardware/arbitrary-precision float `pz` (`Float32`, `Float64`, `BigFloat`) with float inputs keeps
working as today (e.g. `Float32` `pz` with `Float64` inputs converts on store); `Dual` and TPS are not
`AbstractFloat`, and a float array cannot store them. `test_particles` are not part of `Tk`. `csr_kick_pz` is not
changed (it allocates the promoted type).

Docstring: in `csr_kick!`'s argument list, extend the `pz`, `z` bullet with: "`pz` must hold the kick's number type:
with `Dual` or TPS inputs, a `pz` of that element type, or use `csr_kick_pz`."

**Test** (new testset `@testset "csr_kick! pz element type"` in `test/tps_collective_test.jl`, after 6.C and before
the 6.A test):

- `p = CSRParams(n_bin = 40, ds_step = 0.1)`, `geom0 = CSRGeometry(L3, g3, c3)`, `d = Descriptor(2, 2)`,
  `x = @vars(d)` (static).
- Must throw `ErrorException` with a message containing `"cannot hold the kick"`:
  - `csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, electron, p)`;
  - `csr_kick!(zeros(n), z, q .* (1 + x[2]), alive, geom0, 2, 0.17, 200.0, electron, p)`;
  - `csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17, ForwardDiff.Dual{Nothing}(200.0, 1.0), electron, p)`.
- Must not throw, and return a `CSRStepResult`:
  - `csr_kick!(zeros(Float32, n), z, q, alive, geom0, 2, 0.17, 200.0, electron, p)`;
  - `csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17, 200.0, electron, p)`.

### 6.A test (new testset, last in the outer `@testset "Collective TPS"`, after 6.E's test)

```
@testset "Descriptor check"
```

Must be the last testset in the file (it leaves `GTPSA.desc_current` at a descriptor no later code uses; later test
files create their own descriptors, e.g. `track_tps_test.jl:36`). All TPS in this testset are **dynamic** unless
stated otherwise.

1. `p = CSRParams(n_bin = 40, ds_step = 0.1)`. `geom0 = CSRGeometry(L3, g3, c3)` (Float64). `d1 = Descriptor(2, 2)`;
   `x = @vars(d1, dynamic = true)`; `xs = @vars(d1)` (static, for step 7).
2. While `d1` is current, build: `geom1 = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)`; `qx = q .* (1 + x[2])`;
   frozen-wake reference: `pzf = [x[1] + 0.0]`, `zf = [0.0 + x[2]]` (z = 0 lies inside the bunch's bin range: the
   `kernel_test_bunch` particles span about ±6 mm); then
   `csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17, 200.0, electron, p; test_particles = (pzf, zf, [true]))`.
3. `d2 = Descriptor(3, 2)` (now current; a different shape, so a different descriptor).
4. Each of these must throw an `ErrorException` whose message contains `"GTPSA's current Descriptor"` and the
   argument name shown in backticks (e.g. `` "`q`" ``). Test with `@test_throws ErrorException f()` and, separately,
   `occursin` on the message; a helper `errmsg(f) = try f(); "" catch err; sprint(showerror, err) end` is fine.
   | Call | Argument name in the message |
   |---|---|
   | `csr_kick_pz(pz0, z, qx, alive, geom0, 2, 0.17, 200.0, electron, p)` | `q` |
   | `csr_kick_pz(pz0, z, q, alive, geom1, 2, 0.17, 200.0, electron, p)` | `geom` |
   | `csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, electron, p)` | `s_kick` |
   | `csr_bin_particles(z, qx, alive, p)` | `q` |
   | `CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)` | `g` |
   | `planar_survey(L3, [g3[1], 1.5 + x[1], g3[3]])` | `g` |
   | `CSRKickPoint(geom0, 2, 0.17, 200.0 + x[1])` | `gamma` |
5. Not an error (frozen-wake test particles are not checked): repeat step 2's `csr_kick!` call with fresh
   `pzf2 = [x[1] + 0.0]`, `zf2 = [0.0 + x[2]]` as test particles, and test `scalar(pzf2[1]) == scalar(pzf[1])` and
   `GTPSA.jacobian(pzf2) == GTPSA.jacobian(pzf)`.
6. Not an error (all inputs in the current descriptor): `y = @vars(d2, dynamic = true)`;
   `@test length(csr_kick_pz(pz0, z, q .* (1 + y[1]), alive, geom0, 2, 0.17, 200.0, electron, p)) == n`.
7. `@test !CSR.descriptor_mismatch(1.0)`, `@test !CSR.descriptor_mismatch(zeros(3))`,
   `@test !CSR.descriptor_mismatch(geom0)`, `@test CSR.descriptor_mismatch(x[1])`,
   `@test CSR.descriptor_mismatch(geom1)`, `@test !CSR.descriptor_mismatch(y[1])`, and the static exemption
   `@test !CSR.descriptor_mismatch(xs[1])`.

### 6.D Docs (`docs/src/usage.md`)

After the paragraph ending "so kernels with TPS numbers run on the calling thread." and before its code block, the
paragraph:

"CSR.jl creates TPS constants from the number type. Static-descriptor TPS (the `@vars(d)` default) carry their
Descriptor in the type, so this is safe. Dynamic-descriptor TPS (`@vars(d, dynamic = true)`) get GTPSA's current
Descriptor, the one created last: their arguments to `csr_kick!` and `csr_kick_pz` (and to `CSRGeometry`,
`planar_survey`, `CSRKickPoint`, `csr_bin_particles`) must be in it, or these functions raise an error. Use one
Descriptor for all knobs. Test particles are not checked."

## 7. Data formats

No change.

## 8. Numerical requirements

No change to numerics. Tolerances in 6.B; exact equality in 6.C and in 6.A step 5.

## 9. Acceptance criteria

1. Full suite passes on 1 thread and on 4 threads (`-t 1`, `-t 4`), no failures or errors. Previous total 1690;
   report the new total.
2. 6.B: every per-block check passes. Also report, per testset case, the worst `err / (1e-10 * ref + 1e-14 * Hmax)`
   (order 3: `err / (1e-9 * ref + 1e-13 * Tmax)`) and the smallest nonzero `ref / Hmax` (`ref / Tmax`). If a block
   fails, report its indices, `err`, `ref`, `Hmax`/`Tmax`; do not loosen the bounds without approval.
3. 6.B sensitivity: in a scratch script outside the repo, load CSR and GTPSA, apply the temporary mutation
   `CSR.root_order(t::GTPSA.TPS) = 1`, then run the order-2 and order-3 testsets' computations with the new per-block
   checks. They must fail. Report the number of failing blocks per case. The repo is not modified by this.
4. 6.C: all exact-equality tests pass.
5. 6.E: the three error cases throw with the message; the two float cases run.
6. 6.A test: every case in steps 4–7 behaves as specified, and the Julia process does not abort.
7. All pre-existing tests pass unmodified, except the two lines replaced in 6.B.
8. Docs build: `julia --project=docs docs/make.jl` completes with no new warnings (`checkdocs = :all`).

## 10. Build and run

Test environment (scratch, rebuild per session): an env in the scratchpad that `Pkg.develop`s
`~/.julia/dev/CSR.jl`, `~/.julia/dev/BeamTracking.jl`, `~/.julia/dev/Beamlines.jl` and `Pkg.add`s Test, StaticArrays,
AbstractFFTs, ChainRulesCore, ChainRulesTestUtils, GTPSA, AtomicAndPhysicalConstants, KernelAbstractions, Atomix,
ForwardDiff; then `Pkg.resolve()`. Built this session at
`/private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/527ad108-7ac8-4e74-b71d-2ccf3df02e7b/scratchpad/env`.

- Full suite, from `test/`: `julia --project=<env> -t 4 -e 'include("runtests.jl")'` and the same with `-t 1`
  (~5.5 min each).
- One file, from `test/`: `julia --project=<env> -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking,
  Beamlines, ChainRulesCore, ForwardDiff, GTPSA; include("utils.jl"); include("kernels_test.jl");
  include("tps_collective_test.jl")'` (`kernel_test_bunch` is defined in `kernels_test.jl:6`).
- Docs, from the repo root: `julia --project=docs docs/make.jl`.

## 11. Implementation order

One stage: 6.A (core, extension, call sites, docstrings), 6.E, 6.D, then the tests 6.B, 6.C, 6.E and the 6.A test.

## 12. Open items

None affecting implementation. Later stages: descriptor checks for test particles combined with a collective TPS kick
(2b) and for `track_csr!` (2c).

## Spec changes

| Date | Change | Reason | Status |
|---|---|---|---|
| 2026-10-16 | 6.A test uses `@vars(d, dynamic = true)`; static exemption test added (step 7); Section 1, docstrings and 6.D say only dynamic-descriptor TPS are checked. | `@vars(d)` returns static-descriptor TPS in GTPSA 1.5.6 (`src/ctors.jl:130-151`), which the check ignores by design; found by spec-implementer. | Approved by the user |
| 2026-10-16 | Section 2's "`CSRKickPoint` MethodError" item removed; 6.E added (clear error for a float `pz` with `Dual`/TPS kick inputs in `csr_kick!`, docstring, test). | Diagnosis: `CSRKickPoint` with a TPS `gamma` works; the `MethodError` came from `csr_kick!` storing a TPS kick into a `Vector{Float64}` `pz` (`src/kernels/kick.jl:21`); same for `Dual`s. | Approved by the user (clear error option) |
