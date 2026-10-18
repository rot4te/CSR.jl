# Stage 2c: `track_csr!` with a TPS bunch (knobs in elements, charge, reference energy, initial coordinates)

Spec date: 2026-10-17. Base: branch `stage/csr/01-stage2b` (`2fdbe77`). Decisions: D37 (Stage 2 variables), D45
(TPS kernels on one thread), D46 (descriptor checks), D49 (this stage; planning/PLAN.md Section 3), user decisions of
2026-10-17 (Section 13).

## 1. Overview

`track_csr!` must track a bunch whose coordinates, weights, `t_ref` and `p_over_q_ref` are GTPSA TPS (or ForwardDiff
`Dual`s) in "knobs", through lattices whose element parameters may also be TPS/`Dual` knobs, so that the final
coordinates are the collective map in the knobs (D37). The CSR core (`CSRGeometry`, `csr_kick!`, tables, deposit)
already accepts TPS (Stage 2a); this stage changes only the BeamTracking/Beamlines extension.

Probe (`diagnostics/stage2c_tps_params_probe.jl`, 2026-10-17): BeamTracking 0.8.7 with Beamlines 0.11.0 tracks TPS
and `Dual` knobs in `Kn1`, `g_ref`/`Kn0`, `L`, patch `dx`, SaganCavity `voltage`/`dE_ref`, `p_over_q_ref`, initial
coordinates and weights correctly (vs finite differences; `Dual` vs TPS ≤ 4.4e-16), **provided the `Bunch` is built
with `t_ref` and `p_over_q_ref` of the knob type** (BeamTracking assigns `bunch.t_ref = t_exit`,
`ext/BeamTrackingBeamlinesExt/unpack_bl.jl:348`). CSR.jl fails today in `slice_tracking_method` (`round` of a TPS),
`SaganSlice` (`Float64` fields) and `track_csr_line!` (`Float64` arrays).

Changes:
- **A. Slicing** (`slicing.jl`): branch and round on primal values.
- **B. SaganCavity slices** (`sagan.jl`): `SaganSlice` generic in its number type; window/overlap tests on primal
  values.
- **C. Track loop** (`track.jl`): geometry, centroid and energies in the promoted number type; the lcavity ramp for
  every SaganCavity (D49); entry checks (descriptors, reference/coordinate types).
- **D. Docs**: `track_csr!` docstring, `docs/src/usage.md`.
- **E. Tests**: new `test/track_tps_collective_test.jl`.
- **F. Zero-length chord in `CSRGeometry`** (amendment, 2026-10-17): a TPS geometry with a zero-length element
  (marker, zero-length patch) aborted the process in GTPSA (`atan(0, 0)`, `sqrt(0)`); guard in the core.
- **G. `L0²` in `upstream_drift_source`** (amendment, 2026-10-17): a kick on the lattice start aborted GTPSA in
  `sqrt(0)`; use the squared distance directly.
- **H. Test redesign** (amendment, 2026-10-17, 6.H): the FD check E.3 becomes a coarse check with the bin grid pinned
  by zero-weight ghost particles (AD holds the grid fixed by design, FD moved it); E.4 `rtol = 1e-9`; the
  "dE_ref knob at 0" lattice ends with the cavity (BeamTracking 0.8.7 bug after a SaganCavity, see 6.H.3).

## 2. Out of scope

- No change under `src/` (core) except 6.F (`src/geometry.jl`) and 6.G (`src/kernels/source_point.jl`), and none in `ext/CSRGTPSAExt/`, `ext/CSRChainRulesCoreExt/`, `ext/CSRAbstractFFTsExt/`.
- No change to physics, kick formula, tables, deposit, root finding.
- No upstream changes (BeamTracking, Beamlines, GTPSA). Known upstream limitation, documented only (6.D): BeamTracking's
  `rf_step_calc` (`ext/BeamTrackingBeamlinesExt/utils_bl.jl:257`) calls `round` on `rf_omega * L_act` when a
  SaganCavity has automatic `n_cells` (`n_cells < 0`); with a knob in the cavity `L` or RF frequency this fails for TPS.
- `advance_floor`'s series about angle 0 (`src/utils/floor.jl:28`): derivatives of order ≥ 3 in a `g` knob of an element
  with `g = 0` are truncated; documented only (6.D), not fixed (D49).
- No support for time-dependent reference energy (unchanged error). No GPU support for TPS/`Dual` bunches.
- Deferred review findings (PLAN.md Section 4), Phase 8 items.
- No change to `docs/src/validation.md`, `docs/src/decisions.md`, README.
- Do not edit `planning/PLAN.md` (the coordinator does). Do not run git (no commit, branch, stash, checkout).

## 3. Language and toolchain

- Julia 1.13.2 (juliaup `release`); `Project.toml` keeps `julia = "1.10"`. No `Project.toml` changes.
- Style: 2-space indent, sparse comments, a docstring for every new function (the docs build uses
  `checkdocs = :all` and includes `CSRBeamTrackingBeamlinesExt`, `docs/make.jl:5`, `docs/src/internals.md:93`). Match
  the surrounding code.

## 4. Dependencies (existing; use these, do not reimplement)

Versions checked 2026-10-17 against the local General registry (updated 2026-10-17): BeamTracking 0.8.7, Beamlines
0.11.0, GTPSA 1.5.6 are the latest registered; ForwardDiff 1.4.6 in the test env.

| Package | Version | Used for |
|---|---|---|
| BeamTracking | 0.8.7 (compat `"0.8.7"`) | `track!`, `Bunch`, `R_to_E`, `E_to_R`, `STATE_ALIVE`, internals already imported by the extension. `Bunch{B,T,C}` is parametric in `p_over_q_ref` (`B`) and `t_ref` (`T`) (`src/types.jl:39`). |
| Beamlines | 0.11.0 | `deval`, `isactive`, parameter groups. `PatchParams{T}` has one type for its 7 fields `dt, dx, dy, dz, dx_rot, dy_rot, dz_rot` (`src/patch.jl:1`). `isactive(::AbstractParams) = true` (`src/element.jl:12`). |
| CSR core | this repo | `CSR.primal` (`src/utils/backend.jl:117`; identity for plain numbers, scalar part for TPS via CSRGTPSAExt, value for `Dual`), `CSR.primal_max`, `CSR.primal_min` (`backend.jl:128-129`), `CSR.check_descriptors(fname; args...)` (`backend.jl:67`; arrays, numbers, `nothing` accepted; an `AbstractArray` with a non-`Real` eltype, including `Any`, is checked elementwise), `CSR.number_type` (`FloorPosition`), `FloorPosition`, `convert(::Type{FloorPosition{T}}, ::FloorPosition)` (`src/types.jl:20`). |
| GTPSA.jl | 1.5.6 | Tests, docs. `Descriptor(nv, mo)`, `@vars(d)` (static descriptor), `@vars(d, dynamic = true)`, `scalar`, `GTPSA.jacobian`, `GTPSA.hessian` (as in `specs/stage2b.md` Section 4). TPS comparisons (`<`, `>=`, `==` with numbers) compare scalar parts (`src/operators.jl:91-104`); `float(::TPS)` is the identity (`operators.jl:68`). `convert(TPS64{D}, x::Float64)` goes through Base's `convert(::Type{T}, x::Number) = T(x)`. |
| ForwardDiff | 1.4.6 (test env) | Reference Jacobians and nested-`Dual` second derivatives (tests). `Dual` `==`, `<`, `>=` order equal values by their partials (PLAN.md Section 6): every branch must compare `CSR.primal` values. |

## 5. Project layout (files to change or add)

- `src/geometry.jl`: 6.F.1; `test/tps_collective_test.jl`: 6.F.2 (one new testset).
- `src/kernels/source_point.jl`: 6.G.
- `ext/CSRBeamTrackingBeamlinesExt/slicing.jl`: 6.A.
- `ext/CSRBeamTrackingBeamlinesExt/sagan.jl`: 6.B.
- `ext/CSRBeamTrackingBeamlinesExt/track.jl`: 6.C, docstring 6.D.1.
- `docs/src/usage.md`: 6.D.2.
- New `test/track_tps_collective_test.jl`: 6.E. `test/runtests.jl`: add `include("track_tps_collective_test.jl")` as
  the last line inside `@testset "CSR.jl"` (after `include("track_tps_test.jl")`).

## 6. Components

Throughout, `primal`, `primal_max`, `primal_min` mean `CSR.primal`, `CSR.primal_max`, `CSR.primal_min` (qualify them
as `CSR.` in the extension, as the extension already does for `CSR.advance_floor`). For plain floats `primal` is the
identity, so every Float64 code path must produce bitwise the same results as before this stage (except 6.C.6 for
SaganCavity elements with `dE_ref = 0`, criterion 9.4, and 6.G, which changes upstream-drift sources at round-off).

### 6.A Slicing (`slicing.jl`)

1. `slice_tracking_method`: in both places (`AbstractSymplectic` and `RungeKutta` branches) replace
   `round(Int, tm.n_steps * frac)` by `round(Int, tm.n_steps * CSR.primal(frac))`.
2. `CSR.csr_sliceable`: `deval(ele.UniversalParams, context).L != 0` → `CSR.primal(deval(ele.UniversalParams, context).L) != 0`.
3. `CSR.slice_element`: `L > 0 || error(...)` → `CSR.primal(L) > 0 || error(...)` (message unchanged); the map-element
   condition `(first && last && L_slice == L)` → `(first && last && CSR.primal(L_slice) == CSR.primal(L))`.
4. `slice_alignment`: `!isactive(bp) || (bp.g_ref == 0 && bp.tilt_ref == 0)` →
   `!isactive(bp) || (CSR.primal(bp.g_ref) == 0 && CSR.primal(bp.tilt_ref) == 0)`.
5. No other change in `slicing.jl`.

### 6.B SaganCavity slices (`sagan.jl`)

1. **`SaganSlice` type.** Declare `struct SaganSlice{E,TM,M,K,T<:Number}` and give these 12 fields type `T` (instead
   of `Float64`): `E0, dE_ref, P0c0, p_over_q_ref, rf_omega, t_phi0, q_gradient, q_voltage, L_active, L, s0, s1`.
   `mass, q, a, n_cells` stay `Float64`; the other fields are unchanged. Update the docstring: "`T`: number type of
   the energy, RF and length fields (promoted from the cavity's parameters; TPS or `Dual` with knobs)".
2. **`slice_sagan`.**
   - Keep the existing computations, except: the check
     `L_active <= L * (1 + eps(L)) || error(...)` → `CSR.primal(L_active) <= CSR.primal(L) * (1 + eps(CSR.primal(L))) || error(...)`
     (message unchanged).
   - Let `vals = (float(E0), dE_ref, sqrt(E0^2 - mass^2), float(p_over_q_ref), float(rf_omega), float(t_phi0),
     q_gradient, float(q * rp.voltage), float(L_active), L, float(s0), float(s0 + L_slice))` with
     `q_gradient = CSR.primal(L_active) > 0 ? q * rp.voltage / L_active : 0.0` (the 12 values in field order, as the
     current constructor call computes them), `T = float(promote_type(map(typeof, vals)...))`, and pass
     `convert(T, v)` for each `v` in `vals` in the 12 field positions. Keep `float(mass)`, `float(q)`,
     `float(gyromagnetic_anomaly(species))`, `float(n_cells)` as now.
   - For Float64 inputs `T == Float64` and the values are identical to the current ones.
3. **Comparisons on primal values** (bodies otherwise unchanged):
   - `sagan_overlap(a, b, len, s0, s1) = (CSR.primal(a) >= CSR.primal(s0) && CSR.primal(b) <= CSR.primal(s1)) ? len : CSR.primal_max(zero(len), CSR.primal_min(b, s1) - CSR.primal_max(a, s0))`.
   - `sagan_in_window(p, s0, s1, last) = CSR.primal(p) >= CSR.primal(s0) && (CSR.primal(p) < CSR.primal(s1) || last)`.
   - `sagan_P0c_at`: `before(p) = CSR.primal(p) < CSR.primal(s) || (inclusive && CSR.primal(p) <= CSR.primal(s))`.
   - `sagan_segment!`: `len > 0 || return nothing` → `CSR.primal(len) > 0 || return nothing`; `if d > 0` →
     `if CSR.primal(d) > 0`.
   - `sagan_slice_kernel!`: each `L_active > 0` (three places) → `CSR.primal(L_active) > 0`. `n_cells == 0` stays
     (`Float64`).
   - Update the `sagan_in_window`/`sagan_overlap` docstrings: "decided on primal values (`Dual`s at exact boundary ties
     compare partials otherwise; D22 relies on exact ties)".
4. No other change in `sagan.jl` (`track_slice!`, `sagan_dt_ref`, `sagan_ends`, `sagan_exit_P0c` unchanged).

### 6.C Track loop (`track.jl`)

1. **`csr_ref_curvature(bp, ix)`** becomes:
   ```julia
   function csr_ref_curvature(bp, ix)
     isactive(bp) || return 0.0
     g = float(bp.g_ref)
     CSR.primal(g) == 0 && g isa AbstractFloat && return 0.0
     c, s = cos(bp.tilt_ref), sin(bp.tilt_ref)
     abs(CSR.primal(s)) < 1e-12 || error("track_csr!: element $ix bends out of the horizontal plane (tilt_ref = $(bp.tilt_ref))")
     return g * sign(CSR.primal(c))
   end
   ```
   (A TPS/`Dual` `g` with primal 0 keeps its derivative; plain floats as before.) Docstring: add "A TPS or `Dual`
   curvature with primal value 0 is returned as is (its knob derivatives are kept)."
2. **`bunch_centroid`**: `sum(wa) > 0 || error(...)` → `CSR.primal(sum(wa)) > 0 || error(...)`.
3. **`check_bunches!`**: `test_bunch.p_over_q_ref == bunch.p_over_q_ref ||` →
   `CSR.primal(test_bunch.p_over_q_ref) == CSR.primal(bunch.p_over_q_ref) ||` (message unchanged).
4. **New `csr_read_params(line, context)`** (with docstring) returns a `NamedTuple`
   `(; L, g_ref, patch, dE_ref, sagan_rf, p_over_q_ref)` of `Vector{Any}`s with the element parameters CSR.jl reads:
   - `L`: `float(deval(ele.UniversalParams, context).L)` of every element;
   - `g_ref`: `deval(ele.BendParams, context).g_ref` of every element whose `deval(ele.BendParams, context)` is
     `isactive`;
   - `patch`: the 7 fields `dt, dx, dy, dz, dx_rot, dy_rot, dz_rot` (in this order) of every element with an `isactive`
     `deval(ele.PatchParams, context)`;
   - `dE_ref`: `float(deval(ele.BeamlineParams, context).dE_ref)` of every element whose
     `deval(ele.UniversalParams, context).tracking_method isa SaganCavity` (as `csr_ref_energies`);
   - `sagan_rf`: the fields `rate, voltage, phi0` (in this order) of `deval(ele.RFParams, context)` of the same
     SaganCavity elements, when that `RFParams` `isactive` (`RFParams{T}` has one type for the three,
     Beamlines `src/rf.jl:18`). They enter the `SaganSlice` number type (6.B.2) and so the slices' `t_ref`;
   - `p_over_q_ref`: `ele.p_over_q_ref` of every element for which it `isa Number` (others skipped).
5. **New entry checks.** Both `CSR.track_csr!` methods become (the Branch method builds `line` first):
   ```julia
   function CSR.track_csr!(bunch::Bunch, bl::Beamline; test_bunch = nothing, centroid_start = nothing, kwargs...)
     isempty(bl.line) && return bunch
     check_track_inputs(bunch, test_bunch, bl.line, bl.context, centroid_start)
     check_bunches!(bunch, test_bunch, bl)
     check_ref_types(bunch, test_bunch, ref_param_type(bl.line, bl.context))
     return track_csr_line!(bunch, bl.line, bl.context; test_bunch, centroid_start, kwargs...)
   end

   function CSR.track_csr!(bunch::Bunch, br::Branch; test_bunch = nothing, centroid_start = nothing, kwargs...)
     bls = filter(bl -> !isempty(bl.line), br.beamlines)
     isempty(bls) && return bunch
     line = reduce(vcat, [bl.line for bl in bls])
     check_track_inputs(bunch, test_bunch, line, br.context, centroid_start)
     check_bunches!(bunch, test_bunch, first(bls))
     check_ref_types(bunch, test_bunch, ref_param_type(line, br.context))
     return track_csr_line!(bunch, line, br.context; test_bunch, centroid_start, kwargs...)
   end
   ```
   All checks run before any tracking, so an error leaves the bunch and test bunch coordinates unchanged.
   - **`check_track_inputs(bunch, test_bunch, line, context, centroid_start)`** (new, docstring): with
     `r = csr_read_params(line, context)` and `tb = test_bunch` (may be `nothing`), call
     ```julia
     CSR.check_descriptors("track_csr!"; coords = bunch.coords.v, weight = bunch.coords.weight, t_ref = bunch.t_ref,
                           p_over_q_ref = bunch.p_over_q_ref, centroid_start,
                           test_coords = tb isa Bunch ? tb.coords.v : nothing,
                           test_t_ref = tb isa Bunch ? tb.t_ref : nothing,
                           test_p_over_q_ref = tb isa Bunch ? tb.p_over_q_ref : nothing,
                           L = r.L, g_ref = r.g_ref, patch = r.patch, dE_ref = r.dE_ref, sagan_rf = r.sagan_rf,
                           ele_p_over_q_ref = r.p_over_q_ref)
     ```
     in exactly this keyword order (a `test_bunch` that is not a `Bunch` is left to `check_bunches!`'s existing error).
   - **`ref_param_type(line, context)`** (new, docstring): `promote_type(Float64, map(typeof, vcat(r.L, r.patch, r.dE_ref, r.sagan_rf, r.p_over_q_ref))...)`
     with `r = csr_read_params(line, context)` (the parameters that enter `t_ref` and the reference momentum; the
     SaganCavity RF parameters because they set the `SaganSlice` number type, whose `t_ref` the slices write back).
   - **`check_ref_types(bunch, test_bunch, Tp)`** (new, docstring): returns `nothing` without checking if
     `Tp <: AbstractFloat`. Otherwise, for `(name, b)` in `(("bunch", bunch), ("test_bunch", test_bunch))`, skipping
     `nothing`, check in this order, each with "`F` holds `Tp`" meaning `promote_type(F, Tp) == F`:
     1. `F = typeof(b.t_ref)`; else `error("track_csr!: $name.t_ref has type $F, which cannot hold the element parameters' type $Tp (L, p_over_q_ref, dE_ref, patch or SaganCavity RF parameters). Build the Bunch with t_ref, p_over_q_ref and coordinates of a type that holds $Tp, e.g. t_ref = zero(k) for a knob k.")`;
     2. `F = typeof(b.p_over_q_ref)`; else the same message with `$name.p_over_q_ref has type $F` in place of
        `$name.t_ref has type $F`;
     3. `F = eltype(b.coords.v)`; else the same message with `$name coordinates have element type $F` in place of
        `$name.t_ref has type $F`.
6. **`csr_ref_energies(ele, species, p_over_q_ref, context)`** returns `(E_start, E_end, ramp)`: `(E_end, E_end, false)`
   if the tracking method is not `SaganCavity`, else `(E_end - float(dE_ref), E_end, true)` (D49: the ramp applies to
   every SaganCavity, so a `dE_ref` knob at 0 keeps its derivative; with a plain `dE_ref = 0` the ramp is flat).
   Docstring updated accordingly.
7. **`track_csr_line!`.**
   - Reference geometry: fill `Vector{Any}`s `Lv`, `gv`, `fv` (length `n`) in the existing loop with the values now
     stored in `L[i]`, `g[i]`, `ref_exit[i]` (same expressions; `csr_ref_exit(f, Lv[i], gv[i], pp)`), then
     `Tg = promote_type(Float64, map(typeof, Lv)..., map(typeof, gv)...)`, `L = convert(Vector{Tg}, Lv)`,
     `g = convert(Vector{Tg}, gv)`, `Tf = promote_type(Tg, map(CSR.number_type, fv)...)`,
     `ref_exit = FloorPosition{Tf}[convert(FloorPosition{Tf}, f) for f in fv]`. (All Float64: as before.)
   - Centroid start: after `c0 = isnothing(centroid_start) ? bunch_centroid(bunch, w, alive0) : collect(float.(centroid_start))`,
     add `Tc = promote_type(eltype(bunch.coords.v), eltype(c0))` and
     `eltype(c0) == Tc || (c0 = convert(Vector{Tc}, c0))` (a Float64 `centroid_start` with a TPS bunch becomes TPS; GPU
     arrays are never converted because their `Tc` equals `eltype(c0)` in the supported cases).
   - Centroid matrix: `centroid = zeros(n + 1, 6)` → `centroid = Matrix{promote_type(Float64, eltype(c0))}(undef, n + 1, 6)`
     (every row is assigned below; Float32 bunches keep a Float64 matrix as before).
   - Bunch loop: `E_start, E_end = csr_ref_energies(...)` → `E_start, E_end, ramp = csr_ref_energies(...)`, and pass
     `ramp` to `track_csr_element!` as a new positional argument right after `E_end`.
   - Update the docstring signature line if needed; behavior text unchanged.
8. **`track_csr_element!`** gets the new argument:
   `track_csr_element!(bunch, ele, i, L, E_start, E_end, ramp, p_over_q_ref, z, pz, q, geom, params, context, diagnostics, test_bunch = nothing)`.
   Remove `ramp = E_start != E_end`. Replace `ramp && (P0c = oftype(P0c, csr_slice_P0c(s)))` by
   `ramp && (P0c = csr_slice_P0c(s))` (the slice's number type may differ from `E_start`'s, e.g. a voltage knob).
   Everything else unchanged. Update its docstring signature.
9. No other change in `track.jl` (`csr_ref_exit`, `centroid_bunch`, `macro_weights`, `test_kick_particles`,
   `csr_slice`, `track_centroid_sliced!` unchanged).

### 6.D Docs

1. **`CSR.track_csr!` docstring** (`track.jl`): add, after the `test_bunch` item, a paragraph:
   - "Knobs: the bunch may carry GTPSA TPS (or ForwardDiff `Dual`) coordinates, weights, `t_ref` and `p_over_q_ref`,
     and element parameters may be TPS/`Dual` (e.g. `L`, `g_ref`, `Kn1`, patch offsets, RF `voltage`, `dE_ref`, the
     reference energy). The final coordinates are then the collective map in the knobs: the deposit, geometry,
     centroid orbit, γ and kick tables are computed in that number type. Discrete choices (number of CSR steps, slice
     step counts, bin grid, SaganCavity cells and event windows) use the primal values and are held constant."
   - "When `L`, `p_over_q_ref`, `dE_ref`, patch parameters or the RF parameters of a SaganCavity are TPS/`Dual`, the
     reference time is too: the `Bunch` (and `test_bunch`) must be built with `t_ref`, `p_over_q_ref` and coordinates
     of a type that holds them (e.g.
     `t_ref = zero(k)` for a TPS knob `k`); otherwise `track_csr!` raises an error before tracking. Dynamic-descriptor
     TPS in the bunch, `centroid_start`, the test bunch or those element parameters (and `g_ref`) must be in GTPSA's
     current Descriptor (error otherwise)."
   - "Limits: a SaganCavity with automatic `n_cells` and a knob in its length or RF frequency fails in BeamTracking
     (`round` of a TPS); derivatives of order ≥ 3 in a curvature knob of an element with `g = 0` are truncated.
     BeamTracking 0.8.7 leaves `bunch.p_over_q_ref` at the entrance value after a SaganCavity (the coordinates are
     already normalized to the exit reference), so the next element shifts `pz` a second time; CSR.jl's lcavity
     slices follow BeamTracking. With a `dE_ref` knob of primal value 0 that second shift is skipped for the
     derivatives (equal reference energies in value)."
   - In the "Reference energy" paragraph, "except SaganCavity elements with `dE_ref` (lcavity)" →
     "except SaganCavity elements (lcavity; with `dE_ref = 0` the ramp is flat)".
2. **`docs/src/usage.md`**, section "Truncated power series (GTPSA)": after the `track_csr!` test-bunch example block
   and before the paragraph "The collective kick itself also takes TPS: ...", add the paragraph and block:

   "`track_csr!` also takes a whole bunch in TPS, with knobs in element parameters, the bunch charge, the reference
   energy or the initial coordinates; the result is the collective map in the knobs. Build the `Bunch` with TPS
   coordinates, `t_ref` and `p_over_q_ref` (BeamTracking writes the reference time back into it):"

   ```julia
   k = @vars(Descriptor(3, 2))           # knobs: curvature of the first bend, bunch charge, energy chirp
   c(x) = x + zero(k[1])                 # TPS constant
   line = [Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5 + k[1], Kn0 = 1.5 + k[1]), Drift(L = 0.2)]
   bl = Beamline(line; p_over_q_ref = pq, species_ref = e)
   v = c.(v0)                            # v0, w0: Float64 coordinates (N×6) and weights of a line bunch
   v[:, 6] .+= k[3] .* v0[:, 5]          # chirp: pz += k₃ z
   bunch = Bunch(v, nothing, w0 .* (1 + k[2]); species = e, p_over_q_ref = c(pq), t_ref = c(0.0))
   track_csr!(bunch, bl; params = CSRParams(n_bin = 40, ds_step = 0.1))
   GTPSA.jacobian(bunch.coords.v[1, :])  # 6×3: final coordinates of particle 1 w.r.t. the knobs
   ```

### 6.E Tests (`test/track_tps_collective_test.jl`, new)

Header comment: purpose (Stage 2c: `track_csr!` with a TPS bunch in knobs vs ForwardDiff and finite differences),
helpers used (`tc_pq`, `tc_line_bunch`, `TC_E`, `TC_EXT` from `track_csr_test.jl`; `LC_E`, `LC_E0` from
`lcavity_test.jl`; `gaussian`, `E_CHARGE` from `utils.jl` / existing scope), and that the last testset leaves another
GTPSA Descriptor current. All names defined at file scope start with `KT_`/`kt_`/`Knob` (no clashes).

**Helpers (file scope):**
```julia
const KT_NK = 7
const KT_PQ = tc_pq(200)
const KT_PARAMS_A = CSRParams(n_bin = 40, ds_step = 0.1)
const KT_PARAMS_B = CSRParams(n_bin = 40, ds_step = 0.05)
const KT_ROWS = collect(1:37:400)   # 11 sample particles
kt_norm(x) = sqrt(sum(abs2, x))     # 2-norm (LinearAlgebra is not loaded by runtests.jl)

# Final sampled coordinates (11 particles × 6, column-major) and t_ref for knob values p (element type: Float64,
# Dual or TPS). `build(p)` returns (bunch, lattice, keyword NamedTuple for track_csr!).
struct KnobTrack
  build::Any
end
function (f::KnobTrack)(p)
  b, lat, kw = f.build(p)
  track_csr!(b, lat; kw...)
  return vcat(vec(b.coords.v[KT_ROWS, :]), [b.t_ref])
end
struct KnobJacobian
  f::KnobTrack
end
(j::KnobJacobian)(p) = ForwardDiff.jacobian(j.f, p)
```
**Lattice A** (Beamline), knobs p₁…p₇ (all with nominal value 0):
```julia
function kt_build_a(p; csr = true)
  c(x) = x + zero(p[1])
  pq = KT_PQ * (1 + p[6])                                            # p6: reference energy
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[2], g_ref = 1.5 + p[1], Kn0 = 1.5 + p[1]),   # p1: g, p2: L
          Quadrupole(L = 0.2, Kn1 = 2.0 + p[3]), Patch(dx = 1e-4 * (1 + p[4])),          # p3: Kn1, p4: patch dx
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = TC_E)
  b0 = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)
  v = c.(b0.coords.v)
  v[:, 6] .+= p[7] .* b0.coords.v[:, 5]                              # p7: chirp
  w = b0.coords.weight .* (1 + p[5])                                 # p5: bunch charge
  b = Bunch(v, nothing, w; species = TC_E, p_over_q_ref = c(pq), t_ref = c(0.0))
  return b, bl, (; params = KT_PARAMS_A, csr_on = csr ? (ele -> true) : (ele -> false))
end
```
**Lattice B** (Branch with an lcavity):
```julia
function kt_build_b(p; dE0 = false)
  c(x) = x + zero(p[1])
  E0 = LC_E0 * (1 + p[4])                                            # p4: reference energy
  bends = [SBend(L = 0.1, g_ref = 0.5 + p[1], Kn0 = 0.5 + p[1]) for _ in 1:10]   # p1: g of all bends
  cav = RFCavity(L = 1.0 + p[7], voltage = 2e8 * (1 + p[2]), rf_frequency = 1.3e9,  # p7: cavity L, p2: voltage
                 dE_ref = dE0 ? 1e7 * p[3] : 1e8 * (1 + p[3]), phi0 = 0.0,          # p3: dE_ref
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  br = Branch([Marker(E_ref = E0, species_ref = LC_E), bends..., cav, Drift(L = 0.5)])
  n = 400
  z = collect(range(-6e-4, 6e-4; length = n))
  w = gaussian.(z, 1e-4)
  w .*= (1e-9 / E_CHARGE) / sum(w)
  v0 = zeros(n, 6)
  v0[:, 5] .= z
  v = c.(v0)
  v[:, 5] .+= 1e-4 * p[6]                                            # p6: z offset of all particles
  b = Bunch(v, nothing, w .* (1 + p[5]); species = LC_E,             # p5: bunch charge
            p_over_q_ref = c(BeamTracking.E_to_R(LC_E, E0)), t_ref = c(0.0))
  return b, br, (; params = KT_PARAMS_B)
end
```
`d = Descriptor(KT_NK, 2)`, `k = collect(@vars(d))` (static descriptor) at the start of the outer testset.

**Testset "track_csr! with a TPS bunch in knobs"**, with one inner testset per case
`("Beamline", KnobTrack(kt_build_a))`, `("Branch with an lcavity", KnobTrack(kt_build_b))`. For each case `f`:

- **E.1 Scalar part.** `vt = f(k)`, `v0 = f(zeros(KT_NK))`, `m = length(v0)` (67). With `vs = scalar.(vt)`:
  `maximum(abs, vs[1:m-1] .- v0[1:m-1]) <= 1e-12 * maximum(abs, v0[1:m-1])` and
  `abs(vs[m] - v0[m]) <= 1e-14 * abs(v0[m])`.
- **E.2 First order vs ForwardDiff.** `Jt = GTPSA.jacobian(vt)` (m × 7), `Jf = ForwardDiff.jacobian(f, zeros(KT_NK))`.
  For every knob j: `kt_norm(Jf[1:m-1, j]) > 0` and `kt_norm(Jt[1:m-1, j] - Jf[1:m-1, j]) <= 1e-10 * kt_norm(Jf[1:m-1, j])`.
  t_ref row: `abs(Jt[m, j] - Jf[m, j]) <= 1e-10 * maximum(abs, Jf[m, :])` for all j, and `maximum(abs, Jf[m, :]) > 0`.
- **E.3 First order vs central finite differences.** `h = 1e-5`;
  `Jfd[:, j] = (f(h .* e_j) .- f(-h .* e_j)) ./ (2h)` with `e_j` the j-th unit vector (`Float64`). For every j:
  `kt_norm(Jt[1:m-1, j] - Jfd[1:m-1, j]) <= 1e-5 * kt_norm(Jt[1:m-1, j])`; t_ref row:
  `abs(Jt[m, j] - Jfd[m, j]) <= 1e-6 * maximum(abs, Jt[m, :])`.
- **E.4 Second order vs nested Duals.** `H = reshape(ForwardDiff.jacobian(KnobJacobian(f), zeros(KT_NK)), m, KT_NK, KT_NK)`,
  `Hmax = maximum(abs, H[1:m-1, :, :])`, `@test Hmax > 0`. For every output `j in 1:m-1`:
  `isapprox(GTPSA.hessian(vt[j]), H[j, :, :]; rtol = 1e-10, atol = 1e-13 * Hmax)`. For `j = m`:
  `isapprox(GTPSA.hessian(vt[m]), H[m, :, :]; rtol = 1e-10, atol = 1e-13 * maximum(abs, H[m, :, :]))` and
  `maximum(abs, H[m, :, :]) > 0`.

Both cases use the same `KnobTrack`/`KnobJacobian` types, so the `Dual` tag types are shared (compile time, as
`track_tps_test.jl`). Do not define per-case closures for the ForwardDiff calls. Use `kt_norm`, not `LinearAlgebra.norm`; do not add `using LinearAlgebra`.

**Testset "CSR changes the knob map"** (lattice A): `Jon = GTPSA.jacobian(KnobTrack(kt_build_a)(k))`,
`Joff = GTPSA.jacobian(KnobTrack(p -> kt_build_a(p; csr = false))(k))`: `all(iszero, Joff[:, 5])` (charge has no effect
without CSR), `kt_norm(Jon[1:end-1, 5]) > 0`, and `maximum(abs, Jon .- Joff) > 0`.

**Testset "dE_ref knob at 0 (lcavity ramp)"** (lattice B with `dE0 = true`, first order only):
`fb = KnobTrack(p -> kt_build_b(p; dE0 = true))`, `Jt = GTPSA.jacobian(fb(k))`, `Jfd` by central differences as E.3
(`h = 1e-5`) for column 3 only, `Jfd3 = (fb(h .* e3) .- fb(-h .* e3)) ./ (2h)` with `e3` the third unit vector
(the `-h` run has a negative `dE_ref`, which is allowed): `kt_norm(Jt[1:end-1, 3]) > 0` and
`kt_norm(Jt[1:end-1, 3] - Jfd3[1:end-1]) <= 1e-5 * kt_norm(Jt[1:end-1, 3])`.

**Testset "Test bunch in 6D + knobs beside a TPS bunch"** (lattice A, order 1):
- `d13 = Descriptor(13, 1)`, `x = @vars(d13)`, `b, bl, kw = kt_build_a(collect(x[7:13]))` (bunch in knobs 7–13).
- `r = [101, 301]`; test bunch `tb = Bunch(vcat([permutedims([b.coords.v[ri, j] + x[j] for j in 1:6]) for ri in r]...); species = TC_E, p_over_q_ref = b.p_over_q_ref, t_ref = b.t_ref)`.
- `track_csr!(b, bl; kw..., test_bunch = tb)`.
- For `i in 1:2`: `Jt = GTPSA.jacobian(tb.coords.v[i, :])`, `Jb = GTPSA.jacobian(b.coords.v[r[i], :])` (6 × 13):
  `isapprox(Jt[:, 7:13], Jb[:, 7:13]; rtol = 1e-12)`, `kt_norm(Jt[:, 7:13]) > 0`, and
  `maximum(abs, scalar.(tb.coords.v[i, :]) .- scalar.(b.coords.v[r[i], :])) <= 1e-15`.
- Frozen-wake 6D block vs step 5: `y = @vars(Descriptor(6, 1))`; `b5, bl5, kw5 = kt_build_a(zeros(KT_NK))`;
  `tb5 = Bunch(vcat([permutedims([b5.coords.v[ri, j] + y[j] for j in 1:6]) for ri in r]...); species = TC_E, p_over_q_ref = b5.p_over_q_ref)`;
  `track_csr!(b5, bl5; kw5..., test_bunch = tb5)`; for `i in 1:2`:
  `isapprox(GTPSA.jacobian(tb.coords.v[i, :])[:, 1:6], GTPSA.jacobian(tb5.coords.v[i, :]); rtol = 1e-10)`.

**Testset "csr_ref_curvature with knobs"**: `TC_EXT.csr_ref_curvature(BendParams(g_ref = 0.0), 1) === 0.0`;
`gk = TC_EXT.csr_ref_curvature(BendParams(g_ref = k[1]), 1)`: `scalar(gk) == 0` and `GTPSA.jacobian([gk])[1, 1] == 1`;
`gk2 = TC_EXT.csr_ref_curvature(BendParams(g_ref = k[1], tilt_ref = Float64(π)), 1)`: `GTPSA.jacobian([gk2])[1, 1] == -1`.

**Testset "Entry checks"** (lattice A):
1. `bl = kt_build_a(k)[2]`; Float64 bunch `b = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)`, `v_before = copy(b.coords.v)`:
   the error from `track_csr!(b, bl; params = KT_PARAMS_A)` is an `ErrorException` whose message contains
   `"track_csr!: bunch.t_ref has type"`; `b.coords.v == v_before`.
2. TPS bunch `b = kt_build_a(k)[1]`, test bunch `tb = Bunch(reshape([zero(k[1]) for _ in 1:6], 1, 6); species = TC_E, p_over_q_ref = b.p_over_q_ref)`
   (Float64 `t_ref`): message contains `"track_csr!: test_bunch.t_ref has type"`.
3. `b1 = kt_build_a(k)[1]`; `b = Bunch(b1.coords.v, nothing, b1.coords.weight; species = TC_E, p_over_q_ref = KT_PQ, t_ref = b1.t_ref)`
   (Float64 `p_over_q_ref`), lattice `kt_build_a(k)[2]`: message contains `"track_csr!: bunch.p_over_q_ref has type"`.
4. `b0 = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)`; `b = Bunch(b0.coords.v, nothing, b0.coords.weight; species = TC_E, p_over_q_ref = KT_PQ + zero(k[1]), t_ref = zero(k[1]))`
   (Float64 coordinates), lattice `kt_build_a(k)[2]`: message contains `"track_csr!: bunch coordinates have element type"`.
5. `TC_EXT.check_ref_types(Bunch(zeros(Float32, 2, 6); species = TC_E, p_over_q_ref = KT_PQ), nothing, Float64) === nothing`
   (hardware floats are never checked).
Capture messages with `try ... catch err; err; end` and test `err isa ErrorException && occursin(..., err.msg)`.

**Testset "Descriptor check" (last testset of the file; leaves another Descriptor current):**
1. `d1 = Descriptor(7, 1)`, `kd = collect(@vars(d1, dynamic = true))`, `b, bl, kw = kt_build_a(kd)`; then
   `Descriptor(3, 1)`; `track_csr!(b, bl; kw...)` raises an `ErrorException` whose message contains
   ``"track_csr!: TPS in argument `coords`"``.
2. In this order (so no TPS operation mixes Descriptors): `d2 = Descriptor(4, 1)`, `kL = @vars(d2, dynamic = true)[1]`,
   lattice `bl2 = Beamline([Drift(L = 0.2), SBend(L = 0.4 + kL, g_ref = 1.5, Kn0 = 1.5), Drift(L = 0.2)]; p_over_q_ref = KT_PQ, species_ref = TC_E)`;
   then `d3 = Descriptor(5, 1)`, `kb = @vars(d3, dynamic = true)[1]`, and a bunch from `tc_line_bunch(1e-3, 1e-9, 400, KT_PQ)` with coordinates `[x + zero(kb) for x in v]` (matrix),
   `p_over_q_ref = KT_PQ + zero(kb)`, `t_ref = zero(kb)`, same weights: `track_csr!(b2, bl2; params = KT_PARAMS_A)` raises
   an `ErrorException` whose message contains ``"track_csr!: TPS in argument `L`"``.

### 6.F Zero-length chord in `CSRGeometry` (amendment, 2026-10-17)

Found by spec-implementer: lattice A's zero-length `Patch` gives a centroid chord `dr` whose scalar part is exactly
`(0, 0, 0)`; `CSRGeometry` (`src/geometry.jl:46-47`) then calls `atan(dr[1], dr[3])` and `sqrt(dr[1]^2 + dr[3]^2)` on
TPS zeros, and GTPSA aborts the Julia process (`mad_tpsa_div.c:267: invalid domain div(.,+0.0000E+00)`; `sqrt` of a TPS
zero: `mad_tpsa_fun.c:214: invalid domain sqrt(+0.0000E+00)`, checked by the coordinator). Any zero-length element
(e.g. the `Marker` of lattice B) does the same. A chord below Bmad's 1e-8 threshold is not used for the spline or
path excess (the `if abs(primal(L_chord)) >= 1e-8` block), so its direction needs no derivatives.

1. **`src/geometry.jl`, `CSRGeometry(L, g, ref_exit, centroid; ...)`**: replace the two lines
   `theta_chord = atan(dr[1], dr[3])` and `L_chord = sqrt(dr[1]^2 + dr[3]^2)` by
   ```julia
       if primal(dr[1]) == 0 && primal(dr[3]) == 0
         # Zero-length chord (e.g. a marker): TPS atan/sqrt abort at 0 and Dual derivatives are not finite there. The
         # chord is below the 1e-8 threshold below, so its primal direction is enough.
         theta_chord = convert(T, atan(primal(dr[1]), primal(dr[3])))
         L_chord = zero(T)
       else
         theta_chord = atan(dr[1], dr[3])
         L_chord = sqrt(dr[1]^2 + dr[3]^2)
       end
   ```
   Nothing else changes. For Float64 the values are bitwise the same as before (`atan` of the same zeros;
   `sqrt(0.0) == 0.0`).
2. **Test** in `test/tps_collective_test.jl`: a new testset "Zero-length element in a TPS geometry", placed before the
   testset "Descriptor check" (which must stay last). With `k = @vars(Descriptor(2, 2))` (static),
   `L = [0.3, 0.0, 0.3]`, `g64 = [1.0, 0.0, 1.0]`, `gk = [1.0 + k[1], zero(k[1]), 1.0 + k[2]]`,
   `c64 = zeros(4, 6)`, `c64[:, 1] .= [1e-4, 2e-4, 2e-4, 3e-4]` (rows 2 and 3 equal, so element 2's chord is exactly
   zero), `ck = map(x -> x + zero(k[1]), c64)`:
   - `gt = CSRGeometry(L, gk, ck)` runs (no abort); `g0 = CSRGeometry(L, g64, c64)`.
   - For every element `i in 1:3`: `scalar(gt.ele[i].theta_chord) == g0.ele[i].theta_chord` and
     `scalar(gt.ele[i].L_chord) == g0.ele[i].L_chord` (bitwise; the TPS geometry has the Float64 values as scalar
     parts); `g0.ele[2].L_chord == 0`.
   - `gt.ele[2].L_chord` and `gt.ele[2].theta_chord` have all coefficients zero beyond the scalar part:
     `all(iszero, GTPSA.jacobian([gt.ele[2].L_chord, gt.ele[2].theta_chord]))`.
   - Dual: `gd = CSRGeometry(L, [1.0 + ForwardDiff.Dual(0.0, 1.0), 0.0, 1.0], c64)`:
     `all(isfinite, ForwardDiff.partials(gd.ele[2].theta_chord))` and `ForwardDiff.value(gd.ele[2].L_chord) == 0`.
   (Check the field names `theta_chord`, `L_chord` of `CSRElementInfo` in `src/types.jl`; `CSRGeometry(L, g,
   centroid)` is the method at `src/geometry.jl:13`.)

### 6.G `L0²` in `upstream_drift_source` (amendment, 2026-10-17)

Found by spec-implementer after 6.F: every 6.E lattice aborts GTPSA in `upstream_drift_source`
(`src/kernels/source_point.jl:216`, `L0 = sqrt(...)`) at the kick on the lattice start (`s_kick = 0` in element 1, the
first half kick of `track_csr!`), where the kick point and the lattice start coincide (`L0 = 0`). The coordinator's
scan (`diagnostics/stage2c_tps_zero_hazards.jl`: GTPSA's `sqrt!`, `log!`, `atan!`, `div!`, `inv!` overwritten to log
zero arguments) found this as the only remaining site for lattices A, B and B with `dE_ref` at 0 (391 calls each).
`L0` is used only as `L0^2` (line 228).

1. In `upstream_drift_source`, replace `L0 = sqrt((fk.r[1] - f0.r[1])^2 + (fk.r[3] - f0.r[3])^2 + kp.y_source^2)` by
   `L0sq = (fk.r[1] - f0.r[1])^2 + (fk.r[3] - f0.r[3])^2 + kp.y_source^2   # squared distance (no sqrt: TPS abort at 0)`
   and `c = (Lsz0 - dz)^2 - beta2 * L0^2` by `c = (Lsz0 - dz)^2 - beta2 * L0sq`. Nothing else changes (both modes).
   Float64 results change only at round-off (`sqrt(x)^2` vs `x`).
2. No new test: 6.E's lattices kick on the lattice start (first half kick). Sensitivity: criterion 11.

### 6.H Test redesign (amendment, 2026-10-17)

Found by spec-implementer after 6.G (33 of 239 assertions failed: E.3 for both lattices, E.4 for lattice A, "dE_ref
knob at 0") and diagnosed by the coordinator (`diagnostics/stage2c_fd_grid.jl`, `stage2c_dE0_bt.jl`,
`stage2c_sagan_pq_check.jl`):
- `csr_bin_particles` builds the bin grid from primal min/max `z` (`src/binning.jl:25-26`), so TPS and `Dual`
  derivatives hold the grid fixed (Stage 2a design); central FD moves it with the extreme particles. Chirp (A, p₇)
  and z-offset (B, p₆) knobs differ by 29% and 279% for any `h`; with two zero-weight ghost particles fixing the grid,
  9e-9 and 0.25%. Knobs that move the ghosts too (g, L, Kn1, patch) stay at 0.1–10% (Kn1: small derivative, FD
  noise). FD is therefore a coarse check only (it still catches missing or doubled derivative terms).
- E.4 lattice A: TPS vs nested `Dual` second derivatives differ by up to ≈ 4e-10 relative (both AD; accumulated
  round-off through tracking); the user approved `rtol = 1e-9`.
- "dE_ref knob at 0": BeamTracking 0.8.7 bug (see the 6.D.1 limits sentence): the reference particle of an on-crest
  1e8 → 2e8 eV cavity has `pz = 0` after the cavity and `pz = -0.5` after a following drift. With the cavity last,
  TPS vs FD agree to 3.0e-6 (CSR on) and 1.2e-10 (CSR off). A CSR-side workaround is a later stage (Stage 03, user).

Changes to 6.E (the rest of 6.E stays as written):
1. **Ghost particles.** `kt_build_a(p; csr = true, ghosts = 0.0)` and `kt_build_b(p; dE0 = false, ghosts = 0.0)`:
   if `ghosts > 0`, append two particles (rows 401, 402) with weight 0, `z = -ghosts` and `z = +ghosts`, `x = 1e-4`
   (A) or `0.0` (B), all other coordinates 0, made TPS/`Dual` by the same `c(x)` as the other particles; the chirp
   (A, p₇) and z-offset (B, p₆) knobs are applied to rows 1–400 only; the charge knob (p₅) multiplies all weights.
   `KT_ROWS` is unchanged (rows ≤ 400).
2. **E.3 (replaces the E.3 text above).** `k1 = collect(@vars(Descriptor(KT_NK, 1)))` (static, order 1; create it
   once, before the outer testset's per-case loop, after `k`). Per case, with `ghosts = 1e-2` (A) or `1e-3` (B):
   `fg = KnobTrack(p -> kt_build_a(p; ghosts = 1e-2))` (resp. `kt_build_b(p; ghosts = 1e-3)`), `Jg = GTPSA.jacobian(fg(k1))`,
   `h = 1e-4`, `Jfd[:, j] = (fg(h .* e_j) .- fg(-h .* e_j)) ./ (2h)`. For every knob j:
   `kt_norm(Jg[1:m-1, j] - Jfd[1:m-1, j]) <= 0.15 * kt_norm(Jfd[1:m-1, j])` and `kt_norm(Jfd[1:m-1, j]) > 0`; t_ref
   row: `abs(Jg[m, j] - Jfd[m, j]) <= 1e-6 * maximum(abs, Jg[m, :])`. Comment in the test: AD holds the bin grid fixed,
   FD would move it; the ghosts fix it, but knobs that move the ghosts (g, L, Kn1, patch) keep a 0.1–10% difference
   (measured 2026-10-17, worst 0.097 for Kn1), so 0.15 only catches gross errors.
3. **E.4:** `rtol = 1e-10` → `rtol = 1e-9` for the outputs `j in 1:m-1` (the t_ref output keeps `rtol = 1e-10`).
4. **"dE_ref knob at 0":** with `dE0 = true`, `kt_build_b` builds the Branch without the final `Drift(L = 0.5)` (the
   cavity is the last element). Bounds unchanged (`h = 1e-5`, `1e-5`). Comment: the drift is left out because of the
   BeamTracking 0.8.7 double `pz` shift after a SaganCavity.
5. Section 8's tolerance paragraph is superseded for E.3 and E.4 by items 2 and 3.

### 6.I Unit test of the SaganCavity window tests (amendment, 2026-10-18)

spec-tester: with raw comparisons in `sagan_in_window`/`sagan_overlap` (criterion 5b) no test fails; the 6.E
lattices never put `Dual`s at an exact tie. Add to `test/track_tps_collective_test.jl` a testset "SaganCavity window
tests decide on primal values" (anywhere before "Descriptor check", which stays last), with
`D(v, d) = ForwardDiff.Dual(v, d)` and `X = TC_EXT` (the `CSRBeamTrackingBeamlinesExt` module):
1. `X.sagan_in_window(D(0.25, 1.0), D(0.25, 2.0), D(0.5, 0.0), false) == X.sagan_in_window(0.25, 0.25, 0.5, false)`
   (both `true`: the event at the slice start belongs to the slice).
2. `X.sagan_in_window(D(0.5, 0.0), D(0.25, 0.0), D(0.5, 1.0), false) == X.sagan_in_window(0.5, 0.25, 0.5, false)`
   (both `false`: the event at the slice end belongs to the next slice).
3. `r = X.sagan_overlap(D(0.25, 1.0), D(0.5, 0.0), D(0.25, -2.0), D(0.25, 2.0), D(0.75, 0.0))`:
   `ForwardDiff.value(r) == 0.25` and `ForwardDiff.partials(r)[1] == -2.0` (segment fully inside: the nominal length
   `len` is returned with its own partials).
(ForwardDiff orders `Dual`s with equal values by their partials, PLAN.md Section 6, so raw comparisons fail each
item.)
(Item 3's `len` partial is −2, not −1, so the raw fallback `b − a`, partial −1, differs; spec-tester, 2026-10-18.)

### 6.J Zero horizontal distance in `ddz_calc_csr` (amendment, 2026-10-18; code review W4)

Coordinator's scan (`diagnostics/stage2c_tps_zero_hazards.jl`, extended 2026-10-18): lattice A with image charges
**and** `bmad_compat = true` reaches `atan(0, 0)` (`src/kernels/source_point.jl:21`) and `sqrt(0)` (line 76) in
`ddz_calc_csr`, 11 calls per run (GTPSA would abort the process). Compat places those image sources exactly above the
kick point, so `L_vec`'s horizontal part is zero. Default mode with images, `sliced_centroid = false`, a SaganCavity
first in the lattice: no hazard.

1. In `ddz_calc_csr`, replace `L = sqrt(L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2)` and `theta_L = atan(L_vec[1], L_vec[3])` by
   ```julia
     L2 = L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2
     Lp2 = L_vec[1]^2 + L_vec[3]^2
     if primal(Lp2) == 0
       # Source exactly above or below the kick point (image charges in compat mode): TPS atan/sqrt abort at 0 and
       # Dual derivatives are not finite there; take the primal direction and a zero horizontal length.
       theta_L = convert(typeof(Lp2), atan(primal(L_vec[1]), primal(L_vec[3])))
       Lp = zero(Lp2)
     else
       theta_L = atan(L_vec[1], L_vec[3])
       Lp = sqrt(Lp2)
     end
     L = primal(L2) == 0 ? zero(L2) : sqrt(L2)
   ```
   and replace each remaining `sqrt(L_vec[1]^2 + L_vec[3]^2)` in `ddz_calc_csr` (three places: the default-mode
   source-ahead branch, the downstream branch, the `y_source` correction) by `Lp`. Nothing else changes. Float64 values
   are bitwise the same (same expressions; `sqrt(0.0) == 0.0`).
   Note (coordinator's inference, not verified): at these lags the compat source is the kick point itself, computed
   by the same expressions, so the horizontal vector is zero for all knob values and dropping its derivatives is
   exact. The coarse FD check of 6.K.6 is the independent check.
2. Test: see 6.K.6.

### 6.K Review fixes (amendment, 2026-10-18; user selected W1, W2, W5, W6 and S4)

1. **W1** (`test/tps_collective_test.jl`, testset "Zero-length element in a TPS geometry"): for elements 1 and 3,
   `scalar(gt.ele[i].theta_chord) ≈ g0.ele[i].theta_chord` and the same for `L_chord`, with `rtol = 4eps()` (GTPSA's C
   libm vs Julia's may differ in the last bit); element 2 keeps exact `==`.
2. **W2** (`track.jl`): replace the splats by reductions:
   `Tg = promote_type(mapreduce(typeof, promote_type, Lv; init = Float64), mapreduce(typeof, promote_type, gv; init = Float64))`,
   `Tf = mapreduce(CSR.number_type, promote_type, fv; init = Tg)`, and in `ref_param_type`
   `mapreduce(typeof, promote_type, vcat(r.L, r.patch, r.dE_ref, r.sagan_rf, r.p_over_q_ref); init = Float64)`.
3. **W5** (`track_csr!` docstring, the "When `L`, …" item): add "Knobs in other parameters (e.g. `g_ref`, `Kn1`,
   alignment) are not part of this check; they also need TPS/`Dual` bunch coordinates (otherwise BeamTracking or
   `csr_kick!` raises an error)."
4. **S4, type-stable reference momentum** (user: type stability is a primary requirement of the package):
   - `SaganSlice` gets a separate energy number type: `struct SaganSlice{E,TM,M,K,TE<:Number,T<:Number}`; fields
     `E0, dE_ref, P0c0, p_over_q_ref` have type `TE`; the other eight former `T` fields stay `T`. Docstring: "`TE`:
     number type of the energy fields (`E0`, `dE_ref`, `P0c0`, `p_over_q_ref`); `T`: RF and length fields".
   - `slice_sagan`: `ve = (float(E0), dE_ref, sqrt(E0^2 - mass^2), float(p_over_q_ref))`,
     `TE = float(promote_type(map(typeof, ve)...))`; `vt` = the other eight values in field order (`float(rf_omega),
     float(t_phi0), q_gradient, float(q * rp.voltage), float(L_active), L, float(s0), float(s0 + L_slice)`),
     `T = float(promote_type(map(typeof, vt)...))`; pass `convert(TE, ·)` / `convert(T, ·)` in the field positions.
     Then `sagan_P0c_at` and `sagan_exit_P0c` return `TE` (a voltage knob alone no longer makes them TPS).
   - `track_csr_element!`: `TP = float(promote_type(typeof(E_start), typeof(E_end)))`;
     `P0c = convert(TP, sqrt(E_start^2 - mass^2))::TP`; in the loop `ramp && (P0c = convert(TP, csr_slice_P0c(s))::TP)`;
     `pz_scale = ramp ? sqrt(E_kick^2 - mass^2) / P0c : one(E_kick) / one(P0c)` (both branches have the type of
     `E_kick / P0c`; Float64 value `1.0` as before). Float64 results bitwise unchanged.
5. **W6 test** (`test/track_tps_collective_test.jl`, new testset "Float64 centroid_start with a TPS bunch", before
   "Descriptor check"): `kt_build_a` gets a keyword `params = KT_PARAMS_A` (used in its returned NamedTuple).
   With `b, _, _ = kt_build_a(k1)`, `w = CSR.primal.(b.coords.weight)`,
   `c0 = [sum(CSR.primal.(b.coords.v[:, j]) .* w) / sum(w) for j in 1:6]` (Float64), and
   `ft = KnobTrack(p -> ((bb, bl, kw) = kt_build_a(p); (bb, bl, merge(kw, (; centroid_start = c0)))))`:
   `vt = ft(k1)`, `v0 = ft(zeros(KT_NK))`: E.1-style scalar check (coordinates `1e-12` of max, t_ref `1e-14`);
   `Jt = GTPSA.jacobian(vt)` vs `Jf = ForwardDiff.jacobian(ft, zeros(KT_NK))` per column as E.2 (`1e-10`); and
   `maximum(abs, Jt .- GTPSA.jacobian(KnobTrack(kt_build_a)(k1))) > 0` (the knob dependence of the default TPS
   centroid is gone with a Float64 `centroid_start`).
6. **W4 test** (same file, new testset "Images with bmad_compat", before "Descriptor check"):
   `pc = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01, bmad_compat = true)`,
   `fc = KnobTrack(p -> kt_build_a(p; params = pc))`: E.1-style scalar check of `fc(k1)` vs `fc(zeros(KT_NK))`; E.2-style
   columns vs `ForwardDiff.jacobian(fc, zeros(KT_NK))` (`1e-10`); and a 6.H.2-style coarse FD check with ghosts
   (`kt_build_a(p; params = pc, ghosts = 1e-2)`, `h = 1e-4`, bound `0.15`, t_ref `1e-6`). If a bound fails, report the
   numbers (do not loosen).
7. **Type-stability test** (same file, new testset "Type-stable SaganCavity reference momentum", before "Descriptor
   check"): `cav = RFCavity(L = 1.0, voltage = 2e8 + k1[2], rf_frequency = 1.3e9, dE_ref = 1e8, phi0 = 0.0,
   zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)`,
   `ele = Branch([Marker(E_ref = LC_E0, species_ref = LC_E), cav]).beamlines[2].line[1]`,
   `sl = CSR.slice_element(ele, 0.0, 0.25, true, false)`: `sl isa TC_EXT.SaganSlice`, `sl.q_voltage isa GTPSA.TPS`,
   `(@inferred TC_EXT.sagan_exit_P0c(sl)) isa Float64`.
8. Not done (user, 2026-10-18): S1, S2, S3, S7 declined; S5 deferred (unverified, field-map elements); S6 declined;
   W3 dismissed (TPS `atan(1e-4, 0)` runs with correct derivatives, checked by the coordinator).

### 6.L 6.K.6 finalized (amendment, 2026-10-18)

spec-implementer: 6.K.6 failed E.1 (scalar 7.3e-11 vs 1e-12) and the coarse FD for knobs 2 (0.39) and 3 (0.29).
Coordinator's diagnosis (`diagnostics/stage2c_compat_fd.jl`, `stage2c_images_dL.jl`):
- Compat mode: the Float64 map jitters at ~1e-9 under tiny knob changes (FD error × 2h constant ≈ 2.7e-8 for Kn1);
  cause unverified. The TPS scalar parts differ from Float64 by 3e-12 (compat) to 7e-11 (compat + images), within
  that jitter.
- Images (default mode too): the TPS image-table derivative at the first kick in bend 1 matches central FD to
  ≤ 1.5e-7 at h ≤ 1e-5 (knob on L and s_kick together, L only, s_kick only); at h ≥ 1e-4 FD is off by 24–28%: the
  image kick has a kink in `s_kick` within ~1e-5 m (inference: a retarded source crossing a hard bend edge). Map-level
  FD therefore cannot check knobs that move kick points (L) with images, and smaller h drowns in the Float64 jitter.

Changes to 6.K.6 (the rest stays): remove the coarse FD check and the `fg = KnobTrack(...)` line it alone uses; the
E.1-style scalar check uses `1e-9` (relative to max |coordinate|) instead of `1e-12`; t_ref unchanged. Put this
comment as the first lines inside the testset:
```julia
  # Guards the 6.J zero-horizontal-distance guard (images + compat aborted GTPSA before). No finite-difference check:
  # the image kick has kinks in s_kick within ~1e-5 m and the compat Float64 map jitters at ~1e-9, so map-level FD
  # cannot check these knobs (diagnostics/stage2c_images_dL.jl, stage2c_compat_fd.jl). Scalar parts: 1e-9 (compat
  # jitter; measured 7.3e-11).
```

## 7. Data formats

No change. `track_csr!` inputs may now carry TPS/`Dual` numbers as described in 6.D.1.

## 8. Numerical and physics requirements

No physics change: no equation, model, constant or approximation is added or altered; the code is made generic in
the number type, branches use primal values, and the existing lcavity ramp formula (D21, `csr_ref_energy`) is applied
to every SaganCavity (D49; with `dE_ref = 0` it is flat). No `physics-design-sourcer` entry is needed.

Tolerances (6.E): scalar part ≤ 1e-12 of max |coordinate| (atomic deposit order with threads), t_ref 1e-14 relative;
first order vs ForwardDiff column norm 1e-10 relative; vs central FD (`h = 1e-5`) 1e-5 relative (t_ref 1e-6 of the
row max); second order per output `rtol = 1e-10`, `atol = 1e-13 · Hmax`; test bunch knob columns `rtol = 1e-12`,
scalar 1e-15; frozen 6D block `rtol = 1e-10`. Float64 ramp reference (criterion 9.4): ≤ 1e-15 relative.

## 9. Acceptance criteria

1. Full suite passes on `-t 1` and `-t 4`, no failures or errors. Previous total 2215 (7m40s `-t 1`, 7m22s `-t 4`);
   report the new total, run times, and the run time of `track_tps_collective_test.jl` alone (from the full-suite run
   or a separate run).
2. All 6.E assertions pass as written. Report per lattice: E.1 ratio, worst E.2 and E.3 column ratio (value / bound)
   with the knob index, worst E.4 `|ΔH| / (1e-10|H| + 1e-13·Hmax)` with the output index; the "dE_ref knob at 0" ratio;
   the test-bunch ratios. If a bound fails, report the numbers and the knob/output; do not loosen bounds without the
   user's approval.
3. All pre-existing tests pass unmodified (only `runtests.jl` gains one `include`).
4. **Float64 ramp reference.** Run `diagnostics/stage2c_ramp_reference.jl <scratchpad>/ramp_new.txt` on 1 thread
   (`-t 1`) with the new code and compare with `diagnostics/stage2c_ramp_dE0_reference.txt` (made with the code at
   `2fdbe77`): `maximum(abs, new[1:400, :] .- ref[1:400, :]) <= 1e-14 * maximum(abs, ref[1:400, :])` (was 1e-15; 6.G
   changes Float64 results at round-off, measured 2.49e-15, user 2026-10-18), and the t_ref row
   (row 401) equal within `1e-15` relative. Report both numbers. Do not modify the reference file or the script.
5. **Sensitivity** (scratch session, `@eval` redefinitions; the repo is not modified):
   a. `csr_ref_energies` with the old rule (`ramp = E_start != E_end`, i.e. return `ramp = false` when `dE_ref` has
      primal 0): the "dE_ref knob at 0" testset must fail.
   b. `sagan_in_window` and `sagan_overlap` with raw comparisons (no `CSR.primal`): some E.2 or E.4 check of the lcavity
      case must fail (cavity-length knob p₇ puts `Dual`s at exact window ties). If none fails, report it (do not
      change the tests) so the coordinator can decide.
   c. `check_ref_types` as a no-op: "Entry checks" items 1–4 must fail. `check_track_inputs` as a no-op: "Descriptor
      check" must fail (report how: wrong message, other error, or crash).
   d. `csr_ref_curvature` with the old `g == 0 && return 0.0`: the "csr_ref_curvature with knobs" testset must fail.
6. Docs: `julia --project=docs docs/make.jl` completes with no new warnings (existing: the `internals.md` size warning
   and "no deployment environment").
7. The 6.D.2 docs example runs in a scratch script with `pq = tc_pq(200)`, `e = TC_E`, `b0 = tc_line_bunch(1e-3, 1e-9, 400, pq)`,
   `v0 = b0.coords.v`, `w0 = b0.coords.weight` (definitions from `test/track_csr_test.jl:7-19` and `test/utils.jl`
   copied into the script); report that `GTPSA.jacobian(bunch.coords.v[1, :])[:, 2]` is nonzero.

10. **6.F (amendment):** the 6.F.2 testset passes; criteria 1–3 hold with it (the full suite then also runs the 6.E
    lattices). Sensitivity: with the 6.F.1 guard removed (in a separate scratch process, since GTPSA aborts the
    process), `CSRGeometry(L, gk, ck)` from 6.F.2 must abort or error; report which.

11. **6.G (amendment):** criteria 1–4 and 10 with 6.G in place (criterion 4's bound unchanged; report the measured
    difference). Sensitivity, in a separate scratch process: with the old `L0 = sqrt(...)` line restored, lattice A's
    `track_csr!` with TPS knobs must abort (report).

12. **6.H (amendment):** all 6.E assertions with 6.H pass; report per lattice the worst E.3 column ratio (value /
    bound) and knob, the worst E.4 ratio, and the "dE_ref knob at 0" ratio. Criterion 5a (old ramp rule) must still
    make the "dE_ref knob at 0" testset fail.

13. **6.I (amendment):** the new testset passes; sensitivity (scratch, `@eval`): with raw comparisons in
    `sagan_in_window` and `sagan_overlap` (criterion 5b), it fails. Re-run criterion 4 with the 1e-14 bound and the
    new file alone (`-t 4`). A full-suite rerun is not required (only a testset is added).

14. **6.J, 6.K (amendment):** full suite on `-t 1` and `-t 4` passes; docs build as criterion 6; criterion 4 (≤ 1e-14)
    still holds; all new 6.K tests pass (report the 6.K.6 ratios). Sensitivity: (a) 6.J guard removed → the 6.K.6
    testset aborts the process (separate process; report); (b) `track_csr_element!` with the old `P0c` lines and
    `SaganSlice` with one type `T` → the 6.K.7 testset fails; (c) Float64: the existing tests pass unchanged (report
    any exact comparison that changed).

## 10. Build and run

Test environment (scratch; this session's is already built):
`/private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/db72019f-f3f3-4676-9ddd-83eb7cbd1286/scratchpad/env`
(`Pkg.develop`s `~/.julia/dev/CSR.jl`, `~/.julia/dev/BeamTracking.jl`, `~/.julia/dev/Beamlines.jl`; adds Test,
StaticArrays, AbstractFFTs, ChainRulesCore, ChainRulesTestUtils, GTPSA, AtomicAndPhysicalConstants,
KernelAbstractions, Atomix, ForwardDiff, DelimitedFiles; resolved). Scripts and logs in the same scratchpad.

- Full suite, from `test/`: `julia --project=<env> -t 4 -e 'include("runtests.jl")'` and the same with `-t 1`.
- One file, from `test/`: `julia --project=<env> -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking,
  Beamlines, ChainRulesCore, ForwardDiff, GTPSA; include("utils.jl"); include("kernels_test.jl");
  include("track_csr_test.jl"); include("lcavity_test.jl"); include("track_tps_collective_test.jl")'`.
- Ramp reference: `julia --project=<env> -t 1 diagnostics/stage2c_ramp_reference.jl <outfile>` (from the repo root).
- Docs, from the repo root: `julia --project=docs docs/make.jl`.

## 11. Implementation order

One stage: 6.A, 6.B, 6.C (code), 6.D (docstring, usage.md), then 6.E (tests). If tracking with TPS still fails after
6.A–6.C at a place this spec does not name, stop and report the stack trace (do not add other code changes).

## 12. Open items

None affecting implementation.

## 13. Decisions (user, 2026-10-17)

| Item | Decision |
|---|---|
| Scope | Knobs in element parameters (L, g_ref, patch, field knobs such as Kn1), bunch charge (TPS weights), reference energy (`p_over_q_ref`, `dE_ref`), initial coordinates. |
| Probe | Run before the spec (`diagnostics/stage2c_tps_params_probe.jl`). |
| Test reference | TPS order 1 vs ForwardDiff (Dual knobs through the same `track_csr!`) and central FD; order 2 vs nested Duals. |
| `t_ref` type | Clear error up front when the Bunch's `t_ref`/`p_over_q_ref` cannot hold the parameters' type (coordinates checked the same way, and SaganCavity RF parameters included in the type: coordinator's additions). |
| Descriptors | Entry check on bunch coords, weight, t_ref, p_over_q_ref, `centroid_start`, test bunch, and the element parameters CSR reads (L, g_ref, patch, dE_ref, p_over_q_ref; SaganCavity RF parameters added by the coordinator, since `slice_sagan` reads them). |
| Ramp | For every SaganCavity (not `E_start != E_end`); Float64 results with `dE_ref = 0` unchanged to ≤ 1e-15 relative. |
| `g = 0` series | Order ≥ 3 truncation documented, deferred. |
| Zero-length chord | Guard in core `CSRGeometry` (6.F): primal direction, zero length, no derivatives, when the chord is exactly zero in primal. |
| FD check | Coarse FD with the grid pinned by ghost particles, `h = 1e-4`, bound 0.15 per knob column (6.H). |
| E.4 bound | `rtol = 1e-9` for coordinates (6.H). |
| BeamTracking double `pz` shift after a SaganCavity | 2c: cavity last in the dE0 test, documented, upstream report drafted; CSR-side workaround in a separate Stage 03. |
| Sensitivity gaps | 5a by the `csr_ref_energies` unit testset (spec-tester), 5b by the 6.I unit testset; the lattice tests need not detect them (2026-10-18). |
| Criterion 4 bound | 1e-14 relative for coordinates (t_ref 1e-15), 2026-10-18. |
| `L0` at the lattice start | Use `L0²` directly in `upstream_drift_source` (6.G), both modes; Float64 changes at round-off accepted. |

## Spec changes

| Date | Change | Reason | Status |
|---|---|---|---|
| 2026-10-18 | Section 6.L: 6.K.6 without the coarse FD check, scalar bound 1e-9. | spec-implementer: 6.K.6 failed 3 assertions; coordinator's diagnosis: image kick kinks in s_kick (AD verified ≤ 1.5e-7 against FD at small h), compat Float64 jitter ~1e-9 (cause unverified, open item). | Approved by the user |
| 2026-10-18 | Sections 6.J (zero horizontal distance guard in `ddz_calc_csr`) and 6.K (W1, W2, W5, S4 type-stable `P0c` with a `SaganSlice` energy type, W6 and W4 tests, type-stability test) added; criterion 14. | Code review of Stage 2c; W4 confirmed by the coordinator's extended hazard scan (images + compat abort). User selected W1, W2, W4, W5, W6 and asked for S4 (type stability is a primary requirement). | Approved by the user |
| 2026-10-18 | Review S1, S2, S3, S7 (read parameters once; long line; `primal` in error messages; "first bend" wording), S6 (promote error without prefix). | Not selected by the user. | Declined by the user |
| 2026-10-18 | Review S5 (`ShiftedPotential` with TPS on field-map elements). | Unverified; outside 2c's knob set. | Deferred |
| 2026-10-18 | Review W3 (`atan(x, 0)` abort). | Coordinator checked: TPS `atan(1e-4, 0)` runs, derivatives correct. | Dismissed |
| 2026-10-18 | Section 6.I item 3 (proposed): `len` partial −1.0 → −2.0 (`ForwardDiff.Dual(0.25, -2.0)`, expected partial `-2.0`). | spec-tester (criterion 13): with raw comparisons in `sagan_overlap` only, item 3 passes (fallback partial −1 equals `len` partial −1), so the sagan_overlap comparison is undetected; the 6.I testset fails overall via items 1–2 (sagan_in_window). Scratch runs 2026-10-18. | Approved by the user (2026-10-18) |
| 2026-10-18 | Section 6.I added (unit test of `sagan_in_window`/`sagan_overlap` with `Dual` ties); criterion 13; Section 13 rows. | spec-tester: criterion 5b undetected by the lattices. | Approved by the user |
| 2026-10-17 | Section 6.F added (guard for a zero-length chord in `CSRGeometry`, `src/geometry.jl`; test in `tps_collective_test.jl`); Sections 1, 2, 5, 9 (criterion 10), 13 updated. | spec-implementer stopped (Section 11): lattice A's zero-length patch aborts GTPSA in `atan(0, 0)`; the coordinator confirmed `sqrt` of a TPS zero aborts too. Latent since Stage 2a for any zero-length element. | Approved by the user |
| 2026-10-17 | Section 6.H added (E.3 coarse FD with ghost particles, E.4 `rtol = 1e-9`, dE0 lattice without the final drift; docstring limits sentence on the BeamTracking bug); Sections 1, 6.D.1, 9 (criterion 12), 13 updated. | spec-implementer: 33 of 239 failed after 6.G; coordinator's diagnosis: AD holds the bin grid fixed (by design), FD moves it; E.4 round-off ≈ 4e-10; BeamTracking 0.8.7 double `pz` shift after a SaganCavity. | Approved by the user |
| 2026-10-17 | Section 6.G added (`L0²` without `sqrt` in `upstream_drift_source`, `src/kernels/source_point.jl`); Sections 1, 2, 5, 6 (intro), 9 (criterion 11), 13 updated. | spec-implementer stopped again (Section 11): TPS `sqrt(0)` abort at the kick on the lattice start; the coordinator's hazard scan found no other site. | Approved by the user |
| 2026-10-17 | 6.B.3 "three places" for `L_active > 0` in `sagan_slice_kernel!`: the file has five; all five changed. 6.E Descriptor-check item 2: the coordinates built as `reshape([...], size(v))`. | Editorial (spec-implementer's notes); no behavior change. | Applied |
| 2026-10-17 | 6.E: added testset "csr_ref_energies ramp with a dE_ref knob at 0" (before "Entry checks") in `test/track_tps_collective_test.jl`: `csr_ref_energies` returns `ramp = true` for a SaganCavity with a `dE_ref` knob at 0 (6.C.6), `E_start` equals `E_end` in value, and `E_start` is the entrance energy `LC_E0 * (1 + p4)` (independent of `p3`). Added as a unit check; the 6.E testset "dE_ref knob at 0 (lcavity ramp)" is unchanged. | Criterion 5a (sensitivity, `ramp = E_start != E_end`): the "dE_ref knob at 0" testset still passes under the old rule. Under the old rule the column-3 Jacobian moves by 2.3e-6 relative, inside its 1e-5 FD bound (AD-vs-FD ratio 0.51 old, 0.30 new), so the lattice testset cannot separate them. The unit testset fails under the old rule (`ramp` assertion). | Applied (test case following from 6.C.6 and D49) |
| 2026-10-17 | Section 9, criterion 4 (Float64 ramp reference): bound `1e-15` relative, unchanged. Proposed: `1e-14` relative for coordinates (t_ref row unchanged at `1e-15`), because 6.G changes Float64 results at round-off by design. | Criterion 4 measured 2.487e-15 relative (`ramp_new_claude.txt` vs `stage2c_ramp_dE0_reference.txt`); with the `sqrt` line restored in `upstream_drift_source` the difference is 0.0, so the change is from 6.G. 6.G's text says "Float64 results change only at round-off", which the 1e-15 bound does not allow. Needs the user's decision (bound or 6.G form). | Approved by the user (2026-10-18); criterion 4 updated |
| 2026-10-17 | Section 9, criterion 5a and 12 ("the 'dE_ref knob at 0' testset must fail" under the old ramp rule). Proposed: 5a is detected by the new unit testset above, and the lattice testset is not expected to fail at its bound (1e-5). | The lattice testset passes under the old rule (see the row above: 2.3e-6 change against a 1e-5 bound); no bound in 6.H separates them, since FD noise at this step is about 3e-6. Tightening the bound would fail the new code at its own measured 3.0e-6. | Approved by the user (2026-10-18) |
| 2026-10-17 | Section 9, criterion 5b (raw `sagan_in_window`/`sagan_overlap`): report only, per criterion text. Proposed: record that no test detects it. | Sensitivity run: the redefinition is active (a `Dual` tie differs), but the lattice A and lattice B Jacobians are bitwise identical (max relative difference 0.0), and the full file passes 253/253. No test can detect the raw comparisons with these lattices. | Superseded: the user chose a unit test (6.I, 2026-10-18) |
