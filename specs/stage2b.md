# Stage 2b: test particles in 6D + knobs with a collective TPS kick

Spec date: 2026-10-17. Base: branch `stage2a-fixes` (`2ddb2b8`). Decisions: D37 (Stage 2 variables), review finding 4 of
the `phase7-tps` review (planning/PLAN.md Section 4), user decisions of 2026-10-17 (recorded in Section 13).
Amended 2026-10-17 after the Stage 2b code review: Section 6.G (see "Spec changes" at the end). Sections 6.A–6.F are
implemented; 6.G changes some of their code and tests, and where they conflict, 6.G wins.

## 1. Overview

Stage 1 test particles (`csr_kick!(...; test_particles)`, `csr_apply_test_kicks!`) already receive a kick in any
number type. In Stage 2b they carry the 6 phase-space variables plus the knobs, while the bunch, geometry, `gamma`
etc. carry the knobs (Stage 2a), so the kick tables are TPS. This stage makes that path safe and tests it:

- **A. Range check before kicking (finding 4).** Today `csr_apply_test_kicks!` kicks the in-range test particles in the
  same kernel pass that counts the out-of-range ones, then raises an error; a caller that catches it and retries kicks
  them twice. Also `csr_kick!` kicks the bunch before the test particles. New: a count-only kernel runs first, and
  nothing is kicked if any live test particle is outside the range; `csr_kick!` applies the test kicks before the
  bunch kick, so any error from the test-particle path leaves both `pz` and `pz_t` unchanged.
- **B. `pz_t` element type.** A hardware-float `pz_t` (e.g. `Vector{Float64}`) with a TPS or `Dual` kick (or TPS/`Dual`
  `z_t`) fails with a cryptic `MethodError` in the kick kernel (`pz[i] += …`, same failure as D47 for `pz`). New: a
  clear `ErrorException` up front.
- **C. Descriptor check for test particles.** When the kick is TPS (non-`Real` element type), `pz_t` and `z_t` are
  checked with `check_descriptors` (D46). With a `Real` kick (Float64, Float32, `Dual`) they stay unchecked, as now.
- **D. Tests.** Test particles TPS in 6D + 3 knobs vs ForwardDiff (Jacobian, Hessian including the mixed
  phase-space × knob blocks); one testset with the knobs as GTPSA parameters; consistency of test-particle and bunch
  kicks; tests for A, B, C.
- **G. Review fixes (amendment).** Non-finite or far-off `z` counts as outside instead of throwing; zero-length
  launches return early; equal lengths of the test-particle arrays; type-only test-particle checks in `csr_kick!`
  before the table build; documented limits; docstring wording; stronger tests.

## 2. Out of scope

- No change to physics, tables, deposit, root finding, `csr_apply_kicks!`'s behavior for the bunch (it still raises
  only its internal error), or the adjoint/rrule code.
- `track_csr!` / `test_bunch` (Stage 2c). The 2b changes reach `track_csr!` only through `csr_kick!`; do not edit
  `ext/CSRBeamTrackingBeamlinesExt/`.
- Review findings 3, 5, 6 (Section 4 of PLAN.md), finding 2 (test-particle backends, Phase 8). Stage 2b review
  findings 4, 7, 8, 12 (Section 6.G, "Not done").
- No length checks on the bunch's `pz`, `z`, `q`, `alive` (only on `pz_t`, `z_t`, `alive_t`, 6.G.3); no handling of
  mixed static/dynamic descriptors or of two different static descriptors (unchanged, untested).
- No change to `docs/src/validation.md`, `docs/src/decisions.md`, README.
- Do not edit `planning/PLAN.md` (the coordinator does). Do not run git (no commit, branch, stash, checkout).

## 3. Language and toolchain

- Julia 1.13.2 (juliaup `release`); `Project.toml` keeps `julia = "1.10"`. No `Project.toml` changes.
- Style: 2-space indent, sparse comments, a docstring for every new function (the docs build uses
  `checkdocs = :all`; `docs/src/internals.md` autodocs `src/kick.jl` and `kernels/kick.jl`). Match the surrounding code.

## 4. Dependencies (existing; use these, do not reimplement)

| Package | Version | Used for |
|---|---|---|
| KernelAbstractions | as in the test env | `@kernel`, `@index(Global)`, `@Const`, `get_backend`, `KernelAbstractions.zeros`, `KernelAbstractions.synchronize` (already used in `src/kick.jl`, `src/kernels/kick.jl`). |
| Atomix | as in the test env | `Atomix.@atomic` count in the new kernel (as in `csr_apply_kicks_kernel!`). |
| GTPSA.jl | 1.5.6 (latest registered, checked 2026-10-17 in the General registry `G/GTPSA/Versions.toml`; compat `GTPSA = "1.5.6"`) | Tests only, plus the existing `CSRGTPSAExt`. `Descriptor(nv, mo)`; `Descriptor(nv, mo, np, po)` (`src/descriptor.jl:59`: `nv` variables, `np` parameters, `mo` max order incl. parameters, `po` parameter order; sets `GTPSA.desc_current`); `@vars(d)`, `@vars(d, dynamic = true)`, `@params(d)` (`src/ctors.jl:130`, `:161`); `scalar`; `GTPSA.jacobian(m; include_params = false)` (`src/getset.jl:482`; with `include_params = true` the columns are the `nv` variables then the `np` parameters, `getset.jl:447-466`); `GTPSA.hessian(t; include_params = false)` (`getset.jl:681`, same ordering); `isequal` on TPS compares every coefficient (`src/operators.jl:88-124`; `==` compares scalar parts only); `t[[i => 1]]` coefficient indexing (already used in the tests). |
| ForwardDiff | as in the test env | Reference Jacobians and nested-`Dual` Hessians (tests); `ForwardDiff.Dual{Nothing}` (B test). |

## 5. Project layout (files to change; no new files)

- `src/kernels/kick.jl`: add `kick_cell` and `csr_count_outside_kernel!`; `csr_apply_kicks_kernel!` uses `kick_cell`
  (6.A).
- `src/kick.jl`: add `count_outside`; rewrite `csr_apply_test_kicks!` (6.A, 6.B, 6.C); reorder `csr_kick!` (6.A);
  docstrings of `csr_kick!`, `csr_kick_pz`, `csr_apply_test_kicks!` (6.E).
- `test/tps_test.jl`: extend the testset "Test particles outside the bunch" (6.F.1).
- `test/tps_collective_test.jl`: new testsets (6.F.2–6.F.5); extend "Descriptor check" (6.F.6); header comment.
- `docs/src/usage.md`: GTPSA section (6.E).
- 6.G changes `src/kernels/kick.jl`, `src/kick.jl`, `test/tps_test.jl`, `test/tps_collective_test.jl` only.

## 6. Components

### 6.A Range check before kicking

**`src/kernels/kick.jl`.** Add, before `csr_apply_kicks_kernel!`:

```julia
"""
    kick_cell(zp, z_center, dz) -> (i0, inside)

Interpolation cell of a particle at `zp` for `csr_apply_kicks_kernel!`: `i0` from the primal `z` (Fortran `int`
truncation toward zero, as Bmad), and `inside = true` if `1 ≤ i0 < n_bin` and the primal
`r1 = (z - z_center[i0]) / dz` lies in `[-0.01, 1.01]`. Computed from primal values only, so the count
(`csr_count_outside_kernel!`) and the kick agree for every number type.
"""
@inline function kick_cell(zp, z_center, dz)
  zpp = primal(zp)
  i0 = trunc(Int, (zpp - z_center[1]) / dz) + 1   # Fortran int() truncates toward zero
  inside = false
  if 1 <= i0 < length(z_center)
    r1p = (zpp - z_center[i0]) / dz
    inside = oftype(r1p, -0.01) <= r1p <= oftype(r1p, 1.01)
  end
  return i0, inside
end
```

Rewrite the body of `csr_apply_kicks_kernel!` (signature unchanged) to:

```julia
  i = @index(Global)
  @inbounds if alive[i]
    zp = z[i]
    i0, inside = kick_cell(zp, z_center, dz)
    if inside
      r1 = (zp - z_center[i0]) / dz
      r0 = 1 - r1
      pz[i] += pz_scale * (r0 * kick[i0] + r1 * kick[i0+1])
    else
      Atomix.@atomic n_outside[1] += one(eltype(n_outside))
    end
  end
```

and its docstring's first paragraph to: "One work item per particle: `pz += pz_scale (r0 kick[i0] + r1 kick[i0+1])`, linear
interpolation between the bin centres around `z`, with the cell and the range test from `kick_cell` (primal `z`, so TPS
or `Dual` `z` expands within the cell). Particles outside the table are counted in `n_outside` (`Int32`: no 64-bit
atomics on some GPUs)."

Behavior note (accepted by the user): for `Float64`, `Float32` and `Dual` `z` the range test is the same as before
(the primal of `(zp - z_center[i0]) / dz` is the same Float operations). For TPS `z` the test now uses Float arithmetic
on the scalar part instead of the scalar part of TPS arithmetic; they may differ in the last bit at the edges
`r1 = -0.01, 1.01` only.

Add after `csr_apply_kicks_kernel!`:

```julia
"""
    csr_count_outside_kernel!(n_outside, z, alive, z_center, dz)

One work item per particle: count the live particles that `csr_apply_kicks_kernel!` would not kick (outside the
table, `kick_cell`) in `n_outside` (`Int32`). Modifies nothing else.
"""
@kernel function csr_count_outside_kernel!(n_outside, @Const(z), @Const(alive), @Const(z_center), dz)
  i = @index(Global)
  @inbounds if alive[i]
    _, inside = kick_cell(z[i], z_center, dz)
    inside || Atomix.@atomic n_outside[1] += one(eltype(n_outside))
  end
end
```

**`src/kick.jl`.** Add after `apply_kicks_outside!`:

```julia
"""
    count_outside(backend, z, alive, z_center, dz) -> n_outside

Number of live particles outside the kick table (`csr_count_outside_kernel!` on `backend`), without kicking.
"""
function count_outside(backend, z, alive, z_center, dz)
  n_outside = KernelAbstractions.zeros(backend, Int32, 1)   # Int32: no 64-bit atomics on some GPUs
  n = length(z)
  wg = cpu_workgroupsize(n, eltype(z), typeof(dz))
  csr_count_outside_kernel!(backend)(n_outside, z, alive, z_center, dz; ndrange = n, workgroupsize = wg)
  KernelAbstractions.synchronize(backend)
  return Int(Array(n_outside)[1])
end
```

(`cpu_workgroupsize` runs TPS `z` on the calling thread: `primal` of a TPS calls GTPSA, whose thread safety for this
read is not established.)

**`csr_apply_test_kicks!`** (signature unchanged) becomes, in this order:

```julia
function csr_apply_test_kicks!(pz_t::AbstractVector, z_t::AbstractVector, alive_t::AbstractVector{Bool}, bins::CSRBins,
                               kick::AbstractVector; pz_scale = 1)
  Tt = promote_type(eltype(kick), eltype(z_t), typeof(pz_scale))
  eltype(pz_t) <: AbstractFloat && !(Tt <: AbstractFloat) &&
    error("csr_apply_test_kicks!: pz_t has element type $(eltype(pz_t)), which cannot hold the kick in number type " *
          "$Tt; use a pz_t with element type $(promote_type(eltype(pz_t), Tt)).")
  eltype(kick) <: Real || check_descriptors("csr_apply_test_kicks!"; pz_t, z_t)
  cpu(A) = A isa Array ? A : Array(A)
  backend = get_backend(pz_t)
  zc, k = backend isa CPU ? (cpu(bins.z_center), cpu(kick)) : (bins.z_center, kick)
  dz = primal(bins.dz)
  n = count_outside(backend, z_t, alive_t, zc, dz)
  n == 0 || error("csr_apply_test_kicks!: $n test particle(s) outside the bunch's bin range " *
                  "[$(primal(first(zc)) - primal(bins.dz) / 100), $(primal(last(zc))))")
  apply_kicks_outside!(pz_t, z_t, alive_t, zc, k, dz, pz_scale) == 0 ||
    error("csr_apply_test_kicks!: test particle outside the bin range (internal error)")
  return pz_t
end
```

The out-of-range message text is unchanged (an existing test matches `"$(zc[end]))"`). Checks B and C come before
anything else touches `pz_t`, `z_t` or the kick.

**`csr_kick!`.** Swap the last two kick statements, so that the test particles are kicked first:

```julia
  st = csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor)
  test_particles === nothing || csr_apply_test_kicks!(test_particles..., st.bins, st.kick; pz_scale)
  csr_apply_kicks!(pz, z, alive, st.bins, st.kick; pz_scale = device_scalar(get_backend(pz), pz_scale))
  return csr_step_result(st)
```

Everything before `st = …` (descriptor check, D47 check) is unchanged. Results are unchanged (separate arrays).

### 6.B `pz_t` element type

In `csr_apply_test_kicks!` (code in 6.A): `Tt = promote_type(eltype(kick), eltype(z_t), typeof(pz_scale))`; error when
`eltype(pz_t) <: AbstractFloat` and `Tt` is not `<: AbstractFloat`. Message as in 6.A; it contains
`"cannot hold the kick"` and `"pz_t"`. Unchanged: `Float32` `pz_t` with a `Float64` kick (converted on store), TPS
`pz_t` with a `Float64` kick (Stage 1).

### 6.C Descriptor check for test particles

In `csr_apply_test_kicks!` (code in 6.A): when `eltype(kick)` is not `<: Real`, call
`check_descriptors("csr_apply_test_kicks!"; pz_t, z_t)`, i.e. an error naming `` `pz_t` `` (checked first) or
`` `z_t` `` when a dynamic-descriptor TPS among them is not in `GTPSA.desc_current`. With a `Real` kick no check
(D46's existing test that test particles in a non-current descriptor work with a Float64 kick must still pass).

### 6.D No other code changes

`count_outside` and `kick_cell` are internal (not exported), like `apply_kicks_outside!`.

### 6.E Docstrings and docs

`src/kick.jl`:

1. `csr_apply_test_kicks!` docstring: replace the whole text after the signature line with:

   "Frozen-wake kick on test particles: `csr_apply_kicks!` with the bins and kick of a bunch the test particles are not
   part of (they deposit no charge). For TPS (or `Dual`) test particles this gives the kick's expansion in their
   coordinates about the primal `z`, with the bunch's wake held fixed; with a TPS kick (a collective TPS bunch,
   geometry, `gamma`, ...) the expansion is also in the bunch's knobs. The test particles' arrays may live on another
   backend than the bunch (e.g. TPS on the CPU); `bins` and `kick` are copied there.

   Errors, raised before any test particle is kicked (`pz_t` is then unchanged):
   - a live test particle outside the bunch's bin range, `z_center[1] - dz/100 ≤ z < z_center[end]` (the range
     `csr_apply_kicks_kernel!` interpolates over; counted first by `count_outside`);
   - `pz_t` of a hardware float type that cannot hold the kick (TPS or `Dual` kick or `z_t`);
   - with a TPS kick, a dynamic-descriptor TPS in `pz_t` or `z_t` that is not in GTPSA's current Descriptor
     (`check_descriptors`). With a `Real` kick the test particles are not checked."

2. `csr_kick!` docstring, `test_particles` bullet: append the sentence "They are kicked before the bunch, so an error
   from `csr_apply_test_kicks!` (a live test particle outside the bin range, a `pz_t` that cannot hold the kick, a
   descriptor mismatch) leaves `pz` and `pz_t` unchanged."
3. `csr_kick!` docstring, last paragraph: replace "(error otherwise; test particles are not checked)" with "(error
   otherwise; test particles are checked only when the kick is TPS, by `csr_apply_test_kicks!`)".
4. `csr_kick_pz` docstring, last paragraph: replace "(error otherwise; test particles are not checked)" with "(error
   otherwise)".

`docs/src/usage.md`, section "Truncated power series (GTPSA)":

5. In the paragraph beginning "CSR.jl creates TPS constants from the number type.", replace the last sentence "Test
   particles are not checked." with "Test particles are checked only when the kick is TPS; their `pz_t` and `z_t`
   must then be in the same Descriptor."
6. At the end of the section (after the last code block), add:

   ````markdown
   Test particles and a collective TPS kick combine: with the bunch, geometry or `gamma` in TPS knobs and the test
   particles in the 6 phase-space variables of the same Descriptor, the test particles' kick is expanded in both, with
   the wake frozen in phase space and collective in the knobs. If any live test particle is outside the bunch's bin
   range, `csr_kick!` raises an error before kicking anything.

   ```julia
   x = @vars(Descriptor(8, 2))   # 6 phase-space variables, then 2 knobs
   g = [0.0, 1.5 + x[7], 0.0]
   pz_b = [zero(x[1]) for _ in eachindex(pz)]
   pz_t, z_t = [x[6]], [0.2e-3 + x[5]]
   csr_kick!(pz_b, z, q .* (1 + x[8]), alive, CSRGeometry(L, g, centroid), 2, 0.17, 200.0, e, params;
             test_particles = (pz_t, z_t, [true]))
   GTPSA.hessian(pz_t[1])[5, 7]   # ∂²Δpz/∂z∂g₂
   ```
   ````

   (Same variable names as the preceding example: `pz`, `z`, `q`, `alive`, `L`, `centroid`, `e`, `params`.)

### 6.F Tests

Common names, already defined in `test/tps_collective_test.jl`'s outer testset: `electron`, `L3`, `g3`, `c3` (with
columns 1, 2, 6 set), `n = 400`, `z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)`. In `test/tps_test.jl`: `electron`,
`L3`, `g3`, `c3` (column 1 only), `z, q, alive`, `geom`, `x = @vars(Descriptor(6, 3))`.

**6.F.1 Finding 4 (`test/tps_test.jl`, end of testset "Test particles outside the bunch").** Uses that testset's `p`,
`st`, `zc`.

```julia
    # Nothing is kicked when any live test particle is outside (finding 4)
    k0 = CSR.csr_apply_test_kicks!([0.0], [0.0], [true], st.bins, st.kick)[1]
    @test k0 != 0
    pzt = [0.0, 0.0]
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pzt, [0.0, 0.05], [true, true], st.bins, st.kick)
    @test pzt == [0.0, 0.0]
    pzb, pzt = zeros(n), [0.0, 0.0]
    @test_throws ErrorException csr_kick!(pzb, z, q, alive, geom, 2, 0.17, 200.0, electron, p;
                                          test_particles = (pzt, [0.0, 0.05], [true, true]))
    @test all(iszero, pzb) && pzt == [0.0, 0.0]
```

**6.F.2 Main test (`test/tps_collective_test.jl`, new testset after "csr_kick! pz element type", before
"Descriptor check").** Variables `w[1:6]` = test-particle phase space (x, px, y, py, z, pz), `w[7]` = element 2's `g`,
`w[8]` = `gamma`, `w[9]` = bunch knob (`q × (1 + w9)`, `z × (1 + 0.2 w9)`).

```julia
  # Stage 2b (D37): test particles in 6D + knobs with a collective TPS kick (frozen in phase space, collective in knobs)
  m = 50
  z0t = range(-2.5e-3, 2.5e-3; length = m)
  w0 = [zeros(6); 1.5; 200.0; 0.0]
  function G(w, p)
    geom = CSRGeometry(L3, [g3[1], w[7], g3[3]], c3; bmad_compat = p.bmad_compat)
    zt = [z0t[j] + w[5] * (1 + 0.1j / m) + 0.3 * w[1]^2 - 0.2 * w[5] * w[6] for j in 1:m]
    pzt = [w[6] + 1e-4j / m for j in 1:m]
    pzb = [zero(w[1]) for _ in 1:n]
    csr_kick!(pzb, z .* (1 + 0.2w[9]), q .* (1 + w[9]), alive, geom, 2, 0.17, w[8], electron, p;
              test_particles = (pzt, zt, fill(true, m)))
    return pzt
  end
  pzt0 = [1e-4j / m for j in 1:m]

  @testset "Test particles in 6D + knobs (images = $n_img, bmad_compat = $bc)" for n_img in (0, 2), bc in (false, true)
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0,
                  bmad_compat = bc)
    d = Descriptor(9, 2)
    x = @vars(d)
    t = G(w0 .+ x, p)
    y0 = G(w0, p)
    @test maximum(abs, y0 .- pzt0) > 0
    @test maximum(abs, scalar.(t) .- y0) <= 1e-13 * maximum(abs, y0 .- pzt0)
    Jt, Jf = GTPSA.jacobian(t), ForwardDiff.jacobian(w -> G(w, p), w0)
    for c in 1:4
      @test all(iszero, Jt[:, c]) && all(iszero, Jf[:, c])
    end
    for c in 5:9
      @test any(!iszero, Jf[:, c])
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-11
    end
    Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(w -> G(w, p), u), w0), m, 9, 9)
    Ht = permutedims(cat((GTPSA.hessian(ti) for ti in t)...; dims = 3), (3, 1, 2))
    Hmax = maximum(abs, Hf)
    for (a, b) in ((1, 1), (5, 6), (5, 7), (5, 8), (5, 9), (7, 7), (7, 8), (7, 9), (8, 8), (8, 9), (9, 9))
      @test any(!iszero, Hf[:, a, b])   # phase space × knob and knob × knob blocks are exercised
    end
    for a in 1:9, b in a:9
      ref = maximum(abs, Hf[:, a, b])
      err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
      @test err <= 1e-10 * ref + 1e-14 * Hmax
    end
    # Test particles at bunch particles' positions (same TPS z) get the bunch particles' kicks, every coefficient
    w = w0 .+ x
    geom = CSRGeometry(L3, [g3[1], w[7], g3[3]], c3; bmad_compat = bc)
    zb, qb = z .* (1 + 0.2w[9]), q .* (1 + w[9])
    idx = findall(alive)[1:37:end]
    pzb = [zero(x[1]) for _ in 1:n]
    pzt = [zero(x[1]) for _ in idx]
    csr_kick!(pzb, zb, qb, alive, geom, 2, 0.17, w[8], electron, p; test_particles = (pzt, zb[idx], fill(true, length(idx))))
    @test all(i -> isequal(pzt[i], pzb[idx[i]]), eachindex(idx))
    @test any(!iszero, GTPSA.jacobian(pzt))
  end
```

If a nonzero-block assertion in the loop over `(a, b)` fails, stop and report it (do not drop the pair without
approval): it means the test does not exercise what it is meant to.

**6.F.3 Knobs as GTPSA parameters (new testset right after 6.F.2's testset; reuses `G`, `w0`, `m`).**

```julia
  @testset "Test particles in 6D, knobs as GTPSA parameters" begin
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 3e-3)
    d = Descriptor(6, 2, 3, 2)
    x, k = @vars(d), @params(d)
    t = G(vcat(w0[1:6] .+ x, w0[7:9] .+ k), p)
    Jt = GTPSA.jacobian(t; include_params = true)
    Jf = ForwardDiff.jacobian(w -> G(w, p), w0)
    @test size(Jt) == (m, 9)
    for c in 5:9
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-11
    end
    Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(w -> G(w, p), u), w0), m, 9, 9)
    Ht = permutedims(cat((GTPSA.hessian(ti; include_params = true) for ti in t)...; dims = 3), (3, 1, 2))
    Hmax = maximum(abs, Hf)
    for a in 1:9, b in a:9
      ref = maximum(abs, Hf[:, a, b])
      err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
      @test err <= 1e-10 * ref + 1e-14 * Hmax
    end
  end
```

**6.F.4 `pz_t` element type (new testset after 6.F.3).**

```julia
  @testset "Test particles: pz_t element type" begin
    p = CSRParams(n_bin = 40, ds_step = 0.1)
    geom0 = CSRGeometry(L3, g3, c3)
    d = Descriptor(2, 2)
    x = @vars(d)
    geom1 = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)
    errmsg(f) = try f(); "" catch err; sprint(showerror, err) end
    zt3 = [0.0, 1e-3, -1e-3]
    # TPS kick, Float64 pz_t: error, nothing kicked
    pzb, pzt = [zero(x[1]) for _ in 1:n], zeros(3)
    f1 = () -> csr_kick!(pzb, z, q, alive, geom1, 2, 0.17, 200.0, electron, p; test_particles = (pzt, zt3, fill(true, 3)))
    @test_throws ErrorException f1()
    msg = errmsg(f1)
    @test occursin("cannot hold the kick", msg) && occursin("pz_t", msg)
    @test all(t -> isequal(t, zero(x[1])), pzb) && pzt == zeros(3)
    # Dual kick (gamma), Float64 pz_t
    γd = ForwardDiff.Dual{Nothing}(200.0, 1.0)
    pzbd = fill(zero(γd), n)
    f2 = () -> csr_kick!(pzbd, z, q, alive, geom0, 2, 0.17, γd, electron, p; test_particles = (zeros(3), zt3, fill(true, 3)))
    @test_throws ErrorException f2()
    @test occursin("cannot hold the kick", errmsg(f2))
    # Float64 kick, TPS z_t, Float64 pz_t
    st = CSR.csr_binned_kick(z, q, alive, geom0, 2, 0.17, 200.0, electron, p)
    f3 = () -> CSR.csr_apply_test_kicks!(zeros(1), [0.0 + x[2]], [true], st.bins, st.kick)
    @test_throws ErrorException f3()
    @test occursin("cannot hold the kick", errmsg(f3))
    # Float32 pz_t with a Float64 kick still runs
    @test CSR.csr_apply_test_kicks!(zeros(Float32, 3), zt3, fill(true, 3), st.bins, st.kick) isa Vector{Float32}
  end
```

**6.F.5 Header comment of `test/tps_collective_test.jl`.** Append to the first comment block the line
"`# Stage 2b: test particles in 6D + knobs with the collective TPS kick; range, element-type and descriptor checks.`"

**6.F.6 Descriptor check (append at the end of the existing testset "Descriptor check", after its last `@test`).** At
that point `GTPSA.desc_current` is `d2`, `x` is dynamic in `d1` (not current), `y = @vars(d2, dynamic = true)`.

```julia
    # Stage 2b: with a TPS kick, test particles are checked (pz_t first, then z_t); nothing is kicked on error
    qy = q .* (1 + y[1])
    pzb = [zero(y[1]) for _ in 1:n]
    pzt1 = [x[1] + 0.0]
    f1 = () -> csr_kick!(pzb, z, qy, alive, geom0, 2, 0.17, 200.0, electron, p; test_particles = (pzt1, [0.0 + x[2]], [true]))
    @test_throws ErrorException f1()
    msg = errmsg(f1)
    @test occursin("GTPSA's current Descriptor", msg) && occursin("`pz_t`", msg)
    @test all(t -> isequal(t, zero(y[1])), pzb) && isequal(pzt1[1], x[1] + 0.0)
    f2 = () -> csr_kick!(pzb, z, qy, alive, geom0, 2, 0.17, 200.0, electron, p; test_particles = ([y[2] + 0.0], [0.0 + x[2]], [true]))
    @test occursin("`z_t`", errmsg(f2))
    pzt2 = [y[2] + 0.0]
    csr_kick!(pzb, z, qy, alive, geom0, 2, 0.17, 200.0, electron, p; test_particles = (pzt2, [0.0 + y[3]], [true]))
    @test pzt2[1][[1 => 1]] != 0
```

(The descriptor `d1` of `x` and `pzt1` stays alive while `x` is referenced; this matches the existing steps of the
testset, which already use `x` after `d2` was created.)

### 6.G Review fixes (amendment, 2026-10-17)

From the code review of the implemented 6.A–6.F (findings numbered as reported; approved by the user: 1, 2, 3, 5, 6,
9, 10, 11). Implement all of 6.G.1–6.G.8.

**6.G.1 Non-finite or far-off `z` is outside (finding 1).** In `kick_cell` (`src/kernels/kick.jl`), `trunc(Int, ·)`
throws `InexactError` for NaN, ±Inf or values beyond `Int`'s range. Replace the function body with:

```julia
@inline function kick_cell(zp, z_center, dz)
  zpp = primal(zp)
  u = (zpp - z_center[1]) / dz
  # NaN, Inf or far outside the table: outside (trunc(Int, ·) would throw)
  (isfinite(u) && abs(u) < length(z_center) + 1) || return 0, false
  i0 = trunc(Int, u) + 1   # Fortran int() truncates toward zero
  inside = false
  if 1 <= i0 < length(z_center)
    r1p = (zpp - z_center[i0]) / dz
    inside = oftype(r1p, -0.01) <= r1p <= oftype(r1p, 1.01)
  end
  return i0, inside
end
```

(Every finite `u` with `abs(u) ≥ n_bin + 1` gives `i0 ≤ 0` or `i0 ≥ n_bin`, i.e. outside, so in-range results are
unchanged.) Replace the `kick_cell` docstring with:

"Interpolation cell of a particle at `zp` for `csr_apply_kicks_kernel!`: `i0` from the primal `zp` (Fortran `int`
truncation toward zero, as Bmad), and `inside = true` if `1 ≤ i0 < n_bin` and the primal
`r1 = (zp - z_center[i0]) / dz` lies in `[-0.01, 1.01]`. A non-finite primal `zp` (NaN, ±Inf) or one more than
`n_bin + 1` bins from `z_center[1]` gives `(0, false)`. Computed from primal values only, so the count
(`csr_count_outside_kernel!`) and the kick agree for every number type."

This also applies to the bunch's own kick (`csr_apply_kicks!`): a NaN/Inf `z` there now reaches its existing
"internal error" message instead of an `InexactError`. Accepted; no test for the bunch path.

**6.G.2 Zero-length launches (finding 3).** In `count_outside` and in `apply_kicks_outside!` (`src/kick.jl`), return
`0` before allocating `n_outside` when there are no work items: in `count_outside` add `length(z) == 0 && return 0`
as the first statement; in `apply_kicks_outside!` add `length(pz) == 0 && return 0` as the first statement.

**6.G.3 Equal lengths (finding 5) and 6.G.4 type-only checks first (finding 9).** Add to `src/kick.jl`, before
`csr_apply_test_kicks!`:

```julia
"""
    check_test_particles(fname, pz_t, z_t, alive_t, Tkick, pz_scale)

Checks on test particles that need only their types and lengths, for a kick of number type `Tkick` (errors name
function `fname`): equal lengths of `pz_t`, `z_t`, `alive_t`; a `pz_t` that can hold the kick (an `AbstractFloat`
`pz_t` needs an `AbstractFloat` kick, `z_t` and `pz_scale`); and, for a non-`Real` `Tkick` (TPS),
`check_descriptors` on `pz_t` and `z_t`.
"""
function check_test_particles(fname, pz_t, z_t, alive_t, Tkick, pz_scale)
  length(pz_t) == length(z_t) == length(alive_t) ||
    error("$fname: test particle arrays pz_t, z_t, alive_t have lengths $(length(pz_t)), $(length(z_t)), " *
          "$(length(alive_t)); they must be equal.")
  Tt = promote_type(Tkick, eltype(z_t), typeof(pz_scale))
  eltype(pz_t) <: AbstractFloat && !(Tt <: AbstractFloat) &&
    error("$fname: pz_t has element type $(eltype(pz_t)), which cannot hold the kick in number type " *
          "$Tt; use a pz_t with element type $(promote_type(eltype(pz_t), Tt)).")
  Tkick <: Real || check_descriptors(fname; pz_t, z_t)
  return nothing
end
```

In `csr_apply_test_kicks!`, replace its first four statements (the `Tt = …` line through the
`eltype(kick) <: Real || check_descriptors(…)` line) with the single statement
`check_test_particles("csr_apply_test_kicks!", pz_t, z_t, alive_t, eltype(kick), pz_scale)`. The rest of the function
is unchanged.

In `csr_kick!`, insert directly after the D47 check (the `eltype(pz) <: AbstractFloat && … error(…)` statement) and
before `st = csr_binned_kick(…)`:

```julia
  test_particles === nothing || check_test_particles("csr_kick!", test_particles..., Tk, pz_scale)
```

So in `csr_kick!` the length, element-type and descriptor errors for test particles are raised before the tables are
built, with the prefix `csr_kick!:`; `csr_apply_test_kicks!` repeats them (cheap) when `csr_kick!` calls it. `Tk` is
non-`Real` exactly when an input is TPS, i.e. when the kick is TPS. The range error still comes from
`csr_apply_test_kicks!` (it needs the bins).

**6.G.5 Documented limits (finding 2) and wording (finding 11).** In `src/kick.jl`:

1. `csr_apply_test_kicks!` docstring: in the error list, replace the item "`pz_t` of a hardware float type that cannot
   hold the kick (TPS or `Dual` kick or `z_t`);" with "`pz_t`, `z_t`, `alive_t` of different lengths;" followed by a
   new item "`pz_t` with an `AbstractFloat` element type that cannot hold the kick (TPS or `Dual` kick, `z_t` or
   `pz_scale`);". After the list, add the paragraph: "Other element-type combinations that cannot hold the kick (e.g.
   a `Dual` `pz_t` with a TPS kick, a TPS `pz_t` with a `Dual` kick) are not checked and fail inside the kick kernel,
   possibly after part of `pz_t` has been kicked."
2. `csr_kick!` docstring, `test_particles` bullet: replace "(a live test particle outside the bin range, a `pz_t` that
   cannot hold the kick, a descriptor mismatch)" with "(a live test particle outside the bin range, arrays of
   different lengths, a `pz_t` that cannot hold the kick, a descriptor mismatch; the last three are checked before the
   kick tables are built)". Then append to the same bullet: "Other element-type combinations that cannot hold the
   kick (e.g. a `Dual` `pz` or `pz_t` with a TPS kick, a TPS `pz` or `pz_t` with a `Dual` kick) are not checked and
   fail inside the kick kernel, possibly after part of `pz` or `pz_t` has been kicked."

**6.G.6 Tests: range edges, non-finite, empty, lengths (findings 1, 3, 5, 6).** Append to the end of the testset
"Test particles outside the bunch" in `test/tps_test.jl` (after the 6.F.1 lines; uses that testset's `st`, `zc`, `dz`,
`x`):

```julia
    # TPS z_t at the range edges: accepted where Float64 is, same kicked value (review finding 6)
    for zt in (zc[1] - dz / 200, zc[1], zc[end - 1], zc[end] - dz / 4)
      k64 = CSR.csr_apply_test_kicks!([0.0], [zt], [true], st.bins, st.kick)[1]
      kt = CSR.csr_apply_test_kicks!([0.0 + x[6]], [zt + x[5]], [true], st.bins, st.kick)[1]
      @test isapprox(scalar(kt), k64; rtol = 1e-13, atol = 1e-13 * maximum(abs, st.kick))
    end
    for zt in (zc[1] - dz / 50, zc[end] + dz / 4)
      pzt = [0.0 + x[6]]
      @test_throws ErrorException CSR.csr_apply_test_kicks!(pzt, [zt + x[5]], [true], st.bins, st.kick)
      @test isequal(pzt[1], 0.0 + x[6])
    end
    # Non-finite or far-off z: the range error, not an InexactError (review finding 1)
    for zt in (NaN, Inf, -Inf, 1e30)
      pzt = [0.0]
      err = try CSR.csr_apply_test_kicks!(pzt, [zt], [true], st.bins, st.kick); nothing catch e; e end
      @test err isa ErrorException && occursin("outside the bunch's bin range", err.msg)
      @test pzt == [0.0]
      errt = try CSR.csr_apply_test_kicks!([0.0 + x[6]], [zt + x[5]], [true], st.bins, st.kick); nothing catch e; e end
      @test errt isa ErrorException && occursin("outside the bunch's bin range", errt.msg)
    end
    # Empty TPS test particles (review finding 3)
    T6 = typeof(x[6])
    @test isempty(CSR.csr_apply_test_kicks!(T6[], T6[], Bool[], st.bins, st.kick))
    # Different lengths (review finding 5)
    err = try CSR.csr_apply_test_kicks!([0.0, 0.0], [0.0], [true, true], st.bins, st.kick); nothing catch e; e end
    @test err isa ErrorException && occursin("must be equal", err.msg)
    pzb = zeros(n)
    err = try csr_kick!(pzb, z, q, alive, geom, 2, 0.17, 200.0, electron, p;
                        test_particles = ([0.0], [0.0, 0.0], [true, true])); nothing catch e; e end
    @test err isa ErrorException && occursin("csr_kick!: test particle arrays", err.msg)
    @test all(iszero, pzb)
```

If the `isapprox` in the first loop fails, report `scalar(kt)`, `k64` and `zt`; do not loosen without approval.

**6.G.7 Stronger tests (finding 10), in `test/tps_collective_test.jl`.**

1. 6.F.2's consistency block (the lines after the comment "Test particles at bunch particles' positions …"): pass
   `pz_scale = 1.3` to that `csr_kick!` call, keep the two existing `@test` lines, and add after them:

   ```julia
    pzb1 = [zero(x[1]) for _ in 1:n]
    pzt1 = [zero(x[1]) for _ in idx]
    csr_kick!(pzb1, zb, qb, alive, geom, 2, 0.17, w[8], electron, p; test_particles = (pzt1, zb[idx], fill(true, length(idx))))
    @test scalar.(pzt) ≈ 1.3 .* scalar.(pzt1) rtol = 1e-14
    @test GTPSA.jacobian(pzt) ≈ 1.3 .* GTPSA.jacobian(pzt1) rtol = 1e-14
   ```

2. 6.F.4 (testset "Test particles: pz_t element type"):
   - After `msg = errmsg(f1)`, also `@test occursin("csr_kick!: pz_t", msg)` (6.G.4's early check).
   - In the `Dual` case, name the test-particle array: replace `test_particles = (zeros(3), zt3, fill(true, 3))` in `f2`
     with `test_particles = (pzt3, zt3, fill(true, 3))`, define `pzt3 = zeros(3)` before `f2`, and after the existing
     `occursin` test add `@test all(iszero, pzbd) && pzt3 == zeros(3)`.
   - Replace the last test (Float32 `pz_t`, `isa Vector{Float32}`) with:

     ```julia
    pz32 = CSR.csr_apply_test_kicks!(zeros(Float32, 3), zt3, fill(true, 3), st.bins, st.kick)
    @test pz32 isa Vector{Float32}
    @test pz32 == Float32.(CSR.csr_apply_test_kicks!(zeros(3), zt3, fill(true, 3), st.bins, st.kick))
    @test any(!iszero, pz32)
     ```

3. 6.F.6 (end of "Descriptor check"): after `@test occursin("GTPSA's current Descriptor", msg) && occursin("`pz_t`", msg)`
   add `@test occursin("csr_kick!:", msg)`; after the last line (`@test pzt2[1][[1 => 1]] != 0`) add
   `@test pzt2[1][[3 => 1]] != 0   # z_t dependence (∂Δpz/∂y3)`.

**6.G.8 Not done (logged as declined or deferred).** Finding 4 (error message reads `first(zc)` from a GPU array) and
finding 12 (`pz_scale` not passed through `device_scalar` for test particles): Phase 8, with the test-particle
backend work. Finding 7 (no test fails without the one-workgroup guard in `count_outside`): no action; the guard is
conservative. Finding 8 (descriptor check keyed on the kick type, not `pz_scale`): no action; the user's rule is
"the kick is TPS", and `csr_kick!` checks `pz_scale` itself.

## 7. Data formats

No change.

## 8. Numerical and physics requirements

No physics change: no equation, model, constant or approximation is added or altered (the kick formula, tables and
deposit are untouched). The only numerical change is the TPS range test at cell edges (6.A behavior note). Tolerances:
6.F.2/6.F.3 use Stage 2a's bounds (scalar part ≤ 1e-13 relative to the largest kick, Jacobian columns `rtol = 1e-11`,
Hessian per block `err ≤ 1e-10·ref + 1e-14·Hmax`); exact equality (`isequal`, `==`) where stated. 6.G.6: TPS vs Float64
kick at the edges `rtol = 1e-13` (plus `atol = 1e-13·max|kick|`); 6.G.7: `pz_scale` ratio `rtol = 1e-14`.

## 9. Acceptance criteria

1. Full suite passes on 1 thread and on 4 threads (`-t 1`, `-t 4`), no failures or errors. Previous total 1812;
   report the new total and the run times.
2. 6.F.1–6.F.4, 6.F.6: all assertions pass as written. Report the run time of 6.F.2 (all four cases) and 6.F.3.
3. 6.F.2 and 6.F.3: report per case the worst `err / (1e-10·ref + 1e-14·Hmax)` over blocks, the worst Jacobian column
   relative error, and the scalar-part ratio. If a bound fails, report the block and numbers; do not loosen bounds
   without approval.
4. Sensitivity (scratch script outside the repo; the repo is not modified):
   a. `CSR.root_order(t::GTPSA.TPS) = 1`: 6.F.2's Hessian checks must fail in some knob × knob block (report how many
      blocks fail per case).
   b. With `csr_kick!`'s two kick statements in the old order (bunch first), 6.F.1's `all(iszero, pzb)` must fail;
      with `csr_apply_test_kicks!` calling `apply_kicks_outside!` without the preceding `count_outside` check
      (old behavior), 6.F.1's `pzt == [0.0, 0.0]` must fail. Do these by redefining the methods in the scratch
      session (`@eval CSR ...`), not by editing the repo.
5. All pre-existing tests pass unmodified (6.F only adds lines).
6. Docs: `julia --project=docs docs/make.jl` completes with no new warnings (`checkdocs = :all`; existing: the
   `internals.md` size warning and "no deployment environment").
7. The 6.E docs example runs: in a scratch script, define its context from `test/tps_collective_test.jl` (`L = L3`,
   `centroid = c3`, `e = electron`, `params = CSRParams(n_bin = 40, ds_step = 0.1)`, `z, q, alive` from
   `kernel_test_bunch(1e-3, 1e-9, 400)`, `pz = zeros(400)`), run the block, and report that
   `GTPSA.hessian(pz_t[1])[5, 7]` is nonzero.
8. 6.G (amendment): criteria 1, 5 (now: pre-existing tests and the 6.F tests pass, modified only as 6.G.7 says) and
   6 again after 6.G; all 6.G.6 and 6.G.7 assertions pass. Sensitivity in a scratch session: with the old `kick_cell`
   (6.A version), 6.G.6's non-finite loop must fail; with `check_test_particles` removed from `csr_kick!`
   (6.G.4), 6.G.7.2's `occursin("csr_kick!: pz_t", msg)` must fail. Report the new suite total and run times on
   `-t 1` and `-t 4`; if `-t 1` again takes much longer than `-t 4` beyond the added tests, report per-file times.

## 10. Build and run

Test environment (scratch, rebuild per session): an env that `Pkg.develop`s `~/.julia/dev/CSR.jl`,
`~/.julia/dev/BeamTracking.jl`, `~/.julia/dev/Beamlines.jl` and `Pkg.add`s Test, StaticArrays, AbstractFFTs,
ChainRulesCore, ChainRulesTestUtils, GTPSA, AtomicAndPhysicalConstants, KernelAbstractions, Atomix, ForwardDiff; then
`Pkg.resolve()`. Build it at
`/private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/67373b1f-8446-40ba-96be-8d491be62bb3/scratchpad/env`
(scripts and logs in the same scratchpad).

- Full suite, from `test/`: `julia --project=<env> -t 4 -e 'include("runtests.jl")'` and the same with `-t 1`
  (after 6.A–6.F: 7m23s on `-t 4`, 11m03s on `-t 1`).
- One file, from `test/`: `julia --project=<env> -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking,
  Beamlines, ChainRulesCore, ForwardDiff, GTPSA; include("utils.jl"); include("kernels_test.jl");
  include("tps_collective_test.jl")'` (`kernel_test_bunch` is in `kernels_test.jl:6`; same for `tps_test.jl`).
- Docs, from the repo root: `julia --project=docs docs/make.jl`.

## 11. Implementation order

One stage: 6.A–6.C (kernels, `count_outside`, `csr_apply_test_kicks!`, `csr_kick!` order), 6.E (docstrings, docs),
then the tests 6.F.1–6.F.6. Done. Amendment: 6.G.1–6.G.5 (code, docstrings), then the tests 6.G.6, 6.G.7.

## 12. Open items

None affecting implementation. Later: Stage 2c (`track_csr!` with a TPS bunch; descriptor checks there).

## 13. Decisions (user, 2026-10-17)

| Item | Decision |
|---|---|
| Scope | A (finding 4), B, C, D at the `csr_kick!` / `csr_apply_test_kicks!` level; `track_csr!` in 2c. |
| Range check | Count-only kernel first, then the kick kernel only if no live test particle is outside. |
| Range test | One primal helper (`kick_cell`) shared by the count and kick kernels; ULP-level change at the cell edges for TPS `z` accepted. |
| Kick order | `csr_kick!` kicks the test particles before the bunch. |
| Descriptor check | Test particles checked only when the kick is TPS (non-`Real`). |
| Knobs in tests | GTPSA variables (`Descriptor(9, 2)`), plus one testset with GTPSA parameters (`Descriptor(6, 2, 3, 2)`). |
| Test setup | Knobs g (element 2), γ, bunch knob; 50 test particles; images ∈ {0, 2} × `bmad_compat` ∈ {false, true}. |
| Review findings | Act on 1, 2, 3, 5, 6, 9, 10, 11 (6.G); 4, 12 deferred to Phase 8; 7, 8 declined. |

## Spec changes

| Date | Change | Reason | Status |
|---|---|---|---|
| 2026-10-17 | Section 6.G added (6.G.1 non-finite/far-off `z` outside in `kick_cell`; 6.G.2 zero-length early return in `count_outside`, `apply_kicks_outside!`; 6.G.3/6.G.4 `check_test_particles` with a length check, called by `csr_apply_test_kicks!` and early in `csr_kick!`; 6.G.5 docstrings; 6.G.6, 6.G.7 tests); Sections 1, 2, 5, 8, 9 (criterion 8), 10, 11, 13 updated. | Code review of Stage 2b, findings 1, 2, 3, 5, 6, 9, 10, 11. | Approved by the user |
| 2026-10-17 | Review findings 4, 12 (GPU test-particle backend: error message from a device array; `pz_scale` not through `device_scalar`). | Out of scope (test-particle backends). | Deferred to Phase 8 by the user |
| 2026-10-17 | Review finding 7 (no test for the one-workgroup guard in `count_outside`). | Guard is conservative; a failing test would need GTPSA thread corruption on reads. | Declined by the user |
| 2026-10-17 | Review finding 8 (descriptor check keyed on the kick type, not `pz_scale`). | User's rule is "the kick is TPS"; `csr_kick!` checks `pz_scale`. | Declined by the user |
