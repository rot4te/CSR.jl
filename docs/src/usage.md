# Usage

## Tracking a bunch through a beamline

With BeamTracking and Beamlines loaded, [`track_csr!`](@ref) tracks a `Bunch` through a `Beamline` (or the
Beamlines of a `Branch`) and applies a CSR kick between slices of every element. The macroparticle charge is
`weight × |charge of species|`; `weight` is the number of real particles per macroparticle.

```@example usage
using CSR, BeamTracking, Beamlines, AtomicAndPhysicalConstants

e = Species("electron")
pq = BeamTracking.E_to_R(e, 1e4 * massof(e))            # p_over_q_ref for γ = 10⁴

# Line bunch: σ_z = 1 mm, 1 nC, Gaussian weights on a uniform z grid
n = 4001
v = zeros(n, 6); v[:, 5] .= range(-6e-3, 6e-3; length = n)
w = exp.(-v[:, 5] .^ 2 ./ 2e-6); w .*= (1e-9 / E_CHARGE) / sum(w)
bunch = Bunch(v, nothing, w; species = e, p_over_q_ref = pq)

rho = 10.0
bl = Beamline([[SBend(L = 0.1, g_ref = 1 / rho, Kn0 = 1 / rho) for _ in 1:20]; Drift(L = 0.5)];
              p_over_q_ref = pq, species_ref = e)

params = CSRParams(n_bin = 80, ds_step = 0.05)
kicks = Dict{Tuple{Int,Int},Float64}()
track_csr!(bunch, bl; params, diagnostics = (ix, i_step, s, r) -> (kicks[(ix, i_step)] = minimum(r.kick_per_meter)))
extrema(bunch.coords.v[:, 6])
```

Keywords of `track_csr!`:

| Keyword | Meaning |
|---|---|
| `params` | [`CSRParams`](@ref); `ds_step` must be set (step length; each element gets `max(1, round(L / ds_step))` steps). |
| `csr_on(ele)` | Elements for which it returns `false` are tracked whole without CSR (Bmad's per-element `csr_method = off`). Zero-length and map elements are always tracked whole. |
| `centroid_start` | 6-vector where the centroid orbit starts. Default: the charge-weighted mean of the live particles. |
| `sliced_centroid` | `true` (default): the centroid is tracked in the bunch's slices, so it follows a bunch particle's orbit. |
| `diagnostics(ix_ele, i_step, s_kick, result)` | Called after each kick with a [`CSRStepResult`](@ref). |
| `test_bunch` | A `Bunch` (e.g. TPS coordinates) tracked alongside that gets the bunch's kicks without contributing to them; see [Truncated power series (GTPSA)](@ref). |

SaganCavity elements with `dE_ref` (lcavities) are sliced exactly (slices reproduce the whole cavity to round-off);
the reference energy for the CSR kick ramps linearly through the cavity, as in Bmad. Put the cavity in a `Branch`
(which gives it its own Beamline with `dE_ref`) and call `track_csr!(bunch, branch; params)`.

## Shielding by parallel plates

Image charges model two conducting plates a distance `beam_chamber_height` apart, the beam on the midplane:

```julia
params = CSRParams(n_bin = 120, ds_step = 0.05, n_shield_images = 32, beam_chamber_height = 0.01)
```

The image-layer series alternates in sign and converges slowly; by default the sum is averaged
(`image_sum_order = 8`), which makes 32–64 layers enough where a plain sum would need thousands
(see [Physics decisions](@ref)).

## One kick on particle arrays

The core API needs no tracking package: [`CSRGeometry`](@ref) describes the centroid orbit (element lengths,
reference curvatures, centroid coordinates at the element ends), and [`csr_kick!`](@ref) bins the particles,
builds the kick table and applies the kick to `pz`.

```@example usage
z = collect(range(-6e-3, 6e-3; length = 20001))
q = exp.(-z .^ 2 ./ 2e-6); q .*= 1e-9 / sum(q)
alive = fill(true, length(z))
geom = CSRGeometry(fill(0.1, 80), fill(1 / rho, 80), zeros(81, 6))
pz = zeros(length(z))
r = csr_kick!(pz, z, q, alive, geom, 80, 0.05, 1e4, e, CSRParams(n_bin = 200, ds_step = 0.1))
(maximum(abs, r.kick_per_meter), sum(r.charge_per_meter) * (r.z_center[2] - r.z_center[1]))
```

## Bmad compatibility

`CSRParams(bmad_compat = true)` (and the same flag in `CSRGeometry`) reproduces Bmad's expressions wherever CSR.jl's
default differs. Use it for like-for-like comparison with Bmad/Tao output; the default is the more accurate model.

## GPU backends

The kernels run on the backend of the particle arrays (`KernelAbstractions.get_backend`). The kick tables are
always computed on the CPU in `Float64` and copied to the device; on backends without `Float64`
(`KernelAbstractions.supports_float64` is `false`, e.g. Apple Metal) they are copied as `Float32`
([`CSR.csr_float_type`](@ref)). For example, with Metal.jl:

```julia
using Metal
pz_d = Metal.zeros(Float32, length(z))
csr_kick!(pz_d, MtlArray(Float32.(z)), MtlArray(Float32.(q)), MtlArray(alive), geom, 80, 0.05, 1e4, e, params)
```

In `Float32`, strongly shielded kicks (a small residual of large image terms) are accurate to only ~3e-4 relative.

## Derivatives

[`csr_kick_pz`](@ref) is the non-mutating form of `csr_kick!`. ForwardDiff `Dual`s pass through it in the particles
(`pz`, `z`, `q`), in the geometry (build `CSRGeometry` from `Dual` element lengths, curvatures or centroid), in
`s_kick` and in γ. The kick tables then carry derivatives: each retarded source point gets its derivative by implicit
differentiation of its root, `ds/dp = −(∂ddz/∂p)/(∂ddz/∂s)`, not by differentiating the root finder (for nested
`Dual`s, e.g. `ForwardDiff.hessian`, one more Newton step from the root per order). The search
runs on a `Float64` copy of the geometry, and each source is then evaluated once in `Dual`s. One pass with 12
partials costs about 4–5 kicks (ρ = 10 m arc of 0.1 m elements, 1 thread). Held
constant: the bin grid (from the extreme live particles) and the discrete choices (number of steps, the element each
source lies in, the adaptive quadrature's subdivision), so derivatives are one-sided where these switch, e.g. at an
element entrance.

With ChainRulesCore loaded, `rrule`s give the pullbacks of `csr_kick_pz` (with respect to `pz`, `z`, `q`, the
geometry as a `Tangent{CSRGeometry}`, `s_kick` and γ) and of `CSRGeometry(L, g, centroid)` and
`CSRGeometry(L, g, ref_exit, centroid)` (with respect to their positional arguments). The particle part uses explicit
adjoint kernels. The geometry, `s_kick` and γ part is the gradient of `k̄ ⋅ kick` (`k̄` the cotangent of the kick per
bin). Each kick-table entry is a fixed linear combination of source evaluations at fixed lags, so this gradient is a
weighted sum of the evaluations' gradients, with weights from the transposed convolutions. Each evaluation searches
its source in `Float64` and is evaluated once with local `Dual`s: `s_kick`, γ, the kick and source elements, the
lattice ends, and the path-length sums over the elements in between, whose cotangents are passed on to those
elements. The cost is 5–14 kicks, independent of the lattice length (50 ms vs 10 ms for the kick, n_bin = 200,
sources reaching 3 m upstream on a ρ = 10 m arc, 1 thread). Keyword arguments (`kick_factor`, `pz_scale`,
`ref_start`) get no cotangents.

```@example usage
using ForwardDiff, ChainRulesCore
zs, qs, as = z[1:40:end], q[1:40:end], alive[1:40:end]
p = CSRParams(n_bin = 40, ds_step = 0.1)
f(q) = csr_kick_pz(zeros(length(zs)), zs, q, as, geom, 80, 0.05, 1e4, e, p)
J = ForwardDiff.jacobian(f, qs)
y, back = ChainRulesCore.rrule(csr_kick_pz, zeros(length(zs)), zs, qs, as, geom, 80, 0.05, 1e4, e, p)
ȳ = randn(length(zs))
maximum(abs, back(ȳ)[4] .- J' * ȳ) / maximum(abs, J' * ȳ)
```

Derivative of the kick with respect to the bend curvature of the last 10 elements and γ:

```@example usage
dg = ForwardDiff.derivative(t -> csr_kick_pz(zeros(length(zs)), zs, qs, as,
                                             CSRGeometry(fill(0.1, 80), [fill(1 / rho, 70); fill(t, 10)], zeros(81, 6)),
                                             80, 0.05, 1e4, e, p), 1 / rho)
dγ = ForwardDiff.derivative(γ -> csr_kick_pz(zeros(length(zs)), zs, qs, as, geom, 80, 0.05, γ, e, p), 1e4)
maximum(abs, dg), maximum(abs, dγ)
```

## Truncated power series (GTPSA)

With GTPSA loaded, test particles with TPS coordinates can ride along a kick: they receive the bunch's kick
interpolated at their `z` but deposit no charge, so the result is the transfer map of the kick with the bunch's wake
held fixed ("frozen wake"). As in BeamTracking, branch decisions use the TPS scalar part: the expansion is that of the
bin cell the reference point lies in. Bmad's linear interpolation between bin centres makes the kick linear in `z`
within a cell, so orders above 1 in `z` come only from the test particles' own coordinates. Live test particles must
lie within the bunch's bin range.

```julia
using GTPSA
x = @vars(Descriptor(6, 2))
pz_t, z_t = [x[6]], [0.2e-3 + x[5]]
csr_kick!(pz, z, q, alive, geom, 80, 0.05, 1e4, e, CSRParams(n_bin = 200, ds_step = 0.1);
          test_particles = (pz_t, z_t, [true]))
GTPSA.jacobian(pz_t)   # ∂Δpz/∂z of the frozen kick in column 5
```

`track_csr!(bunch, bl; params, test_bunch)` does the same along a whole line: the test bunch is tracked by
BeamTracking through the same slices as the bunch and gets every CSR kick, so a one-particle TPS test bunch gives the
transfer map with the bunch's wake frozen. It needs the bunch's species and reference momentum, and its live particles
must stay within the bunch's bin range.

```julia
tb = Bunch(reshape([0.0, 0.0, 0.0, 0.0, 0.2e-3, 0.0] .+ x, 1, 6); species = e, p_over_q_ref = bunch.p_over_q_ref)
track_csr!(bunch, bl; params, test_bunch = tb)
GTPSA.jacobian(tb.coords.v[1, :])   # 6×6 linear map including the frozen-wake CSR kicks
```

The collective kick itself also takes TPS: `csr_kick_pz` and `csr_kick!` accept TPS particle coordinates and charges,
a geometry built by `CSRGeometry` from TPS lengths, curvatures or centroid, and TPS `s_kick` and `gamma`. The deposit
and the kick tables are then computed in TPS, and each retarded source point gets its expansion by Newton steps from
its primal root (one order per step). The bin grid and the discrete choices are held constant, as with `Dual`s. TPS
arithmetic is not safe on several threads (GTPSA's C library shares scratch space between Julia threads), so kernels
with TPS numbers run on the calling thread.

```julia
x = @vars(Descriptor(2, 2))
g = [0.0, 1.5 + x[1], 0.0]   # bend curvature as a knob
pz_new = csr_kick_pz(pz, z, q .* (1 + x[2]), alive, CSRGeometry(L, g, centroid), 2, 0.17, 200.0, e, params)
GTPSA.jacobian(pz_new)   # ∂Δpz/∂g₂ and ∂Δpz/∂(charge scale) for each particle
```
