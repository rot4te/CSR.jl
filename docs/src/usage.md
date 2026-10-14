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

[`csr_kick_pz`](@ref) is the non-mutating form of `csr_kick!` and is differentiable with respect to `pz`, `z` and
`q`. The bin grid (from the extreme live particles) and the kick tables (from the geometry and γ) are held
constant; derivatives with respect to the geometry and γ are not provided yet.

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
