# Usage

## Tracking a bunch through a beamline

With BeamTracking loaded (extension `CSRBeamTrackingExt`), [`CSREffect`](@ref) is an effect of BeamTracking's
`track!`: `track!(bunch, bl; effects = (CSREffect(params),))` tracks a `Bunch` through a `Beamline` and applies
CSR kicks. The macroparticle charge is `weight × |charge of species|`; `weight` is the number of real particles per
macroparticle (`weight = nothing` means 1).

```@example usage
using CSR, BeamTracking, Beamlines, AtomicAndPhysicalConstants

e = Species("electron")
pq = Beamlines.E_to_R(e, 1e4 * massof(e))   # γ = 10⁴

# Line bunch: σ_z = 1 mm, 1 nC, Gaussian weights on a z grid
function line_bunch(n = 2001)
  v = zeros(n, 6)
  v[:, 5] .= range(-6e-3, 6e-3; length = n)
  w = exp.(-v[:, 5] .^ 2 ./ 2e-6)
  w .*= (1e-9 / E_CHARGE) / sum(w)   # real particles each
  return Bunch(v, nothing, w; species = e, p_over_q_ref = pq)
end

# 2 m arc of radius 10 m (20 bends, 2 slices each), then a drift
rho = 10.0
line = [[SBend(L = 0.1, g_ref = 1 / rho, Kn0 = 1 / rho, n_slices = 2)
         for _ in 1:20]; Drift(L = 0.5, n_slices = 5)]
bl = Beamline(line; p_over_q_ref = pq, species_ref = e)

bunch = line_bunch()
track!(bunch, bl; effects = (CSREffect(CSRParams(n_bin = 80)),))
extrema(bunch.coords.v[:, 6])
```

### Where the kicks are applied

BeamTracking tracks each element in `n_slices` slices of length `L / n_slices` (`n_slices` is a Beamlines element
property, default 1). `CSREffect` applies one kick ([`csr_kick!`](@ref)) at the entrance of each slice, covering the
slice length (`ds = L / n_slices`, `kick_factor = 1`), with the reference energy of the bunch there. An element
with `n_slices = 1` gets one kick at its entrance covering its whole length. The step length is therefore set per
element through `n_slices`; the kick is first order in the slice length.

Kicked are the bending elements (nonzero `g_ref`) and every element after the first bend of the line. Not kicked
are the elements before the first bend, zero-length elements, elements tracked with `SaganCavity`, and elements for
which the keyword `on(ele)` returns `false`.

Before tracking, `CSREffect` surveys the reference orbit of the `Beamline` (with patches) and tracks the centroid
orbit, one particle element by element without CSR, to build the [`CSRGeometry`](@ref). Bends out of the horizontal
plane are an error.

### Keywords of `CSREffect`

| Keyword | Meaning |
|---|---|
| `on(ele)` | Elements for which it returns `false` get no kick (Bmad's per-element `csr_method = off`). Default: `ele -> true`. |
| `centroid_start` | 6-vector where the centroid orbit starts. Default `nothing`: the charge-weighted mean of the live particles. |
| `diagnostics(ix, k, s0, result)` | Called after each kick with the element index, the slice number, the slice entrance [m] from the element entrance and a [`CSRStepResult`](@ref). |
| `record` | A [`CSRRecord`](@ref) that receives every kick, for [`CSRReplay`](@ref). |

Kicks only in the bends, and the minimum kick per meter of each slice:

```@example usage
kicks = Dict{Tuple{Int,Int},Float64}()
diag = (ix, k, s0, r) -> (kicks[(ix, k)] = minimum(r.kick_per_meter))
csr = CSREffect(CSRParams(n_bin = 80);
                on = ele -> ele.kind == "SBend", diagnostics = diag)
bunch = line_bunch()
track!(bunch, bl; effects = (csr,))
length(kicks), kicks[(20, 2)]
```

## Test particles

Test particles get the bunch's kicks without contributing to them. Track the bunch with a [`CSRRecord`](@ref), then
track the test particles through the same `Beamline` with [`CSRReplay`](@ref). Each test particle gets the recorded
kick at its `z`. The elements must have the same `n_slices` as when the kicks were recorded, and live test particles
must lie within the bunch's bin range.

```@example usage
rec = CSRRecord()
bunch = line_bunch()
track!(bunch, bl;
       effects = (CSREffect(CSRParams(n_bin = 80); record = rec),))

# Three test particles at z = -1, 0, 1 mm
vt = zeros(3, 6)
vt[:, 5] .= (-1e-3, 0.0, 1e-3)
test = Bunch(vt; species = e, p_over_q_ref = pq)
track!(test, bl; effects = (CSRReplay(rec),))
test.coords.v[:, 6]
```

The record is emptied at the start of the next `track!` call of its `CSREffect`.

## Shielding by parallel plates

Image charges model two conducting plates a distance `beam_chamber_height` apart, the beam on the midplane:

```@example usage
params = CSRParams(n_bin = 120, n_shield_images = 32,
                   beam_chamber_height = 0.01)
nothing # hide
```

The image-layer series alternates in sign and converges slowly; by default the sum is averaged
(`image_sum_order = 8`), which makes 32–64 layers enough where a plain sum would need thousands
(see [Physics decisions](@ref)).

## One kick on particle arrays

The core API needs no tracking package: [`CSRGeometry`](@ref) describes the centroid orbit (element lengths,
reference curvatures, centroid coordinates at the element ends), and [`csr_kick!`](@ref) bins the particles,
builds the kick table and applies the kick to `pz`. Its arguments after the geometry are the kick's element, the
step length `ds` the kick covers, the kick position `s_kick` from the element entrance and γ.

```@example usage
z = collect(range(-6e-3, 6e-3; length = 20001))
q = exp.(-z .^ 2 ./ 2e-6); q .*= 1e-9 / sum(q)
alive = fill(true, length(z))
geom = CSRGeometry(fill(0.1, 80), fill(1 / rho, 80), zeros(81, 6))
params = CSRParams(n_bin = 200)
pz = zeros(length(z))
r = csr_kick!(pz, z, q, alive, geom, 80, 0.1, 0.0, 1e4, e, params)
dz = r.z_center[2] - r.z_center[1]
maximum(abs, r.kick_per_meter), sum(r.charge_per_meter) * dz
```

## Bmad compatibility

`CSRParams(bmad_compat = true)` (and the same flag in `CSRGeometry`) reproduces Bmad's expressions wherever CSR.jl's
default differs. Use it for like-for-like comparison with Bmad/Tao output; the default is the more accurate model.

## GPU backends

The core kernels run on the backend of the particle arrays (`KernelAbstractions.get_backend`). The kick tables are
always computed on the CPU in `Float64` and copied to the device; on backends without `Float64`
(`KernelAbstractions.supports_float64` is `false`, e.g. Apple Metal) they are copied as `Float32`
([`CSR.csr_float_type`](@ref)). For example, with Metal.jl:

```julia
using Metal
pz_d = Metal.zeros(Float32, length(z))
csr_kick!(pz_d, MtlArray(Float32.(z)), MtlArray(Float32.(q)),
          MtlArray(alive), geom, 80, 0.1, 0.0, 1e4, e, params)
```

In `Float32`, strongly shielded kicks (a small residual of large image terms) are accurate to only ~3e-4 relative.
Tracking with `CSREffect` is CPU only.

## Number types

`csr_kick!`, [`csr_kick_pz`](@ref) and `CSREffect` accept ForwardDiff `Dual` and GTPSA TPS numbers in the particle
coordinates and charges, the geometry, `ds`, `s_kick` and γ (with `CSREffect`: in the bunch coordinates and the
element parameters). These inputs are scalarized (`Beamlines.scalarize`) and the kick is computed in real numbers.
The kick changes only the value (scalar part) of `pz`; its partials or higher TPS coefficients are unchanged. No
derivatives are tracked through the kick.

```@example usage
using ForwardDiff
pzd = ForwardDiff.Dual.(zeros(length(z)), 1.0)
pz1 = csr_kick_pz(pzd, z, q, alive, geom, 80, 0.1, 0.0, 1e4, e, params)
ForwardDiff.value.(pz1) == pz,
  all(ForwardDiff.partials.(pz1) .== ForwardDiff.partials.(pzd))
```

TPS arithmetic is not safe on several threads (GTPSA's C library shares scratch space between Julia threads), so
the kick is applied to a TPS `pz` on the calling thread.
