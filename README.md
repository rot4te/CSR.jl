# CSR

This package computes one-dimensional (line-charge) coherent synchrotron radiation (CSR) kicks on charged particle
bunches. It is a port of Bmad's `csr_method = one_dim`, with optional shielding by parallel plates (image charges),
written with [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels.

`CSR.jl` is intended for the [SciBmad](https://github.com/bmad-sim/SciBmad.jl) ecosystem of accelerator physics
software. With [`BeamTracking.jl`](https://github.com/bmad-sim/BeamTracking.jl) loaded, CSR is an effect of
BeamTracking's `track!`: `track!(bunch, bl; effects = (CSREffect(params),))` tracks a bunch through a
[`Beamlines.jl`](https://github.com/bmad-sim/Beamlines.jl) `Beamline` and applies one CSR kick at the entrance of
each element slice.

Currently implemented and tested (CPU):

- Free-space 1D CSR: retarded source search along the centroid orbit, kick integral, causal convolution, as Bmad.
  Cross-checked against Bmad/Tao to the precision of Bmad's diagnostic output (`bmad_compat = true`).
- Shielding by parallel plates (image charges), including sources ahead of the kick point and past the lattice end.
- Fixes of Bmad approximations and errors in the default mode, each reversible with `bmad_compat = true`: exact
  path-length excess, first-bin and all-bin quadrature, hat-integrated image kernel, averaged image-layer sum, and
  others (see the documentation's "Physics decisions" page).
- Test particles that get the bunch's kicks without contributing to them (`CSRRecord`, `CSRReplay`).

## Installation for development

CSR.jl is not registered. It needs Beamlines 0.12 and BeamTracking 0.9 with features that exist only on local
development branches, not in registered releases:

- BeamTracking.jl, branch `collective/stage-06-slice-effects`: the `effects` keyword of `track!`.
- Beamlines.jl, branch `collective/stage-01b-restore-scalarize-test`: the recursive `Beamlines.scalarize`.
- AtomicAndPhysicalConstants.jl 0.12 (CSR's compat), if no registered 0.12 is available.

Check out these branches, then develop all checkouts into one environment:

```julia
using Pkg
Pkg.develop(path = "path/to/CSR.jl")
Pkg.develop(path = "path/to/Beamlines.jl")
Pkg.develop(path = "path/to/BeamTracking.jl")
Pkg.develop(path = "path/to/AtomicAndPhysicalConstants.jl")
```

BeamTracking is a weak dependency: the extension `CSRBeamTrackingExt`, which provides the tracking effect, loads
when BeamTracking is loaded.

## Example

Electrons with γ = 10⁴ through a 2 m arc of radius 10 m and a drift:

```julia
using CSR, BeamTracking, Beamlines, AtomicAndPhysicalConstants

e = Species("electron")
pq = Beamlines.E_to_R(e, 1e4 * massof(e))   # γ = 10⁴

# Line bunch: σ_z = 1 mm, 1 nC, Gaussian weights on a z grid
n = 2001
v = zeros(n, 6)
v[:, 5] .= range(-6e-3, 6e-3; length = n)
w = exp.(-v[:, 5] .^ 2 ./ 2e-6)
w .*= (1e-9 / E_CHARGE) / sum(w)   # real particles per macroparticle
bunch = Bunch(v, nothing, w; species = e, p_over_q_ref = pq)

# 2 m arc of radius 10 m (20 bends, 2 slices each), then a drift
rho = 10.0
line = [[SBend(L = 0.1, g_ref = 1 / rho, Kn0 = 1 / rho, n_slices = 2)
         for _ in 1:20]; Drift(L = 0.5, n_slices = 5)]
bl = Beamline(line; p_over_q_ref = pq, species_ref = e)

track!(bunch, bl; effects = (CSREffect(CSRParams(n_bin = 80)),))
extrema(bunch.coords.v[:, 6])
```

The step length is set per element with the Beamlines property `n_slices`. Kicked are the bends (nonzero `g_ref`)
and every element after the first bend, except zero-length elements and `SaganCavity` elements.

## Known limitations

- No CSR in cavities: elements tracked with `SaganCavity` are skipped.
- No `Branch` tracking: `CSREffect` works on one `Beamline`.
- CPU only for tracking with `CSREffect`.
- One kick at the entrance of each slice, covering the slice length: first order in the slice length.
- No derivatives through the kick. ForwardDiff `Dual` and GTPSA TPS inputs are accepted, but they are scalarized
  and the kick changes only the value of `pz`.
- Time-dependent reference momentum together with CSR is untested.
- `bmad_compat = true` with shielding images gives isolated large image-charge kicks at the low-charge head bins.
  This was also present before the current tracking driver; the cause is not isolated.
- The scripts in `crosscheck/` and `diagnostics/` still use the removed `track_csr!` call and have not been ported.

## Documentation

Build locally with `julia --project=docs docs/make.jl` (see `docs/README.md`). The pages cover usage, the physics
model, every decision that changes the physics relative to Bmad, and the full API.

## Still to Come

- Three dimensional CSR kicks
