# CSR.jl

One-dimensional (line-charge) coherent synchrotron radiation (CSR) for the
[SciBmad](https://github.com/bmad-sim/SciBmad.jl) ecosystem. The algorithm is a port of Bmad's
`csr_method = one_dim` (D. Sagan, Bmad manual; `bmad/space_charge/csr_and_space_charge_mod.f90`), with optional
shielding by parallel plates (image charges).

- The **core** works on plain particle arrays (`z`, `pz`, charges, alive mask) with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels: [`csr_kick!`](@ref) applies one
  CSR kick.
- The **BeamTracking extension** makes CSR an effect of BeamTracking's `track!`:
  `track!(bunch, bl; effects = (CSREffect(params),))` applies one kick at the entrance of each element slice
  ([`CSREffect`](@ref)). [`CSRRecord`](@ref) and [`CSRReplay`](@ref) give test particles the bunch's kicks.
- **Number types**: ForwardDiff `Dual` and GTPSA TPS inputs are accepted. They are scalarized, and the kick changes
  only the value of `pz`; no derivatives are tracked through the kick.

By default CSR.jl corrects several approximations and errors found in Bmad's implementation;
`CSRParams(bmad_compat = true)` reproduces Bmad. Each such choice is listed, with its effect, on the
[Physics decisions](@ref) page.

## Installation

CSR.jl is not registered. It needs Beamlines 0.12 and, for tracking, BeamTracking 0.9 with features that exist
only on local development branches: BeamTracking `collective/stage-06-slice-effects` (the `effects` keyword of
`track!`) and Beamlines `collective/stage-01b-restore-scalarize-test` (the recursive `Beamlines.scalarize`). Develop
the checkouts into one environment:

```julia
using Pkg
Pkg.develop(path = "path/to/CSR.jl")
Pkg.develop(path = "path/to/Beamlines.jl")
Pkg.develop(path = "path/to/BeamTracking.jl")
Pkg.develop(path = "path/to/AtomicAndPhysicalConstants.jl")   # 0.12
```

## Minimal example

Steady-state CSR kick on a Gaussian line bunch at the entrance of the last element of an 8 m arc of radius 10 m,
using the core API:

```@example index
using CSR, AtomicAndPhysicalConstants

z = collect(range(-6e-3, 6e-3; length = 20001))   # σ_z = 1 mm
q = exp.(-z .^ 2 ./ 2e-6); q .*= 1e-9 / sum(q)    # 1 nC
alive = fill(true, length(z))

# 80 bends of 0.1 m, ρ = 10 m, centroid on the reference orbit
geom = CSRGeometry(fill(0.1, 80), fill(1 / 10.0, 80), zeros(81, 6))
params = CSRParams(n_bin = 200)
pz = zeros(length(z))
# Kick at the entrance of element 80, covering 0.1 m, γ = 10⁴
r = csr_kick!(pz, z, q, alive, geom, 80, 0.1, 0.0, 1e4,
              Species("electron"), params)
extrema(r.kick_per_meter)
```

See [Usage](@ref) for tracking through a `Beamline`.
