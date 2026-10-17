# CSR.jl

One-dimensional (line-charge) coherent synchrotron radiation (CSR) for the
[SciBmad](https://github.com/bmad-sim/SciBmad.jl) ecosystem. The algorithm is a port of Bmad's
`csr_method = one_dim` (D. Sagan, Bmad manual; `bmad/space_charge/csr_and_space_charge_mod.f90`), with optional
shielding by parallel plates (image charges).

- The **core** works on plain particle arrays (`z`, `pz`, charges, alive mask) with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels: [`csr_kick!`](@ref) applies one
  CSR kick.
- The **BeamTracking + Beamlines extension** provides [`track_csr!`](@ref): it slices every element, tracks each
  slice with BeamTracking, and applies a CSR kick between slices.
- **Derivatives** of a kick with respect to the particle coordinates and charges, the geometry (element lengths,
  curvatures, centroid), the kick position and γ: ForwardDiff `Dual`s pass through, and with ChainRulesCore `rrule`s for
  [`csr_kick_pz`](@ref) and `CSRGeometry` give the pullbacks.

By default CSR.jl corrects several approximations and errors found in Bmad's implementation;
`CSRParams(bmad_compat = true)` reproduces Bmad. Each such choice is listed, with its effect, on the
[Physics decisions](@ref) page.

## Installation

CSR.jl is not registered. Add it from its repository; the tracking extension needs BeamTracking ≥ 0.8.7 and
Beamlines 0.10.6 or 0.11:

```julia
using Pkg
Pkg.add(url = "https://github.com/rot4te/CSR.jl")
Pkg.add(["BeamTracking", "Beamlines"])
```

## Minimal example

Steady-state CSR kick on a Gaussian line bunch at the end of an 8 m arc of radius 10 m, using the core API:

```@example index
using CSR, AtomicAndPhysicalConstants

n = 20001
z = collect(range(-6e-3, 6e-3; length = n))                  # 1 mm bunch, ±6σ
q = exp.(-z .^ 2 ./ 2e-6); q .*= 1e-9 / sum(q)              # 1 nC
alive = fill(true, n)

geom = CSRGeometry(fill(0.1, 80), fill(1 / 10.0, 80), zeros(81, 6))   # 80 bends, centroid on axis
params = CSRParams(n_bin = 200, ds_step = 0.1)
pz = zeros(n)
r = csr_kick!(pz, z, q, alive, geom, 80, 0.05, 1e4, Species("electron"), params)
extrema(r.kick_per_meter)
```
