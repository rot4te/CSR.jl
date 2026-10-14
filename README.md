# CSR

This package computes one-dimensional (line-charge) coherent synchrotron radiation (CSR) kicks on charged particle
bunches. It is a port of Bmad's `csr_method = one_dim`, written with
[`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels and differentiable with
ForwardDiff and ChainRules.

`CSR.jl` is intended for the [SciBmad](https://github.com/bmad-sim/SciBmad.jl) ecosystem of accelerator physics
software: with [`BeamTracking.jl`](https://github.com/bmad-sim/BeamTracking.jl) and
[`Beamlines.jl`](https://github.com/bmad-sim/Beamlines.jl) loaded, `track_csr!` tracks a bunch through a beamline
with CSR kicks between element slices.

Currently implemented and tested (CPU; the particle-side kernels also run on GPU backends, e.g. Apple Metal in
`Float32`):

- Free-space 1D CSR: retarded source search along the centroid orbit, kick integral, causal convolution, as Bmad.
  Cross-checked against Bmad/Tao to the precision of Bmad's diagnostic output (`bmad_compat = true`).
- Shielding by parallel plates (image charges), including sources ahead of the kick point and past the lattice end.
- Fixes of Bmad approximations and errors in the default mode, each reversible with `bmad_compat = true`: exact
  path-length excess, first-bin and all-bin quadrature, hat-integrated image kernel, averaged image-layer sum, and
  others (see the documentation's "Physics decisions" page).
- Element slicing following Bmad's rules for all Beamlines element kinds BeamTracking tracks, including RungeKutta
  field maps and SaganCavity (lcavity) elements with a reference-energy ramp.
- Derivatives of a kick with respect to the particle coordinates and charges: ForwardDiff `Dual`s and an `rrule`
  with explicit adjoint kernels.

## Documentation

Build locally with `julia --project=docs docs/make.jl` (see `docs/README.md`); the pages cover usage, the physics
model, every decision that changes the physics relative to Bmad, and the full API.
