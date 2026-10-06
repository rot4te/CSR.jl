# Bmad cross-check (phase 3)

Compares CSR.jl (`track_csr!` on BeamTracking, `bmad_compat = true`) with Bmad/Tao on identical initial particles.
Bmad: conda-forge `bmad` 20260918.0 (its `csr_and_space_charge_mod.f90` is identical to the 20261001 source).

## Run

Needs `tao` (default `~/miniforge3/envs/bmad/bin/tao`, or set `ENV["TAO"]`) and a Julia environment with CSR,
BeamTracking and Beamlines. From this directory:

```
julia --project=<env> run_case.jl <case> <beamline> [cx cy]   # writes <case>/*.dat; output saved in <case>/result.txt
julia --project=<env> geom_check.jl <case> <beamline>          # after run_case.jl
```

| Case | Lattice | Command arguments |
|---|---|---|
| `single_bend` | bend of `regression_tests/csr_and_space_charge_test` (b1_gradient −10, particle_start x 0.2 mm, y 0.1 mm) | `single_bend br0 2e-4 1e-4` |
| `pure_bend` | same bend without b1_gradient or particle_start | `pure_bend br0` |
| `pure_bend_offset` | same bend without b1_gradient | `pure_bend_offset br0 2e-4 1e-4` |
| `csr_beam_tracking` | `bmad-doc/tao_examples/csr_beam_tracking` (drifts, two combined-function bends) | `csr_beam_tracking lat` |
| `csr_beam_tracking_patch` | same with its patch `p1` enabled | `csr_beam_tracking_patch lat` |

`lat.jl` is `bmad_to_scibmad lat.bmad` with `.` in names replaced by `_`. Initial particles: Tao's `beam_init`
(1000 particles, quasi-random), written at BEGINNING and read by both codes. CSR: n_bin 40, particle_bin_span 2,
ds_step 0.01; shielded: n_shield_images 4, beam_chamber_height 0.01. The CSR centroid starts on Tao's design orbit
(`particle_start`), as Tao passes its single-particle orbit to `track1_bunch_csr`.

## What is compared

- `run_case.jl`: (1) tracking without CSR; (2) per-step wake (Bmad `diagnostic_output_file`, 7 significant digits);
  (3) the CSR-induced change of the final coordinates, (CSR on) − (CSR off). CSR.jl's "off" is `track_csr!` with the
  charges scaled by 1e-30 (same slicing), because BeamTracking's integrators depend on the slicing.
- `geom_check.jl`: CSR alone, independent of tracking: CSR.jl's reference floor positions vs Bmad's, and `I_csr` and
  `s_source` per step computed by CSR.jl from Bmad's design orbit.

Notes on Bmad's diagnostic file: with image charges it divides kick and I_csr by the last image layer's
`kick_factor` (`(−1)^n·2` times the free-space one) and prints that layer's source points; the S-position has 6
decimals (use `i_step·ds`).
