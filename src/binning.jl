# Longitudinal binning (host side). Port of the charge part of Bmad's csr_bin_particles.

"""
    csr_bin_particles(z, q, alive, params) -> CSRBins

Bin the live particles (`alive[i] == true`) with longitudinal positions `z` [m] and charges `q` [C].
Each particle is spread as a triangle of base `particle_bin_span * dz`. The grid is centered on the
live-particle z range, with empty padding bins at both ends.

Runs as KernelAbstractions kernels on the backend of `z`. The deposit uses atomic adds for `Float32`/`Float64`
charges, so the summation order (and the last bits of the bin charges) can vary between runs with several
threads. Other element types are deposited serially on the CPU.

The grid (`dz`, `z0_edge`, `z_center`) and `n_particle` are built from primal values (`primal`): with ForwardDiff
Duals only `charge` and `edge_dcharge_density_dz` carry derivatives, and the grid is held constant.
"""
function csr_bin_particles(z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, params::CSRParams)
  n_bin = params.n_bin
  pbs = params.particle_bin_span
  n_bin_eff = n_bin - 2 - (pbs + 1)
  backend = get_backend(z)

  any(alive) || error("csr_bin_particles: no live particles")
  z_maxval = primal(mapreduce((zi, a) -> a ? zi : oftype(zi, -Inf), max, z, alive))
  z_minval = primal(mapreduce((zi, a) -> a ? zi : oftype(zi, Inf), min, z, alive))
  z_maxval > z_minval || error("csr_bin_particles: longitudinal width of beam is zero")

  # 1.0000001 prevents round-off problems (Bmad). Grid in the particles' float type (no Float64 on some GPUs).
  dz = oftype(float(z_maxval), 1.0000001) * (z_maxval - z_minval) / n_bin_eff
  z_center = (z_maxval + z_minval) / 2
  z_min = z_center - n_bin * dz / 2
  dz_particle = pbs * dz

  Tz = typeof(dz)
  Tq = promote_type(Tz, eltype(z), eltype(q))
  z0_edge = KernelAbstractions.allocate(backend, Tz, n_bin)
  z0_edge .= (i -> z_min + i * dz).(0:n_bin-1)   # not range arithmetic: Float32 ranges carry Float64 internals
  bins = CSRBins(dz, z0_edge, z0_edge .+ dz / 2, KernelAbstractions.zeros(backend, Tq, n_bin),
                 KernelAbstractions.zeros(backend, Tz, n_bin), KernelAbstractions.allocate(backend, Tq, n_bin))

  n_outside = KernelAbstractions.zeros(backend, Int32, 1)   # Int32: no 64-bit atomics on some GPUs
  if supports_atomic_add(bins.charge) && supports_atomic_add(bins.n_particle)
    csr_deposit_kernel!(backend)(bins.charge, bins.n_particle, n_outside, z, q, alive, z_min, dz, dz_particle, pbs;
                                 ndrange = length(z))
  else
    backend isa CPU || error("csr_bin_particles: element type $Tq needs the CPU backend (no atomic add)")
    for i in eachindex(z)
      alive[i] && deposit_particle!(PlainAdd(), bins.charge, bins.n_particle, n_outside, z[i], q[i], z_min, dz,
                                    dz_particle, pbs)
    end
  end
  csr_edge_gradient_kernel!(backend)(bins.edge_dcharge_density_dz, bins.charge, dz; ndrange = n_bin)
  KernelAbstractions.synchronize(backend)
  Array(n_outside)[1] == 0 || error("csr_bin_particles: particle outside bin range (internal error)")
  return bins
end
