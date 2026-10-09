# Longitudinal binning. Port of the charge part of Bmad's csr_bin_particles.

"""
    CSRBins

Longitudinal bins. Bin `i` spans `[z0_edge[i], z0_edge[i] + dz)`. `edge_dcharge_density_dz[i]` is the
charge density gradient at the left edge of bin `i`: `(charge[i] - charge[i-1]) / dz²` (zero for `i = 1`).
Arrays live on the backend of the particle coordinates.
"""
struct CSRBins{T,V<:AbstractVector,W<:AbstractVector}
  dz::T
  z0_edge::V
  z_center::V
  charge::W
  n_particle::V
  edge_dcharge_density_dz::W
end

"""
    triangle_overlap(zp, dz_particle, zb0, zb1)

Fraction of a unit-area triangular distribution centered at `zp` with base `dz_particle` that lies in
`[zb0, zb1]`.
"""
function triangle_overlap(zp, dz_particle, zb0, zb1)
  zp0 = zp - dz_particle / 2
  zp1 = zp + dz_particle / 2
  inv_dz2 = 1 / dz_particle^2
  overlap = zero(zp * inv_dz2)
  z1 = max(zp0, zb0)
  z2 = min(zp, zb1)
  z2 > z1 && (overlap += 2 * ((z2 - zp0)^2 - (z1 - zp0)^2) * inv_dz2)
  z1 = max(zp, zb0)
  z2 = min(zp1, zb1)
  z2 > z1 && (overlap += 2 * ((z1 - zp1)^2 - (z2 - zp1)^2) * inv_dz2)
  return overlap
end

# Fortran nint (round half away from zero). The fallback is exact: x - trunc(x) has no rounding error.
nint(x::Union{Float32,Float64}) = round(Int, x, RoundNearestTiesAway)
function nint(x)
  t = trunc(x)
  return Int(t) + (abs(x - t) >= 0.5 ? Int(sign(x)) : 0)
end

# Scatter-add used by the deposit: atomic for hardware floats, plain otherwise (serial path).
struct AtomicAdd end
(::AtomicAdd)(A, i, v) = (Atomix.@atomic A[i] += v; nothing)
struct PlainAdd end
(::PlainAdd)(A, i, v) = (A[i] += v; nothing)

supports_atomic_add(A) = eltype(A) <: Union{Float32,Float64}

"""
    primal(x)

Value of `x` without derivative parts (identity for plain numbers; extensions add methods, e.g. ForwardDiff Duals).
The bin grid is built from primal values, so derivatives treat it as constant.
"""
primal(x) = x

# First bin a particle's triangle can overlap.
@inline deposit_ix0(zp, z_min, dz, dz_particle) = nint((primal(zp) - dz_particle / 2 - z_min) / dz)

# Spread one particle over bins ix0 : ix0 + span + 1. Bins outside the grid are counted in n_outside.
@inline function deposit_particle!(add!, charge, n_particle, n_outside, zp, qp, z_min, dz, dz_particle, pbs)
  n_bin = length(charge)
  ix0 = deposit_ix0(zp, z_min, dz, dz_particle)
  for j in 0:pbs+1
    ib = j + ix0
    if 1 <= ib <= n_bin
      zb0 = z_min + (ib - 1) * dz
      overlap = triangle_overlap(zp, dz_particle, zb0, zb0 + dz)
      add!(n_particle, ib, primal(overlap))
      add!(charge, ib, overlap * qp)
    else
      add!(n_outside, 1, 1)
    end
  end
  return nothing
end

@kernel function csr_deposit_kernel!(charge, n_particle, n_outside, @Const(z), @Const(q), @Const(alive),
                                     z_min, dz, dz_particle, pbs)
  i = @index(Global)
  @inbounds if alive[i]
    deposit_particle!(AtomicAdd(), charge, n_particle, n_outside, z[i], q[i], z_min, dz, dz_particle, pbs)
  end
end

@kernel function csr_edge_gradient_kernel!(edge, @Const(charge), dz)
  ib = @index(Global)
  @inbounds edge[ib] = ib == 1 ? zero(eltype(edge)) : (charge[ib] - charge[ib-1]) / dz^2
end

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

  dz = 1.0000001 * (z_maxval - z_minval) / n_bin_eff   # 1.0000001 prevents round-off problems (Bmad)
  z_center = (z_maxval + z_minval) / 2
  z_min = z_center - n_bin * dz / 2
  dz_particle = pbs * dz

  Tz = typeof(dz)
  Tq = promote_type(Tz, eltype(z), eltype(q))
  z0_edge = KernelAbstractions.allocate(backend, Tz, n_bin)
  z0_edge .= z_min .+ (0:n_bin-1) .* dz
  bins = CSRBins(dz, z0_edge, z0_edge .+ dz / 2, KernelAbstractions.zeros(backend, Tq, n_bin),
                 KernelAbstractions.zeros(backend, Tz, n_bin), KernelAbstractions.allocate(backend, Tq, n_bin))

  n_outside = KernelAbstractions.zeros(backend, Int, 1)
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
