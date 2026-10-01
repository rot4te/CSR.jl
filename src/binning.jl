# Longitudinal binning. Port of the charge part of Bmad's csr_bin_particles.

"""
    CSRBins

Longitudinal bins. Bin `i` spans `[z0_edge[i], z0_edge[i] + dz)`. `edge_dcharge_density_dz[i]` is the
charge density gradient at the left edge of bin `i`: `(charge[i] - charge[i-1]) / dz²` (zero for `i = 1`).
"""
struct CSRBins
  dz::Float64
  z0_edge::Vector{Float64}
  z_center::Vector{Float64}
  charge::Vector{Float64}
  n_particle::Vector{Float64}
  edge_dcharge_density_dz::Vector{Float64}
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
  overlap = 0.0
  z1 = max(zp0, zb0)
  z2 = min(zp, zb1)
  z2 > z1 && (overlap += 2 * ((z2 - zp0)^2 - (z1 - zp0)^2) * inv_dz2)
  z1 = max(zp, zb0)
  z2 = min(zp1, zb1)
  z2 > z1 && (overlap += 2 * ((z1 - zp1)^2 - (z2 - zp1)^2) * inv_dz2)
  return overlap
end

"""
    csr_bin_particles(z, q, alive, params) -> CSRBins

Bin the live particles (`alive[i] == true`) with longitudinal positions `z` [m] and charges `q` [C].
Each particle is spread as a triangle of base `particle_bin_span * dz`. The grid is centered on the
live-particle z range, with empty padding bins at both ends.
"""
function csr_bin_particles(z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, params::CSRParams)
  n_bin = params.n_bin
  pbs = params.particle_bin_span
  n_bin_eff = n_bin - 2 - (pbs + 1)

  any(alive) || error("csr_bin_particles: no live particles")
  z_maxval = maximum(z[i] for i in eachindex(z) if alive[i])
  z_minval = minimum(z[i] for i in eachindex(z) if alive[i])
  z_maxval > z_minval || error("csr_bin_particles: longitudinal width of beam is zero")

  dz = 1.0000001 * (z_maxval - z_minval) / n_bin_eff   # 1.0000001 prevents round-off problems (Bmad)
  z_center = (z_maxval + z_minval) / 2
  z_min = z_center - n_bin * dz / 2
  dz_particle = pbs * dz

  z0_edge = [z_min + (i - 1) * dz for i in 1:n_bin]
  bins = CSRBins(dz, z0_edge, z0_edge .+ dz / 2, zeros(n_bin), zeros(n_bin), zeros(n_bin))

  for i in eachindex(z)
    alive[i] || continue
    zp = z[i]
    zp0 = zp - dz_particle / 2
    ix0 = round(Int, (zp0 - z_min) / dz, RoundNearestTiesAway)   # Fortran nint
    for j in 0:pbs+1
      ib = j + ix0
      1 <= ib <= n_bin || error("csr_bin_particles: particle outside bin range (internal error)")
      overlap = triangle_overlap(zp, dz_particle, z0_edge[ib], z0_edge[ib] + dz)
      bins.n_particle[ib] += overlap
      bins.charge[ib] += overlap * q[i]
    end
  end

  for ib in 2:n_bin
    bins.edge_dcharge_density_dz[ib] = (bins.charge[ib] - bins.charge[ib-1]) / dz^2
  end
  return bins
end
