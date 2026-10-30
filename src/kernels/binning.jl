# Deposit and edge-gradient kernels (triangle shape function, as
# Bmad's csr_bin_particles).

"""
    triangle_overlap(zp, dz_particle, zb0, zb1)

Fraction of a unit-area triangular distribution centered at `zp` with
base `dz_particle` that lies in `[zb0, zb1]`.
"""
function triangle_overlap(zp, dz_particle, zb0, zb1)
  zp0 = zp - dz_particle / 2
  zp1 = zp + dz_particle / 2
  inv_dz2 = 1 / dz_particle^2
  overlap = zero(zp * inv_dz2)
  z1 = Beamlines.scalar_max(zp0, zb0)
  z2 = Beamlines.scalar_min(zp, zb1)
  Beamlines.scalarize_all(z2) > Beamlines.scalarize_all(z1) &&
      (overlap += 2 * ((z2 - zp0)^2 - (z1 - zp0)^2) * inv_dz2)
  z1 = Beamlines.scalar_max(zp, zb0)
  z2 = Beamlines.scalar_min(zp1, zb1)
  Beamlines.scalarize_all(z2) > Beamlines.scalarize_all(z1) &&
      (overlap += 2 * ((z1 - zp1)^2 - (z2 - zp1)^2) * inv_dz2)
  return overlap
end

"""
    deposit_ix0(zp, z_min, dz, dz_particle)

First bin a particle's triangle can overlap.
"""
@inline deposit_ix0(zp, z_min, dz,
                    dz_particle) = nint((Beamlines.scalarize_all(zp) -
                        dz_particle / 2 - z_min) / dz)

"""
    deposit_particle!(add!, charge, n_particle, n_outside, zp, qp,
                      z_min, dz, dz_particle, pbs)

Spread one particle over bins ix0 : ix0 + span + 1. Bins outside the
grid are counted in n_outside.
"""
@inline function deposit_particle!(add!, charge, n_particle,
                                   n_outside, zp, qp, z_min, dz,
                                   dz_particle, pbs)
  n_bin = length(charge)
  ix0 = deposit_ix0(zp, z_min, dz, dz_particle)
  for j in 0:pbs+1
    ib = j + ix0
    if 1 <= ib <= n_bin
      zb0 = z_min + (ib - 1) * dz
      overlap = triangle_overlap(zp, dz_particle, zb0, zb0 + dz)
      add!(n_particle, ib, Beamlines.scalarize_all(overlap))
      add!(charge, ib, overlap * qp)
    else
      add!(n_outside, 1, one(eltype(n_outside)))
    end
  end
  return nothing
end

"""
    csr_deposit_kernel!(charge, n_particle, n_outside, z, q, alive,
                        z_min, dz, dz_particle, pbs)

One work item per particle: deposit live particles with
`deposit_particle!` and atomic adds.
"""
@kernel function csr_deposit_kernel!(charge, n_particle, n_outside,
                                     @Const(z), @Const(q),
                                     @Const(alive),
                                     z_min, dz, dz_particle, pbs)
  i = @index(Global)
  @inbounds if alive[i]
    deposit_particle!(AtomicAdd(), charge, n_particle, n_outside,
                      z[i], q[i], z_min, dz, dz_particle, pbs)
  end
end

"""
    csr_edge_gradient_kernel!(edge, charge, dz)

One work item per bin: charge-density gradient at the left bin edge,
`(charge[i] - charge[i-1]) / dz²` (0 for `i = 1`).
"""
@kernel function csr_edge_gradient_kernel!(edge, @Const(charge), dz)
  ib = @index(Global)
  @inbounds edge[ib] = ib == 1 ? zero(eltype(edge)) : (charge[ib] -
      charge[ib-1]) / dz^2
end
