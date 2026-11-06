# Deposit and edge-gradient kernels (triangle shape function, as
# Bmad's csr_bin_particles).

"""
    triangle_overlap(zp, dz_particle, zb0, zb1)

Fraction of a unit-area triangular distribution centered at `zp` with
base `dz_particle` that lies in `[zb0, zb1]`.

# Arguments
- `zp`: particle position [m], `z[i]`; `Dual` or TPS allowed (branch
  decisions on the primal).
- `dz_particle`: triangle base [m], `particle_bin_span * dz`
  (`CSRParams`, bin width `dz`).
- `zb0`, `zb1`: lower and upper bin edge [m] (for grid bin `ib`:
  `z_min + (ib - 1) * dz` and `zb0 + dz`).

# Returns
- the overlap fraction, in the number type of `zp` and the edges.
"""
function triangle_overlap(zp, dz_particle, zb0, zb1)
  zp0 = zp - dz_particle / 2
  zp1 = zp + dz_particle / 2
  inv_dz2 = 1 / dz_particle^2
  overlap = zero(zp * inv_dz2)
  z1 = max(zp0, zb0)
  z2 = min(zp, zb1)
  Beamlines.scalarize(z2) > Beamlines.scalarize(z1) &&
      (overlap += 2 * ((z2 - zp0)^2 - (z1 - zp0)^2) * inv_dz2)
  z1 = max(zp, zb0)
  z2 = min(zp1, zb1)
  Beamlines.scalarize(z2) > Beamlines.scalarize(z1) &&
      (overlap += 2 * ((z1 - zp1)^2 - (z2 - zp1)^2) * inv_dz2)
  return overlap
end

"""
    deposit_ix0(zp, z_min, dz, dz_particle)

Start of the deposit loop: the bin containing the triangle's left
end, or the one before it.

# Arguments
- `zp`: particle position [m], `z[i]` (only the primal is used).
- `z_min`: lower edge [m] of bin 1 (`bins.z0_edge[1]`).
- `dz`: bin width [m], `bins.dz`.
- `dz_particle`: triangle base [m], `particle_bin_span * dz`.

# Returns
- `Int` bin index (`nint`), start of the deposit loop; may be
  outside `1:n_bin`.
"""
@inline deposit_ix0(zp, z_min, dz,
                    dz_particle) = nint((Beamlines.scalarize(zp) -
                        dz_particle / 2 - z_min) / dz)

"""
    deposit_particle!(add!, charge, n_particle, n_outside, zp, qp,
                      z_min, dz, dz_particle, pbs)

Spread one particle over bins ix0 : ix0 + span + 1. Bins outside the
grid are counted in n_outside.

# Arguments
- `add!`: scatter-add functor `add!(A, i, v)`, `AtomicAdd()` or
  `PlainAdd()`.
- `charge`: bin charges [C], `bins.charge` (modified).
- `n_particle`: primal overlap sum per bin, `bins.n_particle`
  (modified).
- `n_outside`: one-element `Int32` counter (modified: `+1` per bin
  of the loop outside `1:n_bin`).
- `zp`, `qp`: position [m] and charge [C] of the particle, `z[i]`,
  `q[i]`.
- `z_min`: lower edge [m] of bin 1 (`bins.z0_edge[1]`).
- `dz`: bin width [m].
- `dz_particle`: triangle base [m], `pbs * dz`.
- `pbs`: `params.particle_bin_span` (`CSRParams`); the `span`
  above. `ix0` is `deposit_ix0`.
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
      add!(n_particle, ib, Beamlines.scalarize(overlap))
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
`deposit_particle!` and atomic adds. Launched by `csr_bin_particles`
for `Float32`/`Float64` bins (`supports_atomic_add`).

# Arguments
- `charge`, `n_particle`, `n_outside`, `z_min`, `dz`, `dz_particle`,
  `pbs`: as in `deposit_particle!` (`charge`, `n_particle`,
  `n_outside` modified).
- `z`: particle longitudinal positions [m].
- `q`: macroparticle charges [C].
- `alive`: live-particle mask (`Bool`).
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
`(charge[i] - charge[i-1]) / dz²` (0 for `i = 1`). Launched by
`csr_bin_particles`.

# Arguments
- `edge`: gradient per bin (overwritten),
  `bins.edge_dcharge_density_dz` (`CSRBins`).
- `charge`: bin charges [C], `bins.charge` after the deposit.
- `dz`: bin width [m], `bins.dz`.
"""
@kernel function csr_edge_gradient_kernel!(edge, @Const(charge), dz)
  ib = @index(Global)
  @inbounds edge[ib] = ib == 1 ? zero(eltype(edge)) : (charge[ib] -
      charge[ib-1]) / dz^2
end
