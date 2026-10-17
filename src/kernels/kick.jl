# Kick interpolation kernel. Port of the CSR-only branch of Bmad's csr_and_sc_apply_kicks.

"""
    kick_cell(zp, z_center, dz) -> (i0, inside)

Interpolation cell of a particle at `zp` for `csr_apply_kicks_kernel!`: `i0` from the primal `zp` (Fortran `int`
truncation toward zero, as Bmad), and `inside = true` if `1 ≤ i0 < n_bin` and the primal
`r1 = (zp - z_center[i0]) / dz` lies in `[-0.01, 1.01]`. A non-finite primal `zp` (NaN, ±Inf) or one more than
`n_bin + 1` bins from `z_center[1]` gives `(0, false)`. Computed from primal values only, so the count
(`csr_count_outside_kernel!`) and the kick agree for every number type.
"""
@inline function kick_cell(zp, z_center, dz)
  zpp = primal(zp)
  u = (zpp - z_center[1]) / dz
  # NaN, Inf or far outside the table: outside (trunc(Int, ·) would throw)
  (isfinite(u) && abs(u) < length(z_center) + 1) || return 0, false
  i0 = trunc(Int, u) + 1   # Fortran int() truncates toward zero
  inside = false
  if 1 <= i0 < length(z_center)
    r1p = (zpp - z_center[i0]) / dz
    inside = oftype(r1p, -0.01) <= r1p <= oftype(r1p, 1.01)
  end
  return i0, inside
end

"""
    csr_apply_kicks_kernel!(pz, n_outside, z, alive, z_center, kick, dz, pz_scale)

One work item per particle: `pz += pz_scale (r0 kick[i0] + r1 kick[i0+1])`, linear interpolation between the bin
centres around `z`, with the cell and the range test from `kick_cell` (primal `z`, so TPS or `Dual` `z` expands
within the cell). Particles outside the table are counted in `n_outside` (`Int32`: no 64-bit atomics on some GPUs).
"""
@kernel function csr_apply_kicks_kernel!(pz, n_outside, @Const(z), @Const(alive), @Const(z_center), @Const(kick), dz, pz_scale)
  i = @index(Global)
  @inbounds if alive[i]
    zp = z[i]
    i0, inside = kick_cell(zp, z_center, dz)
    if inside
      r1 = (zp - z_center[i0]) / dz
      r0 = 1 - r1
      pz[i] += pz_scale * (r0 * kick[i0] + r1 * kick[i0+1])
    else
      Atomix.@atomic n_outside[1] += one(eltype(n_outside))
    end
  end
end

"""
    csr_count_outside_kernel!(n_outside, z, alive, z_center, dz)

One work item per particle: count the live particles that `csr_apply_kicks_kernel!` would not kick (outside the
table, `kick_cell`) in `n_outside` (`Int32`). Modifies nothing else.
"""
@kernel function csr_count_outside_kernel!(n_outside, @Const(z), @Const(alive), @Const(z_center), dz)
  i = @index(Global)
  @inbounds if alive[i]
    _, inside = kick_cell(z[i], z_center, dz)
    inside || Atomix.@atomic n_outside[1] += one(eltype(n_outside))
  end
end
