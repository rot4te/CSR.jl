# Kick interpolation kernel. Port of the CSR-only branch of Bmad's csr_and_sc_apply_kicks.

"""
    csr_apply_kicks_kernel!(pz, n_outside, z, alive, z_center, kick, dz, pz_scale)

One work item per particle: `pz += pz_scale (r0 kick[i0] + r1 kick[i0+1])`, linear interpolation between the bin
centres around `z` (Fortran `int` truncation for the cell, as Bmad). Particles outside the table are counted in
`n_outside` (`Int32`: no 64-bit atomics on some GPUs).
"""
@kernel function csr_apply_kicks_kernel!(pz, n_outside, @Const(z), @Const(alive), @Const(z_center), @Const(kick), dz, pz_scale)
  i = @index(Global)
  @inbounds if alive[i]
    n_bin = length(kick)
    zp = z[i]
    i0 = trunc(Int, (zp - z_center[1]) / dz) + 1   # Fortran int() truncates toward zero
    if 1 <= i0 < n_bin
      r1 = (zp - z_center[i0]) / dz
      if oftype(r1, -0.01) <= r1 <= oftype(r1, 1.01)
        r0 = 1 - r1
        pz[i] += pz_scale * (r0 * kick[i0] + r1 * kick[i0+1])
      else
        Atomix.@atomic n_outside[1] += one(eltype(n_outside))
      end
    else
      Atomix.@atomic n_outside[1] += one(eltype(n_outside))
    end
  end
end
