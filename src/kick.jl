# Apply the binned kick to particles. Port of the CSR-only branch of Bmad's csr_and_sc_apply_kicks.

"""
    csr_apply_kicks!(pz, z, alive, bins, kick)

Add the kick, linearly interpolated between bin centers, to `pz` of each live particle.
As in Bmad's CSR-only branch, `z` is not rescaled for the change in velocity.
"""
function csr_apply_kicks!(pz::AbstractVector, z::AbstractVector, alive::AbstractVector{Bool}, bins::CSRBins, kick::AbstractVector)
  n_bin = length(kick)
  dz = bins.dz
  zc1 = bins.z_center[1]
  for i in eachindex(pz)
    alive[i] || continue
    zp = z[i]
    i0 = trunc(Int, (zp - zc1) / dz) + 1   # Fortran int() truncates toward zero
    r1 = (zp - bins.z_center[i0]) / dz
    r0 = 1 - r1
    if r1 < -0.01 || r1 > 1.01 || i0 < 1 || i0 >= n_bin
      error("csr_apply_kicks!: particle outside the bin range (internal error)")
    end
    pz[i] += r0 * kick[i0] + r1 * kick[i0+1]
  end
  return pz
end
