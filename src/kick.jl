# Apply the binned kick to particles. Port of the CSR-only branch of Bmad's csr_and_sc_apply_kicks.

@kernel function csr_apply_kicks_kernel!(pz, n_outside, @Const(z), @Const(alive), @Const(z_center), @Const(kick), dz)
  i = @index(Global)
  @inbounds if alive[i]
    n_bin = length(kick)
    zp = z[i]
    i0 = trunc(Int, (zp - z_center[1]) / dz) + 1   # Fortran int() truncates toward zero
    if 1 <= i0 < n_bin
      r1 = (zp - z_center[i0]) / dz
      if -0.01 <= r1 <= 1.01
        r0 = 1 - r1
        pz[i] += r0 * kick[i0] + r1 * kick[i0+1]
      else
        Atomix.@atomic n_outside[1] += 1
      end
    else
      Atomix.@atomic n_outside[1] += 1
    end
  end
end

"""
    csr_apply_kicks!(pz, z, alive, bins, kick)

Add the kick, linearly interpolated between bin centers, to `pz` of each live particle.
As in Bmad's CSR-only branch, `z` is not rescaled for the change in velocity.
One work item per particle, on the backend of `pz`.
"""
function csr_apply_kicks!(pz::AbstractVector, z::AbstractVector, alive::AbstractVector{Bool}, bins::CSRBins, kick::AbstractVector)
  backend = get_backend(pz)
  n_outside = KernelAbstractions.zeros(backend, Int, 1)
  csr_apply_kicks_kernel!(backend)(pz, n_outside, z, alive, bins.z_center, kick, bins.dz; ndrange = length(pz))
  KernelAbstractions.synchronize(backend)
  Array(n_outside)[1] == 0 || error("csr_apply_kicks!: particle outside the bin range (internal error)")
  return pz
end
