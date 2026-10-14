# Reverse-mode pullback of one CSR kick (host side); kernels in kernels/adjoint.jl.

"""
    csr_kick_adjoint(ybar, z, q, alive, st, params; pz_scale = 1) -> (pzbar, zbar, qbar)

Pullback of `csr_kick_pz` with respect to `pz`, `z` and `q` for the output cotangent `ybar`, given the forward
intermediates `st = csr_binned_kick(...)`. The bin grid and kick tables are constants. Dead particles get zero `z`
and `q` cotangents. Runs as kernels on the backend of `z`; the kick-cotangent scatter uses atomic adds for
`Float32`/`Float64` and a serial CPU loop otherwise.
"""
function csr_kick_adjoint(ybar::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, st,
                          params::CSRParams; pz_scale = 1)
  backend = get_backend(z)
  bins = st.bins
  n_bin = length(st.kick)
  T = promote_type(eltype(ybar), eltype(st.kick), typeof(pz_scale))
  kbar = KernelAbstractions.zeros(backend, T, n_bin)
  zbar = KernelAbstractions.allocate(backend, promote_type(T, eltype(q)), length(z))
  if supports_atomic_add(kbar)
    csr_kick_adjoint_kernel!(backend)(kbar, zbar, ybar, z, alive, bins.z_center, st.kick, bins.dz, pz_scale;
                                      ndrange = length(z))
  else
    backend isa CPU || error("csr_kick_adjoint: element type $T needs the CPU backend (no atomic add)")
    for i in eachindex(z)
      if alive[i]
        kick_adjoint_particle!(PlainAdd(), kbar, zbar, i, ybar, z, bins.z_center, st.kick, bins.dz, pz_scale)
      else
        zbar[i] = zero(eltype(zbar))
      end
    end
  end

  gbar = KernelAbstractions.allocate(backend, promote_type(T, eltype(st.I_int), typeof(st.coef)), n_bin)
  csr_convolve_adjoint_kernel!(backend)(gbar, st.I_int, kbar, st.coef; ndrange = n_bin)
  Qbar = similar(gbar)
  csr_edge_gradient_adjoint_kernel!(backend)(Qbar, gbar, bins.dz; ndrange = n_bin)
  for image in st.images
    csr_image_convolve_adjoint_kernel!(backend)(Qbar, image, kbar, st.coef; ndrange = n_bin)
  end

  z_min = bins.z0_edge[1]
  qbar = KernelAbstractions.allocate(backend, eltype(Qbar), length(q))
  csr_deposit_adjoint_kernel!(backend)(qbar, zbar, Qbar, z, q, alive, z_min, bins.dz,
                                       params.particle_bin_span * bins.dz, params.particle_bin_span;
                                       ndrange = length(z))
  KernelAbstractions.synchronize(backend)
  return copy(ybar), zbar, qbar
end
