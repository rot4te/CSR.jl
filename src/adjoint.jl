# Reverse-mode adjoint of one CSR kick with respect to the particles (pz, z, q), the bin grid and kick tables held
# constant (planning/csr_gpu_ad_design.md §3.8). Forward map, for live particles p:
#   Q_j = Σ_p q_p w_j(z_p),  g_j = (Q_j − Q_{j−1}) / dz² (g_1 = 0),
#   k_i = coef Σ_{m≤i} I_int[m] g_{i+1−m} + Σ_layers coef Σ_j image(i − j) Q_j,
#   pz_p += pz_scale (r0 k_{i0} + r1 k_{i0+1}),  r1 = (z_p − z_center[i0]) / dz.

# d/dzp of triangle_overlap(zp, dz_particle, zb0, zb1) = f(zb0 − zp) − f(zb1 − zp), f the triangle's density.
@inline function triangle_overlap_dzp(zp, dz_particle, zb0, zb1)
  f(u) = max(zero(u), 2 / dz_particle - 4 * abs(u) / dz_particle^2)
  return f(zb0 - zp) - f(zb1 - zp)
end

# Interpolation cell of a particle, as in csr_apply_kicks_kernel!.
@inline function kick_cell(zp, zc1, z_center, dz)
  i0 = trunc(Int, (zp - zc1) / dz) + 1
  return i0, (zp - z_center[i0]) / dz
end

# Adjoint of the kick application for one particle: kick cotangent scattered to the two bins, z cotangent from the
# slope of the interpolated kick.
@inline function kick_adjoint_particle!(add!, kbar, zbar, i, ybar, z, z_center, kick, dz, pz_scale)
  i0, r1 = kick_cell(z[i], z_center[1], z_center, dz)
  s = pz_scale * ybar[i]
  add!(kbar, i0, (1 - r1) * s)
  add!(kbar, i0 + 1, r1 * s)
  zbar[i] = s * (kick[i0+1] - kick[i0]) / dz
  return nothing
end

@kernel function csr_kick_adjoint_kernel!(kbar, zbar, @Const(ybar), @Const(z), @Const(alive), @Const(z_center),
                                          @Const(kick), dz, pz_scale)
  i = @index(Global)
  @inbounds if alive[i]
    kick_adjoint_particle!(AtomicAdd(), kbar, zbar, i, ybar, z, z_center, kick, dz, pz_scale)
  else
    zbar[i] = zero(eltype(zbar))
  end
end

# gbar[m] = coef Σ_{i≥m} I_int[i+1−m] kbar[i]  (correlation: transpose of csr_convolve_kernel!).
@kernel function csr_convolve_adjoint_kernel!(gbar, @Const(I_int), @Const(kbar), coef)
  m = @index(Global)
  n_bin = length(gbar)
  acc = zero(eltype(gbar))
  for i in m:n_bin
    @inbounds acc += I_int[i+1-m] * kbar[i]
  end
  @inbounds gbar[m] = coef * acc
end

# Qbar[j] = (gbar[j] [j ≥ 2] − gbar[j+1] [j < n_bin]) / dz²  (transpose of csr_edge_gradient_kernel!).
@kernel function csr_edge_gradient_adjoint_kernel!(Qbar, @Const(gbar), dz)
  j = @index(Global)
  n_bin = length(Qbar)
  acc = zero(eltype(Qbar))
  @inbounds j >= 2 && (acc += gbar[j])
  @inbounds j < n_bin && (acc -= gbar[j+1])
  @inbounds Qbar[j] = acc / dz^2
end

# Qbar[j] += coef Σ_i image(i − j) kbar[i]  (transpose of csr_image_convolve_kernel!).
@kernel function csr_image_convolve_adjoint_kernel!(Qbar, @Const(image), @Const(kbar), coef)
  j = @index(Global)
  n_bin = length(Qbar)
  acc = zero(eltype(Qbar))
  for i in 1:n_bin
    @inbounds acc += image[i-j+n_bin+1] * kbar[i]
  end
  @inbounds Qbar[j] += coef * acc
end

# Adjoint of the deposit: qbar_p = Σ_j w_j(z_p) Qbar_j, zbar_p += q_p Σ_j w_j'(z_p) Qbar_j.
@kernel function csr_deposit_adjoint_kernel!(qbar, zbar, @Const(Qbar), @Const(z), @Const(q), @Const(alive),
                                             z_min, dz, dz_particle, pbs)
  i = @index(Global)
  @inbounds if alive[i]
    n_bin = length(Qbar)
    zp = z[i]
    ix0 = deposit_ix0(zp, z_min, dz, dz_particle)
    aq = zero(eltype(qbar))
    az = zero(eltype(zbar))
    for j in 0:pbs+1
      ib = j + ix0
      if 1 <= ib <= n_bin
        zb0 = z_min + (ib - 1) * dz
        aq += triangle_overlap(zp, dz_particle, zb0, zb0 + dz) * Qbar[ib]
        az += triangle_overlap_dzp(zp, dz_particle, zb0, zb0 + dz) * Qbar[ib]
      end
    end
    qbar[i] = aq
    zbar[i] += q[i] * az
  else
    qbar[i] = zero(eltype(qbar))
  end
end

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
