# Serial reference implementation (phase 1, non-exported): plain loops, Float64, one lag after another with the
# source search starting where the previous lag's search ended (as in Bmad). The kernels in kernels/ (launched by
# binning.jl, tables.jl, kick.jl) are tested against these (test/kernels_test.jl).

"""
    csr_bin_particles_reference(z, q, alive, params)

Serial reference of `csr_bin_particles`.
"""
function csr_bin_particles_reference(z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, params::CSRParams)
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

"""
    csr_convolve_reference!(kick, I_int, edge_grad, coef)

Serial reference of `csr_convolve!`.
"""
function csr_convolve_reference!(kick::AbstractVector, I_int::AbstractVector, edge_grad::AbstractVector, coef)
  n = length(kick)
  for i in 1:n
    acc = zero(eltype(kick))
    for k in 1:i
      acc += I_int[k] * edge_grad[i+1-k]
    end
    kick[i] = coef * acc
  end
  return kick
end

"""
    csr_apply_kicks_reference!(pz, z, alive, bins, kick)

Serial reference of `csr_apply_kicks!`.
"""
function csr_apply_kicks_reference!(pz::AbstractVector, z::AbstractVector, alive::AbstractVector{Bool}, bins::CSRBins, kick::AbstractVector)
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

"""
    csr_kick_reference!(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0)

Serial reference of `csr_kick!` (free space and image charges, same layer weights), without
`pz_scale`. One warm-started source walk over the lags, as Bmad.
"""
function csr_kick_reference!(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                             geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                             params::CSRParams; kick_factor = 1.0)
  n_bin = params.n_bin
  params.bmad_compat == geom.bmad_compat || error("csr_kick!: params.bmad_compat and geom.bmad_compat differ")
  count(alive) >= n_bin || error("csr_kick!: number of live particles ($(count(alive))) is less than n_bin ($n_bin)")

  e = geom.ele[ix_ele_kick]
  _, ds_step = csr_n_step(e.L, params)
  actual_track_step = ds_step * (e.L_chord / e.L)

  bins = csr_bin_particles_reference(z, q, alive, params)
  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)

  # Kick table for lags i*dz, i = 1:n_bin. Each search starts in the element where the previous one ended.
  I_int = zeros(n_bin)
  I_csr = zeros(n_bin)
  s_source = zeros(n_bin)
  k = CSRKick1()
  k.ix_ele_source = ix_ele_kick
  for i in 1:n_bin
    k.dz_particles = i * bins.dz
    s_chord = s_source_calc!(k, geom, kp)
    I_csr!(k, i, bins.dz, geom, kp, i == 1 ? 0.0 : I_csr[i-1])
    I_csr[i] = k.I_csr
    I_int[i] = k.I_int_csr
    s_source[i] = source_s(geom, k.ix_ele_source, s_chord)
  end
  # Bins i ≥ 2 by quadrature (one warm-started walk) instead of the trapezoid
  if params.all_bin_quadrature && !params.bmad_compat
    k = CSRKick1()
    k.ix_ele_source = ix_ele_kick
    for i in 2:n_bin
      I_int[i] = I_csr_quadrature!(k, (i - 1) * bins.dz, i * bins.dz, geom, kp, 8)
    end
  end

  coef = csr_kick_coef(species, gamma, actual_track_step)
  kick = zeros(n_bin)
  csr_convolve_reference!(kick, I_int, bins.edge_dcharge_density_dz, coef)

  # Image charges: lags -n_bin:n_bin, one warm-started walk; kick(i) += coef Σ_j image(i - j) charge(j)
  for ns in 1:params.n_shield_images
    kp_ns = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source = ns * params.beam_chamber_height,
                         kick_factor = (-1)^ns * 2 * image_layer_weight(ns, params) * kick_factor)
    image = zeros(2n_bin + 1)
    k = CSRKick1()
    k.ix_ele_source = ix_ele_kick
    for i in -n_bin:n_bin
      k.dz_particles = i * bins.dz
      s_source_calc!(k, geom, kp_ns)
      image[i+n_bin+1] = image_charge_kick(k, geom, kp_ns)
    end
    # Default: hat-weighted bin integral of the kernel (moments over lag intervals m = -n_bin-1:n_bin)
    if !params.bmad_compat
      u_min = image_u_min(kp_ns)
      tol = image_hat_tol(geom, kp_ns, image, bins.dz, 1e-6, u_min)
      mom = [image_interval_moments!(k, geom, kp_ns, m, bins.dz, tol, u_min) for m in -n_bin-1:n_bin]
      for i in -n_bin:n_bin
        image[i+n_bin+1] = (mom[i+n_bin+1][1] + mom[i+n_bin+2][2]) / bins.dz
      end
    end
    for i in 1:n_bin
      acc = 0.0
      for j in 1:n_bin
        acc += image[i-j+n_bin+1] * bins.charge[j]
      end
      kick[i] += coef * acc
    end
  end
  csr_apply_kicks_reference!(pz, z, alive, bins, kick)

  ds_kick = kick_factor * actual_track_step
  return CSRStepResult(bins.z_center, bins.charge ./ bins.dz, kick ./ ds_kick, I_csr ./ ds_kick, s_source)
end
