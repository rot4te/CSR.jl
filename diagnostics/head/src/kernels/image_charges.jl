# Image-charge kick (Bmad image_charge_kick_calc), its hat-weighted bin integral, and the
# image-table kernels.

"""
    image_charge_kick(k, geom, kp)

Longitudinal field at the kick point of an image charge at the source point stored in `k` (height
`kp.y_source`), times `kp.kick_factor`: the Liénard–Wiechert velocity (`coulomb1`) and acceleration (`radiate`)
terms, without the `q/(4πε₀)` factor. The source moves with speed `β` along the centroid with the centroid's
curvature (zero on the infinite drifts outside the lattice). `1 − N⋅β′ ≈ 1/(2γ²)` and `B⋅(N − β′)` are computed
without cancellation (from `N − t`, `t` the source's direction; D43), in both modes.
"""
function image_charge_kick(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint)
  ix = k.ix_ele_source
  if 1 <= ix <= length(geom.ele)
    sp = geom.ele[ix].spline
    g_bend = -spline_d2(sp, k.s_chord_source) / sqrt(1 + spline_d1(sp, k.s_chord_source)^2)^3
  else
    g_bend = 0.0
  end
  theta = k.floor_s.theta
  t_vec = SVector(sin(theta), 0.0, cos(theta))                         # Direction of motion at the source point
  G_vec = kp.beta^2 * g_bend * SVector(-cos(theta), 0.0, sin(theta))   # Acceleration vector

  N_vec = (kp.floor_k.r - k.floor_s.r) / k.L
  theta = kp.floor_k.theta
  B_vec = SVector(sin(theta), 0.0, cos(theta))                         # β direction at the kick point

  # 1 - N⋅β′ and N - β′ (β′ = β t) without cancellation, from d = N - t (|N| = |t| = 1) and 1 - β = 1/(γ²(1 + β)):
  # 1 - N⋅β′ = (1 - β) + β|d|²/2 and B⋅(N - β′) = (B - t)⋅d - |d|²/2 + (1 - β) B⋅t
  d_vec = N_vec - t_vec
  dd = dot3(d_vec, d_vec)
  one_m_beta = 1 / (kp.gamma2 * (1 + kp.beta))
  OneNBp3 = (one_m_beta + kp.beta * dd / 2)^3
  NBp_vec = d_vec + one_m_beta * t_vec
  rad_cross_vec = cross3(N_vec, cross3(NBp_vec, G_vec))

  radiate = dot3(B_vec, rad_cross_vec) / (k.L * OneNBp3)
  B_NBp = dot3(B_vec - t_vec, d_vec) - dd / 2 + one_m_beta * dot3(B_vec, t_vec)
  coulomb1 = B_NBp / (kp.gamma2 * k.L^2 * OneNBp3)
  return kp.kick_factor * (radiate + coulomb1)
end

"""
    csr_image_kernel!(image, s_source, geom, kp, dz, n_lag, search)

Image kick for lags i = -n_bin:n_bin, stored at index i + n_bin + 1. One work item per block of `n_lag`
consecutive lags, warm-started within a block as csr_wake_kernel!. `search` as in `s_source_calc`.
"""
@kernel function csr_image_kernel!(image, s_source, geom, kp, dz, n_lag, search)
  b = @index(Global)
  n_bin = (length(image) - 1) ÷ 2
  k = CSRKick1(geom, kp)
  for j in (b - 1) * n_lag + 1:min(b * n_lag, length(image))
    k, s_chord = s_source_calc(with_fields(k; dz_particles = (j - n_bin - 1) * dz), geom, kp, search)
    @inbounds begin
      image[j] = image_charge_kick(k, geom, kp)
      s_source[j] = source_s(geom, k.ix_ele_source, s_chord)
    end
  end
end

"""
    image_kernel_at(k, u, geom, kp, search = nothing) -> (k, K)

Image kick at lag u (the source search starts from the element in `k`; `search` as in `s_source_calc`), and the
work struct after the search.
"""
function image_kernel_at(k::CSRKick1, u, geom::CSRGeometry, kp::CSRKickPoint, search = nothing)
  k, _ = s_source_calc(with_fields(k; dz_particles = u), geom, kp, search)
  return k, image_charge_kick(k, geom, kp)
end

"""
    image_simpson(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, depth, leaf = nothing, search = nothing) -> (k, S)

Adaptive Simpson for S = (∫K, ∫K t) over [a, b], t = (u - u0) / dz; fa, fm, fb are K at a, (a + b)/2, b.
Intervals narrower than `h_min` are not split (the kernel has round-off noise from the source search). Each accepted
interval `[a, b]` contributes Boole's rule on its 5 nodes; `leaf(a, b)` is called for it unless `leaf === nothing`
(`csr_kick_param_adjoint`). `search` as in `s_source_calc`.
"""
function image_simpson(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, depth, leaf = nothing, search = nothing)
  m = (a + b) / 2
  k, fl = image_kernel_at(k, (a + m) / 2, geom, kp, search)
  k, fr = image_kernel_at(k, (m + b) / 2, geom, kp, search)
  g(u, f) = SVector(f, f * (u - u0) / dz)
  h = b - a
  S1 = h / 6 * (g(a, fa) + 4g(m, fm) + g(b, fb))
  S2 = h / 12 * (g(a, fa) + 4g((a + m) / 2, fl) + 2g(m, fm) + 4g((m + b) / 2, fr) + g(b, fb))
  if depth <= 0 || h <= h_min || maximum(abs, primal.(S2 - S1)) <= 15primal(tol)
    leaf === nothing || leaf(a, b)
    return k, S2 + (S2 - S1) / 15
  end
  k, Sl = image_simpson(k, geom, kp, a, m, fa, fl, fm, u0, dz, tol / 2, h_min, depth - 1, leaf, search)
  k, Sr = image_simpson(k, geom, kp, m, b, fm, fr, fb, u0, dz, tol / 2, h_min, depth - 1, leaf, search)
  return k, Sl + Sr
end

"""
    image_segment(k, geom, kp, a, b, u0, dz, tol, h_min, leaf = nothing, search = nothing) -> (k, S)

`S = (∫K, ∫K t)` over `[a, b]`, `t = (u - u0)/dz`, by `image_simpson` from the kernel at `a`, the midpoint and `b`.
"""
function image_segment(k, geom, kp, a, b, u0, dz, tol, h_min, leaf = nothing, search = nothing)
  k, fa = image_kernel_at(k, a, geom, kp, search)
  k, fm = image_kernel_at(k, (a + b) / 2, geom, kp, search)
  k, fb = image_kernel_at(k, b, geom, kp, search)
  return image_simpson(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, 40, leaf, search)
end

"""
    image_interval_moments(k, geom, kp, m, dz, tol, u_min, leaf = nothing, search = nothing) -> (k, moments)

Hat-function moments of the image kernel over lag interval m, [m dz, (m+1) dz]: (∫K t, ∫K (1 - t)),
t = u/dz - m. `tol` is the absolute tolerance on the interval. The intervals touching u = 0 are split
geometrically down to `u_min` first, so that structure finer than the first Simpson nodes is not missed
(on a drift the kernel peaks at |u| = y/(√2 γ)). `leaf` and `search` as in `image_simpson`.
"""
function image_interval_moments(k, geom, kp, m, dz, tol, u_min, leaf = nothing, search = nothing)
  a = m * dz
  if m == 0 || m == -1
    sgn = m == 0 ? 1 : -1
    acc = zero(SVector{2,number_type(k)})
    x = dz
    while x / 2 > u_min
      lo, hi = minmax(sgn * x / 2, sgn * x)
      k, S = image_segment(k, geom, kp, lo, hi, a, dz, tol * (x / 2) / dz, u_min, leaf, search)
      acc += S
      x /= 2
    end
    lo, hi = minmax(0.0, sgn * x)
    k, S = image_segment(k, geom, kp, lo, hi, a, dz, tol * x / dz, u_min, leaf, search)
    acc += S
  else
    k, acc = image_segment(k, geom, kp, a, a + dz, a, dz, tol, u_min, leaf, search)
  end
  return k, SVector(acc[2], acc[1] - acc[2])
end

"""
    csr_image_hat_kernel!(rise, fall, geom, kp, dz, tol, u_min, n_int, search)

Moments for lag intervals m = -n_bin-1:n_bin (index m + n_bin + 2), blocks of `n_int` consecutive intervals.
`search` as in `s_source_calc`.
"""
@kernel function csr_image_hat_kernel!(rise, fall, geom, kp, dz, tol, u_min, n_int, search)
  b = @index(Global)
  n_bin = (length(rise) - 2) ÷ 2
  k = CSRKick1(geom, kp)
  for j in (b - 1) * n_int + 1:min(b * n_int, length(rise))
    k, mom = image_interval_moments(k, geom, kp, j - n_bin - 2, dz, tol, u_min, nothing, search)
    @inbounds rise[j] = mom[1]
    @inbounds fall[j] = mom[2]
  end
end

"""
    csr_image_hat_combine_kernel!(image, rise, fall, dz)

Hat-weighted image table from the interval moments: image(l) = (∫ over [(l-1)dz, l dz] of K t
+ ∫ over [l dz, (l+1)dz] of K (1 - t)) / dz.
"""
@kernel function csr_image_hat_combine_kernel!(image, @Const(rise), @Const(fall), dz)
  j = @index(Global)
  @inbounds image[j] = (rise[j] + fall[j+1]) / dz
end
