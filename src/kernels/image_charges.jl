# Image-charge kick (Bmad image_charge_kick_calc), its hat-weighted bin integral, and the
# image-table kernels.

"""
    image_charge_kick(k, geom, kp)

Longitudinal field at the kick point of an image charge at the source point stored in `k` (height
`kp.y_source`), times `kp.kick_factor`: the Liénard–Wiechert velocity (`coulomb1`) and acceleration (`radiate`)
terms, without the `q/(4πε₀)` factor. The source moves with speed `β` along the centroid with the centroid's
curvature (zero on the infinite drifts outside the lattice).
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
  Bp_vec = kp.beta * SVector(sin(theta), 0.0, cos(theta))              # β vector at the source point
  G_vec = kp.beta^2 * g_bend * SVector(-cos(theta), 0.0, sin(theta))   # Acceleration vector

  N_vec = (kp.floor_k.r - k.floor_s.r) / k.L
  theta = kp.floor_k.theta
  B_vec = SVector(sin(theta), 0.0, cos(theta))                         # β direction at the kick point

  OneNBp3 = (1 - dot3(N_vec, Bp_vec))^3
  NBp_vec = N_vec - Bp_vec
  rad_cross_vec = cross3(N_vec, cross3(NBp_vec, G_vec))

  radiate = dot3(B_vec, rad_cross_vec) / (k.L * OneNBp3)
  coulomb1 = dot3(B_vec, NBp_vec) / (kp.gamma2 * k.L^2 * OneNBp3)
  return kp.kick_factor * (radiate + coulomb1)
end

"""
    csr_image_kernel!(image, s_source, geom, kp, dz, n_lag)

Image kick for lags i = -n_bin:n_bin, stored at index i + n_bin + 1. One work item per block of `n_lag`
consecutive lags, warm-started within a block as csr_wake_kernel!.
"""
@kernel function csr_image_kernel!(image, s_source, geom, kp, dz, n_lag)
  b = @index(Global)
  n_bin = (length(image) - 1) ÷ 2
  k = CSRKick1(geom, kp)
  for j in (b - 1) * n_lag + 1:min(b * n_lag, length(image))
    k, s_chord = s_source_calc(with_fields(k; dz_particles = (j - n_bin - 1) * dz), geom, kp)
    @inbounds begin
      image[j] = image_charge_kick(k, geom, kp)
      s_source[j] = source_s(geom, k.ix_ele_source, s_chord)
    end
  end
end

"""
    image_kernel_at(k, u, geom, kp) -> (k, K)

Image kick at lag u (the source search starts from the element in `k`), and the work struct after the search.
"""
function image_kernel_at(k::CSRKick1, u, geom::CSRGeometry, kp::CSRKickPoint)
  k, _ = s_source_calc(with_fields(k; dz_particles = u), geom, kp)
  return k, image_charge_kick(k, geom, kp)
end

"""
    image_simpson(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, depth) -> (k, S)

Adaptive Simpson for S = (∫K, ∫K t) over [a, b], t = (u - u0) / dz; fa, fm, fb are K at a, (a + b)/2, b.
Intervals narrower than `h_min` are not split (the kernel has round-off noise from the source search).
"""
function image_simpson(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, depth)
  m = (a + b) / 2
  k, fl = image_kernel_at(k, (a + m) / 2, geom, kp)
  k, fr = image_kernel_at(k, (m + b) / 2, geom, kp)
  g(u, f) = SVector(f, f * (u - u0) / dz)
  h = b - a
  S1 = h / 6 * (g(a, fa) + 4g(m, fm) + g(b, fb))
  S2 = h / 12 * (g(a, fa) + 4g((a + m) / 2, fl) + 2g(m, fm) + 4g((m + b) / 2, fr) + g(b, fb))
  (depth <= 0 || h <= h_min || maximum(abs, primal.(S2 - S1)) <= 15primal(tol)) && return k, S2 + (S2 - S1) / 15
  k, Sl = image_simpson(k, geom, kp, a, m, fa, fl, fm, u0, dz, tol / 2, h_min, depth - 1)
  k, Sr = image_simpson(k, geom, kp, m, b, fm, fr, fb, u0, dz, tol / 2, h_min, depth - 1)
  return k, Sl + Sr
end

"""
    image_segment(k, geom, kp, a, b, u0, dz, tol, h_min) -> (k, S)

`S = (∫K, ∫K t)` over `[a, b]`, `t = (u - u0)/dz`, by `image_simpson` from the kernel at `a`, the midpoint and `b`.
"""
function image_segment(k, geom, kp, a, b, u0, dz, tol, h_min)
  k, fa = image_kernel_at(k, a, geom, kp)
  k, fm = image_kernel_at(k, (a + b) / 2, geom, kp)
  k, fb = image_kernel_at(k, b, geom, kp)
  return image_simpson(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, 40)
end

"""
    image_interval_moments(k, geom, kp, m, dz, tol, u_min) -> (k, moments)

Hat-function moments of the image kernel over lag interval m, [m dz, (m+1) dz]: (∫K t, ∫K (1 - t)),
t = u/dz - m. `tol` is the absolute tolerance on the interval. The intervals touching u = 0 are split
geometrically down to `u_min` first, so that structure finer than the first Simpson nodes is not missed
(on a drift the kernel peaks at |u| = y/(√2 γ)).
"""
function image_interval_moments(k, geom, kp, m, dz, tol, u_min)
  a = m * dz
  if m == 0 || m == -1
    sgn = m == 0 ? 1 : -1
    acc = zero(SVector{2,number_type(k)})
    x = dz
    while x / 2 > u_min
      lo, hi = minmax(sgn * x / 2, sgn * x)
      k, S = image_segment(k, geom, kp, lo, hi, a, dz, tol * (x / 2) / dz, u_min)
      acc += S
      x /= 2
    end
    lo, hi = minmax(0.0, sgn * x)
    k, S = image_segment(k, geom, kp, lo, hi, a, dz, tol * x / dz, u_min)
    acc += S
  else
    k, acc = image_segment(k, geom, kp, a, a + dz, a, dz, tol, u_min)
  end
  return k, SVector(acc[2], acc[1] - acc[2])
end

"""
    csr_image_hat_kernel!(rise, fall, geom, kp, dz, tol, u_min, n_int)

Moments for lag intervals m = -n_bin-1:n_bin (index m + n_bin + 2), blocks of `n_int` consecutive intervals.
"""
@kernel function csr_image_hat_kernel!(rise, fall, geom, kp, dz, tol, u_min, n_int)
  b = @index(Global)
  n_bin = (length(rise) - 2) ÷ 2
  k = CSRKick1(geom, kp)
  for j in (b - 1) * n_int + 1:min(b * n_int, length(rise))
    k, mom = image_interval_moments(k, geom, kp, j - n_bin - 2, dz, tol, u_min)
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
