# CSR kick integral. Port of Bmad's I_csr.

"Kick integrand for the source geometry stored in `k` (Bmad I_csr formula)."
function I_csr_value(k::CSRKick1, kp::CSRKickPoint)
  z = k.dz_particles
  return -kp.kick_factor * 2 * (k.dL / z + kp.gamma2 * k.theta_sl * k.theta_lk / (1 + kp.gamma2 * k.theta_sl^2)) / k.L
end

"""
    I_csr!(k, i_bin, dz_slice, geom, kp, I_csr_prev)

Compute the CSR kick integral `k.I_csr` for separation `k.dz_particles` and its integral over the bin,
`k.I_int_csr`. For `i_bin = 1` the integral from 0 uses Bmad's closed form over the kick element, which
handles the near-field log singularity, plus (if the source is upstream of the kick element) the rest of
the bin by quadrature (`I_csr_integral`; Bmad's approximation with `geom.bmad_compat`). For `i_bin > 1`
it is a trapezoid with the previous bin's `I_csr_prev`.
"""
function I_csr!(k::CSRKick1, i_bin, dz_slice, geom::CSRGeometry, kp::CSRKickPoint, I_csr_prev)
  k.I_int_csr = 0.0
  k.I_csr = 0.0

  # No kick when the source particle is ahead of the kicked particle
  z = k.dz_particles
  z <= 0 && return k

  gamma2 = kp.gamma2
  k.I_csr = I_csr_value(k, kp)

  if i_bin == 1
    ix_ele_kick = kp.ix_ele_kick
    s_chord_kick = kp.s_chord_kick
    while s_chord_kick == 0   # At the element entrance: use the end of the previous element
      ix_ele_kick -= 1
      ix_ele_kick == 0 && return k   # No kick from before the beginning of the lattice
      s_chord_kick = geom.ele[ix_ele_kick].L_chord
    end

    ek = geom.ele[ix_ele_kick]
    spl = ek.spline
    g_bend = -spline_d2(spl, s_chord_kick) / sqrt(1 + spline_d1(spl, s_chord_kick)^2)^3

    if k.ix_ele_source == ix_ele_kick
      Ls = k.L + k.dL
      k.I_int_csr = -kp.kick_factor * ((g_bend * Ls / 2)^2 - log(2gamma2 * z / Ls) / gamma2)
    else
      # Source point in another element: first integrate over the element containing the kick point,
      # then add the rest.
      L = s_chord_kick
      dtheta_L = spl[2] + spl[3] * L + spl[4] * L^2
      dL = path_excess(0.0, L, spl, dtheta_L; bmad_compat = geom.bmad_compat)   # = Ls - L
      Ls = L + dL
      zz = L / (2gamma2) + dL
      k.I_int_csr = -kp.kick_factor * ((g_bend * Ls / 2)^2 - log(2gamma2 * zz / Ls) / gamma2)
      if geom.bmad_compat
        # Bmad approximates the rest of the integral with I_csr held at its value at z. In steady state
        # (I ∝ u^(-1/3), zz ≪ z) this gives 2/3 of the integral.
        k.I_int_csr += k.I_csr * (z - zz)
      else
        k.I_int_csr += I_csr_integral(zz, z, geom, kp)
      end
    end
  else
    k.I_int_csr = (k.I_csr + I_csr_prev) * dz_slice / 2
  end
  return k
end

# One work item per block of `n_lag` consecutive lags i (separation i*dz). Within a block each source search
# starts where the previous lag's ended (as in Bmad); the first lag of a block starts in the kick element.
# Searches end in the same element, and call zbrent on the same bracket, whatever the start element.
# For i > 1, I_int is overwritten by csr_trapezoid_kernel!.
@kernel function csr_wake_kernel!(I_csr, I_int, s_source, geom, kp, dz, n_lag)
  b = @index(Global)
  k = CSRKick1()
  k.ix_ele_source = kp.ix_ele_kick
  for i in (b - 1) * n_lag + 1:min(b * n_lag, length(I_csr))
    k.dz_particles = i * dz
    s_chord = s_source_calc!(k, geom, kp)
    I_csr!(k, i, dz, geom, kp, 0.0)
    @inbounds begin
      I_csr[i] = k.I_csr
      I_int[i] = k.I_int_csr
      s_source[i] = source_s(geom, k.ix_ele_source, s_chord)
    end
  end
end

@kernel function csr_trapezoid_kernel!(I_int, @Const(I_csr), dz)
  j = @index(Global)
  i = j + 1
  @inbounds I_int[i] = (I_csr[i] + I_csr[i-1]) * dz / 2
end

# Bins i ≥ 2 by quadrature: one work item per block of `n_lag` consecutive bins (bin i is block element i - 1);
# the source search is warm-started from node to node and bin to bin within a block.
@kernel function csr_bin_quadrature_kernel!(I_int, geom, kp, dz, n_lag, n_quad)
  b = @index(Global)
  k = CSRKick1()
  k.ix_ele_source = kp.ix_ele_kick
  for i in (b - 1) * n_lag + 2:min(b * n_lag + 1, length(I_int))
    @inbounds I_int[i] = I_csr_quadrature!(k, (i - 1) * dz, i * dz, geom, kp, n_quad)
  end
end

"""
    csr_wake_table(geom, kp, dz, n_bin, backend; n_block, all_bin_quadrature = false) -> (I_csr, I_int, s_source)

Kick integral `I_csr`, its integral over each bin `I_int`, and the reference s of the source point, for lags
`i * dz`, `i = 1:n_bin`, at kick point `kp`. The lags are split into `n_block` blocks (one work item and
workgroup each); a block reuses the previous lag's source element as the start of the next search.
Default: 4 blocks per thread. CPU backend only: the source search uses a mutable work struct and the
non-isbits `CSRGeometry`.

`I_int` for bins `i ≥ 2`: trapezoid in `I_csr`, or with `all_bin_quadrature` Simpson in `u^(1/3)` with 8 nodes per
bin (`I_csr_quadrature!`; 8 more source solves per bin, in blocks as above).
"""
function csr_wake_table(geom::CSRGeometry, kp::CSRKickPoint, dz, n_bin, backend;
                        n_block = min(n_bin, 4 * Threads.nthreads()), all_bin_quadrature::Bool = false)
  I_csr = KernelAbstractions.zeros(backend, Float64, n_bin)
  I_int = KernelAbstractions.zeros(backend, Float64, n_bin)
  s_source = KernelAbstractions.zeros(backend, Float64, n_bin)
  n_lag = cld(n_bin, n_block)
  # Workgroup size passed at launch: a static size (`csr_wake_kernel!(backend, 1)`) is not inferred.
  csr_wake_kernel!(backend)(I_csr, I_int, s_source, geom, kp, dz, n_lag; ndrange = cld(n_bin, n_lag), workgroupsize = 1)
  KernelAbstractions.synchronize(backend)
  if n_bin > 1
    if all_bin_quadrature
      n_q = cld(n_bin - 1, min(n_bin - 1, n_block))
      csr_bin_quadrature_kernel!(backend)(I_int, geom, kp, dz, n_q, 8; ndrange = cld(n_bin - 1, n_q), workgroupsize = 1)
    else
      csr_trapezoid_kernel!(backend)(I_int, I_csr, dz; ndrange = n_bin - 1)
    end
    KernelAbstractions.synchronize(backend)
  end
  return I_csr, I_int, s_source
end

#---------------------------------------------------------------------------------------------------
# Image charges (shielding). Port of Bmad's image_charge_kick_calc and the image part of csr_bin_kicks.

cross3(a, b) = SVector(a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

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

# Image kick for lags i = -n_bin:n_bin, stored at index i + n_bin + 1. One work item per block of `n_lag`
# consecutive lags, warm-started within a block as csr_wake_kernel!.
@kernel function csr_image_kernel!(image, s_source, geom, kp, dz, n_lag)
  b = @index(Global)
  n_bin = (length(image) - 1) ÷ 2
  k = CSRKick1()
  k.ix_ele_source = kp.ix_ele_kick
  for j in (b - 1) * n_lag + 1:min(b * n_lag, length(image))
    k.dz_particles = (j - n_bin - 1) * dz
    s_chord = s_source_calc!(k, geom, kp)
    @inbounds begin
      image[j] = image_charge_kick(k, geom, kp)
      s_source[j] = source_s(geom, k.ix_ele_source, s_chord)
    end
  end
end

# Image kick at lag u (the source search starts from the element in `k`).
function image_kernel_at!(k::CSRKick1, u, geom::CSRGeometry, kp::CSRKickPoint)
  k.dz_particles = u
  s_source_calc!(k, geom, kp)
  return image_charge_kick(k, geom, kp)
end

# Adaptive Simpson for (∫K, ∫K t) over [a, b], t = (u - u0) / dz; fa, fm, fb are K at a, (a + b)/2, b.
# Intervals narrower than `h_min` are not split (the kernel has round-off noise from the source search).
function image_simpson!(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, depth)
  m = (a + b) / 2
  fl = image_kernel_at!(k, (a + m) / 2, geom, kp)
  fr = image_kernel_at!(k, (m + b) / 2, geom, kp)
  g(u, f) = SVector(f, f * (u - u0) / dz)
  h = b - a
  S1 = h / 6 * (g(a, fa) + 4g(m, fm) + g(b, fb))
  S2 = h / 12 * (g(a, fa) + 4g((a + m) / 2, fl) + 2g(m, fm) + 4g((m + b) / 2, fr) + g(b, fb))
  (depth <= 0 || h <= h_min || maximum(abs, S2 - S1) <= 15tol) && return S2 + (S2 - S1) / 15
  return image_simpson!(k, geom, kp, a, m, fa, fl, fm, u0, dz, tol / 2, h_min, depth - 1) +
         image_simpson!(k, geom, kp, m, b, fm, fr, fb, u0, dz, tol / 2, h_min, depth - 1)
end

function image_segment!(k, geom, kp, a, b, u0, dz, tol, h_min)
  fa = image_kernel_at!(k, a, geom, kp)
  fm = image_kernel_at!(k, (a + b) / 2, geom, kp)
  fb = image_kernel_at!(k, b, geom, kp)
  return image_simpson!(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, h_min, 40)
end

# Hat-function moments of the image kernel over lag interval m, [m dz, (m+1) dz]: (∫K t, ∫K (1 - t)),
# t = u/dz - m. `tol` is the absolute tolerance on the interval. The intervals touching u = 0 are split
# geometrically down to `u_min` first, so that structure finer than the first Simpson nodes is not missed
# (on a drift the kernel peaks at |u| = y/(√2 γ)).
function image_interval_moments!(k, geom, kp, m, dz, tol, u_min)
  a = m * dz
  if m == 0 || m == -1
    sgn = m == 0 ? 1 : -1
    acc = zero(SVector{2,Float64})
    x = dz
    while x / 2 > u_min
      lo, hi = minmax(sgn * x / 2, sgn * x)
      acc += image_segment!(k, geom, kp, lo, hi, a, dz, tol * (x / 2) / dz, u_min)
      x /= 2
    end
    lo, hi = minmax(0.0, sgn * x)
    acc += image_segment!(k, geom, kp, lo, hi, a, dz, tol * x / dz, u_min)
  else
    acc = image_segment!(k, geom, kp, a, a + dz, a, dz, tol, u_min)
  end
  return SVector(acc[2], acc[1] - acc[2])
end

# Finest lag scale resolved by the hat-weighted image table: 1% of y/γ, the width of the drift kernel's peak.
image_u_min(kp::CSRKickPoint) = 0.01 * abs(kp.y_source) / kp.gamma

# Absolute tolerance per lag interval for the hat-weighted image table: `rtol * dz` times the largest |K| at the
# lags (`image`, point-sampled) and at the nodes ±dz/2^j > u_min. The lags alone miss the peak near u = 0 when
# dz ≫ y/γ, which would give a tolerance below the kernel's round-off noise.
function image_hat_tol(geom::CSRGeometry, kp::CSRKickPoint, image, dz, rtol, u_min)
  k = CSRKick1()
  k.ix_ele_source = kp.ix_ele_kick
  K = maximum(abs, image)
  x = dz / 2
  while x > u_min
    K = max(K, abs(image_kernel_at!(k, x, geom, kp)), abs(image_kernel_at!(k, -x, geom, kp)))
    x /= 2
  end
  return rtol * K * dz
end

# Moments for lag intervals m = -n_bin-1:n_bin (index m + n_bin + 2), blocks of `n_int` consecutive intervals.
@kernel function csr_image_hat_kernel!(rise, fall, geom, kp, dz, tol, u_min, n_int)
  b = @index(Global)
  n_bin = (length(rise) - 2) ÷ 2
  k = CSRKick1()
  k.ix_ele_source = kp.ix_ele_kick
  for j in (b - 1) * n_int + 1:min(b * n_int, length(rise))
    mom = image_interval_moments!(k, geom, kp, j - n_bin - 2, dz, tol, u_min)
    @inbounds rise[j] = mom[1]
    @inbounds fall[j] = mom[2]
  end
end

# Hat-weighted image table from the interval moments: image(l) = (∫ over [(l-1)dz, l dz] of K t
# + ∫ over [l dz, (l+1)dz] of K (1 - t)) / dz.
@kernel function csr_image_hat_combine_kernel!(image, @Const(rise), @Const(fall), dz)
  j = @index(Global)
  @inbounds image[j] = (rise[j] + fall[j+1]) / dz
end

"""
    csr_image_table(geom, kp, dz, n_bin, backend; n_block, bin_integrate = false, rtol = 1e-6) -> (image, s_source)

Image-charge kick (`image_charge_kick`) and reference s of the source point for lags `i * dz`,
`i = -n_bin:n_bin` (index `i + n_bin + 1`), at kick point `kp` with `kp.y_source ≠ 0`. Blocks of lags as in
`csr_wake_table`. CPU backend only.

`bin_integrate = false` samples the kernel at the lags (Bmad). `bin_integrate = true` integrates it against the hat
function of width `2dz` centred on each lag (the charge density interpolated linearly between bin centres), by
adaptive Simpson with absolute tolerance `rtol * dz * max|K|` per interval (`image_hat_tol`), down to lag widths of
`0.01 y/γ`. The kernel's structure can be much finer than a bin (`~y/γ` on a drift, `~Ls³/24ρ²` in a bend), which
point sampling does not resolve (`planning/PLAN.md` Section 13). `s_source` is at the lags in both cases.
"""
function csr_image_table(geom::CSRGeometry, kp::CSRKickPoint, dz, n_bin, backend;
                         n_block = min(2n_bin + 1, 4 * Threads.nthreads()), bin_integrate::Bool = false,
                         rtol = 1e-6)
  n = 2n_bin + 1
  image = KernelAbstractions.zeros(backend, Float64, n)
  s_source = KernelAbstractions.zeros(backend, Float64, n)
  n_lag = cld(n, n_block)
  csr_image_kernel!(backend)(image, s_source, geom, kp, dz, n_lag; ndrange = cld(n, n_lag), workgroupsize = 1)
  KernelAbstractions.synchronize(backend)
  bin_integrate || return image, s_source

  u_min = image_u_min(kp)
  tol = image_hat_tol(geom, kp, image, dz, rtol, u_min)
  rise = KernelAbstractions.zeros(backend, Float64, n + 1)
  fall = KernelAbstractions.zeros(backend, Float64, n + 1)
  n_int = cld(n + 1, min(n + 1, n_block))
  csr_image_hat_kernel!(backend)(rise, fall, geom, kp, dz, tol, u_min, n_int; ndrange = cld(n + 1, n_int),
                                 workgroupsize = 1)
  KernelAbstractions.synchronize(backend)
  csr_image_hat_combine_kernel!(backend)(image, rise, fall, dz; ndrange = n)
  KernelAbstractions.synchronize(backend)
  return image, s_source
end

#---------------------------------------------------------------------------------------------------

"""
    I_csr_integral(z0, z1, geom, kp; n = 16)

`∫ I_csr(u) du` from `z0` to `z1` by composite Simpson in `t = u^(1/3)`, which makes the near-field
`u^(-1/3)` behaviour smooth. Each node needs its own source-point solve.
"""
function I_csr_integral(z0, z1, geom::CSRGeometry, kp::CSRKickPoint; n = 16)
  k = CSRKick1()
  function f(t)
    u = t^3
    k.ix_ele_source = kp.ix_ele_kick
    k.dz_particles = u
    s_source_calc!(k, geom, kp)
    return I_csr_value(k, kp) * 3t^2
  end
  t0 = cbrt(z0)
  t1 = cbrt(z1)
  h = (t1 - t0) / n
  acc = f(t0) + f(t1)
  for j in 1:n-1
    acc += (isodd(j) ? 4 : 2) * f(t0 + j * h)
  end
  return acc * h / 3
end

"""
    I_csr_quadrature!(k, z0, z1, geom, kp, n)

As `I_csr_integral`, with the work struct `k` supplied: each source search starts where the previous one ended
(nodes in increasing order). Searches end in the same element whatever the start element, so the result does not
depend on the warm start.
"""
function I_csr_quadrature!(k::CSRKick1, z0, z1, geom::CSRGeometry, kp::CSRKickPoint, n)
  t0 = cbrt(z0)
  h = (cbrt(z1) - t0) / n
  acc = 0.0
  for j in 0:n
    t = j == n ? cbrt(z1) : t0 + j * h
    k.dz_particles = t^3
    s_source_calc!(k, geom, kp)
    w = (j == 0 || j == n) ? 1 : (isodd(j) ? 4 : 2)
    acc += w * I_csr_value(k, kp) * 3t^2
  end
  return acc * h / 3
end
