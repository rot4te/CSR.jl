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
      dL = dspline_len(0.0, L, spl, dtheta_L; bmad_compat = geom.bmad_compat)   # = Ls - L
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
      s_source[i] = k.ix_ele_source == 0 ? s_chord : geom.ele[k.ix_ele_source].s_start + s_chord
    end
  end
end

@kernel function csr_trapezoid_kernel!(I_int, @Const(I_csr), dz)
  j = @index(Global)
  i = j + 1
  @inbounds I_int[i] = (I_csr[i] + I_csr[i-1]) * dz / 2
end

"""
    csr_wake_table(geom, kp, dz, n_bin, backend) -> (I_csr, I_int, s_source)

Kick integral `I_csr`, its integral over each bin `I_int`, and the reference s of the source point, for lags
`i * dz`, `i = 1:n_bin`, at kick point `kp`. The lags are split into `n_block` blocks (one work item and
workgroup each); a block reuses the previous lag's source element as the start of the next search.
Default: 4 blocks per thread. CPU backend only: the source search uses a mutable work struct and the
non-isbits `CSRGeometry`.
"""
function csr_wake_table(geom::CSRGeometry, kp::CSRKickPoint, dz, n_bin, backend;
                        n_block = min(n_bin, 4 * Threads.nthreads()))
  I_csr = KernelAbstractions.zeros(backend, Float64, n_bin)
  I_int = KernelAbstractions.zeros(backend, Float64, n_bin)
  s_source = KernelAbstractions.zeros(backend, Float64, n_bin)
  n_lag = cld(n_bin, n_block)
  csr_wake_kernel!(backend, 1)(I_csr, I_int, s_source, geom, kp, dz, n_lag; ndrange = cld(n_bin, n_lag))
  KernelAbstractions.synchronize(backend)
  csr_trapezoid_kernel!(backend)(I_int, I_csr, dz; ndrange = n_bin - 1)
  KernelAbstractions.synchronize(backend)
  return I_csr, I_int, s_source
end

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
