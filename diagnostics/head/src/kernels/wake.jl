# Free-space kick integrand I_csr, its bin integrals, and the wake-table kernels (Bmad I_csr).

"Kick integrand for the source geometry stored in `k` (Bmad I_csr formula)."
function I_csr_value(k::CSRKick1, kp::CSRKickPoint)
  z = k.dz_particles
  return -kp.kick_factor * 2 * (k.dL / z + kp.gamma2 * k.theta_sl * k.theta_lk / (1 + kp.gamma2 * k.theta_sl^2)) / k.L
end

"""
    I_csr_calc(k, i_bin, dz_slice, geom, kp, I_csr_prev, search = nothing) -> k

Bmad `I_csr`: `k` with the CSR kick integral `I_csr` for separation `k.dz_particles` and its integral over the bin,
`I_int_csr`. For `i_bin = 1` the integral from 0 uses Bmad's closed form over the kick element, which
handles the near-field log singularity, plus (if the source is upstream of the kick element) the rest of
the bin by quadrature (`I_csr_integral`; Bmad's approximation with `geom.bmad_compat`). For `i_bin > 1`
it is a trapezoid with the previous bin's `I_csr_prev`. `search` as in `s_source_calc` (for the quadrature's sources).
"""
function I_csr_calc(k::CSRKick1, i_bin, dz_slice, geom::CSRGeometry, kp::CSRKickPoint, I_csr_prev, search = nothing)
  # No kick when the source particle is ahead of the kicked particle
  z = k.dz_particles
  primal(z) <= 0 && return with_fields(k; I_csr = 0.0, I_int_csr = 0.0)

  gamma2 = kp.gamma2
  I_csr = I_csr_value(k, kp)
  I_int_csr = zero(I_csr)

  if i_bin == 1
    ix_ele_kick = kp.ix_ele_kick
    s_chord_kick = kp.s_chord_kick
    while primal(s_chord_kick) == 0   # At the element entrance: use the end of the previous element
      ix_ele_kick -= 1
      ix_ele_kick == 0 && return with_fields(k; I_csr, I_int_csr)   # No kick from before the beginning of the lattice
      s_chord_kick = geom.ele[ix_ele_kick].L_chord
    end

    ek = geom.ele[ix_ele_kick]
    spl = ek.spline
    g_bend = -spline_d2(spl, s_chord_kick) / sqrt(1 + spline_d1(spl, s_chord_kick)^2)^3
    # Ls/z of a straight line: 2γ² with Bmad's retardation, γ²(1 + β) exact (D42), so the log vanishes there
    straight = geom.bmad_compat ? 2gamma2 : gamma2 * (1 + kp.beta)

    if k.ix_ele_source == ix_ele_kick
      Ls = k.L + k.dL
      I_int_csr = -kp.kick_factor * ((g_bend * Ls / 2)^2 - log(straight * z / Ls) / gamma2)
    else
      # Source point in another element: first integrate over the element containing the kick point,
      # then add the rest.
      if geom.bmad_compat
        L = s_chord_kick
        dtheta_L = spl[2] + spl[3] * L + spl[4] * L^2
        dL = path_excess(0.0, L, spl, dtheta_L; bmad_compat = true)   # = Ls - L
        Ls = L + dL
        zz = L / (2gamma2) + dL
      else
        # zz: the lag of a source at the element entrance (ddz_calc_csr, so that the quadrature starts where the
        # source search crosses into the previous element); Ls: arc length from the entrance to the kick point.
        k_entry, zz = ddz_calc_csr(with_fields(k; ix_ele_source = ix_ele_kick, dz_particles = zero(k.dz_particles)),
                                   0.0, geom, kp)
        Ls = path_excess(0.0, s_chord_kick, spl) + s_chord_kick
      end
      I_int_csr = -kp.kick_factor * ((g_bend * Ls / 2)^2 - log(straight * zz / Ls) / gamma2)
      if geom.bmad_compat
        # Bmad approximates the rest of the integral with I_csr held at its value at z. In steady state
        # (I ∝ u^(-1/3), zz ≪ z) this gives 2/3 of the integral.
        I_int_csr += I_csr * (z - zz)
      else
        # First node: the entrance source itself, not a search for lag zz (which may return the previous element's end,
        # the same point but other numbers: two branches for derivatives w.r.t. the raw geometry numbers)
        I_int_csr += I_csr_integral(zz, z, geom, kp, search; k0 = with_fields(k_entry; dz_particles = zz))
      end
    end
  else
    I_int_csr = (I_csr + I_csr_prev) * dz_slice / 2
  end
  return with_fields(k; I_csr, I_int_csr)
end

"""
    I_csr_integral(z0, z1, geom, kp, search = nothing; n = 16, k0 = nothing)

`∫ I_csr(u) du` from `z0` to `z1` by composite Simpson in `t = u^(1/3)`, which makes the near-field
`u^(-1/3)` behaviour smooth. Each node needs its own source-point solve, started in the kick element (`search` as in
`s_source_calc`). `k0`: the source at `z0`, if known (then not searched).
"""
function I_csr_integral(z0, z1, geom::CSRGeometry, kp::CSRKickPoint, search = nothing; n = 16, k0 = nothing)
  function f(t)
    k, _ = s_source_calc(CSRKick1(geom, kp; dz_particles = t^3), geom, kp, search)
    return I_csr_value(k, kp) * 3t^2
  end
  t0 = cube_root(z0)
  t1 = cube_root(z1)
  h = (t1 - t0) / n
  acc = (k0 === nothing ? f(t0) : I_csr_value(k0, kp) * 3t0^2) + f(t1)
  for j in 1:n-1
    acc += (isodd(j) ? 4 : 2) * f(t0 + j * h)
  end
  return acc * h / 3
end

"""
    I_csr_quadrature(k, z0, z1, geom, kp, n, search = nothing) -> (k, integral)

As `I_csr_integral`, with each source search starting where the previous one ended (from `k`; nodes in increasing
order). Searches end in the same element whatever the start element, so the result does not depend on the warm
start. Returns the work struct after the last node, for the next warm start. `search` as in `s_source_calc`.
"""
function I_csr_quadrature(k::CSRKick1, z0, z1, geom::CSRGeometry, kp::CSRKickPoint, n, search = nothing)
  t0 = cube_root(z0)
  h = (cube_root(z1) - t0) / n
  acc = zero(number_type(k))
  for j in 0:n
    t = j == n ? cube_root(z1) : t0 + j * h
    k, _ = s_source_calc(with_fields(k; dz_particles = t^3), geom, kp, search)
    w = (j == 0 || j == n) ? 1 : (isodd(j) ? 4 : 2)
    acc += w * I_csr_value(k, kp) * 3t^2
  end
  return k, acc * h / 3
end

"""
    csr_wake_kernel!(I_csr, I_int, s_source, geom, kp, dz, n_lag, search)

One work item per block of `n_lag` consecutive lags i (separation i*dz). Within a block each source search
starts where the previous lag's ended (as in Bmad); the first lag of a block starts in the kick element.
Searches end in the same element, and call zbrent on the same bracket, whatever the start element.
For i > 1, I_int is overwritten by csr_trapezoid_kernel!. `search` as in `s_source_calc`.
"""
@kernel function csr_wake_kernel!(I_csr, I_int, s_source, geom, kp, dz, n_lag, search)
  b = @index(Global)
  k = CSRKick1(geom, kp)
  for i in (b - 1) * n_lag + 1:min(b * n_lag, length(I_csr))
    k, s_chord = s_source_calc(with_fields(k; dz_particles = i * dz), geom, kp, search)
    k = I_csr_calc(k, i, dz, geom, kp, 0.0, search)
    @inbounds begin
      I_csr[i] = k.I_csr
      I_int[i] = k.I_int_csr
      s_source[i] = source_s(geom, k.ix_ele_source, s_chord)
    end
  end
end

"""
    csr_trapezoid_kernel!(I_int, I_csr, dz)

Trapezoid bin integrals `I_int[i] = (I_csr[i] + I_csr[i-1]) dz / 2` for bins `i ≥ 2` (Bmad; `all_bin_quadrature =
false` or `bmad_compat`). One work item per bin.
"""
@kernel function csr_trapezoid_kernel!(I_int, @Const(I_csr), dz)
  j = @index(Global)
  i = j + 1
  @inbounds I_int[i] = (I_csr[i] + I_csr[i-1]) * dz / 2
end

"""
    csr_bin_quadrature_kernel!(I_int, geom, kp, dz, n_lag, n_quad, search)

Bins i ≥ 2 by quadrature: one work item per block of `n_lag` consecutive bins (bin i is block element i - 1);
the source search is warm-started from node to node and bin to bin within a block. `search` as in `s_source_calc`.
"""
@kernel function csr_bin_quadrature_kernel!(I_int, geom, kp, dz, n_lag, n_quad, search)
  b = @index(Global)
  k = CSRKick1(geom, kp)
  for i in (b - 1) * n_lag + 2:min(b * n_lag + 1, length(I_int))
    k, v = I_csr_quadrature(k, (i - 1) * dz, i * dz, geom, kp, n_quad, search)
    @inbounds I_int[i] = v
  end
end
