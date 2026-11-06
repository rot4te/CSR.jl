# Free-space kick integrand I_csr, its bin integrals, and the
# wake-table kernels (Bmad I_csr).

"""Kick integrand for the source geometry stored in `k` (Bmad I_csr
formula).

# Arguments
- `k`: work struct (`CSRKick1`) describing the source (uses
  `dz_particles` (nonzero), `dL`, `L`, `theta_sl`, `theta_lk`), from
  `s_source_calc`.
- `kp`: kick point, from `CSRKickPoint` (uses `kp.gamma2` and
  `kp.kick_factor`).

# Returns
- the integrand value, including `kp.kick_factor`.
"""
function I_csr_value(k::CSRKick1, kp::CSRKickPoint)
  z = k.dz_particles
  return -kp.kick_factor * 2 * (k.dL / z + kp.gamma2 * k.theta_sl *
                                k.theta_lk / (1 + kp.gamma2 *
      k.theta_sl^2)) / k.L
end

"""
    I_csr_calc(k, i_bin, dz_slice, geom, kp, I_csr_prev,
               search = nothing) -> k

Bmad `I_csr`: `k` with the CSR kick integral `I_csr` for separation
`k.dz_particles` and its integral over the bin, `I_int_csr`. For
`i_bin = 1` the integral from 0 uses Bmad's closed form over the kick
element, which handles the near-field log singularity, plus (if the
source is upstream of the kick element) the rest of the bin by
quadrature (`I_csr_integral`; Bmad's approximation with
`geom.bmad_compat`). For `i_bin > 1` it is a trapezoid with the
previous bin's `I_csr_prev`. `search` as in `s_source_calc` (for the
quadrature's sources).

# Arguments
- `k`: work struct (`CSRKick1`) describing the source for the lag
  `k.dz_particles` [m], from `s_source_calc`.
- `i_bin`: bin (lag) index, `≥ 1`.
- `dz_slice`: bin width [m] (used for `i_bin > 1`).
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint`.
- `I_csr_prev`: `I_csr` of bin `i_bin - 1` (used for `i_bin > 1`;
  `csr_wake_kernel!` passes 0, its `I_int` for `i > 1` being
  overwritten later).
- `search = nothing`: `nothing` or a `ScalarizeAllSearch`.

# Returns
- `k` with `I_csr` and `I_int_csr` set (both 0 for
  `k.dz_particles ≤ 0`).
"""
function I_csr_calc(k::CSRKick1, i_bin, dz_slice, geom::CSRGeometry,
                    kp::CSRKickPoint, I_csr_prev, search = nothing)
  # No kick when the source particle is ahead of the kicked particle
  z = k.dz_particles
  Beamlines.scalarize(z) <= 0 && return with_fields(k;
      I_csr = 0.0, I_int_csr = 0.0)

  gamma2 = kp.gamma2
  I_csr = I_csr_value(k, kp)
  I_int_csr = zero(I_csr)

  if i_bin == 1
    ix_ele_kick = kp.ix_ele_kick
    s_chord_kick = kp.s_chord_kick
    # At the element entrance: use the end of the previous element
    while Beamlines.scalarize(s_chord_kick) == 0
      ix_ele_kick -= 1
      # No kick from before the beginning of the lattice
      ix_ele_kick == 0 && return with_fields(k; I_csr, I_int_csr)
      s_chord_kick = geom.ele[ix_ele_kick].L_chord
    end

    ek = geom.ele[ix_ele_kick]
    spl = ek.spline
    g_bend = -spline_d2(spl, s_chord_kick) / sqrt(1 + spline_d1(spl,
        s_chord_kick)^2)^3
    # Ls/z of a straight line: 2γ² with Bmad's retardation, γ²(1 +
    # β) exact (D42), so the log vanishes there
    straight = geom.bmad_compat ? 2gamma2 : gamma2 * (1 + kp.beta)

    if k.ix_ele_source == ix_ele_kick
      Ls = k.L + k.dL
      I_int_csr = -kp.kick_factor * ((g_bend * Ls / 2)^2 -
                                     log(straight * z / Ls) / gamma2)
    else
      # Source point in another element: first integrate over the
      # element containing the kick point, then add the rest.
      if geom.bmad_compat
        L = s_chord_kick
        dtheta_L = spl[2] + spl[3] * L + spl[4] * L^2
        # = Ls - L
        dL = path_excess(0.0, L, spl, dtheta_L; bmad_compat = true)
        Ls = L + dL
        zz = L / (2gamma2) + dL
      else
        # zz: the lag of a source at the element entrance
        # (ddz_calc_csr, so that the quadrature starts where the
        # source search crosses into the previous element); Ls: arc
        # length from the entrance to the kick point.
        k_entry, zz = ddz_calc_csr(with_fields(k;
            ix_ele_source = ix_ele_kick,
            dz_particles = zero(k.dz_particles)),
                                   0.0, geom, kp)
        Ls = path_excess(0.0, s_chord_kick, spl) + s_chord_kick
      end
      I_int_csr = -kp.kick_factor * ((g_bend * Ls / 2)^2 -
                                     log(straight * zz / Ls) / gamma2)
      if geom.bmad_compat
        # Bmad approximates the rest of the integral with I_csr held
        # at its value at z. In steady state (I ∝ u^(-1/3), zz ≪
        # z) this gives 2/3 of the integral.
        I_int_csr += I_csr * (z - zz)
      else
        # First node: the entrance source itself, not a search for lag
        # zz (which may return the previous element's end, the same
        # point but other numbers: two branches for derivatives w.r.t.
        # the raw geometry numbers)
        I_int_csr += I_csr_integral(zz, z, geom, kp, search;
                                    k0 = with_fields(k_entry;
            dz_particles = zz))
      end
    end
  else
    I_int_csr = (I_csr + I_csr_prev) * dz_slice / 2
  end
  return with_fields(k; I_csr, I_int_csr)
end

"""
    I_csr_integral(z0, z1, geom, kp, search = nothing; n = 16,
                   k0 = nothing)

`∫ I_csr(u) du` from `z0` to `z1` by composite Simpson in `t =
u^(1/3)`, which makes the near-field `u^(-1/3)` behaviour smooth. Each
node needs its own source-point solve, started in the kick element
(`search` as in `s_source_calc`). `k0`: the source at `z0`, if known
(then not searched).

# Arguments
- `z0`, `z1`: lag bounds [m].
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint`.
- `search = nothing`: `nothing` or a `ScalarizeAllSearch`.
- `n = 16`: number of Simpson intervals in `t` (must be even; not
  checked).
- `k0 = nothing`: `nothing` or a `CSRKick1` describing the source at
  `z0` (`I_csr_calc` passes the source at the kick element's
  entrance, from `ddz_calc_csr`).

# Returns
- the integral (a number).
"""
function I_csr_integral(z0, z1, geom::CSRGeometry, kp::CSRKickPoint,
                        search = nothing; n = 16, k0 = nothing)
  function f(t)
    k, _ = s_source_calc(CSRKick1(geom, kp; dz_particles = t^3), geom,
                         kp, search)
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
    I_csr_quadrature(k, z0, z1, geom, kp, n, search = nothing) -> (k,
        integral)

As `I_csr_integral`, with each source search starting where the
previous one ended (from `k`; nodes in increasing order). Searches end
in the same element whatever the start element, so the result does not
depend on the warm start. Returns the work struct after the last node,
for the next warm start. `search` as in `s_source_calc`.

# Arguments
- `k`: work struct (`CSRKick1`) whose `ix_ele_source` starts the
  first search, from `CSRKick1` or a previous call.
- `z0`, `z1`: lag bounds [m].
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint`.
- `n`: number of Simpson intervals in `t = u^(1/3)` (must be even;
  not checked; callers: `N_BIN_QUAD`).
- `search = nothing`: `nothing` or a `ScalarizeAllSearch`.

# Returns
- `(k, integral)`: the work struct after the last node, and
  `∫ I_csr(u) du` from `z0` to `z1`.
"""
function I_csr_quadrature(k::CSRKick1, z0, z1, geom::CSRGeometry,
                          kp::CSRKickPoint, n, search = nothing)
  t0 = cube_root(z0)
  h = (cube_root(z1) - t0) / n
  acc = zero(number_type(k))
  for j in 0:n
    t = j == n ? cube_root(z1) : t0 + j * h
    k, _ = s_source_calc(with_fields(k; dz_particles = t^3), geom, kp,
                         search)
    w = (j == 0 || j == n) ? 1 : (isodd(j) ? 4 : 2)
    acc += w * I_csr_value(k, kp) * 3t^2
  end
  return k, acc * h / 3
end

"""
    csr_wake_kernel!(I_csr, I_int, s_source, geom, kp, dz, n_lag,
                     search)

One work item per block of `n_lag` consecutive lags i (separation
i*dz). Within a block each source search starts where the previous
lag's ended (as in Bmad); the first lag of a block starts in the kick
element. Searches end in the same element, and call zbrent on the same
bracket, whatever the start element. For i > 1, I_int is overwritten
in `csr_wake_table` by `csr_trapezoid_kernel!` or, with
`all_bin_quadrature`, by `csr_bin_quadrature_kernel!`. `search` as in
`s_source_calc`.

# Arguments
- `I_csr`, `I_int`, `s_source`: vectors of length `n_bin`
  (modified), allocated by `csr_wake_table`: per lag `i`, `k.I_csr`
  and `k.I_int_csr` of `I_csr_calc`, and the reference s [m] of the
  source (`source_s`).
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint`.
- `dz`: bin width [m].
- `n_lag`: lags per work item (`csr_wake_table`: `cld(n_bin,
  n_block)`).
- `search`: `nothing` or a `ScalarizeAllSearch`, from
  `scalarize_all_search`.
"""
@kernel function csr_wake_kernel!(I_csr, I_int, s_source, geom, kp,
                                  dz, n_lag, search)
  b = @index(Global)
  k = CSRKick1(geom, kp)
  for i in (b - 1) * n_lag + 1:min(b * n_lag, length(I_csr))
    k, s_chord = s_source_calc(with_fields(k; dz_particles = i * dz),
                               geom, kp, search)
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

Trapezoid bin integrals `I_int[i] = (I_csr[i] + I_csr[i-1]) dz / 2`
for bins `i ≥ 2` (Bmad; `all_bin_quadrature = false` or
`bmad_compat`). One work item per bin.

# Arguments
- `I_int`: bin integrals (modified for `i ≥ 2`), from
  `csr_wake_kernel!`.
- `I_csr`: kick integrals per lag, from `csr_wake_kernel!`.
- `dz`: bin width [m].
"""
@kernel function csr_trapezoid_kernel!(I_int, @Const(I_csr), dz)
  j = @index(Global)
  i = j + 1
  @inbounds I_int[i] = (I_csr[i] + I_csr[i-1]) * dz / 2
end

"""
    csr_bin_quadrature_kernel!(I_int, geom, kp, dz, n_lag, n_quad,
                               search)

Bins i ≥ 2 by quadrature: one work item per block of `n_lag`
consecutive bins (bin i is block element i - 1); the source search is
warm-started from node to node and bin to bin within a block. `search`
as in `s_source_calc`.

# Arguments
- `I_int`: bin integrals (modified for `i ≥ 2`), from
  `csr_wake_kernel!`.
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint`.
- `dz`: bin width [m]; bin `i` spans lags `(i - 1) dz` to `i dz`.
- `n_lag`: bins per work item.
- `n_quad`: Simpson intervals per bin (`csr_wake_table`:
  `N_BIN_QUAD`), passed to `I_csr_quadrature`.
- `search`: `nothing` or a `ScalarizeAllSearch`, from
  `scalarize_all_search`.
"""
@kernel function csr_bin_quadrature_kernel!(I_int, geom, kp, dz,
    n_lag, n_quad, search)
  b = @index(Global)
  k = CSRKick1(geom, kp)
  for i in (b - 1) * n_lag + 2:min(b * n_lag + 1, length(I_int))
    k, v = I_csr_quadrature(k, (i - 1) * dz, i * dz, geom, kp, n_quad,
                            search)
    @inbounds I_int[i] = v
  end
end
