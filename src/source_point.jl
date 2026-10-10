# Retarded source point. Port of Bmad's s_source_calc, ddz_calc_csr and super_zbrent.
# Match and floor_shift elements are not supported (no Beamlines equivalent).

"""
    CSRKickPoint

Kick-point information that is the same for all source lags (part of Bmad's `csr_struct`).
"""
struct CSRKickPoint
  ix_ele_kick::Int        # Element containing the kick point
  s_kick::Float64         # Kick point, reference s from the element entrance
  s_chord_kick::Float64   # Kick point along the centroid chord
  floor_k::FloorPosition  # Floor position of the kick point
  gamma::Float64
  gamma2::Float64
  beta::Float64
  y_source::Float64       # Height of the source particle (image charges; 0 for free space)
  kick_factor::Float64    # Scales the kick (1/2 at element ends)
end

function CSRKickPoint(geom::CSRGeometry, ix_ele_kick, s_kick, gamma; y_source = 0.0, kick_factor = 1.0)
  e = geom.ele[ix_ele_kick]
  s_chord_kick = s_ref_to_s_chord(s_kick, e)
  x = spline_val(e.spline, s_chord_kick)
  c, s = cos(e.theta_chord), sin(e.theta_chord)
  r = SVector(x * c + s_chord_kick * s, 0.0, -x * s + s_chord_kick * c) + e.floor0.r
  theta = e.theta_chord + spline_d1(e.spline, s_chord_kick)
  beta = sqrt(1 - 1 / gamma^2)
  return CSRKickPoint(ix_ele_kick, s_kick, s_chord_kick, FloorPosition(r, theta), gamma, gamma^2, beta, y_source, kick_factor)
end

"""
    CSRKick1

Kick quantities for one source-to-kick separation (Bmad `csr_kick1_struct`).
"""
mutable struct CSRKick1
  I_csr::Float64              # Kick integral
  I_int_csr::Float64          # Integrated kick integral
  L_vec::SVector{3,Float64}   # Source-to-kick vector in floor coordinates
  L::Float64                  # Distance between source and kick points
  dL::Float64                 # Ls - L
  dz_particles::Float64       # Kicked minus source particle position at constant time
  s_chord_source::Float64     # Source point along the source element chord
  theta_L::Float64            # Angle of L_vec
  theta_sl::Float64           # Angle between source velocity and L_vec
  theta_lk::Float64           # Angle between L_vec and kicked particle velocity
  ix_ele_source::Int          # Source element (0 = upstream of the lattice)
  floor_s::FloorPosition      # Floor position of the source point
end

CSRKick1() = CSRKick1(0.0, 0.0, SVector(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, FloorPosition())

"""
    ddz_calc_csr!(k, s_chord_source, geom, kp)

Separation (at constant time) between the kicked particle and a source at chord position `s_chord_source`
in element `k.ix_ele_source`, minus the target `k.dz_particles`. Updates the geometry fields of `k`.
"""
function ddz_calc_csr!(k::CSRKick1, s_chord_source, geom::CSRGeometry, kp::CSRKickPoint)
  es = geom.ele[k.ix_ele_source]
  bc = geom.bmad_compat
  x = spline_val(es.spline, s_chord_source)
  c, s = cos(es.theta_chord), sin(es.theta_chord)
  rs = SVector(x * c + s_chord_source * s, kp.y_source, -x * s + s_chord_source * c) + es.floor0.r

  k.L_vec = kp.floor_k.r - rs
  k.L = sqrt(k.L_vec[1]^2 + k.L_vec[2]^2 + k.L_vec[3]^2)
  k.theta_L = atan(k.L_vec[1], k.L_vec[3])

  s0 = s_chord_source
  s1 = kp.s_chord_kick

  if k.ix_ele_source == kp.ix_ele_kick
    ds = s1 - s0
    # dtheta_L = angle of the L line in the centroid chord frame
    sp = es.spline
    dtheta_L = sp[2] + sp[3] * (2s0 + ds) + sp[4] * (3s0^2 + 3s0 * ds + ds^2)
    dL = dspline_len(s0, s1, sp, dtheta_L; bmad_compat = bc)   # = Ls - L
    # Ls is negative if the source point is ahead of the kick point (ds < 0), but L is always positive.
    ds < 0 && (dL += 2ds)
    k.theta_sl = spline_d1(sp, s0) - dtheta_L
    k.theta_lk = dtheta_L - spline_d1(sp, s1)
    theta_s = k.theta_sl + k.theta_L
    # For a source ahead of the kick point L_vec points backwards, and theta_sl + theta_L is the source direction
    # + π (Bmad, planning/PLAN.md V8). Only the image-charge kick uses it.
    ds < 0 && !bc && (theta_s = modulo2(theta_s + pi, pi))
  elseif k.ix_ele_source > kp.ix_ele_kick
    # Source downstream of the kick element (image charges, source ahead of the kick point; not in Bmad).
    # Ls = -(path length from the kick point to the source), so dL = Ls - L = -(excess + 2L) in the plane.
    theta_L = k.theta_L
    ek = geom.ele[kp.ix_ele_kick]
    excess = dspline_len(s1, ek.L_chord, ek.spline, modulo2(theta_L - ek.theta_chord, pi / 2); bmad_compat = bc)
    for i in kp.ix_ele_kick+1:k.ix_ele_source-1
      ce = geom.ele[i]
      excess += dspline_len(0.0, ce.L_chord, ce.spline, modulo2(theta_L - ce.theta_chord, pi / 2); bmad_compat = bc)
    end
    excess += dspline_len(0.0, s0, es.spline, modulo2(theta_L - es.theta_chord, pi / 2); bmad_compat = bc)
    dL = -(excess + 2sqrt(k.L_vec[1]^2 + k.L_vec[3]^2))
    theta_s = spline_d1(es.spline, s0) + es.theta_chord
    k.theta_sl = modulo2(theta_s - theta_L, pi / 2)
    k.theta_lk = modulo2(theta_L - (spline_d1(ek.spline, s1) + ek.theta_chord), pi / 2)
  else
    # Where the centroid takes a backstep, theta_chord is anti-parallel to the other angles, so angles
    # are folded into [-π/2, π/2).
    theta_L = k.theta_L
    dL = dspline_len(s0, es.L_chord, es.spline, modulo2(theta_L - es.theta_chord, pi / 2); bmad_compat = bc)
    for i in k.ix_ele_source+1:kp.ix_ele_kick-1
      ce = geom.ele[i]
      dL += dspline_len(0.0, ce.L_chord, ce.spline, modulo2(theta_L - ce.theta_chord, pi / 2); bmad_compat = bc)
    end
    ce = geom.ele[kp.ix_ele_kick]
    dL += dspline_len(0.0, s1, ce.spline, modulo2(theta_L - ce.theta_chord, pi / 2); bmad_compat = bc)
    k.theta_sl = modulo2((spline_d1(es.spline, s0) + es.theta_chord) - theta_L, pi / 2)
    k.theta_lk = modulo2(theta_L - (spline_d1(ce.spline, s1) + ce.theta_chord), pi / 2)
    theta_s = k.theta_sl + k.theta_L
  end

  k.floor_s = FloorPosition(rs, theta_s)

  # The dL calculation above neglects y_source, so correct for it.
  kp.y_source != 0 && (dL -= k.L - sqrt(k.L_vec[1]^2 + k.L_vec[3]^2))
  k.dL = dL
  return k.L / (2kp.gamma2) + dL - k.dz_particles
end

"""
    zbrent(f, x1, x2, rel_tol, abs_tol) -> (x_zero, ok)

Brent's root finder, following Bmad's `super_zbrent` (including the final linear interpolation).
"""
function zbrent(f, x1, x2, rel_tol, abs_tol)
  itmax = 100
  a = x1; b = x2
  fa = f(a); fb = f(b)
  if (fa > 0 && fb > 0) || (fa < 0 && fb < 0)
    return 1e100, false
  end
  c = b; fc = fb
  d = b - a; e = d
  for _ in 1:itmax
    if (fb > 0 && fc > 0) || (fb < 0 && fc < 0)
      c = a; fc = fa
      d = b - a; e = d
    end
    if abs(fc) < abs(fb)
      a = b; b = c; c = a
      fa = fb; fb = fc; fc = fa
    end
    tol1 = 0.5 * (rel_tol * abs(b) + abs_tol)
    xm = 0.5 * (c - b)
    if abs(xm) <= tol1 || fb == 0
      return (fb == 0 ? b : (b * fc - c * fb) / (fc - fb)), true   # Linear interpolation
    end
    if abs(e) >= tol1 && abs(fa) > abs(fb)
      s = fb / fa
      if a == c
        p = 2xm * s
        q = 1 - s
      else
        q = fa / fc
        r = fb / fc
        p = s * (2xm * q * (q - r) - (b - a) * (r - 1))
        q = (q - 1) * (r - 1) * (s - 1)
      end
      p > 0 && (q = -q)
      p = abs(p)
      if 2p < min(3xm * q - abs(tol1 * q), abs(e * q))
        e = d
        d = p / q
      else
        d = xm
        e = d
      end
    else
      d = xm
      e = d
    end
    a = b
    fa = fb
    b += abs(d) > tol1 ? d : copysign(tol1, xm)
    fb = f(b)
  end
  return b, false
end

"Reference s of a source at `s_chord` in element `ix_ele_source` (0: upstream drift, `n + 1`: downstream drift)."
function source_s(geom::CSRGeometry, ix_ele_source, s_chord)
  ix_ele_source == 0 && return s_chord
  if ix_ele_source > length(geom.ele)
    e = geom.ele[end]
    return e.s_start + e.L + s_chord
  end
  return geom.ele[ix_ele_source].s_start + s_chord
end

"""
    s_source_calc!(k, geom, kp) -> s_chord_source

Find the source point for target separation `k.dz_particles`, starting the search in element
`k.ix_ele_source` and walking element by element. Updates `k`. A source upstream of the lattice is
placed on an infinite drift (`k.ix_ele_source = 0`, negative s).

A source ahead of the kick point (`dz < 0`, image charges) beyond the kick element is searched for downstream,
and past the lattice end on an infinite drift (`k.ix_ele_source = length(geom.ele) + 1`, returning the distance
past the end). Bmad does not do this; `geom.bmad_compat` restores Bmad's behaviour.
"""
function s_source_calc!(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint)
  dz = k.dz_particles
  beta2 = kp.beta^2
  last_step = 0
  ddz(s) = ddz_calc_csr!(k, s, geom, kp)

  while true
    # Source upstream of the lattice: infinite drift, closed-form root.
    if k.ix_ele_source == 0
      fk = kp.floor_k
      f0 = geom.start
      L0 = sqrt((fk.r[1] - f0.r[1])^2 + (fk.r[3] - f0.r[3])^2 + kp.y_source^2)
      # Lz is the z-component from the lattice start to the kick point
      Lz = (fk.r[1] - f0.r[1]) * sin(f0.theta) + (fk.r[3] - f0.r[3]) * cos(f0.theta)
      # Lsz0 is Ls from the lattice start to the kick point
      ek = geom.ele[kp.ix_ele_kick]
      Lsz0 = dspline_len(0.0, kp.s_chord_kick, ek.spline; bmad_compat = geom.bmad_compat) + kp.s_chord_kick
      for i in 1:kp.ix_ele_kick-1
        Lsz0 += geom.ele[i].dL_s + geom.ele[i].L_chord
      end
      a = 1 / kp.gamma2
      b = 2 * (Lsz0 - dz - beta2 * Lz)
      c = (Lsz0 - dz)^2 - beta2 * L0^2
      ds_source = -(-b + sqrt(b^2 - 4a * c)) / (2a)
      # A search warm-started here (another lag) may belong inside the lattice: go back and walk.
      if ds_source > 0 && last_step == 0
        k.ix_ele_source = 1
        continue
      end

      rs =SVector(f0.r[1] + ds_source * sin(f0.theta), kp.y_source, f0.r[3] + ds_source * cos(f0.theta))
      k.floor_s = FloorPosition(rs, f0.theta)
      k.L_vec = fk.r - rs
      k.L = sqrt(k.L_vec[1]^2 + k.L_vec[2]^2 + k.L_vec[3]^2)
      k.dL = Lsz0 - ds_source - k.L   # ds_source is negative
      k.theta_L = atan(k.L_vec[1], k.L_vec[3])
      k.theta_sl = f0.theta - k.theta_L
      k.theta_lk = k.theta_L - (spline_d1(ek.spline, kp.s_chord_kick) + ek.theta_chord)
      k.s_chord_source = ds_source
      return ds_source
    end

    # Source downstream of the lattice (image charges; not in Bmad): infinite drift along the centroid exit
    # direction. Exact retardation Ls - βL = dz with Ls = -(Lke + t), t ≥ 0 the distance past the lattice end.
    if k.ix_ele_source > length(geom.ele)
      fk = kp.floor_k
      fe = geom.ele[end].floor1
      ek = geom.ele[kp.ix_ele_kick]
      # Lke is Ls from the kick point to the lattice end
      Lke = dspline_len(kp.s_chord_kick, ek.L_chord, ek.spline; bmad_compat = geom.bmad_compat) + ek.L_chord - kp.s_chord_kick
      for i in kp.ix_ele_kick+1:length(geom.ele)
        Lke += geom.ele[i].dL_s + geom.ele[i].L_chord
      end
      ez = z_axis(fe.theta)
      dr = fk.r - fe.r
      # (u - t)² = β² L² with u = -(Lke + dz), L² = L0² - 2t Lz + t²: t²/γ² - 2b t + c = 0, smaller root.
      u = -(Lke + dz)
      b = u - beta2 * (dr[1] * ez[1] + dr[3] * ez[3])
      c = u^2 - beta2 * (dr[1]^2 + dr[3]^2 + kp.y_source^2)
      t = c / (b + sqrt(b^2 - c / kp.gamma2))
      # A search warm-started here (previous lag) may belong inside the lattice: go back and walk.
      if t < 0 && last_step == 0
        k.ix_ele_source = length(geom.ele)
        continue
      end

      rs = fe.r + t * ez + SVector(0.0, kp.y_source, 0.0)
      k.floor_s = FloorPosition(rs, fe.theta)
      k.L_vec = fk.r - rs
      k.L = sqrt(k.L_vec[1]^2 + k.L_vec[2]^2 + k.L_vec[3]^2)
      k.dL = -(Lke + t) - k.L
      k.theta_L = atan(k.L_vec[1], k.L_vec[3])
      k.theta_sl = modulo2(fe.theta - k.theta_L, pi / 2)
      k.theta_lk = modulo2(k.theta_L - (spline_d1(ek.spline, kp.s_chord_kick) + ek.theta_chord), pi / 2)
      k.s_chord_source = t
      return t
    end

    es = geom.ele[k.ix_ele_source]
    # Check whether the source point is within this element. Generally ddz decreases with increasing s,
    # but this may not hold for patch elements.
    ddz0 = ddz(0.0)
    ddz1 = ddz(es.L_chord)

    if last_step == -1 && ddz1 > 0   # Round-off error makes ddz1 positive
      # Bmad returns the ddz value here, not a position (csr_and_space_charge_mod.f90:952).
      k.s_chord_source = ddz(es.L_chord)
      return k.s_chord_source
    end
    if last_step == 1 && ddz0 < 0    # Round-off error makes ddz0 negative
      k.s_chord_source = ddz(0.0)
      return k.s_chord_source
    end

    if ddz0 < 0 && ddz1 < 0
      last_step == 1 && break
      last_step = -1
      k.ix_ele_source -= 1
      continue
    end

    if ddz0 > 0 && ddz1 > 0
      # Source beyond this element. Beyond the kick element it is ahead of the kick point, which needs dz < 0
      # (image charges). Bmad ignores it there (k is left describing the source at the end of the kick element);
      # CSR.jl continues downstream (planning/PLAN.md, V8).
      if k.ix_ele_source == kp.ix_ele_kick && (geom.bmad_compat || dz >= 0)
        k.s_chord_source = 0.0
        return 0.0
      end
      last_step == -1 && break
      last_step = 1
      k.ix_ele_source += 1
      continue
    end

    # Root is bracketed
    s_source, ok = zbrent(ddz, 0.0, es.L_chord, 1e-12, 1e-8)
    ok || error("s_source_calc!: root finder failed in element $(k.ix_ele_source)")
    # zbrent leaves k describing its last evaluation, not the returned root (Bmad does the same). These differ
    # by up to the tolerance, or completely when the root is an endpoint with ddz exactly 0 (zbrent then
    # returns without evaluating there). Re-evaluate at the root (planning/PLAN.md, V7).
    geom.bmad_compat || ddz(s_source)
    k.s_chord_source = s_source
    return s_source
  end
  error("s_source_calc!: CSR source-point search failed (element walk did not converge)")
end
