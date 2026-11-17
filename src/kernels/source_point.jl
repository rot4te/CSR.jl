# Retarded source point for a lag. Port of Bmad's s_source_calc and
# ddz_calc_csr; runs inside the table kernels. Match and floor_shift
# elements are not supported (no Beamlines equivalent).

"""
    ddz_calc_csr(k, s_chord_source, geom, kp) -> (k, ddz)

Separation (at constant time) between the kicked particle and a source
at chord position `s_chord_source` in element `k.ix_ele_source`, minus
the target `k.dz_particles`. Returns `k` with the source geometry
fields and `s_chord_source` set, and the separation.

# Arguments
- `k`: work struct (`CSRKick1`) with the source element
  `k.ix_ele_source` (in `1:length(geom.ele)`) and the target lag
  `k.dz_particles` [m]. From `CSRKick1` or a previous source-point
  call.
- `s_chord_source`: source position [m] along the chord of element
  `k.ix_ele_source`.
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint` (uses `kp.floor_k`,
  `kp.y_source`, `kp.s_chord_kick`, `kp.ix_ele_kick`, `kp.gamma2`,
  `kp.beta`).

# Returns
- `(k, ddz)`: `k` with `L_vec`, `L`, `theta_L`, `theta_sl`,
  `theta_lk`, `dL`, `s_chord_source` and `floor_s` set, and the
  separation minus `k.dz_particles` [m].
"""
@inline function ddz_calc_csr(k::CSRKick1, s_chord_source,
                              geom::CSRGeometry, kp::CSRKickPoint)
  es = geom.ele[k.ix_ele_source]
  bc = geom.bmad_compat
  x = spline_val(es.spline, s_chord_source)
  c, s = cos(es.theta_chord), sin(es.theta_chord)
  rs = SVector(x * c + s_chord_source * s, kp.y_source,
               -x * s + s_chord_source * c) + es.floor0.r

  L_vec = kp.floor_k.r - rs
  L2 = L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2
  Lp2 = L_vec[1]^2 + L_vec[3]^2
  theta_L = atan(L_vec[1], L_vec[3])
  Lp = sqrt(Lp2)
  L = sqrt(L2)

  s0 = s_chord_source
  s1 = kp.s_chord_kick

  if k.ix_ele_source == kp.ix_ele_kick
    ds = s1 - s0
    sp = es.spline
    if bc
      # Bmad: dtheta_L = slope of the L line in the centroid chord
      # frame (the secant of the spline)
      dtheta_L = sp[2] + sp[3] * (2s0 + ds) + sp[4] * (3s0^2 + 3s0 *
          ds + ds^2)
      # = Ls - L
      dL = path_excess(s0, s1, sp, dtheta_L; bmad_compat = bc)
      # Ls is negative if the source point is ahead of the kick point
      # (ds < 0), but L is always positive.
      ds < 0 && (dL += 2ds)
      theta_sl = spline_d1(sp, s0) - dtheta_L
      theta_lk = dtheta_L - spline_d1(sp, s1)
      # For a source ahead of the kick point L_vec points backwards,
      # and this is the source direction + π (PLAN.md, V8). Only the
      # image-charge kick uses it.
      theta_s = theta_sl + theta_L
    else
      # The angle of the L line in the chord frame, as in the other
      # branches, so that ddz is continuous across element boundaries
      # (one root; Bmad's secant slope gives a jump ~ (α³/3)² ds at
      # the boundary).
      dtheta_L = modulo2(theta_L - es.theta_chord, pi / 2)
      dL = path_excess(s0, s1, sp, dtheta_L)
      # Source ahead: Ls = -(excess + L_plane), so dL = Ls - L =
      # -(excess + 2 L_plane) (as for downstream sources; Bmad's 2ds
      # is the chord-coordinate difference).
      ds < 0 && (dL -= 2Lp)
      theta_sl = spline_d1(sp, s0) - dtheta_L
      theta_lk = dtheta_L - spline_d1(sp, s1)
      theta_s = spline_d1(sp, s0) + es.theta_chord
    end
  elseif k.ix_ele_source > kp.ix_ele_kick
    # Source downstream of the kick element (image charges, source
    # ahead of the kick point; not in Bmad). Ls = -(path length from
    # the kick point to the source), so dL = Ls - L = -(excess + 2L)
    # in the plane.
    ek = geom.ele[kp.ix_ele_kick]
    excess = path_excess(s1, ek.L_chord, ek.spline,
                         modulo2(theta_L - ek.theta_chord, pi / 2);
                         bmad_compat = bc)
    excess = middle_excess(excess, geom,
                           kp.ix_ele_kick+1:k.ix_ele_source-1,
                           theta_L)
    excess += path_excess(0.0, s0, es.spline,
                          modulo2(theta_L - es.theta_chord, pi / 2);
                          bmad_compat = bc)
    dL = -(excess + 2Lp)
    theta_s = spline_d1(es.spline, s0) + es.theta_chord
    theta_sl = modulo2(theta_s - theta_L, pi / 2)
    theta_lk = modulo2(theta_L - (spline_d1(ek.spline,
        s1) + ek.theta_chord), pi / 2)
  else
    # Where the centroid takes a backstep, theta_chord is
    # anti-parallel to the other angles, so angles are folded into
    # [-π/2, π/2).
    dL = path_excess(s0, es.L_chord, es.spline,
                     modulo2(theta_L - es.theta_chord, pi / 2);
                     bmad_compat = bc)
    dL = middle_excess(dL, geom, k.ix_ele_source+1:kp.ix_ele_kick-1,
                       theta_L)
    ce = geom.ele[kp.ix_ele_kick]
    dL += path_excess(0.0, s1, ce.spline,
                      modulo2(theta_L - ce.theta_chord, pi / 2);
                      bmad_compat = bc)
    theta_sl = modulo2((spline_d1(es.spline,
                                  s0) + es.theta_chord) - theta_L,
                       pi / 2)
    theta_lk = modulo2(theta_L - (spline_d1(ce.spline,
        s1) + ce.theta_chord), pi / 2)
    theta_s = theta_sl + theta_L
  end

  # The dL calculation above neglects y_source, so correct for it.
  kp.y_source != 0 && (dL -= L - Lp)
  k = with_fields(k; L_vec, L, theta_L, theta_sl, theta_lk, dL,
                  s_chord_source, floor_s = FloorPosition(rs,
      theta_s))
  # Retardation Ls - βL = dL + L(1 - β). Bmad: L(1 - β) ≈
  # L/(2γ²) (error L/(8γ⁴)); the exact form matches the infinite
  # drifts outside the lattice, so ddz is continuous at the lattice
  # ends.
  ret = bc ? L / (2kp.gamma2) : L / (kp.gamma2 * (1 + kp.beta))
  return k, ret + dL - k.dz_particles
end

"""
    middle_excess(acc, geom, r, theta_L)

`acc` plus, element by element, the path-length excess `path_excess(0,
L_chord, spline, modulo2(theta_L - theta_chord, π/2))` of the
elements `r` (strictly between source and kick element in
`ddz_calc_csr`).

# Arguments
- `acc`: path length [m] accumulated so far.
- `geom`: centroid geometry, from `CSRGeometry`.
- `r`: range of element indices into `geom.ele`.
- `theta_L`: floor angle [rad] of the source-to-kick vector
  (`theta_L` in `ddz_calc_csr`).

# Returns
- `acc` plus the elements' path-length excesses [m].
"""
@inline function middle_excess(acc, geom::CSRGeometry, r, theta_L)
  for i in r
    ce = geom.ele[i]
    acc += path_excess(0.0, ce.L_chord, ce.spline,
                       modulo2(theta_L - ce.theta_chord, pi / 2);
                       bmad_compat = geom.bmad_compat)
  end
  return acc
end

"""
    s_source_calc(k, geom, kp) -> (k, s_chord_source)

Find the source point for target separation `k.dz_particles`, starting
the search in element `k.ix_ele_source` and walking element by
element. Returns `k` describing the source and the source position
along its element's chord. A source upstream of the lattice is placed
on an infinite drift (`k.ix_ele_source = 0`, negative s).

A source ahead of the kick point (`dz < 0`, image charges) beyond the
kick element is searched for downstream, and past the lattice end on
an infinite drift (`k.ix_ele_source = length(geom.ele) + 1`, returning
the distance past the end). Bmad does not do this; `geom.bmad_compat`
restores Bmad's behaviour.

# Arguments
- `k`: work struct (`CSRKick1`) with the start element
  `k.ix_ele_source` and the lag `k.dz_particles` [m]. From
  `CSRKick1(geom, kp; dz_particles)`, or a previous call's `k` with
  a new lag (`with_fields`) for a warm start.
- `geom`: centroid geometry, from `CSRGeometry`.
- `kp`: kick point, from `CSRKickPoint`.

# Returns
- `(k, s_chord_source)`: `k` describing the source (as set by
  `ddz_calc_csr`, `k.ix_ele_source` its element) and the source
  position [m] along that element's chord (on the drifts: from the
  lattice start or end; for Bmad's round-off branches, the residual
  of `source_at`; 0 for a source beyond the kick element).
"""
function s_source_calc(k::CSRKick1, geom::CSRGeometry,
                       kp::CSRKickPoint)
  dz = k.dz_particles
  last_step = 0

  while true
    # Source upstream of the lattice: infinite drift, closed-form
    # root.
    if k.ix_ele_source == 0
      k1, ds_source = upstream_drift_source(k, geom, kp)
      # A search warm-started here (another lag) may belong inside the
      # lattice: go back and walk.
      if ds_source > 0 && last_step == 0
        k = with_fields(k; ix_ele_source = 1)
        continue
      end
      return k1, ds_source
    end

    # Source downstream of the lattice (image charges; not in Bmad).
    if k.ix_ele_source > length(geom.ele)
      k1, t = downstream_drift_source(k, geom, kp)
      # A search warm-started here (previous lag) may belong inside
      # the lattice: go back and walk.
      if t < 0 && last_step == 0
        k = with_fields(k; ix_ele_source = length(geom.ele))
        continue
      end
      return k1, t
    end

    ix = k.ix_ele_source
    es = geom.ele[ix]
    # Check whether the source point is within this element. Generally
    # ddz decreases with increasing s, but this may not hold for patch
    # elements.
    k, ddz0 = ddz_calc_csr(k, 0.0, geom, kp)
    k, ddz1 = ddz_calc_csr(k, es.L_chord, geom, kp)

    # Round-off error makes ddz1 positive
    if last_step == -1 && ddz1 > 0
      k, r = source_at(k, es.L_chord, geom, kp)
      return k, r
    end
    # Round-off error makes ddz0 negative
    if last_step == 1 && ddz0 < 0
      k, r = source_at(k, 0.0, geom, kp)
      return k, r
    end

    if ddz0 < 0 && ddz1 < 0
      last_step == 1 && break
      last_step = -1
      k = with_fields(k; ix_ele_source = ix - 1)
      continue
    end

    if ddz0 > 0 && ddz1 > 0
      # Source beyond this element. Beyond the kick element it is
      # ahead of the kick point, which needs dz < 0 (image charges).
      # Bmad ignores it there (k is left describing the source at the
      # end of the kick element); CSR.jl continues downstream
      # (PLAN.md, V8).
      if ix == kp.ix_ele_kick && (geom.bmad_compat || dz >= 0)
        return with_fields(k; s_chord_source = 0.0),
               zero(number_type(k))
      end
      last_step == -1 && break
      last_step = 1
      k = with_fields(k; ix_ele_source = ix + 1)
      continue
    end

    # Root is bracketed.
    kk = k
    s_root, ok, s_last = zbrent(s -> last(ddz_calc_csr(kk, s, geom,
                                                       kp)), 0.0,
                                es.L_chord, 1e-12, 1e-8)
    ok || error("s_source_calc: root finder failed in element $ix")
    k, s_source = root_source(k, s_root, s_last, geom, kp)
    return k, s_source
  end
  error("s_source_calc: CSR source-point search failed (element walk did not converge)")
end

"""
    upstream_drift_source(k, geom, kp) -> (k, ds_source)

Source on the infinite drift upstream of the lattice (closed-form
root; `ds_source < 0` from the lattice start).

# Arguments
- `k`: work struct (`CSRKick1`) with the lag `k.dz_particles` [m]
  (`k.ix_ele_source` is not read or changed; callers have it 0).
- `geom`: centroid geometry, from `CSRGeometry` (uses `geom.start`).
- `kp`: kick point, from `CSRKickPoint`.

# Returns
- `(k, ds_source)`: `k` with the source geometry fields and
  `s_chord_source = ds_source` set, and the source position [m]
  from the lattice start along `geom.start.theta`.
"""
function upstream_drift_source(k::CSRKick1, geom::CSRGeometry,
                               kp::CSRKickPoint)
  dz = k.dz_particles
  beta2 = kp.beta^2
  fk = kp.floor_k
  f0 = geom.start
  # squared distance (only its square enters the quadratic below)
  L0sq = (fk.r[1] - f0.r[1])^2 + (fk.r[3] - f0.r[3])^2 + kp.y_source^2
  # Lz is the z-component from the lattice start to the kick point
  Lz = (fk.r[1] - f0.r[1]) * sin(f0.theta) + (fk.r[3] - f0.r[3]) *
      cos(f0.theta)
  # Lsz0 is Ls from the lattice start to the kick point
  ek = geom.ele[kp.ix_ele_kick]
  Lsz0 = path_excess(0.0, kp.s_chord_kick, ek.spline;
                     bmad_compat = geom.bmad_compat) + kp.s_chord_kick
  for i in 1:kp.ix_ele_kick-1
    Lsz0 += geom.ele[i].dL_s + geom.ele[i].L_chord
  end
  a = 1 / kp.gamma2
  b = 2 * (Lsz0 - dz - beta2 * Lz)
  c = (Lsz0 - dz)^2 - beta2 * L0sq
  ds_source = -(-b + sqrt(b^2 - 4a * c)) / (2a)

  rs = SVector(f0.r[1] + ds_source * sin(f0.theta), kp.y_source,
               f0.r[3] + ds_source * cos(f0.theta))
  L_vec = fk.r - rs
  L = sqrt(L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2)
  theta_L = atan(L_vec[1], L_vec[3])
  k = with_fields(k; floor_s = FloorPosition(rs, f0.theta), L_vec, L,
                  dL = Lsz0 - ds_source - L,   # ds_source is negative
                  theta_L, theta_sl = f0.theta - theta_L,
                  theta_lk = theta_L - (spline_d1(ek.spline,
      kp.s_chord_kick) + ek.theta_chord),
                  s_chord_source = ds_source)
  return k, ds_source
end

"""
    downstream_drift_source(k, geom, kp) -> (k, t)

Source on the infinite drift past the lattice end along the centroid
exit direction (image charges; not in Bmad), `t ≥ 0` the distance
past the end. Exact retardation `Ls - βL = dz` with `Ls = -(Lke +
t)`.

# Arguments
- `k`: work struct (`CSRKick1`) with the lag `k.dz_particles` [m]
  (`k.ix_ele_source` is not read or changed).
- `geom`: centroid geometry, from `CSRGeometry` (uses
  `geom.ele[end].floor1`).
- `kp`: kick point, from `CSRKickPoint`.

# Returns
- `(k, t)`: `k` with the source geometry fields and
  `s_chord_source = t` set, and `t` [m].
"""
function downstream_drift_source(k::CSRKick1, geom::CSRGeometry,
                                 kp::CSRKickPoint)
  dz = k.dz_particles
  beta2 = kp.beta^2
  fk = kp.floor_k
  fe = geom.ele[end].floor1
  ek = geom.ele[kp.ix_ele_kick]
  # Lke is Ls from the kick point to the lattice end
  Lke = path_excess(kp.s_chord_kick, ek.L_chord, ek.spline;
                    bmad_compat = geom.bmad_compat) + ek.L_chord -
                        kp.s_chord_kick
  for i in kp.ix_ele_kick+1:length(geom.ele)
    Lke += geom.ele[i].dL_s + geom.ele[i].L_chord
  end
  ez = z_axis(fe.theta)
  dr = fk.r - fe.r
  # (u - t)² = β² L² with u = -(Lke + dz), L² = L0² - 2t Lz +
  # t²: t²/γ² - 2b t + c = 0, smaller root.
  u = -(Lke + dz)
  b = u - beta2 * (dr[1] * ez[1] + dr[3] * ez[3])
  c = u^2 - beta2 * (dr[1]^2 + dr[3]^2 + kp.y_source^2)
  t = c / (b + sqrt(b^2 - c / kp.gamma2))

  rs = fe.r + t * ez + SVector(0.0, kp.y_source, 0.0)
  L_vec = fk.r - rs
  L = sqrt(L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2)
  theta_L = atan(L_vec[1], L_vec[3])
  k = with_fields(k; floor_s = FloorPosition(rs, fe.theta), L_vec, L,
                  dL = -(Lke + t) - L, theta_L,
                  theta_sl = modulo2(fe.theta - theta_L, pi / 2),
                  theta_lk = modulo2(theta_L - (spline_d1(ek.spline,
      kp.s_chord_kick) + ek.theta_chord), pi / 2),
                  s_chord_source = t)
  return k, t
end

"""
    source_at(k, s, geom, kp) -> (k, r)

Bmad's round-off branches: `k` from `ddz_calc_csr` at chord position
`s`, with `s_chord_source` set to the residual `r` there (Bmad returns
the ddz value, not a position; csr_and_space_charge_mod.f90:952).

# Arguments
- `k`, `geom`, `kp`: as in `ddz_calc_csr`.
- `s`: chord position [m] in element `k.ix_ele_source` (callers
  pass 0 or its `L_chord`).

# Returns
- `(k, r)`: `k` with `s_chord_source = r`, and the residual `r` [m]
  of `ddz_calc_csr` at `s`.
"""
@inline function source_at(k::CSRKick1, s, geom::CSRGeometry,
                           kp::CSRKickPoint)
  k, r = ddz_calc_csr(k, s, geom, kp)
  return with_fields(k; s_chord_source = r), r
end

"""
    root_source(k, s_root, s_last, geom, kp) -> (k, s_source)

Source at the root `s_root` of `ddz_calc_csr` in element
`k.ix_ele_source`. Bmad leaves `k` describing zbrent's last evaluation
`s_last`, not the returned root. These differ by up to the tolerance,
or completely when the root is an endpoint with ddz exactly 0 (zbrent
then returns without evaluating there). CSR.jl evaluates at the root
unless compat (docs, "Physics decisions", V7).

# Arguments
- `k`: work struct (`CSRKick1`) with the source element
  `k.ix_ele_source` and the lag `k.dz_particles` [m].
- `s_root`: root [m] along the element's chord, from `zbrent`
  (`s_source_calc`).
- `s_last`: zbrent's last evaluation [m] (`zbrent`'s `x_last`).
- `geom`, `kp`: as in `ddz_calc_csr`.

# Returns
- `(k, s_source)`: `k` describing the source with
  `s_chord_source = s_source`, and the root [m] in the number type
  promoted from `s_root` and the separation.
"""
@inline function root_source(k::CSRKick1, s_root, s_last,
                             geom::CSRGeometry, kp::CSRKickPoint)
  k, ddz_root = ddz_calc_csr(k, s_root, geom, kp)
  # Mixed precision: zbrent's root can have another type than ddz
  # (e.g. a Float64 root from the 0.0 end of the bracket for a
  # BigFloat geometry). The root is returned in the promoted type,
  # and k is evaluated again at it when the types differ.
  s_source = convert(promote_type(typeof(s_root), typeof(ddz_root)),
                     s_root)
  if geom.bmad_compat || !(s_source isa typeof(s_root))
    k, _ = ddz_calc_csr(k,
                        (geom.bmad_compat ? s_last : s_root) +
                        (s_source - s_root), geom, kp)
  end
  return with_fields(k; s_chord_source = s_source), s_source
end
