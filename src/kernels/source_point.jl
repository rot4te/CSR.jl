# Retarded source point for a lag. Port of Bmad's s_source_calc and ddz_calc_csr; runs inside the table kernels.
# Match and floor_shift elements are not supported (no Beamlines equivalent).

"""
    ddz_calc_csr(k, s_chord_source, geom, kp, dmid = nothing) -> (k, ddz)

Separation (at constant time) between the kicked particle and a source at chord position `s_chord_source`
in element `k.ix_ele_source`, minus the target `k.dz_particles`. Returns `k` with the source geometry fields and
`s_chord_source` set, and the separation. `dmid` (if not `nothing`) is added to the path-length excess of the
elements strictly between the source and kick elements (`add_mid`; used by `csr_kick_param_adjoint`).
"""
@inline function ddz_calc_csr(k::CSRKick1, s_chord_source, geom::CSRGeometry, kp::CSRKickPoint, dmid = nothing)
  es = geom.ele[k.ix_ele_source]
  bc = geom.bmad_compat
  x = spline_val(es.spline, s_chord_source)
  c, s = cos(es.theta_chord), sin(es.theta_chord)
  rs = SVector(x * c + s_chord_source * s, kp.y_source, -x * s + s_chord_source * c) + es.floor0.r

  L_vec = kp.floor_k.r - rs
  L = sqrt(L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2)
  theta_L = atan(L_vec[1], L_vec[3])

  s0 = s_chord_source
  s1 = kp.s_chord_kick

  if k.ix_ele_source == kp.ix_ele_kick
    ds = s1 - s0
    # dtheta_L = angle of the L line in the centroid chord frame
    sp = es.spline
    dtheta_L = sp[2] + sp[3] * (2s0 + ds) + sp[4] * (3s0^2 + 3s0 * ds + ds^2)
    dL = path_excess(s0, s1, sp, dtheta_L; bmad_compat = bc)   # = Ls - L
    # Ls is negative if the source point is ahead of the kick point (ds < 0), but L is always positive.
    primal(ds) < 0 && (dL += 2ds)
    theta_sl = spline_d1(sp, s0) - dtheta_L
    theta_lk = dtheta_L - spline_d1(sp, s1)
    theta_s = theta_sl + theta_L
    # For a source ahead of the kick point L_vec points backwards, and theta_sl + theta_L is the source direction
    # + π (Bmad, planning/PLAN.md V8). Only the image-charge kick uses it.
    primal(ds) < 0 && !bc && (theta_s = modulo2(theta_s + pi, pi))
  elseif k.ix_ele_source > kp.ix_ele_kick
    # Source downstream of the kick element (image charges, source ahead of the kick point; not in Bmad).
    # Ls = -(path length from the kick point to the source), so dL = Ls - L = -(excess + 2L) in the plane.
    ek = geom.ele[kp.ix_ele_kick]
    excess = path_excess(s1, ek.L_chord, ek.spline, modulo2(theta_L - ek.theta_chord, pi / 2); bmad_compat = bc)
    excess = add_mid(middle_excess(excess, geom, kp.ix_ele_kick+1:k.ix_ele_source-1, theta_L), dmid)
    excess += path_excess(0.0, s0, es.spline, modulo2(theta_L - es.theta_chord, pi / 2); bmad_compat = bc)
    dL = -(excess + 2sqrt(L_vec[1]^2 + L_vec[3]^2))
    theta_s = spline_d1(es.spline, s0) + es.theta_chord
    theta_sl = modulo2(theta_s - theta_L, pi / 2)
    theta_lk = modulo2(theta_L - (spline_d1(ek.spline, s1) + ek.theta_chord), pi / 2)
  else
    # Where the centroid takes a backstep, theta_chord is anti-parallel to the other angles, so angles
    # are folded into [-π/2, π/2).
    dL = path_excess(s0, es.L_chord, es.spline, modulo2(theta_L - es.theta_chord, pi / 2); bmad_compat = bc)
    dL = add_mid(middle_excess(dL, geom, k.ix_ele_source+1:kp.ix_ele_kick-1, theta_L), dmid)
    ce = geom.ele[kp.ix_ele_kick]
    dL += path_excess(0.0, s1, ce.spline, modulo2(theta_L - ce.theta_chord, pi / 2); bmad_compat = bc)
    theta_sl = modulo2((spline_d1(es.spline, s0) + es.theta_chord) - theta_L, pi / 2)
    theta_lk = modulo2(theta_L - (spline_d1(ce.spline, s1) + ce.theta_chord), pi / 2)
    theta_s = theta_sl + theta_L
  end

  # The dL calculation above neglects y_source, so correct for it.
  primal(kp.y_source) != 0 && (dL -= L - sqrt(L_vec[1]^2 + L_vec[3]^2))
  k = with_fields(k; L_vec, L, theta_L, theta_sl, theta_lk, dL, s_chord_source, floor_s = FloorPosition(rs, theta_s))
  return k, L / (2kp.gamma2) + dL - k.dz_particles
end

"""
    middle_excess(acc, geom, r, theta_L)

`acc` plus, element by element, the path-length excess `path_excess(0, L_chord, spline, modulo2(theta_L -
theta_chord, π/2))` of the elements `r` (strictly between source and kick element in `ddz_calc_csr`).
"""
@inline function middle_excess(acc, geom::CSRGeometry, r, theta_L)
  for i in r
    ce = geom.ele[i]
    acc += path_excess(0.0, ce.L_chord, ce.spline, modulo2(theta_L - ce.theta_chord, pi / 2); bmad_compat = geom.bmad_compat)
  end
  return acc
end

"""
    add_mid(x, dmid)

`x + dmid`, or `x` for `dmid = nothing`: the offset of the middle-element path-length sums (`ddz_calc_csr`).
"""
@inline add_mid(x, ::Nothing) = x
@inline add_mid(x, dmid) = x + dmid

"""
    mid_root(dmid), mid_eval(dmid)

`dmid` for the root's implicit derivative and for the evaluated source in `root_source`: a pair `(root, eval)` is
split, anything else used for both (the other source branches use `mid_eval`).
"""
@inline mid_root(dmid) = dmid
@inline mid_root(dmid::Tuple) = dmid[1]
@inline mid_eval(dmid) = dmid
@inline mid_eval(dmid::Tuple) = dmid[2]

"""
    SourceOutcome

Where `s_source_walk` placed a source, from primal decisions only, so that `s_source_eval` can evaluate the same
source in another number type without searching. `code`: `SRC_UPSTREAM`/`SRC_DOWNSTREAM` (infinite drifts),
`SRC_ROOT` (root of `ddz_calc_csr` in element `ix`, zbrent's root `s_root` and last evaluation `s_last`),
`SRC_END`/`SRC_START` (Bmad's round-off branches), `SRC_BEYOND` (source beyond the kick element, ignored).
"""
struct SourceOutcome
  code::Int
  ix::Int
  s_root::Float64
  s_last::Float64
end

const SRC_UPSTREAM = 1
const SRC_DOWNSTREAM = 2
const SRC_ROOT = 3
const SRC_END = 4
const SRC_START = 5
const SRC_BEYOND = 6

"""
    s_source_calc(k, geom, kp) -> (k, s_chord_source)

Find the source point for target separation `k.dz_particles`, starting the search in element
`k.ix_ele_source` and walking element by element. Returns `k` describing the source and the source position along
its element's chord. A source upstream of the lattice is placed on an infinite drift (`k.ix_ele_source = 0`,
negative s).

A source ahead of the kick point (`dz < 0`, image charges) beyond the kick element is searched for downstream,
and past the lattice end on an infinite drift (`k.ix_ele_source = length(geom.ele) + 1`, returning the distance
past the end). Bmad does not do this; `geom.bmad_compat` restores Bmad's behaviour.
"""
@inline function s_source_calc(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint)
  k, s, _ = s_source_walk(k, geom, kp)
  return k, s
end

"""
    PrimalSearch(geom, kp)

Primal copies (`primal`) of a geometry and kick point, in which `s_source_calc(k, geom, kp, search)` searches the
source before evaluating it once in the number types of `geom` and `kp` (e.g. `Dual`s): the search's evaluations
are not differentiated. Built by `primal_search`.
"""
struct PrimalSearch{G<:CSRGeometry,K<:CSRKickPoint}
  geom::G
  kp::K
end

"""
    primal_search(geom, kp, T) -> PrimalSearch or nothing

`PrimalSearch(primal(geom), primal(kp))` for tables in number type `T <: Dual`, else `nothing` (search and evaluate
in the same number type).
"""
primal_search(geom::CSRGeometry, kp::CSRKickPoint, ::Type{T}) where {T} =
  T <: Dual ? PrimalSearch(primal(geom), primal(kp)) : nothing

"""
    s_source_calc(k, geom, kp, search) -> (k, s_chord_source)

`s_source_calc(k, geom, kp)` for `search === nothing`. With a `PrimalSearch`, the source is searched in its primal
geometry (`s_source_walk`, from element `k.ix_ele_source`) and evaluated in `geom` and `kp` (`s_source_eval`); same
result, as the search decides on primal values only.
"""
@inline s_source_calc(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint, ::Nothing) = s_source_calc(k, geom, kp)

@inline function s_source_calc(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint, search::PrimalSearch)
  k0 = CSRKick1(; ix_ele_source = k.ix_ele_source, dz_particles = primal(k.dz_particles))
  _, _, o = s_source_walk(k0, search.geom, search.kp)
  return s_source_eval(o, k, geom, kp)
end

"""
    upstream_drift_source(k, geom, kp, dmid) -> (k, ds_source)

Source on the infinite drift upstream of the lattice (closed-form root; `ds_source < 0` from the lattice start).
"""
function upstream_drift_source(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint, dmid)
  dz = k.dz_particles
  beta2 = kp.beta^2
  fk = kp.floor_k
  f0 = geom.start
  L0 = sqrt((fk.r[1] - f0.r[1])^2 + (fk.r[3] - f0.r[3])^2 + kp.y_source^2)
  # Lz is the z-component from the lattice start to the kick point
  Lz = (fk.r[1] - f0.r[1]) * sin(f0.theta) + (fk.r[3] - f0.r[3]) * cos(f0.theta)
  # Lsz0 is Ls from the lattice start to the kick point
  ek = geom.ele[kp.ix_ele_kick]
  Lsz0 = path_excess(0.0, kp.s_chord_kick, ek.spline; bmad_compat = geom.bmad_compat) + kp.s_chord_kick
  for i in 1:kp.ix_ele_kick-1
    Lsz0 += geom.ele[i].dL_s + geom.ele[i].L_chord
  end
  Lsz0 = add_mid(Lsz0, dmid)
  a = 1 / kp.gamma2
  b = 2 * (Lsz0 - dz - beta2 * Lz)
  c = (Lsz0 - dz)^2 - beta2 * L0^2
  ds_source = -(-b + sqrt(b^2 - 4a * c)) / (2a)

  rs = SVector(f0.r[1] + ds_source * sin(f0.theta), kp.y_source, f0.r[3] + ds_source * cos(f0.theta))
  L_vec = fk.r - rs
  L = sqrt(L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2)
  theta_L = atan(L_vec[1], L_vec[3])
  k = with_fields(k; floor_s = FloorPosition(rs, f0.theta), L_vec, L,
                  dL = Lsz0 - ds_source - L,   # ds_source is negative
                  theta_L, theta_sl = f0.theta - theta_L,
                  theta_lk = theta_L - (spline_d1(ek.spline, kp.s_chord_kick) + ek.theta_chord),
                  s_chord_source = ds_source)
  return k, ds_source
end

"""
    downstream_drift_source(k, geom, kp, dmid) -> (k, t)

Source on the infinite drift past the lattice end along the centroid exit direction (image charges; not in Bmad),
`t ≥ 0` the distance past the end. Exact retardation `Ls - βL = dz` with `Ls = -(Lke + t)`.
"""
function downstream_drift_source(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint, dmid)
  dz = k.dz_particles
  beta2 = kp.beta^2
  fk = kp.floor_k
  fe = geom.ele[end].floor1
  ek = geom.ele[kp.ix_ele_kick]
  # Lke is Ls from the kick point to the lattice end
  Lke = path_excess(kp.s_chord_kick, ek.L_chord, ek.spline; bmad_compat = geom.bmad_compat) + ek.L_chord - kp.s_chord_kick
  for i in kp.ix_ele_kick+1:length(geom.ele)
    Lke += geom.ele[i].dL_s + geom.ele[i].L_chord
  end
  Lke = add_mid(Lke, dmid)
  ez = z_axis(fe.theta)
  dr = fk.r - fe.r
  # (u - t)² = β² L² with u = -(Lke + dz), L² = L0² - 2t Lz + t²: t²/γ² - 2b t + c = 0, smaller root.
  u = -(Lke + dz)
  b = u - beta2 * (dr[1] * ez[1] + dr[3] * ez[3])
  c = u^2 - beta2 * (dr[1]^2 + dr[3]^2 + kp.y_source^2)
  t = c / (b + sqrt(b^2 - c / kp.gamma2))

  rs = fe.r + t * ez + SVector(0.0, kp.y_source, 0.0)
  L_vec = fk.r - rs
  L = sqrt(L_vec[1]^2 + L_vec[2]^2 + L_vec[3]^2)
  theta_L = atan(L_vec[1], L_vec[3])
  k = with_fields(k; floor_s = FloorPosition(rs, fe.theta), L_vec, L, dL = -(Lke + t) - L, theta_L,
                  theta_sl = modulo2(fe.theta - theta_L, pi / 2),
                  theta_lk = modulo2(theta_L - (spline_d1(ek.spline, kp.s_chord_kick) + ek.theta_chord), pi / 2),
                  s_chord_source = t)
  return k, t
end

"""
    source_at(k, s, geom, kp, dmid) -> (k, r)

Bmad's round-off branches: `k` from `ddz_calc_csr` at chord position `s`, with `s_chord_source` set to the residual
`r` there (Bmad returns the ddz value, not a position; csr_and_space_charge_mod.f90:952).
"""
@inline function source_at(k::CSRKick1, s, geom::CSRGeometry, kp::CSRKickPoint, dmid)
  k, r = ddz_calc_csr(k, s, geom, kp, dmid)
  return with_fields(k; s_chord_source = r), r
end

"""
    root_source(k, s_root, s_last, geom, kp, dmid) -> (k, s_source)

Source at the root `s_root` of `ddz_calc_csr` in element `k.ix_ele_source` (found from primal values), with the
root's derivatives with respect to the parameters attached by implicit differentiation (`root_tangent`), so that
zbrent's iterations are not differentiated. Bmad leaves `k` describing zbrent's last evaluation `s_last`, not the
returned root. These differ by up to the tolerance, or completely when the root is an endpoint with ddz exactly 0
(zbrent then returns without evaluating there). CSR.jl evaluates at the root unless compat (planning/PLAN.md, V7).
With derivatives, `k` is evaluated again with the root's derivatives attached.
"""
@inline function root_source(k::CSRKick1, s_root, s_last, geom::CSRGeometry, kp::CSRKickPoint, dmid)
  kk = k
  dr = mid_root(dmid)
  k, ddz_root = ddz_calc_csr(k, s_root, geom, kp, dr)
  s_source = root_tangent(s -> last(ddz_calc_csr(csr_kick1_like(kk, s), s, geom, kp, dr)), s_root, ddz_root)
  if geom.bmad_compat || !(s_source isa typeof(s_root))
    k, _ = ddz_calc_csr(k, (geom.bmad_compat ? s_last : s_root) + (s_source - s_root), geom, kp, mid_eval(dmid))
  end
  return with_fields(k; s_chord_source = s_source), s_source
end

"""
    s_source_walk(k, geom, kp) -> (k, s_chord_source, outcome::SourceOutcome)

`s_source_calc`, also returning where the source was placed (`SourceOutcome`). All branch decisions use primal
values.
"""
function s_source_walk(k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint)
  dz = k.dz_particles
  last_step = 0

  while true
    # Source upstream of the lattice: infinite drift, closed-form root.
    if k.ix_ele_source == 0
      k1, ds_source = upstream_drift_source(k, geom, kp, nothing)
      # A search warm-started here (another lag) may belong inside the lattice: go back and walk.
      if primal(ds_source) > 0 && last_step == 0
        k = with_fields(k; ix_ele_source = 1)
        continue
      end
      return k1, ds_source, SourceOutcome(SRC_UPSTREAM, 0, 0.0, 0.0)
    end

    # Source downstream of the lattice (image charges; not in Bmad).
    if k.ix_ele_source > length(geom.ele)
      k1, t = downstream_drift_source(k, geom, kp, nothing)
      # A search warm-started here (previous lag) may belong inside the lattice: go back and walk.
      if primal(t) < 0 && last_step == 0
        k = with_fields(k; ix_ele_source = length(geom.ele))
        continue
      end
      return k1, t, SourceOutcome(SRC_DOWNSTREAM, k.ix_ele_source, 0.0, 0.0)
    end

    ix = k.ix_ele_source
    es = geom.ele[ix]
    # Check whether the source point is within this element. Generally ddz decreases with increasing s,
    # but this may not hold for patch elements.
    # Branch decisions on primal values: ForwardDiff orders Duals with equal values by their partials.
    k, ddz0 = ddz_calc_csr(k, 0.0, geom, kp)
    k, ddz1 = ddz_calc_csr(k, es.L_chord, geom, kp)
    ddz0, ddz1 = primal(ddz0), primal(ddz1)

    if last_step == -1 && ddz1 > 0   # Round-off error makes ddz1 positive
      k, r = source_at(k, es.L_chord, geom, kp, nothing)
      return k, r, SourceOutcome(SRC_END, ix, 0.0, 0.0)
    end
    if last_step == 1 && ddz0 < 0    # Round-off error makes ddz0 negative
      k, r = source_at(k, 0.0, geom, kp, nothing)
      return k, r, SourceOutcome(SRC_START, ix, 0.0, 0.0)
    end

    if ddz0 < 0 && ddz1 < 0
      last_step == 1 && break
      last_step = -1
      k = with_fields(k; ix_ele_source = ix - 1)
      continue
    end

    if ddz0 > 0 && ddz1 > 0
      # Source beyond this element. Beyond the kick element it is ahead of the kick point, which needs dz < 0
      # (image charges). Bmad ignores it there (k is left describing the source at the end of the kick element);
      # CSR.jl continues downstream (planning/PLAN.md, V8).
      if ix == kp.ix_ele_kick && (geom.bmad_compat || primal(dz) >= 0)
        return with_fields(k; s_chord_source = 0.0), zero(number_type(k)), SourceOutcome(SRC_BEYOND, ix, 0.0, 0.0)
      end
      last_step == -1 && break
      last_step = 1
      k = with_fields(k; ix_ele_source = ix + 1)
      continue
    end

    # Root is bracketed. The search uses primal values (root_source attaches the root's derivatives).
    kk = k
    s_root, ok, s_last = zbrent(s -> primal(last(ddz_calc_csr(kk, s, geom, kp))), 0.0, primal(es.L_chord), 1e-12, 1e-8)
    ok || error("s_source_calc: root finder failed in element $ix")
    k, s_source = root_source(k, s_root, s_last, geom, kp, nothing)
    return k, s_source, SourceOutcome(SRC_ROOT, ix, s_root, s_last)
  end
  error("s_source_calc: CSR source-point search failed (element walk did not converge)")
end

"""
    s_source_eval(o, k, geom, kp, dmid = nothing) -> (k, s_chord_source)

The source that `s_source_walk` placed (`o`), evaluated without searching for lag `k.dz_particles` in the number
types of `k`, `geom` and `kp` (e.g. local `Dual`s in `csr_kick_param_adjoint`, with the search done in `Float64`).
`dmid` as in `ddz_calc_csr`, or a pair (see `mid_root`).
"""
function s_source_eval(o::SourceOutcome, k::CSRKick1, geom::CSRGeometry, kp::CSRKickPoint, dmid = nothing)
  k = with_fields(k; ix_ele_source = o.ix)
  o.code == SRC_ROOT && return root_source(k, o.s_root, o.s_last, geom, kp, dmid)
  de = mid_eval(dmid)
  o.code == SRC_UPSTREAM && return upstream_drift_source(k, geom, kp, de)
  o.code == SRC_DOWNSTREAM && return downstream_drift_source(k, geom, kp, de)
  L_chord = geom.ele[o.ix].L_chord
  o.code == SRC_END && return source_at(k, L_chord, geom, kp, de)
  o.code == SRC_START && return source_at(k, 0.0, geom, kp, de)
  k, _ = ddz_calc_csr(k, L_chord, geom, kp, de)   # SRC_BEYOND
  return with_fields(k; s_chord_source = 0.0), zero(number_type(k))
end

"""
    csr_kick1_like(k, s)

Zeroed work struct for the number types of `k` and `s`, with `k`'s source element and lag (for evaluating
`ddz_calc_csr` at an `s` of another number type, e.g. a nested `Dual` in `root_tangent`).
"""
csr_kick1_like(k::CSRKick1, s) =
  CSRKick1{promote_type(number_type(k), typeof(s))}(; ix_ele_source = k.ix_ele_source, dz_particles = k.dz_particles)
