# Conversions between reference s and the centroid chord coordinate.

"""
    s_ref_to_s_chord(s_ref, e::CSRElementInfo; bmad_compat = false)

Distance along the centroid chord corresponding to reference position
`s_ref` from the element entrance.

Straight elements (`|g| ≤ 1e-5`): `s_ref * L_chord / L`, as Bmad.
Bends: the chord coordinate where the centroid spline crosses the
reference normal plane at `s_ref` (`chord_normal_offset` = 0; Newton
steps); `0` and `L_chord` at the element ends. `bmad_compat = true`:
Bmad's formula `|ρ| atan(s_ref cos dθ, |ρ + x + s_ref sin dθ|)`,
which on an on-axis bend gives `ρ atan(s/ρ)` instead of the chord
coordinate (docs, "Physics decisions", V3).

# Arguments
- `s_ref`: reference position [m] from the entrance of the element
  (e.g. `s_kick` in `CSRKickPoint`).
- `e`: the element's centroid geometry, `geom.ele[i]` of a
  `CSRGeometry`.
- `bmad_compat`: `CSRKickPoint` passes `geom.bmad_compat`.

# Returns
- chord coordinate [m] in element `e` at `s_ref`.
"""
function s_ref_to_s_chord(s_ref, e::CSRElementInfo;
                          bmad_compat::Bool = false)
  abs(e.g) > 1e-5 || return s_ref * e.L_chord / e.L
  # Zero-length chord: no spline (CSRGeometry), Bmad's formula
  if bmad_compat || abs(e.L_chord) < 1e-8
    rho = 1 / e.g
    dtheta = e.floor0.theta - e.ref_floor0.theta
    dr = e.ref_floor0.r - e.floor0.r
    t = e.ref_floor0.theta + pi / 2
    x = dr[1] * cos(t) + dr[3] * sin(t)
    return abs(rho) * atan(s_ref * cos(dtheta),
                           abs(rho + x + s_ref * sin(dtheta)))
  end
  ref = advance_floor(e.ref_floor0, s_ref, e.g)
  slope(u) = dot3(z_axis(e.theta_chord) + spline_d1(e.spline,
      u) * x_axis(e.theta_chord), z_axis(ref.theta))
  if s_ref == 0
    u = 0.0
  elseif s_ref == e.L
    u = e.L_chord
  else
    u = s_ref * e.L_chord / e.L
    converged = false
    for _ in 1:20
      du = chord_normal_offset(u, e, ref) / slope(u)
      u -= du
      # Quadratic convergence: the error left is ~du²/L
      converged = abs(du) <= 1e-12 * abs(e.L_chord)
      converged && break
    end
    converged || error("s_ref_to_s_chord: Newton iteration did not converge (s_ref = $s_ref)")
  end
  # Return the root in the number type promoted with the offset's:
  # u is the literal 0.0 (Float64) at s_ref = 0, e.g. for a BigFloat
  # geometry, and the offset is Float64 for a Float32 geometry (the
  # 0.0 in z_axis and x_axis), whose u = e.L_chord at s_ref = L is a
  # Float32.
  f0 = chord_normal_offset(u, e, ref)
  return convert(promote_type(typeof(u), typeof(f0)), u)
end

"""
    chord_normal_offset(u, e::CSRElementInfo, ref::FloorPosition)

Distance along the reference direction of motion at `ref` from `ref`
to the centroid spline point at chord coordinate `u` (zero where the
spline crosses the reference normal plane).

# Arguments
- `u`: chord coordinate [m] in element `e`.
- `e`: the element's centroid geometry, `geom.ele[i]`.
- `ref`: reference floor position, e.g. `advance_floor(e.ref_floor0,
  s_ref, e.g)` at reference position `s_ref` in the element.

# Returns
- signed distance [m].
"""
function chord_normal_offset(u, e::CSRElementInfo, ref::FloorPosition)
  x = spline_val(e.spline, u)
  dr = (e.floor0.r - ref.r) + u * z_axis(e.theta_chord) + x *
      x_axis(e.theta_chord)
  return dot3(dr, z_axis(ref.theta))
end

"""Reference s of a source at `s_chord` in element `ix_ele_source` (0:
upstream drift, `n + 1`: downstream drift).

# Arguments
- `geom`: centroid geometry, from `CSRGeometry`; `n =
  length(geom.ele)`.
- `ix_ele_source`: source element, `k.ix_ele_source` of the
  `CSRKick1` from `s_source_calc`.
- `s_chord`: source position [m] from `s_source_calc`: along the
  element's chord, or on a drift the distance from the lattice start
  (negative) or past the lattice end.
"""
function source_s(geom::CSRGeometry, ix_ele_source, s_chord)
  ix_ele_source == 0 && return s_chord
  if ix_ele_source > length(geom.ele)
    e = geom.ele[end]
    return e.s_start + e.L + s_chord
  end
  return geom.ele[ix_ele_source].s_start + s_chord
end
