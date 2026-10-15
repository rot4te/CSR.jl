# Conversions between reference s and the centroid chord coordinate.

"""
    s_ref_to_s_chord(s_ref, e::CSRElementInfo; bmad_compat = false)

Distance along the centroid chord corresponding to reference position `s_ref` from the element entrance.

Straight elements (`|g| ≤ 1e-5`): `s_ref * L_chord / L`, as Bmad. Bends: the chord coordinate where the centroid
spline crosses the reference normal plane at `s_ref` (`chord_normal_offset` = 0; Newton steps on primal values, then
`root_tangent`); `0` and `L_chord` at the element ends. `bmad_compat = true`: Bmad's formula
`|ρ| atan(s_ref cos dθ, |ρ + x + s_ref sin dθ|)`, which on an on-axis bend gives `ρ atan(s/ρ)` instead of the chord
coordinate (`planning/PLAN.md`, V3).
"""
function s_ref_to_s_chord(s_ref, e::CSRElementInfo; bmad_compat::Bool = false)
  abs(primal(e.g)) > 1e-5 || return s_ref * e.L_chord / e.L
  if bmad_compat || abs(primal(e.L_chord)) < 1e-8   # Zero-length chord: no spline (CSRGeometry), Bmad's formula
    rho = 1 / e.g
    dtheta = e.floor0.theta - e.ref_floor0.theta
    dr = e.ref_floor0.r - e.floor0.r
    t = e.ref_floor0.theta + pi / 2
    x = dr[1] * cos(t) + dr[3] * sin(t)
    return abs(rho) * atan(s_ref * cos(dtheta), abs(rho + x + s_ref * sin(dtheta)))
  end
  ref = advance_floor(e.ref_floor0, s_ref, e.g)
  e0, ref0 = primal(e), primal(ref)
  slope(u) = dot3(z_axis(e0.theta_chord) + spline_d1(e0.spline, u) * x_axis(e0.theta_chord), z_axis(ref0.theta))
  s0 = primal(s_ref)
  if s0 == 0
    u = 0.0
  elseif s0 == e0.L
    u = e0.L_chord
  else
    u = s0 * e0.L_chord / e0.L
    converged = false
    for _ in 1:20
      du = chord_normal_offset(u, e0, ref0) / slope(u)
      u -= du
      converged = abs(du) <= 1e-12 * abs(e0.L_chord)   # Quadratic convergence: the error left is ~du²/L
      converged && break
    end
    converged || error("s_ref_to_s_chord: Newton iteration did not converge (s_ref = $s0)")
  end
  f = u -> chord_normal_offset(u, e, ref)
  return root_tangent(f, u, f(u), slope(u))
end

"""
    chord_normal_offset(u, e::CSRElementInfo, ref::FloorPosition)

Distance along the reference direction of motion at `ref` from `ref` to the centroid spline point at chord coordinate
`u` (zero where the spline crosses the reference normal plane).
"""
function chord_normal_offset(u, e::CSRElementInfo, ref::FloorPosition)
  x = spline_val(e.spline, u)
  dr = (e.floor0.r - ref.r) + u * z_axis(e.theta_chord) + x * x_axis(e.theta_chord)
  return dot3(dr, z_axis(ref.theta))
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
