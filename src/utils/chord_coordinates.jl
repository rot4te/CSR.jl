# Conversions between reference s and the centroid chord coordinate.

"""
    s_ref_to_s_chord(s_ref, e::CSRElementInfo)

Distance along the centroid chord corresponding to reference position `s_ref` from the element entrance.
Literal port of Bmad's `s_ref_to_s_chord`; see `planning/PLAN.md` (open question V3) about the bend branch.
"""
function s_ref_to_s_chord(s_ref, e::CSRElementInfo)
  if abs(e.g) > 1e-5
    rho = 1 / e.g
    dtheta = e.floor0.theta - e.ref_floor0.theta
    dr = e.ref_floor0.r - e.floor0.r
    t = e.ref_floor0.theta + pi / 2
    x = dr[1] * cos(t) + dr[3] * sin(t)
    return abs(rho) * atan(s_ref * cos(dtheta), abs(rho + x + s_ref * sin(dtheta)))
  else
    return s_ref * e.L_chord / e.L
  end
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
