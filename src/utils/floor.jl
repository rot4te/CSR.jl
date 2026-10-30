# Planar floor coordinates: survey, patches, centroid offset.
# Floor conventions (Bmad): r = (X, Y, Z). In the horizontal plane, an orientation angle θ has direction of motion
# ẑ = (sin θ, 0, cos θ) and local x-axis x̂ = (cos θ, 0, -sin θ). A bend with g > 0 curves towards -x̂ and θ
# decreases by g*L.

"""
    x_axis(theta)

Local horizontal x-axis `(cos θ, 0, -sin θ)` of a floor orientation `theta` (Bmad convention).
"""
x_axis(theta) = SVector(cos(theta), 0.0, -sin(theta))

"""
    z_axis(theta)

Direction of motion `(sin θ, 0, cos θ)` of a floor orientation `theta` (Bmad convention).
"""
z_axis(theta) = SVector(sin(theta), 0.0, cos(theta))

"""
    advance_floor(f, L, g) -> FloorPosition

Reference floor position after an element of length `L` and reference curvature `g`, entered at `f`.
"""
function advance_floor(f::FloorPosition, L, g)
  angle = g * L
  if Beamlines.scalarize_all(angle) == 0
    # Expansion about angle 0 (values exact; derivatives w.r.t. g and L correct to second order)
    dx = -L * angle / 2
    dz = L * (1 - angle^2 / 6)
  else
    dx = -2 * sin(angle / 2)^2 / g   # (cos(angle) - 1) / g
    dz = sin(angle) / g
  end
  return FloorPosition(f.r + dx * x_axis(f.theta) + dz * z_axis(f.theta), f.theta - angle)
end

"""
    advance_floor_patch(f, dr, ez) -> FloorPosition

Reference floor position after a patch entered at `f`. `dr` is the exit origin and `ez` the exit z axis,
both in the entrance coordinates `(x, y, z)`. Projected on the horizontal plane (the `y` parts are dropped,
as for all CSR geometry).
"""
function advance_floor_patch(f::FloorPosition, dr, ez)
  r = f.r + dr[1] * x_axis(f.theta) + dr[3] * z_axis(f.theta)
  return FloorPosition(r, f.theta + atan(ez[1], ez[3]))
end

"""
    planar_survey(L, g; start = FloorPosition()) -> Vector{FloorPosition}

Reference floor positions at the exit of each element, for elements of length `L[i]` and
reference curvature `g[i]` (zero for straight elements). Horizontal plane only.

Dynamic-descriptor TPS arguments must be in GTPSA's current Descriptor.
"""
function planar_survey(L::AbstractVector, g::AbstractVector; start::FloorPosition = FloorPosition())
  check_descriptors("planar_survey"; L, g, start)
  length(L) == length(g) || error("planar_survey: L and g must have the same length")
  floors = Vector{FloorPosition{float(promote_type(eltype(L), eltype(g), number_type(start)))}}(undef, length(L))
  f = start
  for i in eachindex(L)
    f = advance_floor(f, L[i], g[i])
    floors[i] = f
  end
  return floors
end

"""
    centroid_floor(ref, x, px, pz) -> FloorPosition

Floor position of a centroid with horizontal offset `x` and momentum `px`, `pz` relative to the
reference floor position `ref` (the vertical coordinate is dropped, as in Bmad).
"""
function centroid_floor(ref::FloorPosition, x, px, pz)
  r = ref.r + x * x_axis(ref.theta)
  # The 1e-14 follows Bmad (avoids 0/0 at the beginning of an e_gun).
  return FloorPosition(SVector(r[1], 0.0, r[3]), ref.theta + asin(px / (1 + 1e-14 + pz)))
end
