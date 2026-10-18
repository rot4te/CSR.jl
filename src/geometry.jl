# Centroid geometry for a beamline and the per-kick constants (Bmad csr_ele_info_struct, csr_struct).

"""
    CSRGeometry(L, g, centroid; ref_start = FloorPosition(), bmad_compat = false)

Build the centroid geometry for elements of length `L[i]` and reference curvature `g[i]`.

`centroid` is an `(n+1) × 6` matrix of centroid phase-space coordinates `[x px y py z pz]`: row 1 at the
start of the lattice, row `i+1` at the exit of element `i` (Bmad's `centroid(0:n)`).

Dynamic-descriptor TPS arguments must be in GTPSA's current Descriptor.
"""
function CSRGeometry(L::AbstractVector, g::AbstractVector, centroid::AbstractMatrix;
                     ref_start::FloorPosition = FloorPosition(), bmad_compat::Bool = false)
  check_descriptors("CSRGeometry"; L, g, centroid, ref_start)
  return CSRGeometry(L, g, planar_survey(L, g; start = ref_start), centroid; ref_start, bmad_compat)
end

"""
    CSRGeometry(L, g, ref_exit, centroid; ref_start = FloorPosition(), bmad_compat = false)

As above, with the reference floor positions `ref_exit[i]` at the exit of each element given
(e.g. for lattices with patches; see `advance_floor`, `advance_floor_patch`).
"""
function CSRGeometry(L::AbstractVector, g::AbstractVector, ref_exit::AbstractVector{<:FloorPosition},
                     centroid::AbstractMatrix; ref_start::FloorPosition = FloorPosition(), bmad_compat::Bool = false)
  check_descriptors("CSRGeometry"; L, g, ref_exit, centroid, ref_start)
  n = length(L)
  length(g) == n && length(ref_exit) == n || error("CSRGeometry: L, g and ref_exit must have the same length")
  size(centroid) == (n + 1, 6) || error("CSRGeometry: centroid must be an (n+1) × 6 matrix with n = length(L)")
  T = float(promote_type(eltype(L), eltype(g), mapreduce(number_type, promote_type, ref_exit; init = Float64),
                         eltype(centroid), number_type(ref_start)))

  cfloor(ref, row) = centroid_floor(ref, centroid[row, 1], centroid[row, 2], centroid[row, 6])
  start = cfloor(ref_start, 1)

  ele = Vector{CSRElementInfo{T}}(undef, n)
  s = zero(T)
  for i in 1:n
    ref_floor0 = i == 1 ? ref_start : ref_exit[i-1]
    ref_floor1 = ref_exit[i]
    floor0 = i == 1 ? start : ele[i-1].floor1
    floor1 = cfloor(ref_floor1, i + 1)

    dr = floor1.r - floor0.r
    if primal(dr[1]) == 0 && primal(dr[3]) == 0
      # Zero-length chord (e.g. a marker): TPS atan/sqrt abort at 0 and Dual derivatives are not finite there. The
      # chord is below the 1e-8 threshold below, so its primal direction is enough.
      theta_chord = convert(T, atan(primal(dr[1]), primal(dr[3])))
      L_chord = zero(T)
    else
      theta_chord = atan(dr[1], dr[3])
      L_chord = sqrt(dr[1]^2 + dr[3]^2)
    end
    spline = zero(SVector{4,T})
    dL_s = zero(T)

    if abs(primal(L_chord)) >= 1e-8   # Bmad's threshold for a zero-length chord
      # An element with a negative step (e.g. off-center beam in a patch with only an x-pitch) can have
      # floor%theta and theta_chord 180° apart. Restrict theta0 and theta1 to [-π/2, π/2).
      theta0 = modulo2(floor0.theta - theta_chord, pi / 2)
      theta1 = modulo2(floor1.theta - theta_chord, pi / 2)
      parallel0 = abs(modulo2(primal(floor0.theta - theta_chord), pi)) < pi / 2
      parallel1 = abs(modulo2(primal(floor1.theta - theta_chord), pi)) < pi / 2
      parallel0 == parallel1 || error("CSRGeometry: inconsistent centroid direction in element $i")
      parallel0 || (L_chord = -L_chord)
      spline = chord_spline(L_chord, theta0, theta1)
      dL_s = path_excess(0.0, L_chord, spline; bmad_compat)
    end

    ele[i] = CSRElementInfo{T}(L[i], g[i], s, ref_floor0, ref_floor1, floor0, floor1, theta_chord, L_chord, spline, dL_s)
    s += L[i]
  end
  return CSRGeometry{T}(start, ele, bmad_compat)
end

"""
    CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source = 0.0, kick_factor = 1.0)

Kick point at reference position `s_kick` from the entrance of element `ix_ele_kick` (chord coordinate from
`s_ref_to_s_chord`, floor position on the centroid spline), for a reference particle with Lorentz factor `gamma`.
`y_source` is the height of the source particle (image charges; 0 in free space), `kick_factor` multiplies every
kick computed with this point (half steps, image sign and weight).
"""
function CSRKickPoint(geom::CSRGeometry, ix_ele_kick, s_kick, gamma; y_source = 0.0, kick_factor = 1.0)
  check_descriptors("CSRKickPoint"; geom, s_kick, gamma, y_source, kick_factor)
  e = geom.ele[ix_ele_kick]
  s_chord_kick = s_ref_to_s_chord(s_kick, e; geom.bmad_compat)
  x = spline_val(e.spline, s_chord_kick)
  c, s = cos(e.theta_chord), sin(e.theta_chord)
  r = SVector(x * c + s_chord_kick * s, 0.0, -x * s + s_chord_kick * c) + e.floor0.r
  theta = e.theta_chord + spline_d1(e.spline, s_chord_kick)
  beta = sqrt(1 - 1 / gamma^2)
  return CSRKickPoint(ix_ele_kick, s_kick, s_chord_kick, FloorPosition(r, theta), gamma, gamma^2, beta, y_source, kick_factor)
end

"""
    primal(f::FloorPosition), primal(e::CSRElementInfo), primal(geom::CSRGeometry), primal(kp::CSRKickPoint)

Copy with the primal value of every number (`Float64` from `Dual`s). A `CSRGeometry{Float64}` is returned as is.
"""
primal(f::FloorPosition) = FloorPosition(primal.(f.r), primal(f.theta))
primal(e::CSRElementInfo) =
  CSRElementInfo(primal(e.L), primal(e.g), primal(e.s_start), primal(e.ref_floor0), primal(e.ref_floor1), primal(e.floor0),
                 primal(e.floor1), primal(e.theta_chord), primal(e.L_chord), primal.(e.spline), primal(e.dL_s))
primal(geom::CSRGeometry{Float64}) = geom
primal(geom::CSRGeometry) = CSRGeometry(primal(geom.start), map(primal, geom.ele), geom.bmad_compat)
primal(kp::CSRKickPoint) =
  CSRKickPoint(kp.ix_ele_kick, primal(kp.s_kick), primal(kp.s_chord_kick), primal(kp.floor_k), primal(kp.gamma),
               primal(kp.gamma2), primal(kp.beta), primal(kp.y_source), primal(kp.kick_factor))
