# Planar centroid geometry for the 1D CSR calculation.
# Port of the geometry parts of Bmad's csr_and_space_charge_mod.f90 (track1_bunch_csr setup,
# dspline_len, s_ref_to_s_chord).
#
# Floor conventions (Bmad): r = (X, Y, Z). In the horizontal plane, an orientation angle θ has
# direction of motion ẑ = (sin θ, 0, cos θ) and local x-axis x̂ = (cos θ, 0, -sin θ).
# A bend with g > 0 curves towards -x̂ and θ decreases by g*L.

"""
    FloorPosition(r, theta)

Position `r = (X, Y, Z)` [m] and horizontal orientation angle `theta` [rad] in floor coordinates.
"""
struct FloorPosition
  r::SVector{3,Float64}
  theta::Float64
end

FloorPosition() = FloorPosition(SVector(0.0, 0.0, 0.0), 0.0)

x_axis(theta) = SVector(cos(theta), 0.0, -sin(theta))
z_axis(theta) = SVector(sin(theta), 0.0, cos(theta))

"""
    advance_floor(f, L, g) -> FloorPosition

Reference floor position after an element of length `L` and reference curvature `g`, entered at `f`.
"""
function advance_floor(f::FloorPosition, L, g)
  angle = g * L
  if angle == 0
    dx = 0.0
    dz = float(L)
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
"""
function planar_survey(L::AbstractVector, g::AbstractVector; start::FloorPosition = FloorPosition())
  length(L) == length(g) || error("planar_survey: L and g must have the same length")
  floors = Vector{FloorPosition}(undef, length(L))
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

"""
    modulo2(x, amp)

`x` folded into the range `[-amp, amp)`. Same as Bmad's `modulo2`.
"""
function modulo2(x, amp)
  m = mod(x, 2amp)
  return m >= amp ? m - 2amp : m
end

#---------------------------------------------------------------------------------------------------
# Cubic spline x(s) = c0 + c1 s + c2 s^2 + c3 s^3 with origin s = 0 (Bmad spline_struct with x0 = 0).

spline_val(c, s) = ((c[4] * s + c[3]) * s + c[2]) * s + c[1]
spline_d1(c, s) = (3c[4] * s + 2c[3]) * s + c[2]
spline_d2(c, s) = 6c[4] * s + 2c[3]

"""
    chord_spline(L_chord, slope0, slope1) -> SVector{4}

Coefficients of the cubic with `x(0) = x(L_chord) = 0` and end slopes `slope0`, `slope1`.
Equivalent to Bmad's `create_a_spline([0, 0], [L_chord, 0], slope0, slope1)`.
"""
function chord_spline(L_chord, slope0, slope1)
  return SVector(0.0, slope0, (-2slope0 - slope1) / L_chord, (slope0 + slope1) / L_chord^2)
end

"""
    dspline_len(s0, s1, c, dtheta_ref = 0; bmad_compat = false)

Small-slope length difference between the spline curve and a reference line, `½∫ (x′ - dtheta_ref)² ds`
from `s0` to `s1`. The reference line is the spline axis rotated by `dtheta_ref`.

With `bmad_compat = true`, the s⁴ term uses Bmad's coefficient `9c₃²` instead of the integral's `9c₃²/5`
(see `planning/dspline_coeff.md`).
"""
function dspline_len(s0, s1, c, dtheta_ref = 0.0; bmad_compat::Bool = false)
  c1 = c[2] - dtheta_ref
  c2 = c[3]
  c3 = c[4]
  ds = s1 - s0
  c3_coef = bmad_compat ? 9c3^2 : 9c3^2 / 5
  return (ds / 2) * (
    c1^2 +
    (2s0 + ds) * 2c1 * c2 +
    (3s0^2 + 3s0 * ds + ds^2) * (6c1 * c3 + 4c2^2) / 3 +
    (4s0^3 + 6s0^2 * ds + 4s0 * ds^2 + ds^3) * 3c2 * c3 +
    (5s0^4 + 10s0^3 * ds + 10s0^2 * ds^2 + 5s0 * ds^3 + ds^4) * c3_coef
  )
end

const GL4_NODES = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
const GL4_WEIGHTS = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)

"""
    path_excess(s0, s1, c, dtheta_ref = 0; bmad_compat = false)

Length of the spline curve minus its projection onto a reference line rotated by `dtheta_ref` from the spline axis,
from `s0` to `s1` (signed: negative for `s1 < s0`). Since the projection of a path onto a line is
`∫ cos(θ - θ_ref) ds_arc`, this is `∫ (√(1 + x′²) - cos dθ - x′ sin dθ) ds`
`= ∫ x′² / (1 + √(1 + x′²)) ds + 2 sin²(dθ/2) Δs - sin(dθ) Δx`: exact in `dtheta_ref`, with the slope term by 4-point
Gauss–Legendre (exact for its `x′²/2` part; `x′` is the small slope relative to the element chord).

With `bmad_compat = true`, Bmad's small-angle form `dspline_len(...; bmad_compat = true)`. That form drops the
`dθ⁴` terms: on a circle of radius ρ it overestimates the excess over an arc Ls by `Ls⁵/(1920ρ⁴)`, which shifts image
sources at large heights (`planning/PLAN.md`).
"""
function path_excess(s0, s1, c, dtheta_ref = 0.0; bmad_compat::Bool = false)
  bmad_compat && return dspline_len(s0, s1, c, dtheta_ref; bmad_compat)
  h = (s1 - s0) / 2
  m = (s0 + s1) / 2
  acc = 0.0
  for i in 1:4
    xp = spline_d1(c, m + h * GL4_NODES[i])
    acc += GL4_WEIGHTS[i] * xp^2 / (1 + sqrt(1 + xp^2))
  end
  sh, ch = sincos(dtheta_ref / 2)
  return h * acc + 2sh^2 * (s1 - s0) - 2sh * ch * (spline_val(c, s1) - spline_val(c, s0))
end

#---------------------------------------------------------------------------------------------------

"""
    CSRElementInfo

Per-element centroid geometry (Bmad `csr_ele_info_struct`). The centroid "chord" is the line from the
centroid position at the element entrance to the centroid position at the exit. `spline` gives the
centroid offset from the chord as a function of distance along the chord.
"""
struct CSRElementInfo
  L::Float64                   # Element (reference) length
  g::Float64                   # Reference curvature (0 for straight elements)
  s_start::Float64             # Reference s at the element entrance
  ref_floor0::FloorPosition    # Reference floor position at entrance
  ref_floor1::FloorPosition    # Reference floor position at exit
  floor0::FloorPosition        # Centroid floor position at entrance
  floor1::FloorPosition        # Centroid floor position at exit
  theta_chord::Float64         # Angle of the chord in the floor Z-X plane
  L_chord::Float64             # Chord length. Negative if the centroid moves backwards in the element
  spline::SVector{4,Float64}   # Centroid offset from the chord vs. chord distance
  dL_s::Float64                # Centroid path length minus L_chord
end

"""
    CSRGeometry

Centroid geometry for a beamline: `start` is the centroid floor position at the start of the lattice
(Bmad `eleinfo(0)%floor1`), and `ele[i]` the information for element `i`.
"""
struct CSRGeometry
  start::FloorPosition
  ele::Vector{CSRElementInfo}
  bmad_compat::Bool
end

"""
    CSRGeometry(L, g, centroid; ref_start = FloorPosition(), bmad_compat = false)

Build the centroid geometry for elements of length `L[i]` and reference curvature `g[i]`.

`centroid` is an `(n+1) × 6` matrix of centroid phase-space coordinates `[x px y py z pz]`: row 1 at the
start of the lattice, row `i+1` at the exit of element `i` (Bmad's `centroid(0:n)`).
"""
function CSRGeometry(L::AbstractVector, g::AbstractVector, centroid::AbstractMatrix;
                     ref_start::FloorPosition = FloorPosition(), bmad_compat::Bool = false)
  return CSRGeometry(L, g, planar_survey(L, g; start = ref_start), centroid; ref_start, bmad_compat)
end

"""
    CSRGeometry(L, g, ref_exit, centroid; ref_start = FloorPosition(), bmad_compat = false)

As above, with the reference floor positions `ref_exit[i]` at the exit of each element given
(e.g. for lattices with patches; see `advance_floor`, `advance_floor_patch`).
"""
function CSRGeometry(L::AbstractVector, g::AbstractVector, ref_exit::AbstractVector{FloorPosition},
                     centroid::AbstractMatrix; ref_start::FloorPosition = FloorPosition(), bmad_compat::Bool = false)
  n = length(L)
  length(g) == n && length(ref_exit) == n || error("CSRGeometry: L, g and ref_exit must have the same length")
  size(centroid) == (n + 1, 6) || error("CSRGeometry: centroid must be an (n+1) × 6 matrix with n = length(L)")

  cfloor(ref, row) = centroid_floor(ref, centroid[row, 1], centroid[row, 2], centroid[row, 6])
  start = cfloor(ref_start, 1)

  ele = Vector{CSRElementInfo}(undef, n)
  s = 0.0
  for i in 1:n
    ref_floor0 = i == 1 ? ref_start : ref_exit[i-1]
    ref_floor1 = ref_exit[i]
    floor0 = i == 1 ? start : ele[i-1].floor1
    floor1 = cfloor(ref_floor1, i + 1)

    dr = floor1.r - floor0.r
    theta_chord = atan(dr[1], dr[3])
    L_chord = sqrt(dr[1]^2 + dr[3]^2)
    spline = SVector(0.0, 0.0, 0.0, 0.0)
    dL_s = 0.0

    if abs(L_chord) >= 1e-8   # Bmad's threshold for a zero-length chord
      # An element with a negative step (e.g. off-center beam in a patch with only an x-pitch) can have
      # floor%theta and theta_chord 180° apart. Restrict theta0 and theta1 to [-π/2, π/2).
      theta0 = modulo2(floor0.theta - theta_chord, pi / 2)
      theta1 = modulo2(floor1.theta - theta_chord, pi / 2)
      parallel0 = abs(modulo2(floor0.theta - theta_chord, pi)) < pi / 2
      parallel1 = abs(modulo2(floor1.theta - theta_chord, pi)) < pi / 2
      parallel0 == parallel1 || error("CSRGeometry: inconsistent centroid direction in element $i")
      parallel0 || (L_chord = -L_chord)
      spline = chord_spline(L_chord, theta0, theta1)
      dL_s = path_excess(0.0, L_chord, spline; bmad_compat)
    end

    ele[i] = CSRElementInfo(L[i], g[i], s, ref_floor0, ref_floor1, floor0, floor1, theta_chord, L_chord, spline, dL_s)
    s += L[i]
  end
  return CSRGeometry(start, ele, bmad_compat)
end

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
