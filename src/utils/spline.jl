# Centroid chord spline and path-length excess over a line (Bmad create_a_spline, dspline_len).

# Cubic spline x(s) = c0 + c1 s + c2 s^2 + c3 s^3 with origin s = 0 (Bmad spline_struct with x0 = 0).
"""
    spline_val(c, s)
    spline_d1(c, s)
    spline_d2(c, s)

Value, first and second derivative at `s` of the cubic `x(s) = c[1] + c[2] s + c[3] s² + c[4] s³` (Bmad
`spline_struct` with `x0 = 0`). Centroid offsets from an element's chord are such cubics (`chord_spline`).
"""
spline_val(c, s) = ((c[4] * s + c[3]) * s + c[2]) * s + c[1]

"""
    spline_d1(c, s)

First derivative of the cubic `c` at `s` (see `spline_val`).
"""
spline_d1(c, s) = (3c[4] * s + 2c[3]) * s + c[2]

"""
    spline_d2(c, s)

Second derivative of the cubic `c` at `s` (see `spline_val`).
"""
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

"""
    GL4_NODES, GL4_WEIGHTS

Nodes and weights of 4-point Gauss–Legendre quadrature on [-1, 1] (used by `path_excess`).
"""
const GL4_NODES = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)

"Weights of 4-point Gauss–Legendre quadrature on [-1, 1] (see `GL4_NODES`)."
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
sources at large heights (`planning/PLAN.md`, V9).
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
