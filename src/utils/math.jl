# Small math helpers.

# Fortran nint (round half away from zero). The fallback is exact: x -
# trunc(x) has no rounding error.
"""
    nint(x)

Nearest integer, half away from zero (Fortran `nint`, used by Bmad's
binning). Exact for any real type; hardware floats use
`RoundNearestTiesAway`, other types (e.g. `BigFloat`, which lacks that
mode) a `trunc`-based fallback.

# Arguments
- `x`: real number (e.g. a position in bin widths, as in
  `deposit_ix0`).

# Returns
- `Int`.
"""
nint(x::Union{Float32,Float64}) = round(Int, x, RoundNearestTiesAway)

function nint(x)
  t = trunc(x)
  return Int(t) + (abs(x - t) >= 0.5 ? Int(sign(x)) : 0)
end

"""
    modulo2(x, amp)

`x` folded into the range `[-amp, amp)`. Same as Bmad's `modulo2`.

# Arguments
- `x`: value to fold (e.g. an angle difference [rad]). For a
  `Dual` only the value is folded (partials unchanged);
  CSRGTPSAExt adds a TPS method.
- `amp`: half-width of the range (`> 0`; e.g. `pi / 2`).
"""
function modulo2(x, amp)
  m = mod(x, 2amp)
  return m >= amp ? m - 2amp : m
end

# Fold the value only; the derivative of the fold is 1 (ForwardDiff's
# `mod` gives NaN partials at a value of exactly 0).
# CSRGTPSAExt adds a TPS method (GTPSA has no `mod`).
modulo2(x::Dual{T}, amp) where {T} = Dual{T}(modulo2(value(x), amp),
    partials(x))

"""
    cube_root(x)

Real cube root, `cbrt(x)`. CSRGTPSAExt adds a TPS method (GTPSA
has no `cbrt`).

# Arguments
- `x`: real number, `Dual` or TPS (the TPS method errors on a zero
  scalar part).
"""
cube_root(x) = cbrt(x)

"""
    cross3(a, b)

Cross product of two 3-vectors (indexable), as an `SVector`.

# Arguments
- `a`, `b`: 3-vectors, indexed `1:3` (e.g. `SVector`s).
"""
cross3(a, b) = SVector(a[2] * b[3] - a[3] * b[2],
                       a[3] * b[1] - a[1] * b[3],
                       a[1] * b[2] - a[2] * b[1])

"""
    dot3(a, b)

Dot product of two 3-vectors (indexable).

# Arguments
- `a`, `b`: 3-vectors, indexed `1:3` (e.g. `SVector`s from
  `x_axis`, `z_axis`, or `FloorPosition` `r` differences).
"""
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
