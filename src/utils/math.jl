# Small math helpers.

# Fortran nint (round half away from zero). The fallback is exact: x -
# trunc(x) has no rounding error.
"""
    nint(x)

Nearest integer, half away from zero (Fortran `nint`, used by Bmad's
binning). Exact for any real type; hardware floats use
`RoundNearestTiesAway`, other types (e.g. `BigFloat`, which lacks that
mode) a `trunc`-based fallback.
"""
nint(x::Union{Float32,Float64}) = round(Int, x, RoundNearestTiesAway)

function nint(x)
  t = trunc(x)
  return Int(t) + (abs(x - t) >= 0.5 ? Int(sign(x)) : 0)
end

"""
    modulo2(x, amp)

`x` folded into the range `[-amp, amp)`. Same as Bmad's `modulo2`.
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
"""
cube_root(x) = cbrt(x)

"""
    cross3(a, b)

Cross product of two 3-vectors (indexable), as an `SVector`.
"""
cross3(a, b) = SVector(a[2] * b[3] - a[3] * b[2],
                       a[3] * b[1] - a[1] * b[3],
                       a[1] * b[2] - a[2] * b[1])

"""
    dot3(a, b)

Dot product of two 3-vectors (indexable).
"""
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
