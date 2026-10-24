# Bracketed root finder (Bmad super_zbrent).

"""
    zbrent(f, x1, x2, rel_tol, abs_tol) -> (x_zero, ok, x_last)

Brent's root finder, following Bmad's `super_zbrent` (including the final linear interpolation). `x_last` is the
last point at which `f` was evaluated.
"""
function zbrent(f, x1, x2, rel_tol, abs_tol)
  itmax = 100
  a = x1; b = x2
  fa = f(a); fb = f(b)
  x_last = b
  if (fa > 0 && fb > 0) || (fa < 0 && fb < 0)
    return 1e100, false, x_last
  end
  c = b; fc = fb
  d = b - a; e = d
  for _ in 1:itmax
    if (fb > 0 && fc > 0) || (fb < 0 && fc < 0)
      c = a; fc = fa
      d = b - a; e = d
    end
    if abs(fc) < abs(fb)
      a = b; b = c; c = a
      fa = fb; fb = fc; fc = fa
    end
    tol1 = 0.5 * (rel_tol * abs(b) + abs_tol)
    xm = 0.5 * (c - b)
    if abs(xm) <= tol1 || fb == 0
      return (fb == 0 ? b : (b * fc - c * fb) / (fc - fb)), true, x_last   # Linear interpolation
    end
    if abs(e) >= tol1 && abs(fa) > abs(fb)
      s = fb / fa
      if a == c
        p = 2xm * s
        q = 1 - s
      else
        q = fa / fc
        r = fb / fc
        p = s * (2xm * q * (q - r) - (b - a) * (r - 1))
        q = (q - 1) * (r - 1) * (s - 1)
      end
      p > 0 && (q = -q)
      p = abs(p)
      if 2p < min(3xm * q - abs(tol1 * q), abs(e * q))
        e = d
        d = p / q
      else
        d = xm
        e = d
      end
    else
      d = xm
      e = d
    end
    a = b
    fa = fb
    b += abs(d) > tol1 ? d : copysign(tol1, xm)
    fb = f(b)
    x_last = b
  end
  return b, false, x_last
end

"""
    root_tangent(f, s0, f0, dfds = nothing) -> s

Root `s0` of `f(s) = 0`, found from primal values, with the root's dependence on the parameters of `f` attached:
Newton steps `s ← s - (f(s) - r0) / dfds` from `s0` in the number type of `f0 = f(s0)`, with `r0 = primal(f0)` (so the
primal root stays `s0`) and the primal slope `dfds = ∂f/∂s` at `s0` (computed by `root_slope` if `nothing`). With a
constant slope each step gains one order, so `root_order(f0)` steps are taken: one for `Dual`s (implicit
differentiation, `ds/dp = -(∂f/∂p) / (∂f/∂s)`), two for nested `Dual`s (Hessians), the truncation order for a TPS.
Returns `s0` (converted to the promoted type of `s0` and `f0`) when `f0` carries no derivatives.
"""
function root_tangent(f, s0, f0, dfds = nothing)
  n = root_order(f0)
  n == 0 && return convert(promote_type(typeof(s0), typeof(f0)), s0)
  d = dfds === nothing ? root_slope(f, s0) : dfds
  r0 = primal(f0)
  s = s0 - (f0 - r0) / d
  for _ in 2:n
    s -= (f(s) - r0) / d
  end
  return s
end

"""
    root_order(x)

Number of Newton steps `root_tangent` needs for `x`'s number type: 0 for plain numbers, 1 per level of `Dual`
nesting (a TPS extension adds its truncation order). Other number types (e.g. `Complex`) have no method.
"""
root_order(x::Number) = root_order(typeof(x))
root_order(::Type{<:Real}) = 0
root_order(::Type{<:Dual{T,V}}) where {T,V} = 1 + root_order(V)

"""
    root_slope(f, s0)

Primal `∂f/∂s` at `s0`, from a `Dual` with tag `RootTag` (outside the parameters' `Dual`s of any nesting).
"""
root_slope(f, s0) = primal(partials(f(Dual{RootTag}(primal(s0), one(primal(s0)))), 1))

"""
    RootTag

ForwardDiff tag of the derivative `∂f/∂s` in `root_slope`. Ordered after any other tag (including `Nothing`, for
untagged `Dual`s), so that its `Dual` wraps the parameters' `Dual`s.
"""
struct RootTag end
ForwardDiff.:≺(::Type, ::Type{RootTag}) = true
ForwardDiff.:≺(::Type{RootTag}, ::Type) = false
ForwardDiff.:≺(::Type{RootTag}, ::Type{RootTag}) = false
