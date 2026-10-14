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
    root_tangent(f, s0, f0) -> s

Root `s0` of `f(s) = 0`, found from primal values, with the root's derivatives with respect to the parameters of
`f` attached by implicit differentiation: `ds/dp = -(∂f/∂p) / (∂f/∂s)`, where `f0 = f(s0)` carries `∂f/∂p` at
fixed `s` and `∂f/∂s` comes from a nested `Dual` at `s0` (tag `RootTag`). Returns `s0` when `f0` carries no
derivatives. First order only: nested `Dual`s (e.g. Hessians) are not supported.
"""
root_tangent(f, s0, f0) = s0

function root_tangent(f, s0::Real, f0::Dual{Tg,V,N}) where {Tg,V<:AbstractFloat,N}
  sv = Dual{Tg}(V(s0), zero(Partials{N,V}))
  fs = f(Dual{RootTag}(sv, Partials((one(sv),))))
  dfds = value(partials(fs, 1))
  return Dual{Tg}(V(s0), -partials(f0) / dfds)
end

root_tangent(f, s0::Real, f0::Dual{Tg,V}) where {Tg,V} =
  error("root_tangent: nested Duals (value type $V) are not supported")

"""
    RootTag

ForwardDiff tag of the derivative `∂f/∂s` in `root_tangent`. Ordered inside any other tag (including `Nothing`, for
untagged `Dual`s), so that it nests within the parameters' `Dual`s.
"""
struct RootTag end
ForwardDiff.:≺(::Type, ::Type{RootTag}) = true
ForwardDiff.:≺(::Type{RootTag}, ::Type) = false
ForwardDiff.:≺(::Type{RootTag}, ::Type{RootTag}) = false
