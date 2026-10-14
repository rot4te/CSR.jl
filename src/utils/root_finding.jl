# Bracketed root finder (Bmad super_zbrent).

"""
    zbrent(f, x1, x2, rel_tol, abs_tol) -> (x_zero, ok)

Brent's root finder, following Bmad's `super_zbrent` (including the final linear interpolation).
"""
function zbrent(f, x1, x2, rel_tol, abs_tol)
  itmax = 100
  a = x1; b = x2
  fa = f(a); fb = f(b)
  if (fa > 0 && fb > 0) || (fa < 0 && fb < 0)
    return 1e100, false
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
      return (fb == 0 ? b : (b * fc - c * fb) / (fc - fb)), true   # Linear interpolation
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
  end
  return b, false
end
