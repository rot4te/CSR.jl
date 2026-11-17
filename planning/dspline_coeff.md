# `dspline_len`: s⁴ term appears to be missing a factor 1/5

**Component:** Bmad 1D CSR (`csr_method = one_dim`)
**File:** `bmad/space_charge/csr_and_space_charge_mod.f90`, function `dspline_len`, line 1796
**Version checked:** `bmad-ecosystem` at `ee81f2a9e` (release 20261001-0). The file was last changed in
`89493969e` (2026-08-24). `git log -L` traces the line back to `219f308ed` ("More csr devel.", 2016-04-21).
**Status:** analysis and a proposed fix. Not yet confirmed by Bmad maintainers.

---

## Summary

`dspline_len` returns the path-length excess of a cubic spline over a reference line,
`½∫ (x′ − dθ)² ds`. In the closed-form result, the s⁴ term is `9*c(3)**2`. Integrating `9c₃²s⁴` gives
`9c₃²/5 · (s₁⁵ − s₀⁵)`, so the term should be `9*c(3)**2/5`. The other four terms carry their 1/(n+1)
factors correctly.

The error is zero when `c(3) = 0`. Otherwise `dlen` is too large.

---

## Current code

```fortran
! x' = c(1) + 2*c2*s + 3*c3*s^2
! dlen = Integral: x'^2/2 ds

c = spline%coef
s0 = s_chord0
ds = s_chord1 - s_chord0

if (present(dtheta_ref)) then
  c(1) = c(1) - dtheta_ref
endif

dlen = (ds / 2) * ( &
        c(1)**2 + &
        (2*s0 + ds) * 2*c(1)*c(2) + &
        (3*s0*s0 + 3*s0*ds + ds*ds) * (6*c(1)*c(3) + 4*c(2)**2) / 3 + &
        (4*s0**3 + 6*s0*s0*ds + 4*s0*ds*ds + ds**3) * 3*c(2)*c(3) + &
        (5*s0**4 + 10*s0**3*ds + 10*s0*s0*ds*ds + 5*s0*ds**3 + ds**4) * 9*c(3)**2 &
       )
```

---

## Derivation

With `x(s) = c₀ + c₁s + c₂s² + c₃s³` (`c₁` already shifted by `dtheta_ref`):

```
x′    = c₁ + 2c₂ s + 3c₃ s²
(x′)² = c₁² + 4c₁c₂ s + (4c₂² + 6c₁c₃) s² + 12c₂c₃ s³ + 9c₃² s⁴
```

Each power integrates as

```
∫_{s₀}^{s₁} sⁿ ds = (s₁ⁿ⁺¹ − s₀ⁿ⁺¹)/(n+1) = ds · Pₙ/(n+1),     ds = s₁ − s₀
```

with

```
P₀ = 1
P₁ = 2s₀ + ds
P₂ = 3s₀² + 3s₀ds + ds²
P₃ = 4s₀³ + 6s₀²ds + 4s₀ds² + ds³
P₄ = 5s₀⁴ + 10s₀³ds + 10s₀²ds² + 5s₀ds³ + ds⁴
```

These are exactly the polynomials in the code. Therefore

```
dlen = ½ ∫ (x′)² ds
     = (ds/2) · [ c₁²·P₀ + (4/2)c₁c₂·P₁ + ((4c₂² + 6c₁c₃)/3)·P₂ + (12/4)c₂c₃·P₃ + (9/5)c₃²·P₄ ]
```

Term by term:

| Power | Coefficient in (x′)² | ÷ (n+1) | Code | |
|---|---|---|---|---|
| s⁰ | `c₁²` | `c₁²` | `c(1)**2` | ✓ |
| s¹ | `4c₁c₂` | `2c₁c₂` | `2*c(1)*c(2)` | ✓ |
| s² | `4c₂² + 6c₁c₃` | `(4c₂² + 6c₁c₃)/3` | `(6*c(1)*c(3) + 4*c(2)**2) / 3` | ✓ |
| s³ | `12c₂c₃` | `3c₂c₃` | `3*c(2)*c(3)` | ✓ |
| s⁴ | `9c₃²` | **`9c₃²/5`** | `9*c(3)**2` | ✗ |

**Minimal case.** `c₁ = c₂ = 0`, `s₀ = 0`:
exact `½∫₀^ds 9c₃²s⁴ ds = 9c₃²ds⁵/10`; code gives `(ds/2)·ds⁴·9c₃² = 9c₃²ds⁵/2`, a factor 5 too large.

---

## When it matters

The centroid spline is built by `create_a_spline([0,0], [L_chord,0], theta0, theta1)`
(`csr_and_space_charge_mod.f90:289`). With `dy = 0`, `sim_utils/math/spline_mod.f90:127-130` gives

```
c(2) = −(2·theta0 + theta1)/L_chord
c(3) =  (theta0 + theta1)/L_chord²
```

So `c(3) = 0` exactly when the centroid's entrance and exit angles are symmetric about the chord
(`theta1 = −theta0`). An on-axis sector bend is such a case, which may be why this has not shown up.

It is nonzero for asymmetric centroid paths:
- off-axis or angled centroid orbits,
- bends with asymmetric edge angles or combined-function fields that steer the centroid,
- quadrupoles or other elements where the centroid is offset,
- drifts entered at an angle with a different exit angle (after a kick upstream).

`dspline_len` feeds every path-length difference in the 1D CSR calculation:

| Line | Use |
|---|---|
| 290 | `eleinfo%dL_s`, per-element path excess |
| 907 | `Lsz0`, path length from lattice start (upstream-drift source) |
| 1055, 1065, 1069, 1072 | `dL` in `ddz_calc_csr`, the retardation residual `L/(2γ²) + dL − dz` |
| 1420 | `dL` in the first-bin integral `I_int_csr` |

An overestimated `dL` shifts the retarded source point and enters `I_csr` through `dL/z`.
I have not quantified the effect on the final CSR wake.

---

## Numerical check

Base Julia. `bmad = true` reproduces the current code.

```julia
simpson(f, a, b, n) = (h = (b - a) / n;
  h / 3 * (f(a) + f(b) + 4 * sum(f(a + (2k - 1) * h) for k in 1:n÷2) + 2 * sum(f(a + 2k * h) for k in 1:n÷2-1)))

function dspline_len(s0, s1, c, dtheta; bmad = false)
  c1 = c[2] - dtheta; c2 = c[3]; c3 = c[4]; ds = s1 - s0
  return (ds / 2) * (c1^2 + (2s0 + ds) * 2c1 * c2 + (3s0^2 + 3s0 * ds + ds^2) * (6c1 * c3 + 4c2^2) / 3 +
                     (4s0^3 + 6s0^2 * ds + 4s0 * ds^2 + ds^3) * 3c2 * c3 +
                     (5s0^4 + 10s0^3 * ds + 10s0^2 * ds^2 + 5s0 * ds^3 + ds^4) * 9c3^2 / (bmad ? 1 : 5))
end

Lc = 1.0
for (t0, t1) in ((0.01, -0.01), (0.01, -0.012), (0.01, 0.0))
  c = [0.0, t0, -(2t0 + t1) / Lc, (t0 + t1) / Lc^2]          # create_a_spline coefficients
  q = simpson(s -> (c[2] + 2c[3] * s + 3c[4] * s^2)^2 / 2, 0.0, Lc, 2000)
  println((t0, t1), "  Bmad/quad = ", dspline_len(0.0, Lc, c, 0.0; bmad = true) / q,
                    "  corrected/quad = ", dspline_len(0.0, Lc, c, 0.0) / q)
end
```

Output (Julia 1.12.7):

| theta0 | theta1 | c(3) | Bmad / quadrature | Corrected / quadrature |
|---|---|---|---|---|
| 0.010 | −0.010 | 0 | 1.0000 | 1.000000 |
| 0.010 | −0.012 | −0.002 | 1.7105 | 1.000000 |
| 0.010 | 0.000 | 0.010 | 55.0000 | 1.000000 |

For random coefficients and intervals (with `dtheta_ref ≠ 0`), the corrected form matches Simpson
quadrature to about 1e-14. The Bmad form is off by factors of 4 to 45.

---

## Resolution

Proposed one-line fix at line 1796:

```diff
-        (5*s0**4 + 10*s0**3*ds + 10*s0*s0*ds*ds + 5*s0*ds**3 + ds**4) * 9*c(3)**2 &
+        (5*s0**4 + 10*s0**3*ds + 10*s0*s0*ds*ds + 5*s0*ds**3 + ds**4) * 9*c(3)**2 / 5 &
```

Suggested follow-up for the Bmad side (not done):
1. Add a unit test comparing `dspline_len` with numerical quadrature for a spline with `c(3) ≠ 0`.
2. Expect `regression_tests/csr_and_space_charge_test/output.correct` to change if the test bend's centroid
   spline has `c(3) ≠ 0` (its particle start has `x = 2e-4`, and the bend has a gradient). Regenerate after
   confirming the fix.

### Caveat

This analysis assumes the code is meant to compute what its comment says, `½∫ x′² ds` (the small-slope form of
spline length minus chord length). The other four terms follow that integral exactly, so a dropped factor is the
likely explanation, but it needs confirmation from the Bmad maintainers.

---

## Handling in CSR.jl

- Use the corrected formula (`9c₃²/5`) by default.
- Provide a `bmad_compat` switch that reproduces the current Bmad expression, for like-for-like comparison
  with Bmad output in phase 3.
- Unit test: closed form vs quadrature for random coefficients with `c₃ ≠ 0` (check C8 in `verify_analytic.jl`).
