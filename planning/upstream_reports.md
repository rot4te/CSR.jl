# Draft upstream reports (not filed)

Drafted 2026-10-09 for the user to review and file. Nothing here has been sent anywhere.
Line numbers: Bmad master `00b4557af` (2026-10-08); BeamTracking.jl `fe92aac` (2026-10-02).

Already fixed upstream, no report needed: **V6** (swapped energy-interpolation weights), bmad-ecosystem PR #2199,
merged 2026-10-06. Note: open PR #2085 (wiggler subdivision) rewrites `s_source_calc`/`ddz_calc_csr`; V7 and V8 sites
may move if it merges.

Not proposed for reporting: V2 (ion kick prefactor; unresolved on our side), V3 (`s_ref_to_s_chord`; intent unclear),
V4 (round-off branches; harmless for symmetric bends). Could be added as questions.

---

## Bmad: one issue, four items

**Title:** 1D CSR (`csr_and_space_charge_mod.f90`): four possible errors (dspline_len, first-bin integral, source
geometry after zbrent, source direction for sources ahead)

Found while porting `csr_method = one_dim` to Julia (CSR.jl). Each item was checked numerically against quadrature
or an analytic limit; scripts available on request. Line numbers are for master `00b4557af`.

**1. `dspline_len`, line 1796: missing factor 1/5 in the s⁴ term.**
The term is `(5*s0**4 + … + ds**4) * 9*c(3)**2`. Integrating `9 c₃² s⁴` over `[s0, s0+ds]` gives
`9 c₃²/5 · (s₁⁵ − s₀⁵)`, and the bracket is `(s₁⁵ − s₀⁵)/ds`, so the coefficient should be `9*c(3)**2/5`.
Exact when `c(3) = 0` (centroid symmetric about its chord). For an asymmetric chord spline (end slopes 0.010,
−0.012) the returned dL is 1.71× the quadrature value. Affects the retardation root and `I_csr` for off-axis or
asymmetric centroid orbits. (Write-up: attach `dspline_coeff.md`.)

**2. `I_csr`, lines 1414–1425: first bin when the source is in an upstream element.**
`k%I_int_csr = k%I_int_csr + k%I_csr * (z - zz)` holds `I_csr` constant from `zz` to `z`. `I_csr` varies strongly
there: the first-bin integral is 2/3 of the quadrature value in steady state and 3–4× too small in the
entrance transient. Quadrature of `I_csr` over `[zz, z]` (Simpson in u^(1/3)) fixes it. In our cross-check against Tao, the
CSR Δpz of the `csr_and_space_charge_test` bend differs from Bmad's by about 50%, mostly from this item. (Write-up: attach `upstream-bin.md`.)

**3. `s_source_calc`, line 992: geometry in `kick1` is from the last `ddz_calc_csr` evaluation, not the root.**
`ddz_calc_csr` stores the source geometry in `kick1` as a side effect. `super_zbrent` ends with an interpolation step
(and returns at once when `f(a) == 0` after evaluating `f(b)`), so `kick1` describes the last trial point. Normally
within the 1e-8 m tolerance. When `ddz(0) == 0` exactly, `kick1` describes the far end of the element and `I_csr`
evaluated from it is wrong (example: 9.6e6 instead of −40). Fix: call `ddz_calc_csr(s_source, status)` once after
`super_zbrent`. Bmad's own first-bin code may never hit the catastrophic case; the within-tolerance inconsistency remains.

**4. `ddz_calc_csr`, line 1077: source direction reversed for sources ahead of the kick point.**
`floor_s%theta = theta_sl + theta_L`, with `theta_L = atan2(L_vec)` the direction of the vector kick − source. For a
source ahead (`ds < 0`) `L_vec` points backwards, so `theta` is the true direction + π. Only
`image_charge_kick_calc` reads it (line 1469), so source velocity and acceleration are reversed for the image kick
from sources ahead. Check on a drift (γ = 100, y = 0.01, coulomb term): sources behind match the uniform-motion field
`γΔz/(γ²Δz² + y²)^{3/2}`; sources ahead at Δz = −0.011, −0.03 give 1.21, 8.99 vs −0.826, −0.111 analytic. Using the
true direction reproduces the analytic values to 5 digits. Reached only when the bunch is longer than about the
chamber height (`|lag| > ns·beam_chamber_height`); `csr_and_space_charge_test` does not reach it. Line dates from
`ff57a4dda` (2016).

---

## BeamTracking.jl: one issue, five items

**Title:** SaganCavity `universal!` and energy conversions: t_ref, error messages, p_over_q_ref, R_to_gamma/R_to_v for ions

Line numbers for `fe92aac`.

**1. `universal!` (SaganCavity), `ext/BeamTrackingBeamlinesExt/unpack_bl.jl:395, 460`: `t_ref` reset.**
`t_exit = dt_ref` and then `bunch.t_ref = t_exit`, so the entrance `bunch.t_ref` is dropped (expected
`t_enter + dt_ref`). Every other `universal!` adds to `t_ref` (e.g. line 85).

**2. Same function, lines 356–361: `$ele_name` is undefined** (not an argument of `universal!`), so each of these
checks throws `UndefVarError` instead of its message. Also a stray `\` ends line 359: it parses as left division
between the two `||` expressions on lines 359 and 360 (harmless at present, but unintended).

**3. `bunch.p_over_q_ref` is not updated after a SaganCavity** when `p_over_q_ref` is not time dependent (only the
`TimeDependentParam` branch, line ~413, calls `setfield!`).

**4. `src/utils/energy.jl:6`, `R_to_gamma` omits the charge state:** `sqrt((R*C_LIGHT/massof)^2 + 1)` should be
`sqrt((R*C_LIGHT*chargeof/massof)^2 + 1)` (as `R_to_E`, `R_to_beta_gamma`). Correct for |Z| = 1 only.
Checked: He²⁺ at R = 1 T·m gives 1.00323 vs `R_to_E/massof` = 1.01286.

**5. `src/utils/energy.jl:9`, `R_to_v` misplaces the charge:** `abs(chargeof)*C_LIGHT / sqrt(1 + (m/(R c))^2)` should
be `C_LIGHT / sqrt(1 + (m/(R c chargeof))^2)`. Correct for |Z| = 1 only; for |Z| > 1 it can exceed c. Checked: He²⁺ at
R = 1 gives v/c = 0.16034 vs 0.15882 from `beta_gamma_to_v(R_to_beta_gamma)`; ¹²C⁶⁺ 0.16091 vs 0.15892. Used by
`rf_omega_calc` (`utils_bl.jl:205`), so RF frequencies set by harmonic number are wrong for multiply charged ions.
