# Physics decisions

Decisions that affect the physics CSR.jl computes, with their reasons and measured effects. The guiding rule
(D5): **the default fixes demonstrated errors or approximations in Bmad's implementation; `bmad_compat = true`
reproduces Bmad.** Where Bmad's intent is unclear, its code is ported literally. Labels `D…` are project decisions,
`V…` findings in Bmad's `csr_and_space_charge_mod.f90`; line numbers refer to bmad-ecosystem `00b4557af`.

## Differences from Bmad in the default mode

| | Topic | Default (CSR.jl) | `bmad_compat = true` (Bmad) | Measured effect |
|---|---|---|---|---|
| V9 | Path-length excess `ΔL` in the retardation condition | Exact: `∫ x′²/(1+√(1+x′²)) + 2 sin²(dθ/2) Δt − sin dθ Δx` | Small-angle `½∫(x′ − dθ)²` (`dspline_len`) | Bmad's excess is too large by `L_s⁵/(1920ρ⁴)` on a circle (1.9e-4 m at `L_s` = 5.2 m, ρ = 10 m). Image kernel near zero lag at heights ≥ 3 cm: 14% low in Bmad, ≤ 3e-5 from an exact-retardation Liénard–Wiechert field in CSR.jl. Free-space `I_csr` at a bend entrance: up to 1.8% (cross-check). Cost +1%. |
| V1 | `dspline_len` s⁴ coefficient | Superseded by the exact form (V9) | `9c₃²` instead of `9c₃²/5` | Bmad's `ΔL` 1.71× the quadrature value for an asymmetric chord spline; exact for symmetric (on-axis) orbits. |
| V5, D15 | Integral of `I_csr` over the bins | First bin: closed form over the kick element plus quadrature beyond it; every other bin by Simpson quadrature in `u^{1/3}` (`all_bin_quadrature = true`) | First bin holds `I_csr` constant beyond the kick element; other bins trapezoid | Bmad's first bin is 2/3 of the integral in steady state and 3–4× too small in the entrance transient. Final Δpz vs Bmad: 48–49% (single bends), 4.6% (Tao example). Kick error at n_bin 400, entrance transient: 0.050 (trapezoid) vs 0.0026. Cost 6.6× per free-space kick (1 thread). `all_bin_quadrature = false` keeps the quadrature for the first bin only. |
| V7 | Source geometry after the root search | Re-evaluated at the returned root | Last trial point of `super_zbrent` | Normally within the 1e-8 m tolerance; when `ddz(0) = 0` exactly, `I_csr` from the stale point was 9.6e6 instead of −40. |
| V8, D13 | Direction of a source ahead of the kick point (image kick) | True direction of motion | Direction + π | Image field from sources ahead: 1.21 / 8.99 (Bmad) vs −0.826 / −0.111 (analytic, reproduced to 5 digits). Max kick change 33% of the peak, energy loss ≤ 6% (σ = 1 mm, h = 1 cm). |
| D12 | Image sources ahead beyond the kick element | Searched downstream, and past the lattice end on an infinite drift (exact retardation) | Not searched | Matters when the bunch is longer than about the chamber height; needed for long, space-charge-dominated bunches. |
| D20 | Image kernel over a bin | Integrated against the hat function of the interpolated charge density (adaptive Simpson, `rtol = 1e-6`) | Sampled at the lag | Bmad's result depends on n_bin unless bins resolve `~y/γ` (h = 3 mm, ρ = 10 m: 2.7 / 3.4 / 4.9 / 0.37 × free space at 60/120/240/480 bins). Hat: 0.345 → 0.340 over the same range (16 layers). Cost ~5× the source solves. |
| V2 | Kick coefficient `csr_kick_coef` | `ds r_e abs(q) / ((m/m_e) e β² γ)` from `Δpz = ΔE/(β²E)` | `ds r_c / ((m/m_e) e abs(q) γ)`, `r_c = r_e q²/(m/m_e)`: an extra `1/(m/m_e)` (since Bmad replaced `r_e` by `classical_radius`, commit `50b90dee33`, 2020) and no `1/β²` | Protons: Bmad's kick 1/1836 of the default. Electrons and positrons: `1/β²` only (1e-4 at γ = 100). |
| Seam | Retardation condition `ddz` for sources in the kick element, and `L(1 − β)` | Angle of the source–kick line in the chord frame (as for sources in other elements), `−2 L_plane` for sources ahead, `L/(γ²(1+β))`: `ddz` continuous at every element boundary and at the lattice ends, one source per lag | Secant slope of the spline as that angle, `2Δs_chord` for sources ahead, `L/(2γ²)` | Bmad's `ddz` jumps at the kick element's entrance by ~(α³/3)²Δs (8.6e-9 m at the test lattice's bend, 1e-7 m near its entrance), so lags in the overlap have two sources (image kernel 43 vs 3209 at one hat node: the bend's curvature starts there) and the result depends on where the search starts. With the overlap, the rrule and ForwardDiff disagreed by up to 2e-5. Also removes the `L/(8γ⁴)` error of Bmad's retardation (drift image kick at zero lag, y = 1 cm: 0.031 at γ = 200 and 0.50 at γ = 50 instead of 0; now 0 to round-off). |
| V3 | Kick point along the centroid chord in a bend (`s_ref_to_s_chord`) | Where the centroid spline crosses the reference normal plane at the kick's `s` | `ρ atan(s/ρ)` on an on-axis bend (the chord coordinate is `ρ[sin(s/ρ − θ/2) + sin(θ/2)]`) | Kick point upstream by `≈ 7L³/(24ρ²)` at the exit of an on-axis bend (29 µm for L = 0.1 m, ρ = 10 m). |
| D24 | Sum over image layers | Binomial average of the last `image_sum_order = 8` partial sums (fixed per-layer weights) | Plain sum | Terms alternate and decay as `~1/n`. Plain sum of 16 layers: 170× (h = 3 mm) and 6× (h = 1 cm) the converged kick. Averaged, 32 layers agree with a 1024-layer reference to ~1e-6. |

## Choices with no Bmad counterpart, or Bmad kept

| | Topic | Decision | Reason / effect |
|---|---|---|---|
| D1 | Coupling to tracking | CSR.jl slices each element and tracks each slice with BeamTracking, kicking between slices (no change to BeamTracking). | BeamTracking runs an element as one fused per-particle kernel; a whole-bunch histogram can only run between launches. |
| D7 | Meaning of "same as Bmad" | CSR-level equivalence (slicing rules, geometry, γ, kicks) on top of BeamTracking's own tracking. | Cross-check: CSR.jl reproduces Bmad's CSR to Bmad's diagnostic precision (compat); end-to-end differences come from tracking. |
| D14 | Centroid orbit | Tracked in the bunch's slices (`sliced_centroid = true`). | BeamTracking's integration depends on the slicing; the centroid then follows a bunch particle's orbit. Regression bend vs Bmad: wake 9.9e-5 → 5.9e-6. |
| D2 | Macroparticle charge | `weight × |charge|`, `weight = nothing` means 1. | Same convention as BeamTracking's IBS. |
| D9 | Which elements get CSR | All by default; `csr_on(ele)` to select. | Bmad's per-element default is off; tracking whole elements without CSR is available. |
| D8, D21 | γ in an lcavity | Linear ramp from `E_ref − dE_ref` to `E_ref`; applied Δpz rescaled to BeamTracking's stepped reference momentum. | Bmad's ramp; its swapped interpolation weights (V6) were fixed upstream (PR #2199), so both modes follow current Bmad. 2 vs 8 cells agree to 5e-4 with the rescale, 4% without. |
| D10, D22 | SaganCavity slices | Built from BeamTracking's Sagan kernels; the half multipole and ponderomotive kicks of each drift segment are point events at its ends, only drifts are split. | Slices reproduce the whole cavity to round-off (standing wave 2e-7 → 7e-19; with a solenoid 2e-16, the solenoid map's own round-off). |
| D11, V4 | Round-off branches of the source search | Ported literally (they return a residual, not a position). | Affects only the image kick's bend curvature at the source; vanishes for symmetric bends. |
| — | GPU number type | `Float32` where the backend lacks `Float64` (Metal); kick tables computed on the CPU in `Float64`. | Free-space kick on Metal vs CPU (Float32): 2.7e-6; strongly shielded kick: 3e-4 (cancellation). |

## Where the details are

Derivations, scripts and the full measurement history are in the repository: `planning/PLAN.md` (sections
"Findings in Bmad", "Results log"), `planning/upstream_reports.md` (reports drafted for Bmad and BeamTracking),
`planning/dspline_coeff.md` (V1), `planning/upstream-bin.md` (V5), and the scripts in `planning/diagnostics/`.
