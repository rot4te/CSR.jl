# Draft upstream reports (not filed)

Drafted 2026-10-09 for the user to review and file. Nothing here has been sent anywhere.
Line numbers: Bmad master `00b4557af` (2026-10-08; `csr_and_space_charge_mod.f90` unchanged through `7c9df4059`,
REL 20261014-0). Item 5 added 2026-10-14, items 6–8 2026-10-15.

**Refreshed 2026-10-15:**
- Bmad: `origin/main` is still `7c9df4059` after a fetch; every cited line was re-read there and matches its quoted
  content (lines 290, 339, 769, 907, 951–959, 992, 1054–1057, 1077, 1083, 1141, 1414–1426, 1469, 1609, 1770–1799,
  1817–1842).
- Bmad PR #2085 (wiggler subdivision, David Sagan): **still open** (GitHub page, 2026-10-15), head `3dd783a67`
  (2026-07-04), no formal review; a long review comment by C. Mayes (2026-07-04) lists eight findings, one of them
  "a roundoff-recovery path that sets the wrong `s_source` value" (our V4, below). It changes 417+/127− lines of
  `csr_and_space_charge_mod.f90` but none of the code of items 1–8: every site is unchanged in the PR, only moved.
  Line numbers there are in the table at the end of the Bmad section; quote both if it merges before filing.
- BeamTracking.jl: `origin/main` = v0.8.7 `65e343f`. Former items 1–2 (SaganCavity `t_ref` reset, undefined
  `$ele_name`, stray `\`) were fixed upstream by `8c302c6` (2026-09-18, "Fix bugs in SaganCavity universal!", in
  v0.8.7, not in `fe92aac`, which the draft was checked against). Former item 3 (`p_over_q_ref` not updated after
  a SaganCavity) does not occur with v0.8.7 (`diagnostics/sagan_tref_check.jl`: −0.33356 → −0.66713 after an
  lcavity with dE_ref = 1e8 eV). They are removed below; the two ion items remain.
  **Correction 2026-10-17:** former item 3 does occur with v0.8.7 (the −0.33356 → −0.66713 doubling above was most
  likely this bug, misread; inference). Re-drafted as a separate issue below ("BeamTracking.jl: second issue").
- Existing issues (GitHub issue search via WebFetch, titles only, summarized by a model: moderately reliable): none
  matching in bmad-ecosystem ("csr", 36 results) or BeamTracking.jl (`R_to_gamma`, `R_to_v`, `SaganCavity`: 0).

Already fixed upstream, no report needed: **V6** (swapped energy-interpolation weights), bmad-ecosystem PR #2199,
merged 2026-10-06.

Not proposed for reporting: V4 (round-off branches that return a `ddz` residual as the source position; harmless
for symmetric bends). Item 8 describes when they are reached; V4 could be added there as a question. Mayes's PR #2085
comment appears to raise the same point (not read in full: summary of the PR page).

---

## Bmad: one issue, eight items

**Title:** 1D CSR (`csr_and_space_charge_mod.f90`): eight possible errors (dspline_len coefficient, first-bin integral,
source geometry after zbrent, source direction for sources ahead, small-angle path-length excess, kick prefactor for
ions, kick position in bends, discontinuous retardation condition)

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

**5. `dspline_len` (1770–1799) and its callers: small-angle path-length excess.**
`dspline_len` returns `½∫(x′ − dθ)² ds`, the second-order expansion of the path excess over the chord (or over the
source–kick line, angle `dθ` in the element's chord frame). `ddz_calc_csr` sums it over elements to get
`dL = Ls − L` for the retardation condition (lines 1055, 1065–1072; also 290, 907, 1420). Since the projection of a
path onto a line is `∫cos(θ − θ_L) ds_arc`, the exact excess per element is
`∫ x′²/(1 + √(1 + x′²)) ds + 2 sin²(dθ/2) Δs − sin(dθ) Δx` (Δs, Δx: chord and offset change over the interval);
`½∫(x′ − dθ)²` drops the dθ⁴ terms. Over the long source distances of image charges `dθ` is not small (≈ 0.26 rad at
Ls = 5 m on ρ = 10 m). On a circle the small-angle form overestimates the excess by `Ls⁵/(1920ρ⁴)` (the next term of
`ρφ − 2ρ sin(φ/2)`), i.e. a relative error `Ls²/(80ρ²)`. Checked (ρ = 10 m, γ = 1e4, kick 5 cm before the end of a
40 m arc of 0.1 m sector bends, centroid on axis): excess too large by 1.8e-5, 1.9e-4, 1.9e-3 m at Ls = 3.2, 5.2,
8.2 m, matching the formula. It acts as a shift of the source lag, so the retarded image source sits too close (Ls
5.165 vs 5.169 m at y = 0.768 m). The image kernel (`image_charge_kick_calc`) evaluated at Bmad's source point equals
an independent Liénard–Wiechert field with exact retardation evaluated at that same point (≤ 2e-3), so the field
formula is fine and the error is the source position. Near lag 0 the kernel is steep, and at every height
y ≥ 3 cm it comes out ~14% low (0.86 of the exact-retardation field); high image layers sit entirely in that region.
Small heights and free space are affected far less: the image kernel at y = 3 mm, lag 6 mm is 0.1% high, and
free-space `I_csr` for a 1 mm bunch (lags to 6 mm) on a ρ = 10 m arc changes by ≤ 4e-4 relative. With the exact form (4-point Gauss–Legendre
for the `x′` term) the kernel agrees with the exact field to ≤ 3e-5 at y = 3 cm–1.9 m, at ~1% extra run time in our
port. The exact form also makes item 1 moot.

**6. Kick prefactor, line 769 (same pattern at 1141 and 1609): `rel_mass` counted twice for non-electrons.**
`coef = actual_track_step * classical_radius(species) / (rel_mass * e_charge * abs(charge_of(species)) * gamma)`,
with `classical_radius = classical_radius_factor * charge_of**2 / mass_of` = `r_e Q² m_e/m` and `rel_mass = m/m_e`.
This gives `ds r_e |Q| / (rel_mass² e γ)`. The energy change of a particle of charge `Qe` in the CSR field of a line
charge (bin charges in C, unsigned) is `ΔE = |Q| e E_z ds` with `E_z = (1/4πε₀) × (kernel ⋆ λ′)` and
`1/4πε₀ = r_e m_e c²/e²`, so `Δpz = ΔE/(β²E) = ds r_e |Q| / (rel_mass e β² γ) × (kernel ⋆ λ′)`. Bmad's kick is
smaller by `rel_mass` (1/1836 for protons); electrons and positrons are unaffected. The extra factor dates from
commit `50b90dee33` (2020-03-24, "removing r_e in favor of classical_radius() function"), which replaced `r_e` by
`classical_radius(species)` and kept the explicit `rel_mass` (before it the line read `r_e / (rel_mass e |Q| γ)`:
right mass, `1/|Q|` instead of `|Q|`). Other `classical_radius` users (Touschek, envelope) have no extra `rel_mass`.
Lines 1141 and 1609 (LSC) have the same pattern; we have not derived those. Minor: the `1/β²` of `Δpz = ΔE/(β²E)` is
dropped (≈ 1/γ² for electrons, larger for slow ions).

**7. `s_ref_to_s_chord`, lines 1817–1842 (called at 339): kick point upstream of the bunch in a bend.**
The bend branch, `|ρ| atan2(s_ref cos dθ, |ρ + x + s_ref sin dθ|)`, treats `s_ref` as a straight distance along the
entrance tangent and returns ρ times the angle seen from the bend centre. For a centroid on the reference orbit it
returns `ρ atan(s/ρ)`; the chord coordinate of the reference point at `s` is `ρ[sin(s/ρ − θ/2) + sin(θ/2)]` (θ = L/ρ).
At `s = L` it should be `L_chord = 2ρ sin(θ/2)` (0.09996 vs 0.09967 for L/ρ = 0.1, ρ = 1 m). The kick point then lies
upstream by up to `≈ 7L³/(24ρ²)` (29 µm for L = 0.1 m, ρ = 10 m; 2.9 mm for a 1 m bend). This moves where the field is
evaluated, so it matters mainly in entrance/exit transients (not measured). We use the chord coordinate where the
centroid spline crosses the reference normal plane at `s_ref` (a few Newton steps); for an on-axis bend it agrees with
the formula above to the accuracy of the cubic chord spline.

**8. `ddz_calc_csr`, lines 1053–1057 and 1083; `I_csr` 1419–1422: retardation condition discontinuous at element
boundaries.**
For a source in the kick element, line 1054 passes the secant slope `(x(s1) − x(s0))/ds` of the chord spline as
`dtheta_L` ("angle of L line"); the cross-element branch uses `theta_L − theta_chord`, an angle. Line 1057 uses `2 ds`
(a chord-coordinate difference) where the source–kick distance belongs. Line 1083 uses `L/(2γ²)` for `L(1 − β)`
(error `L/(8γ⁴)`), while the upstream infinite drift solves `Ls − βL = dz` exactly. So `ddz` jumps at the kick
element's entrance and at the lattice start (there by `L/(8γ⁴)`, from the formulas; not measured). Measured with Bmad's formulas (our port, compat mode) on a test lattice
(drift 0.2 m, bend 0.3 m with g = 1.5, drift 0.2 m; off-axis centroid; γ = 200; image layer y = 1 cm): jump at the bend
entrance −1.1e-7, −3.0e-8, −8.6e-9, −5.9e-13 m for kicks 0.03, 0.13, 0.17, 0.29 m into the bend. A negative jump leaves
lags with no root (the round-off branches at lines 951–959 then return a `ddz` value as `s_source`); a positive jump
gives lags with two roots, one on each side of the boundary, and the source found depends on where the element walk
starts. We hit the second case when we used the exact path excess with the secant slope: at one lag the image kernel
was 43 or 3209 depending on the start element (the source is just outside or just inside the bend). We did not look
for a positive jump with Bmad's own formulas. `zz` in `I_csr` (line 1422, the lag of a source at the kick element's
entrance) is built the same way, so the first-bin split does not match the root search exactly. Fix in our port:
the angle `theta_L − theta_chord` in the kick element as elsewhere, `−2 L_plane` for sources ahead, `L/(γ²(1 + β))`,
and `zz` from `ddz_calc_csr` at the entrance. `ddz` is then continuous to round-off (≤ 1e-16 m) at element boundaries
and at the lattice ends, each lag has one source, and the drift image kick at zero lag becomes 0 (y = 1 cm, γ = 200:
0.031 with `L/(2γ²)`). Cost ~1%.

**Line numbers in open PR #2085** (head `3dd783a67`; code at each site unchanged, found by its text):

| Item | master `7c9df4059` | PR #2085 |
|---|---|---|
| 1 `dspline_len` s⁴ term | 1796 (function 1770–1799) | 1840 (1814–1843) |
| 2 `I_csr` first bin | 1414–1426 (`zz` 1422, update 1425) | `zz` 1462, update 1465 (subroutine `I_csr` from 1409) |
| 3 `super_zbrent` in `s_source_calc` | 992 | 1024 |
| 4 `floor_s%theta` | 1077; read at 1469 | 1118; 1509 |
| 5 `dspline_len` callers | 290, 907, 1055, 1065–1072, 1420 | 1969 (`chord%dL_s`), 2113 (`Lsz0`, moved to a helper), 1089, 1101–1113, 1460 |
| 6 kick prefactor | 769, 1141, 1609 | 801, 1182, 1653 |
| 7 `s_ref_to_s_chord` | function 1817–1842, call 339 | subroutine 1863–1901 (per sub-chord; bend branch unchanged), call 364 |
| 8 `ddz_calc_csr` | 1054, 1057, 1083 (function 1023–1086) | 1088, 1091, 1124 (1055–1127) |
| V4 round-off branches | 951–959 | 982–990 |

---

## BeamTracking.jl: one issue, two items

**Title:** `R_to_gamma` and `R_to_v` wrong for ions with |Z| ≠ 1

Line numbers for v0.8.7 `65e343f` (re-read 2026-10-15). (Former items 1–3 on SaganCavity `universal!`: fixed
upstream or not reproducible with v0.8.7; see the header.)

**1. `src/utils/energy.jl:6`, `R_to_gamma` omits the charge state:** `sqrt((R*C_LIGHT/massof)^2 + 1)` should be
`sqrt((R*C_LIGHT*chargeof/massof)^2 + 1)` (as `R_to_E`, `R_to_beta_gamma`). Correct for |Z| = 1 only.
Checked: He²⁺ at R = 1 T·m gives 1.00323 vs `R_to_E/massof` = 1.01286.

**2. `src/utils/energy.jl:9`, `R_to_v` misplaces the charge:** `abs(chargeof)*C_LIGHT / sqrt(1 + (m/(R c))^2)` should
be `C_LIGHT / sqrt(1 + (m/(R c chargeof))^2)`. Correct for |Z| = 1 only; for |Z| > 1 it can exceed c. Checked: He²⁺ at
R = 1 gives v/c = 0.16034 vs 0.15882 from `beta_gamma_to_v(R_to_beta_gamma)`; ¹²C⁶⁺ 0.16091 vs 0.15892. Used by
`rf_omega_calc` (`utils_bl.jl:205`), so RF frequencies set by harmonic number are wrong for multiply charged ions.

---

## BeamTracking.jl: second issue (drafted 2026-10-17)

**Title:** `pz` shifted twice after a SaganCavity (`bunch.p_over_q_ref` not updated for constant reference energy)

Version: v0.8.7 `65e343f` (= `main` on 2026-10-17 per the GitHub commit list; no matching issue found by a
GitHub search for "SaganCavity", WebFetch summary, moderately reliable).

**What happens.** `universal!` for `SaganCavity` (`ext/BeamTrackingBeamlinesExt/unpack_bl.jl:357` ff.) sets
`bunch.p_over_q_ref` only in the `TimeDependentParam` branch (`unpack_bl.jl:421`); at the end it sets only
`bunch.t_ref` (`unpack_bl.jl:465`). The cavity kernel already renormalizes the coordinates to the exit reference
momentum cell by cell (`reference_momentum_shift!` in the Sagan kernels). So after the cavity the coordinates refer to
the exit reference, but `bunch.p_over_q_ref` still holds the entrance value, and the next element's `universal!`
(`unpack_bl.jl:133`) sees different reference energies and applies `reference_momentum_shift!` a second time.

**Reproduction** (`diagnostics/stage2c_sagan_pq_check.jl` in CSR.jl): `Branch([Marker(E_ref = 1e8, species_ref = e),
RFCavity(L = 1.0, voltage = 1e8, rf_frequency = 1.3e9, dE_ref = 1e8, phi0 = 0.0, zero_phase = PhaseRef.Accelerating,
tracking_method = SaganCavity(n_cells = 4), traveling_wave = true), Drift(L = 0.5), Drift(L = 0.5)])`, electron
`Bunch(zeros(1, 6))` at 1e8 eV, `track!` element by element:

| after | `pz` | `bunch` E_ref | element E_ref |
|---|---|---|---|
| RFCavity | 0.0 | 1.0e8 | 2.0e8 |
| Drift | −0.500004896 | 2.0e8 | 2.0e8 |
| Drift | −0.500004896 | 2.0e8 | 2.0e8 |

The on-crest reference particle gains exactly `dE_ref`, so `pz` must stay ≈ 0; a drift cannot change it. Same with
`n_cells = 0`. Expected fix (suggestion): set `bunch.p_over_q_ref = p_over_q_ref` (the cavity's exit value) at the end
of the SaganCavity `universal!` for constant `p_over_q_ref`.

Side effect seen with GTPSA/ForwardDiff: with a `dE_ref` knob of primal value 0, the `≈` test at `unpack_bl.jl:133`
compares equal values and skips the second shift for the derivatives only, so derivatives and finite differences
disagree by a factor 2 there (`diagnostics/stage2c_dE0_bt.jl`).

---

## GTPSA.jl: one issue (drafted 2026-10-16; cc BeamTracking.jl)

GitHub issue search (API, titles, 2026-10-16) for "thread" and "segfault" in `bmad-sim/GTPSA.jl`: no issue on this.
Closed #117 ("Make `@FastGTPSA` thread safe") is about the Julia-side buffer, not the C temporaries.

**Title:** Segfault: nonlinear TPS functions are not safe on several Julia threads (C temporaries indexed by OpenMP
thread number)

**Versions:** GTPSA.jl 1.5.6, GTPSA_jll 1.7.1+0, Julia 1.13.2, macOS arm64 (Apple M3). MAD-NG source read at master
`9f591769c` (2026-10-16); that GTPSA_jll is built from this exact source was not checked (its `libgtpsa.dylib`
links `libgomp` and imports `omp_get_thread_num`).

**Reproducer** (`diagnostics/gtpsa_threads.jl`):
```julia
using GTPSA
d = Descriptor(3, 2); x = @vars(d)
t0 = 0.3 + x[1] + 0.1x[2] * x[3]
Threads.@threads for i in 1:200_000
  y = sqrt(t0 + i * 1e-9)
end
```
`julia -t 4`: segfault (signal 11), 3/3 runs. `julia -t 1`: runs. With 4 threads, `+`, `*` and `abs` run;
`/` (TPS/TPS), `c/TPS`, `sqrt`, `log`, `sin`, `exp`, `atan(y, x)`, `^(1/3)` all segfault.

**Cause (from the C source):** nonlinear operations take scratch TPSs from the descriptor,
`mad_tpsa_impl.h:168-170`:
```c
int tid = omp_get_thread_num();
assert(d->ti[tid] < DESC_MAX_TMP);
tpsa_t *tmp = d->t[ tid*DESC_MAX_TMP + d->ti[tid]++ ];
```
(released at 182–186). Outside an OpenMP parallel region `omp_get_thread_num()` returns 0, so every Julia thread
uses slot 0 and increments `ti[0]` without synchronization: two threads get the same temporary or corrupt the counter.

**Consequence in BeamTracking.jl (0.8.7):** `track!(bunch, bl; use_cpu_multithreading = true)` with a TPS bunch
(2000 particles, 6 variables, order 2, bend + drift + quadrupole, 4 threads) segfaults; with the default `false` it
runs (`diagnostics/bt_tps_threads.jl`). `@FastGTPSA` indexes its own buffer by `Threads.threadid()`, but the C
functions it calls still take temporaries by OpenMP thread number.

**Possible fixes (for the maintainers to judge):** let the Julia side pass the thread index to the C temporaries
(e.g. a per-call or thread-local index set from `Threads.threadid()`), or build without `DESC_USE_TMP` (temporaries
allocated per call, `mad_desc_impl.h:133-139`), or document that TPS arithmetic must not run on several threads.

Also noticed (inference, not tested): `nth = 10` on this machine (a descriptor field; apparently
`omp_get_max_threads()`), while `@FastGTPSA` indexes by `Threads.threadid()`, which can exceed it with more Julia
threads than cores.

**CSR.jl workaround (2026-10-16, branch `stage2-tps`):** kernels with TPS numbers run as one workgroup on the
calling thread (`thread_safe_number`, `cpu_workgroupsize`); TPS kick tables in one block.
