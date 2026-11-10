# Test moves and changed expectations (CSR.jl stage S2)

Baseline: CSR.jl 44a4e72, verbose `-t 1` run, 2681/2681 (`.modwork/s2-logs/baseline_verbose_t1.log`).
Result of this stage: CSR.jl `-t 1`, 1870/1870 (includes the 2 new step-length asserts) pass (`.modwork/s2-logs/suite_t1.log`).

## Testsets that are not in CSR.jl (baseline pass/total)

| Baseline testset (CSR.jl file) | Count | Goes to |
|---|---|---|
| `CSR kick > csr_n_step` (kick_test.jl) | 3/3 | BeamTracking, `slice_n_step` test (spec 4.6.3, stage S6) |
| `Element slicing (CSR off)` (slicing_test.jl) | 159/159 | BeamTracking slicing port (spec 4.4.4, S4); its SaganCavity case inside `RF cavity` (slicing_test.jl:289-292) goes with S5 |
| `lcavity (SaganCavity) mode > Slices reproduce the whole cavity: *` (11 testsets, 19 each), `Reference momentum steps at the cell kicks` (3), `Errors` (1) | 213 | BeamTracking SaganCavity slice tests (spec 4.5.5, S5) |
| `lcavity (SaganCavity) mode > track_csr! through a Branch` (14), `csr_kick! pz_scale` (2) | 16/16 | CSR.jl CSR-effect tests (spec 4.7.4, S7) |
| `track_csr!` (track_csr_test.jl, all children incl. `csr_ref_energy`) | 41/41 | CSR.jl CSR-effect tests (S7) |
| `track_csr! with a test bunch` (track_tps_test.jl) | 37/37 | CSR.jl CSR-effect tests (S7) |
| `track_csr! with a TPS bunch in knobs` (track_tps_collective_test.jl) | 310/310 | CSR.jl CSR-effect tests (S7) |
| `Package extensions load`: `CSRBeamTrackingBeamlinesExt` check | 1 | CSR.jl BeamTracking extension (S7) |
| `Review fixes 2026-10-18`: BT-dependent parts (W3 `shifted_to_slice`, S2 `track_csr!` entry checks, S10 last SaganCavity slice, S7 SaganSlice keywords, D.10 `slice_end`, the lcavity lines of "Final review fixes", the `shifted_to_slice`/`slice_end` lines of "TPS cases") | 33 of 66 | Spec 4.7.4 says "the BT parts of review_fixes_test" go to CSR.jl S7. The per-part split is not in the baseline summary, so the removed count is the total difference only. |

## Testsets that stay (core), with the removed parts

- `CSR kick`: 25 of 28 (the `csr_n_step` testset removed).
- `Review fixes 2026-10-18`: 33 of 66. Kept: W4 (7/7), W5 (13/13), S10 reference range (2/2), S9 (4/4), Final review
  fixes (5 of its asserts; the lcavity lines removed), TPS cases (2 of its asserts; the shifted/slice lines removed).
- `Package extensions load`: 3 of 4 (the BeamTracking extension check removed).
- All other baseline testsets of geometry, binning, wake, kick, kernels, shielding, ad, ad_geometry, ad_rrule, tps,
  tps_collective and type_stability: same pass/total as the baseline (compared testset by testset).

## Tests whose expected values changed

None. `ds` is a differentiable argument: the ForwardDiff and finite-difference references are derived from the same
(Dual) element length as in CSR.jl, and the rrule returns the `ds` cotangent (`dsbar`), which the tests chain into the
element length with d(ds)/dL = 1/n. The reverse-mode geometry cotangent therefore includes the same ∂ds/∂L term as
ForwardDiff. `vs ForwardDiff`, `Parameter adjoint == ForwardDiff` and `test_rrule` pass at their original tolerances.

Other changes in the AD tests (not values):
- rrule of `csr_kick_pz` returns 12 entries (the `ds` slot, `dsbar`, is at position 8); tuple indices for `s_kick` and
  `gamma` moved by one (9 and 10).
- `test_rrule` row names changed because the printed `CSRParams` text no longer shows `ds_step`; counts unchanged
  (73/62/11 in each case).

Added tests (kick_test.jl, "Step length"): `ds = 0` raises an `ErrorException`; `CSRParams(; ds_step = …)` raises a
`MethodError`.

## Pruned in S2b

Applied: [proposal](/Users/alexcoxe/.julia/dev/CSR.jl/.modwork/s2b/proposal.md) section 2, items 1-27 and 29. Kept by user decision: item 28 (tps_collective "Test particles in 6D, knobs as GTPSA parameters") and item 30 (ad_geometry "Hessian: hat-integrated image table"). Criterion-3 items B1-B5: no test edits. Counts are testset pass totals from `.modwork/s2b/suite_t1.log`: 1870 before, 1157 after.

| File | Asserts before | Asserts after | Proposal items |
|---|---|---|---|
| geometry_test.jl | 97 | 73 | 1, 2 |
| binning_test.jl | 32 | 28 | 3 |
| wake_test.jl | 105 | 59 | 5, 6, 7, 8, 9 |
| kernels_test.jl | 116 | 85 | 4, 10 |
| shielding_test.jl | 112 | 104 | 11, 12 |
| ad_test.jl | 219 | 49 | 13, 14 |
| ad_geometry_test.jl | 193 | 132 | 15, 16, 17, 18, 19 |
| ad_rrule_test.jl | 208 | 134 | 20, 21 |
| tps_test.jl | 71 | 46 | 22, 23, 24 |
| tps_collective_test.jl | 574 | 307 | 25, 26, 27 |
| review_fixes_test.jl | 33 | 30 | 29 |

Unchanged: runtests.jl (3), kick_test.jl (27), type_stability_test.jl (80).

Notes:
- Item 20 removes 73 asserts, not the 84 in the proposal. Each `test_rrule` case totals 73 in the log (62 + 11).
- Item 14 keeps interior particles 151 and 263 (the proposal's "e.g.").
- Item 11 and item 25 use explicit (case, bmad_compat) tuples; the removed combinations are not rectangular.

## S2f: derivative-through-kick tests removed

Spec 4.2b item 5 (scalar kick, derivative machinery removed). Counts are pass totals: before from
`.modwork/s2f-ref/suite/table_t1.tsv` (0450fbb's counts equal these, 1157), after from the S2f part 2 `-t 1`
run: 447/447 (same at `-t 4`). All other testsets: pass totals unchanged. New Dual/TPS tests (spec 4.2b item 5,
"Add") are not part of this list.

| File | Testset | Removed | Kept |
|---|---|---|---|
| runtests.jl | `Package extensions load` (all three extensions deleted) | 3 | 0 |
| ad_test.jl (deleted) | `kick in bend` | 23 | 3, moved |
| ad_test.jl (deleted) | `kick at element entrance, images` | 23 | 3, moved |
| ad_test.jl (deleted) | `Dual derivative of the charge in q; grid constant` | 3 | 0 |
| ad_geometry_test.jl (deleted) | all testsets (`advance_floor at g = 0` 4, `modulo2 at a fold boundary` 3, `Source root: residual stationary` 4 + 4, `... to second order` 4 + 4, `Primal search == search in Duals` 21 + 21, `Hat-integrated image table` 8, `free space` 14, `images` 13, `images, bmad_compat` 13, `kick at element entrance, images` 13, `Hessian: free space, trapezoid` 2, `Hessian: images, bmad_compat` 2, `Hessian: hat-integrated image table` 2) | 132 | 0 |
| ad_rrule_test.jl (deleted) | all testsets (`geometry_to_vector round trip` 4, `vs ForwardDiff` 10 + 10 + 10, `s_kick with images, round-off (D43)` 2, `Parameter adjoint == ForwardDiff` 8 × 3, `Zero cotangents` 1, `test_rrule: free space` 73) | 134 | 0 |
| tps_test.jl | `primal, root_order` | 3 | 0 |
| tps_test.jl | `Frozen-wake test particles (images = 0)`: TPS vs ForwardDiff value, Jacobian, Hessian, third order | 16 | 1 (Float64 test particles at bunch positions) |
| tps_collective_test.jl | `Helpers` | 8 | 0 |
| tps_collective_test.jl | `Order 2 (images = 0, bmad_compat = false)`, `(2, false)`, `(2, true)` | 34 × 3 | 0 |
| tps_collective_test.jl | `Order 3 (images = 0)` | 5 | 0 |
| tps_collective_test.jl | `csr_kick! with TPS pz` | 4 | 0 |
| tps_collective_test.jl | `csr_kick! pz element type`: "cannot hold the kick" errors | 6 | 2 (Float32 and Float64 `pz`) |
| tps_collective_test.jl | `Test particles in 6D + knobs (images = 0, bmad_compat = false)` | 76 | 0 |
| tps_collective_test.jl | `Test particles in 6D, knobs as GTPSA parameters` | 51 | 0 |
| tps_collective_test.jl | `Test particles: pz_t element type`: "cannot hold the kick" errors | 9 | 3 (Float32 `pz_t`) |
| tps_collective_test.jl | `Zero-length element in a TPS geometry`: TPS and Dual geometry | 9 | 1 (Float64 `L_chord == 0`; testset renamed `Zero-length element: L_chord == 0`) |
| tps_collective_test.jl | `Descriptor check` | 31 | 0 |
| type_stability_test.jl | `Dual`, `nested Dual`, `TPS`, `TPS dynamic` | 13 × 4 | 0 (`Float64`, `Float32`, `Float32 spline` unchanged) |
| geometry_test.jl | `s_ref_to_s_chord in a bend (V3)`: ForwardDiff derivatives in `s` and `g` | 6 | 17 |
| wake_test.jl | `First-bin quadrature: one branch for each raw geometry number` | 4 | 0 |
| kernels_test.jl | `Work struct and geometry types are isbits (GPU-ready)`: the `Dual` case | 3 | 7 |
| review_fixes_test.jl | `W4 device_number_type`: `device_number_type` (deleted) 4, rrule forward pass 2, duplicate 1 | 7 | 0 (the last assert, renamed `W4 csr_kick_pz element type`, duplicated tps_collective_test.jl `csr_kick_pz returns eltype(pz)`; changed expectation below) |
| review_fixes_test.jl | `W5 gamma check`: `Dual` gamma | 2 | 8 (one `Dual` gamma assert kept: `CSRKickPoint`) |
| review_fixes_test.jl | `S9 no boxed state` (`geometry_active_indices`, `widen_source_range`: deleted with `param_adjoint.jl`) | 4 | 0 |

Moved: the Float64 asserts of ad_test.jl's two `csr_kick_pz` cases (`csr_kick_pz` leaves `pz` unchanged, equals
`csr_kick!`, kick nonzero; 3 per case) are in kernels_test.jl, `csr_kick_pz == csr_kick!: kick in bend` and
`csr_kick_pz == csr_kick!: kick at element entrance, images` (same bunch, geometry, parameters).

Changed expectation: `csr_kick_pz returns eltype(pz)` (tps_collective_test.jl): `csr_kick_pz` with a `Float32`
`pz` and `Float64` inputs on the CPU returns `Float32` (spec 4.2b item 2, `eltype(pz)`); it returned `Float64`
before.

Removed Float64 asserts of deleted functions (no code left to test): `device_number_type` (3 of the 4 W4 asserts
had `Float64`/`Float32` arguments), `geometry_to_vector` round trip, `geometry_active_indices`,
`widen_source_range`.
