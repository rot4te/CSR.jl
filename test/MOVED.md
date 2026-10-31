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
