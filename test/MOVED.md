# Test moves and changed expectations (ModularCSR stage S2)

Baseline: CSR.jl 44a4e72, verbose `-t 1` run, 2681/2681 (`.modwork/s2-logs/baseline_verbose_t1.log`).
Result of this stage: ModularCSR `-t 1`, 1870/1870 (includes the 2 new step-length asserts) pass (`.modwork/s2-logs/suite_t1.log`). The top-level testset is
renamed `CSR.jl` → `ModularCSR`.

## Testsets that are not in ModularCSR (baseline pass/total)

| Baseline testset (CSR.jl file) | Count | Goes to |
|---|---|---|
| `CSR kick > csr_n_step` (kick_test.jl) | 3/3 | BeamTracking, `slice_n_step` test (spec 4.6.3, stage S6) |
| `Element slicing (CSR off)` (slicing_test.jl) | 159/159 | BeamTracking slicing port (spec 4.4.4, S4); its SaganCavity case inside `RF cavity` (slicing_test.jl:289-292) goes with S5 |
| `lcavity (SaganCavity) mode > Slices reproduce the whole cavity: *` (11 testsets, 19 each), `Reference momentum steps at the cell kicks` (3), `Errors` (1) | 213 | BeamTracking SaganCavity slice tests (spec 4.5.5, S5) |
| `lcavity (SaganCavity) mode > track_csr! through a Branch` (14), `csr_kick! pz_scale` (2) | 16/16 | ModularCSR CSR-effect tests (spec 4.7.4, S7) |
| `track_csr!` (track_csr_test.jl, all children incl. `csr_ref_energy`) | 41/41 | ModularCSR CSR-effect tests (S7) |
| `track_csr! with a test bunch` (track_tps_test.jl) | 37/37 | ModularCSR CSR-effect tests (S7) |
| `track_csr! with a TPS bunch in knobs` (track_tps_collective_test.jl) | 310/310 | ModularCSR CSR-effect tests (S7) |
| `Package extensions load`: `CSRBeamTrackingBeamlinesExt` check | 1 | ModularCSR BeamTracking extension (S7) |
| `Review fixes 2026-10-18`: BT-dependent parts (W3 `shifted_to_slice`, S2 `track_csr!` entry checks, S10 last SaganCavity slice, S7 SaganSlice keywords, D.10 `slice_end`, the lcavity lines of "Final review fixes", the `shifted_to_slice`/`slice_end` lines of "TPS cases") | 33 of 66 | Spec 4.7.4 says "the BT parts of review_fixes_test" go to ModularCSR S7. The per-part split is not in the baseline summary, so the removed count is the total difference only. |

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
