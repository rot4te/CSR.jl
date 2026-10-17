# CSR.jl plan and handoff

One document for planning and session handoff (merged from `PLAN.md` and `PLAN_continued.md` on 2026-10-14).

**How to use it.** A new session reads Parts 1–3 (status, how to work, decisions), then whatever later part the task
needs. Part 9 is the dated results log; older entries there are history, and where they conflict with Parts 1–8, the
earlier parts win.

Contents
1. Status and next steps
2. How to work on this project (preferences, git, paths, test environment)
3. Decisions (D1–D47)
4. Open questions and pending upstream reports
5. Code map
6. Physics and implementation notes
7. Findings in Bmad and BeamTracking
8. Validation: analytic checks and unverified items
9. Results log, by phase
10. Roadmap, limitations, risks
11. Reference sources and planning files

---

## 1. Status and next steps

**Goal.** A 1D (line-charge) coherent synchrotron radiation (CSR) module for the SciBmad Julia ecosystem, ported from
Bmad's Fortran `csr_method = one_dim`; GPU-capable (KernelAbstractions) and differentiable (ForwardDiff, ChainRules,
GTPSA TPS).

**Branches (2026-10-14, none pushed by Claude):**
- `stage-c` (from `phase6-open-questions` `468fc83`): the physics work of 2026-10-14 (commits below).
- **`reorg`** (from `stage-c` `f3c69bf`), head `5ee107c` + this handoff edit: source layout as BeamTracking.jl
  (`src/kernels/`, `src/utils/`), a docstring for every function, README.md, Documenter docs (`docs/`, GitHub
  Actions workflow). 1035/1035 tests on 1 and 4 threads; docs build clean (`checkdocs = :all`).
- **`phase6-geom_speed-deltas`** (from `reorg` `08da04e`, pushed by the user): Phase 6 step 3 (forward mode and rrules
  w.r.t. geometry, `s_kick`, γ), ForwardDiff hard dependency, immutable `CSRKick1`. 1355/1355 on 1 and 4 threads.
- **`geom-pullback`** (from `phase6-geom_speed-deltas` `90e0b77`): cheaper geometry pullback (D30).
- **`fwd-primal-search`** (from `geom-pullback` `82de7c6`, pushed by the user): forward mode with the source search on
  primal geometry (D31), flaky FD test fix (D32), 3e-9 investigation (Section 4).
- **`phase7-tps`** (from `fwd-primal-search` `0a4cfbc`; `origin/phase7-tps` exists, so pushed by the user): Phase 7
  steps 1–3 (D33–D36).
- **`phase7-step5`** (from `phase7-tps` `aff3059`, 2026-10-15): this handoff (D37, D38). (Local `fwd-primal-search` no
  longer exists; its head is `0a4cfbc`.)
- **`phase7-review-fixes`** (from `phase7-step5` `e5a0ede`, 2026-10-15): fixes from the `phase7-tps` review
  (findings 3, 5, 6, 7; Section 9) `4a6140f`; D32 flake fixed (γ scaled in `test_rrule`, s_kick restored; Section 4)
  `2748fb9`.
- **`v2-v3-physics`** (from `phase7-review-fixes` `da6650d`, 2026-10-15): V2 and V3 fixed in default mode (D39, D40),
  continuous retardation condition, one source per lag (D41).
- **`phase7-step5`** (at `v2-v3-physics` `12ec910` when the session started, moved there outside the session), then Phase 7 step 5 (`track_csr!` `test_bunch`;
  user: continue on this existing branch, 2026-10-15).
- **`upstream-refresh`** (from `phase7-step5` `7d344c9`, 2026-10-15): session item 4 (upstream reports, docs
  workflow).
- **`sagan-tref`** (from `upstream-refresh` `0ec4424`, 2026-10-15): SaganCavity slices accumulate `t_ref` as
  BeamTracking 0.8.7 (Section 4).
- **`reruns-d41`** (from `sagan-tref` `bb34954`, 2026-10-15): reruns after D39–D41 (Bmad cross-check,
  `image_kernel_exact.jl`); found and fixed the first-bin log term (D42).
- **`s-kick-residual`** (from `reruns-d41` `1073953`, 2026-10-15): s_kick rrule residual traced to ForwardDiff
  round-off (Section 4, Section 9), `b24fd58`.
- **`image-kernel-d43`** (from `s-kick-residual` `b24fd58`, 2026-10-15): image kernel without cancellation (D43),
  `1d76391`.
- **`fd-flake-d44`** (from `image-kernel-d43` `1d76391`, 2026-10-15): free-space `test_rrule` flake fixed (D44).
- **`tighten-rrule-rtol`** (from `fd-flake-d44` `a1780f0`, by the user, pushed): `test_rrule` rtol per case, seed
  study script started (`cddf927`).
- **`tighten-rrule-tol-atol`** (from `tighten-rrule-rtol` `cddf927`, 2026-10-16): tolerance tightening study
  (Section 9): `test_approx` grouping in `d32_flake.jl`, (rtol, atol) per case.
- **`stage2-tps`** (from `tighten-rrule-tol-atol` `8359698`, 2026-10-16): Stage 2a (collective TPS core, D45) and
  docstrings without references to planning files, `2ad0ca8`.
- **`stage2a-fixes`** (created by the user from `stage2-tps` `2ad0ca8`, 2026-10-16; name as proposed in the session,
  verify): fixes from the Stage 2a review (D46, D47; Section 9), committed by the user. **Latest state.**

| Commit | Branch | Content |
|---|---|---|
| `59e86fa` | stage-c | Step 0: compat `Beamlines = "0.10.6, 0.11"`, `BeamTracking = "0.8.7"`; `~` paths in planning |
| `1b92650` | stage-c | Particle-side kernels run on Metal (Float32 grid, Int32 counters, typed literals) |
| `6d1b8ea` | stage-c | D24 diagnostics |
| `9cbc8bf` | stage-c | D24 averaged image-layer sum (`image_sum_order = 8`); GPU float-type fallback; large-y diagnosis |
| `ffb4004` | stage-c | Exact path-length excess in default mode (`path_excess`, V9); D24 layer-term data tracked |
| `a908ef9` | stage-c | Merged PLAN.md |
| `bf3d3c9` | stage-c | D22: exact SaganCavity split |
| `f3c69bf` | stage-c | Cross-check rerun (Section 9, Phase 3); V9 upstream report draft |
| `5ee107c` | reorg | Reorganized src, docstrings, README, docs |

**Done:** phases 0–5, phase 3 (Bmad cross-check, rerun 2026-10-14), phase 6 steps 0–2 (AD w.r.t. particles),
lcavity mode, Stage B (D20, D21, D23), D24, GPU number types, exact path excess (V9), D22, source reorganization,
README, Documenter docs.

**Phase 6 step 3 done** (3a forward mode, D28/D29, 3b rrules) on branch `phase6-geom_speed-deltas`; **cheaper
geometry pullback done** (D30, branch `geom-pullback`): 5–14 kicks instead of 32–367; **forward mode with primal
search done** (D31, branch `fwd-primal-search`): 12 partials ≈ 4–5 kicks instead of 7–13; see Section 9. **Phase 7 in
progress** (branch `phase7-tps`, steps below). Later: Phase 8 GPU table work (Adapt-based `CSRGeometry` with the element
vector as a type parameter; errors in the source search as return codes; iterative adaptive Simpson).

**Session 2026-10-14 (evening), user decisions:** forward mode with primal search (done, D31); drop s_kick from the
flaky FD check (done, D32); investigate the 3e-9 issue (done: not reproducible, no blocking dependence, Section 4);
keep testing on Julia 1.13.1.

**Phase 7 (started 2026-10-14, branch `phase7-tps`), plan agreed with the user (D33–D36):**
1. Generic source root by Newton steps from the primal root (nested Duals, TPS). **Done** (`a5f28e6`).
2. TPS basics in `CSRGTPSAExt` (`primal` = scalar part, `root_order` = truncation order). **Done.**
3. Stage 1: frozen-wake kick on TPS test particles (`csr_apply_test_kicks!`, `csr_kick!(...; test_particles)`). **Done.**
4. Stage 2: full collective TPS (TPS deposit, TPS tables). Variables decided: D37 (knobs + 6D on test particles). Needs `<:Real` relaxed to `<:Number` in the types (TPS is a
   `Number`), a TPS `modulo2`, and a TPS deposit (serial path). Also review findings 4 and 8 (Section 4).
   **2a done** (D45), reviewed and fixed (D46, D47); 2b, 2c not started.
5. `track_csr!` with a TPS test bunch tracked alongside the bunch (BeamTracking per slice). **Done 2026-10-15**
   (keyword `test_bunch`; Section 9).

**Session 2026-10-15 — handoff (the user restarted the session to load a custom subagent). Agreed plan, in order:**
1. **Done 2026-10-15** (diff was 9 files, +234/−39; findings and fixes in Section 9, deferred items in Section 4).
   **Code review of `phase7-tps`** by the project subagent `sonnet-reviewer` (`.claude/agents/sonnet-reviewer.md`,
   `model: sonnet`, tools Read + Bash, read-only; untracked file, the user decides whether to commit or ignore it), at
   effort `high`, foreground. The user approved this subagent plan; launch it directly. If the type is still not found
   after the restart, tell the user (do not fall back silently). Prompt essentials:
   - Diff `git diff 0a4cfbc..phase7-tps -- src ext test docs` (3 commits, 10 files, +287/−53; PLAN.md excluded): files
     `src/utils/root_finding.jl`, `src/kernels/source_point.jl`, `src/kernels/kick.jl`, `src/kick.jl`,
     `ext/CSRGTPSAExt/CSRGTPSAExt.jl`, `test/tps_test.jl`, `test/ad_geometry_test.jl`, `docs/src/usage.md`.
   - Context: D33–D35 and the Section 6 Dual/TPS pitfalls (branch on `primal`; Dual `==`/`<` compare partials at
     ties; `modulo2` Dual method; TPS is a `Number`; KA CPU splits ndrange > 1024 across threads).
   - Check: order-n Newton steps with a constant primal slope in `root_tangent` (order gained per step, step count,
     `root_order` dispatch for Float64/Dual/nested Dual/TPS); branches on non-primal values; thread safety of TPS in
     the test-kick kernel; dead/out-of-range test particles; backend mismatch; whether the tests can fail
     (independence of the Jacobian/Hessian references, tolerances, gaps such as TPS order > 2); docs example vs API.
   - Output: findings ranked by severity, file:line, failure scenario, confidence. Claude verifies each finding before
     reporting; **no fixes until the user decides.**
2. **Done 2026-10-15** (branch `v2-v3-physics`; D39–D41, Section 9). V2 (ion kick prefactor) and V3
   (`s_ref_to_s_chord`). Fixing V3 moved the kick point of `ad_rrule_test.jl` onto a lag with two sources (the seam);
   the user chose to make `ddz` continuous (D41).
3. **Done 2026-10-15: Phase 7 step 5** (`track_csr!` with a TPS test bunch; D38; Section 9). Tests: order 1 =
   ForwardDiff Jacobian, order 2 = nested-ForwardDiff second derivatives of the same Float64 tracking; Float64 test
   particles follow bunch particles (to 1e-15, not bitwise: BeamTracking SIMD, Section 7b).
4. **Done 2026-10-15** (branch `upstream-refresh`; Section 9): upstream reports refreshed (Bmad lines unchanged,
   PR #2085 still open with a line map; BeamTracking items 1–3 fixed upstream or not reproducible, removed);
   docs workflow on Documenter 1.19's template versions. **Found:** CSR.jl's SaganCavity slices still copy the old
   BeamTracking `t_ref` reset, which 0.8.7 fixed; user chose to match 0.8.7, fixed on branch `sagan-tref` (Section 4).
   The user files the reports and
   enables GitHub Pages.
5. Later (not today): Stage 2 with variables per D37.
6. **Done 2026-10-15** (user's choice of next task; branch `reruns-d41`; Section 9): reruns after D39–D41. Found a
   spurious default-mode first-bin kick on straight lines since D41; user chose the fix D42.
7. **Done 2026-10-15** (user's choice; branches `s-kick-residual`, `image-kernel-d43`; Section 9): s_kick rrule residual. Round-off in the
   image derivative, not a pullback error. User then chose to rewrite the image kernel's `1 − N·β′` (D43).
8. **Done 2026-10-15** (user's choice; branch `fd-flake-d44`; Section 9): free-space `test_rrule` FD flake. Not a step
   size problem (a D32-style rescaling cannot help); user chose to pin the first-bin quadrature's first node (D44).
9. **Done 2026-10-16** (branches `tighten-rrule-rtol` by the user, `tighten-rrule-tol-atol`; Section 9): `test_rrule`
   tolerance tightening. User chose (rtol, atol) = (1e-6, 1e-9) free space, (3e-3, 1e-8) compat, with the
   atol/cotangent ratio measured and reported. Per-field tolerances (a custom `test_approx` for the geometry
   `Tangent`, so element 2's `g` and `ref_floor0.theta` are held to 1e-6 rather than ~8.5e-5 / ~5e-6 through atol):
   **user: keep the general (rtol, atol) for now** (2026-10-16).

10. **Session 2026-10-16 (evening), branch `stage2-tps`.** User: Stage 2 per D37, in parts: **2a** (core collective
    TPS) **done**, then stop for the user's review; **2b** (Stage 1 test particles carrying 6D + knobs, with TPS
    tables; finding 4 fixed by checking the range **before** kicking, user's choice) and **2c** (`track_csr!` with a
    TPS bunch; check that Beamlines/BeamTracking accept TPS element parameters) not started. The TPS thread crash was
    diagnosed first (user's choice) and fixed by D45; GTPSA.jl report drafted. Docstrings and docs pages no longer
    cite planning files (user: they are for development only; benchmark figures kept); they cite the docs page
    "Physics decisions" by label instead. Code and test comments still name `diagnostics/...` scripts (not public).

11. **Session 2026-10-16 (late), branch `stage2a-fixes`. Done.** Review of Stage 2a by Claude (reading the code; no
    Bash in the coordinating session), user's choice "review 2a first". Findings (numbered as reported; Section 4 for
    the deferred ones):
    1. TPS made from a type (`zero(T)`, `KernelAbstractions.zeros(backend, T, n)`, `T(x)`) land in
       `GTPSA.desc_current`. **Only dynamic-descriptor TPS** (`@vars(d, dynamic = true)`) are affected: `@vars(d)`
       returns static-descriptor TPS (descriptor in the type), which the Stage 2a tests and the docs use. → D46.
    2. Hessian/order-3 tests normalized by the overall max → per-block bounds (done, Section 9).
    3. D45 regression only caught with `-t > 1` (deferred).
    4. `csr_kick!` with TPS `pz` untested → test (done).
    5. `thread_safe_number` counts `Dual{_,TPS}` as safe (deferred). 6. `root_order(::TPS)` docstring says
       "truncation order"; it is GTPSA's `mo` "Max order" (deferred).
    Found while implementing: `csr_kick!` with a float `pz` and `Dual`/TPS inputs fails with a cryptic `MethodError`
    in the kick kernel → D47. (First attributed to `CSRKickPoint`; the diagnosis showed `CSRKickPoint` works.)
    **New workflow (this session):** Claude writes a spec (`specs/stage2a-fixes.md`), the `spec-implementer` agent
    implements it, the `spec-tester` agent tests it against the spec's acceptance criteria; spec changes need the
    user's approval and are logged in the spec's "Spec changes" table. Two spec changes approved (dynamic TPS in the
    descriptor test; D47 added). Suite 1812/1812 on 4 threads (5m22s) and 1 thread (5m33s); docs build clean
    (existing warnings).

**Handoff 2026-10-16 (late).** Latest state: branch `stage2a-fixes` (committed by the user), not pushed. Suite
1812/1812 on 1 and 4 threads; docs build clean. Next (user decides): Stage 2b, 2c (each via a spec, Section 2);
deferred review findings 3, 5, 6 (Section 4); Phase 8 items (Section 10). The test environment is a scratch env
(Section 2), rebuilt each session.

Original notes for step 3 (from the reorg handoff, 2026-10-14): Goal: derivatives of a kick w.r.t. the geometry and the reference speed/energy (γ), and `s_kick`.
Starting points (suggestions, not decided):
- Today the `rrule` (`ext/CSRChainRulesCoreExt/CSRChainRulesCoreExt.jl`) returns `@not_implemented` for `geom`,
  `s_kick`, `gamma`; ForwardDiff passes only through the particle side (tables are `Float64`, built on the CPU in
  `src/tables.jl` with the `CSRKick1` work struct).
- The tables depend on the parameters through the retarded source point: implicit differentiation of the root of
  `ddz(s; p) = L/(2γ²) + ΔL − dz = 0` (`kernels/source_point.jl`), `ds/dp = −∂ddz/∂p / ∂ddz/∂s`, then the chain rule
  through `I_csr_value` / `I_csr_calc` / quadratures (`kernels/wake.jl`) and `image_charge_kick` + hat integration
  (`kernels/image_charges.jl`). Also `csr_kick_coef` (∝ 1/γ, `utils/kick_setup.jl`).
- Decide with the user first: which parameters count as "geometry" (centroid orbit rows? element L, g? `s_kick`),
  forward (Duals through the table kernels: needs the work struct and tables generic in the number type) vs reverse
  (adjoint of the table w.r.t. parameters), and test strategy (D18: by hand vs ForwardDiff and finite differences).
- Docs: update `docs/src/usage.md` ("Derivatives") and Section 3 / "Physics decisions" if any physics changes.

Later: Phase 7 (TPS), Phase 8 (FFT convolution, benchmarks, GPU wake table, upstream hook proposal).

**For the user:** push `tighten-rrule-tol-atol`, `stage2-tps`, `stage2a-fixes`; copy this PLAN.md to the Drive
planning folder; file the upstream reports in `planning/upstream_reports.md` (Section 4, incl. V9 and GTPSA.jl); enable GitHub
Pages (`gh-pages`) after pushing. Action versions in `.github/workflows/Documentation.yml` now follow Documenter
1.19's hosting page (checked 2026-10-15; the workflow itself has not run on GitHub).

---

## 2. How to work on this project

### User preferences (also in the user's global CLAUDE.md)
- Accuracy over helpfulness; flag uncertainty ("verify this", "from memory"); never invent APIs or sources.
- Fewest words; lead with the answer; paragraphs ≤ 5 sentences.
- Code changes: report (1) what was wrong, (2) what changed, (3) tested/verified?, (4) what the user must do next.
- Git: never commit to or push existing branches; create a new branch and commit there; never push.
- Show a detailed plan and get approval before launching any subagent.
- Save every diagnostic script written for this project in `diagnostics/` of the planning folder (user, 2026-10-15).
- Planning folder (user, 2026-10-16): `~/Library/CloudStorage/GoogleDrive-rot4te@gmail.com/My Drive/code/Bmad+/CSR`
  (Google Drive; was `planning/` in git until `8359698`). PLAN.md lives at the repo root **and** as a copy there:
  edit the repo copy, then copy it to the Drive folder at each handoff (user: keep both in sync).
- Physics-model and upstream choices are the user's: ask with options and a recommendation.
- Style: 2-space indent, `@kwdef`, docstrings with `## Properties` (BeamTracking style), sparse comments.
- Only PLAN.md is tracked in git; the rest of the planning folder is on Google Drive (above). Paths below written
  `planning/...` or `diagnostics/...` refer to that folder.

### Spec-driven workflow (since 2026-10-16)
- The coordinating Claude session researches, designs, and writes a spec; it does not write project code. Specs live in
  the repo's `specs/` (user's choice for `specs/stage2a-fixes.md`; untracked unless the user commits them).
- `spec-implementer` implements a spec stage exactly; it stops with questions on ambiguity. `spec-tester` tests the
  result against the spec's acceptance criteria, reports code defects, and proposes spec changes (behaviour changes
  need the user's approval; logged in the spec's "Spec changes" table). At most 3 implement–test rounds per stage.
- Agents never commit, push, or change branches; the user creates the branch beforehand and commits afterwards.
- Agents run in the background; scripts and logs in the session scratchpad (diagnostic scripts worth keeping: copy
  to the planning folder's `diagnostics/`).

### Paths (two machines: `$HOME` is `/Users/alexcoxe` or `/Users/coxea3`; always write `~`)
- Package `~/.julia/dev/CSR.jl`; dev checkouts beside it: `BeamTracking.jl` (v0.8.7, `65e343f`), `Beamlines.jl`
  (v0.11.0, `35fe016`), `AtomicAndPhysicalConstants.jl`, `GTPSA.jl` (v1.5.6, the latest registered on 2026-10-16),
  `SciBmad.jl`, `SpaceCharge.jl`.
- Bmad source `~/bmad-ecosystem` (`7c9df4059`, REL 20261014-0 on one machine); conda env `~/miniforge3/envs/bmad`
  on one machine, `~/miniconda3/envs/bmad` on the other (set `ENV["TAO"]` for the cross-check there).
- Julia 1.13.1 via juliaup `release` (user, 2026-10-14; earlier phases ran on 1.12.7); 1.13.2 on 2026-10-16.
  `Project.toml`: `julia = "1.10"`.

### Git
- Remote `origin` (`github.com/rot4te/CSR.jl`). Branches: `main` (`ae0ae77`), `phase6-open-questions` (`468fc83`, pushed
  by the user), `stage-c`, `reorg` (unpushed). Latest: `stage2a-fixes` from `stage2-tps` (Section 1).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Test environment
A scratch environment (wiped between sessions) that `Pkg.develop`s CSR.jl, BeamTracking.jl, Beamlines.jl (from `~`)
and adds Test, StaticArrays, AbstractFFTs, ChainRulesCore, ChainRulesTestUtils, GTPSA, AtomicAndPhysicalConstants,
KernelAbstractions, Atomix, ForwardDiff (+ DelimitedFiles for diagnostics). Run `Pkg.resolve()` after changing CSR's
weakdeps, or new extensions silently do not load.
- Full suite, from `test/`: `julia --project=<env> -t 4 -e 'include("runtests.jl")'` (~5.5 min on 2026-10-16, mostly
  compilation; also run `-t 1`). `track_csr_test.jl` alone needs `maxrel` from `kernels_test.jl`;
  `tps_collective_test.jl` alone needs `kernel_test_bunch` from `kernels_test.jl`.
- One file: `julia --project=<env> -e 'using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines,
  ChainRulesCore, GTPSA; include("utils.jl"); include("<file>_test.jl")'`.
- BeamTracking prints a harmless "Straight multipoles … curved reference system" warning.
- Metal checks (not in the suite): a separate env with Metal.jl + CSR (dev) + AtomicAndPhysicalConstants, StaticArrays.

---

## 3. Decisions

| # | Date | Decision |
|---|---|---|
| D1 | 10-01 | Integration route (a): CSR.jl slices each element and tracks each slice with BeamTracking, applying the CSR kick between slices. No upstream BeamTracking changes (reaffirmed for the lcavity, D10). |
| D2 | 10-01 | Macroparticle charge from `bunch.coords.weight` (real particles per macroparticle): `q_p = w_p * chargeof(species; C=true)`; `weight === nothing` means 1. |
| D3 | 10-06 | Bmad reference: conda-forge bmad (Tao only, no Fortran compilation). |
| D4 | 10-01 | KernelAbstractions CPU backend only "for now"; relaxed 2026-10-14: particle-side kernels and `csr_kick!` run on GPU backends with the tables computed on the CPU (Section 9, Stage C). |
| D5 | 10-01 | Default behavior fixes proven Bmad errors; `bmad_compat = true` reproduces Bmad. Ambiguous Bmad code (V3, V4) is ported literally and logged. |
| D6 | 10-01 | First-bin quadrature option A (superseded by D15). |
| D7 | 10-01 | "Same as Bmad" means CSR-level equivalence (slicing rules, geometry, γ, kicks) on top of BeamTracking's own tracking, not particle-level. |
| D8 | 10-01 | RF γ: Bmad's linear reference-energy ramp; Δpz rescaled to BeamTracking's reference (V6 part superseded by D21). |
| D9 | 10-01 | Per-element CSR opt-in (`csr_on`), default all elements. |
| D10 | 10-02 | lcavity mode needed; implemented CSR-side from BeamTracking's internal Sagan kernels (not an upstream `fringe_at`, not Symplectic RF + reference shifts). |
| D11 | 10-06 | V4 (s_source_calc round-off branches) kept as Bmad in default mode too (bends are typically symmetric). |
| D12 | 10-06 | Image charges: sources ahead of the kick point beyond the kick element are searched downstream, and past the lattice end on an infinite drift (generalizes to long bunches). Not in Bmad; compat restores Bmad. |
| D13 | 10-06 | V8 (source direction reversed for sources ahead) fixed by default, compat reproduces. |
| D14 | 10-06 | `track_csr!(...; sliced_centroid = true)`: centroid tracked in the bunch's slices. |
| D15 | 10-06 | `CSRParams(all_bin_quadrature = true)`: every bin by quadrature (D6 option B) "for now". |
| D16 | 10-08 | Phase 6 = steps 0–2 first (done); step 3 (geometry/γ derivatives) later. |
| D17 | 10-08 | Non-mutating function `csr_kick_pz`. |
| D18 | 10-08 | No ChainRulesTestUtils; pullbacks tested by hand vs ForwardDiff `J'ȳ` and finite differences. |
| D19 | 10-08 | All open questions handled before phase 6 step 3. |
| D20 | 10-09 | Image kernel integrated against the bin hat function (adaptive quadrature) in default mode; compat keeps point sampling. |
| D21 | 10-09 | V6 fixed in Bmad (PR #2199, 2026-10-06): `bmad_compat` follows current Bmad; V6 compat branch removed. |
| D22 | 10-09 | SaganCavity standing-wave ~1e-7 sliced-vs-unsliced difference: investigate an exact split (not accepted as is). Done 2026-10-14: half kicks as point events at drift-segment ends; slices == whole cavity to round-off. |
| D23 | 10-09 | Compat entries: Beamlines when registered with `EMFieldParams`, BeamTracking when a release has `RungeKutta`. Done 2026-10-14: `Beamlines = "0.10.6, 0.11"`, `BeamTracking = "0.8.7"`. |
| D24 | 10-10 | Image-layer sum by a transformed sum, default mode only. Done 2026-10-14: `image_sum_order = 8` (binomial averaging of the last partial sums). |
| — | 10-14 | GPU: fall back to `Float32` where `KernelAbstractions.supports_float64(backend)` is false. |
| — | 10-14 | Exact path-length excess (`path_excess`) in default mode, accepted because it costs < 10% (measured +1%). |
| D25 | 10-14 | Phase 6 step 3 parameters: all geometry builder inputs (centroid rows, element L and g), `s_kick`, γ. |
| D26 | 10-14 | Phase 6 step 3 route: forward first (number-type-generic tables, primal root + implicit tangent), then a hybrid `rrule` (convolution adjoint × table Jacobian by chunked Duals) plus an `rrule` for `CSRGeometry(L, g, centroid)`. |
| D27 | 10-14 | Tests may use ChainRulesTestUtils (reverses D18 for new rrule tests; existing hand tests stay). |
| D28 | 10-14 | ForwardDiff is a hard dependency (already one of BeamTracking; CSR.jl is rarely used alone). `CSRForwardDiffExt` removed; `primal(::Dual)`, `root_tangent`, `RootTag` in core. Compat `ForwardDiff = "1.3.3"` (BeamTracking's; only 1.4.6 tested). |
| D29 | 10-14 | `CSRKick1` immutable (user: prefer immutable types); the source-point functions return updated copies. |
| D30 | 10-14 | Cheaper geometry pullback: source search on primal `Float64` + per-evaluation local `Dual`s (user chose this over per-lag seeding, Step 1 only, and hand-written reverse adjoints; also over "the BeamTracking.jl method"). |
| D31 | 10-14 | Forward mode: `Dual` tables search sources on a primal copy of geometry and kick point (`PrimalSearch`), then evaluate once in `Dual`s (user chose this as the next item). |
| D32 | 10-14 | Flaky FD test: s_kick dropped from the `test_rrule` FD check with images in compat (not an RNG seed); the exact rrule == ForwardDiff check keeps it. **Superseded 10-15:** the failing cotangent was γ, not s_kick (Section 4); γ enters the test wrapper as γ/Γ, Γ = 1e5, and s_kick is back in the FD check. |
| D33 | 10-14 | Phase 7 staged: Stage 1 frozen wake (bunch in Float64 makes the wake; separate TPS test particles get the kick and deposit nothing), then Stage 2 full collective TPS. Stage 1 API: TPS test particles (not the whole bunch in TPS). |
| D34 | 10-14 | TPS shape function: consistent with SciBmad/BeamTracking: branch decisions on the scalar part (`vifelse`, GTPSA comparisons), Bmad's triangle deposit and linear interpolation kept, no smooth-shape option; orders ≥ 2 are the expansion within the reference point's bin cell. |
| D35 | 10-14 | Source root for any number type: Newton steps from the primal root in the full number type (one order per step), replacing the first-order-only implicit tangent. |
| D36 | 10-14 | `track_csr!` with a TPS (test) bunch is in Phase 7, after the core. |
| D37 | 10-15 | Stage 2 TPS variables, option (b): parameters/knobs shared by all particles (geometry L, g, centroid; γ; s_kick; bunch charge; upstream element knobs), every particle's z, q TPS in them, deposit and tables in TPS; plus Stage 1 test particles carrying the 6 phase-space variables as well (frozen in phase space, collective in parameters). |
| D38 | 10-15 | Phase 7 order: step 5 (`track_csr!` with a TPS test bunch) before Stage 2; V2 and V3 decided before step 5. |
| D39 | 10-15 | V2: default kick coefficient `ds r_e abs(q) / (rel_mass e β² γ)` (`Δpz = ΔE/(β²E)`; user chose to include 1/β²); compat keeps Bmad's `ds r_c / (rel_mass e abs(q) γ)`. |
| D40 | 10-15 | V3: default kick point in a bend where the centroid spline crosses the reference normal plane at `s_kick` (Newton in Float64, `root_tangent` for Dual/TPS); straight elements and compat as Bmad. |
| D42 | 10-15 | First-bin closed form in default mode: `log(γ²(1+β) z/Ls)` instead of Bmad's `log(2γ² z/Ls)`, so it vanishes on a straight line with D41's exact retardation (was `1/(4γ⁴)`); both branches of `I_csr_calc`. User chose this over reverting `zz` to Bmad's and over documenting it. Compat unchanged. |
| D43 | 10-15 | Image kernel (both modes): `1 − N·β′ = 1/(γ²(1+β)) + β\|N − t\|²/2`, `N − β′ = (N − t) + (1−β)t`, `B·(N − β′) = (B − t)·(N − t) − \|N − t\|²/2 + (1−β)B·t` (same formula, no cancellation). |
| D44 | 10-15 | First-bin quadrature (default mode): the node at `zz` is the kick element's entrance source from `ddz_calc_csr` (which defines `zz`), not a source search for lag `zz`. Same point in a consistent geometry; one branch for derivatives w.r.t. the raw geometry numbers. |
| D45 | 10-16 | TPS and threads: after diagnosing GTPSA (user chose diagnosis first), every kernel launch with a non-`Real` (TPS) number type runs as one workgroup on the calling thread (`thread_safe_number`, `cpu_workgroupsize`), TPS tables in one block; chosen over serializing the tables only. GTPSA.jl report drafted (user files). |
| D41 | 10-15 | Seam (user: "only one root physically, so the simulation should only have one"): default `ddz_calc_csr` continuous at every element boundary and at the lattice ends: angle (not secant slope) of the L line for sources in the kick element, `−2 L_plane` for sources ahead in it, `L/(γ²(1+β))` for `L(1 − β)`; the first-bin split lag `zz` from `ddz_calc_csr`. Compat keeps Bmad. Chosen over a tie-break in the walk and over replaying the forward search in the pullback. |
| D46 | 10-16 | Descriptor check (review finding 1, user chose "entry check" over switching `GTPSA.desc_current` temporarily or building every TPS from an input value): `planar_survey`, both `CSRGeometry(L, g, ...)` methods, `CSRKickPoint(geom, ...)`, `csr_bin_particles`, `csr_kick!`, `csr_kick_pz` first call `check_descriptors`, which raises an `ErrorException` naming the argument when a **dynamic-descriptor** TPS (`TPS{Float64,GTPSA.Dynamic}`) is not in `GTPSA.desc_current`. Static-descriptor TPS (the `@vars` default), complex TPS, and test particles are not checked. Uses GTPSA internals `getdesc`, `desc_current` (1.5.6). |
| D47 | 10-16 | `csr_kick!` raises a clear `ErrorException` up front when `eltype(pz) <: AbstractFloat` but the kick's number type (promoted from z, q, geometry, s_kick, γ, kick_factor, pz_scale) is not (`Dual`, TPS): "use csr_kick_pz, or a pz with element type …". User chose this over docs only. Float `pz` with float inputs unchanged. |

---

## 4. Open questions and pending upstream reports

**Open (user decisions or unresolved):**
- V2, V3: decided 2026-10-15 (D39, D40). The seam is removed in default mode (D41).
- Seam/V4: closed by the rerun of `image_kernel_exact.jl` after D41 (2026-10-15): y = 3 mm, lag −3 mm now
  −1.111111e-3 vs exact −1.111111e-3 (was −1.3e-4). Lag 0 at y = 3 mm is still 0.945 of exact, but the absolute error
  (0.07) is no larger than at the neighbouring lags ±1e-4 (0.08, 0.27; kernel ~1e4 there): the value sits near a sign
  change, so the ratio shows the ~1e-5 relative accuracy of the other lags (inference from the table, not traced).
- rrule vs ForwardDiff for `s_kick` with images (default mode), 4e-11–5e-10 relative at s = 0.19, 0.23: **closed
  2026-10-15, round-off** (Section 9, `diagnostics/s_kick_residual.jl`). ForwardDiff alone moves by as much when its
  seed is rescaled; the rrule lies inside that spread. Reduced to ≤ 3.6e-13 by D43.
- V4 round-off branches (`SRC_END`/`SRC_START`, D11) remain; with D41 they are reached only at round-off level.
- At h = 1 cm the hat-integrated shielded result depends on n_bin below ~240 bins (likely charge-density resolution
  on a small residual; not verified).
- Test issues found 2026-10-14 (Julia 1.13.1), both closed the same evening:
  (1) "composed checks fail at 2.86e-9 when `ad_rrule_test.jl` runs alone on 1 thread": **not reproducible** at
  `5461eaa` (alone with `kernels_test.jl` for its helpers, 1 thread: pass; scripts on 1 and 4 threads: composed vs
  ForwardDiff(g) 7.2e-11 images, 1.5e-11 compat, bound 1e-9). How the earlier "alone" run was set up is unknown.
  **No blocking dependence:** `Dual` wake and image tables are bitwise identical for `n_block` 1, 3, 4, 16, 81 (values
  and partials, default and compat). Single-direction derivatives through the two routes (Dual g vs Dual geometry
  vector along J[:, i]) agree exactly, and seeds scaled by powers of 2 give bitwise identical derivatives (no branch
  reads partials). Other seed scales spread a derivative by ~2e-11–5e-11 relative, with images only (free space
  ≤ 3e-14): round-off amplified in the image derivative (likely `1 − N·β′ ≈ 1/(2γ²)`, cubed, and the root slope
  `∂ddz/∂s = O(1/γ²)`; the attribution is an inference, not measured term by term). The 7e-11 is that round-off.
  (2) `test_rrule` "images, bmad_compat" s_kick FD flake: fixed by D32 (3/3 runs pass). **Recurred 2026-10-15**
  (branch `phase7-review-fixes`, full suite, 1 thread): 1 failure in that testset, 1483/1484; the next full 1-thread
  run and 3 runs of `ad_rrule_test.jl` alone passed. Failing run's RNG: `Xoshiro(0x620b862c3f3a50f7,
  0x31a050bb29ec984c, 0xd0ff6bd94546534b, 0xd24373f881d7c19c, 0xcbc1bf67a4460596)`. Which comparison failed was not
  captured. **Reopened and fixed 2026-10-15** (`diagnostics/d32_flake.jl`, `d32_steps.jl`): on 1 thread the FD Jacobian is
  deterministic and the only randomness is ȳ (`rand(-9:0.01:9)` per output); `test_approx` falls back to elementwise
  `isapprox(rtol = 1e-2, atol = 1e-10)`. Over 20000 ȳ: "images, bmad_compat" fails 22.8%, all in the **γ** cotangent
  (FD column error 4.2e-3: `max_range = 1e-6` is an absolute step, 5e-9 of γ = 200, lost in the source-root noise);
  pz, q, geometry, s_kick 0 failures; free space 0%. FD error of the γ column vs `max_range` 1e-6/1e-3/1e-1: 4.2e-3/
  1.1e-4/1.0e-5 (compat), 7e-6/5e-9/8e-10 (free); s_kick 3e-7 (compat) at all steps. So D32 most likely removed the
  wrong input (inference: 3/3 passes after D32 has probability ~0.46 at a 23% failure rate). Fix: γ enters the
  wrapper as γ/Γ, Γ = 1e5 (pullback × Γ), s_kick restored: failure rate 1.5e-4 (3/20000, near-exact cancellations
  of ȳ⋅∂kick/∂γ); `ad_rrule_test.jl` 10/10 seeds on 1 thread, 3/3 on 4 threads (206 tests).
  **After D43 (2026-10-15):** compat 0/20000; free space 2e-4 (4/20000), hit once in a full 4-thread run. **Fixed by
  D44** (free space 0/20000; Section 9): not source-root noise but a branch ambiguity at the first-bin quadrature's
  first node.
  **Tolerances tightened 2026-10-16** (Section 9): the rates above used a too lenient grouping (whole geometry block
  by norm); with `test_approx`'s field-by-field grouping, (rtol, atol) = (1e-6, 1e-9) free, (3e-3, 1e-8) compat:
  0/20000 each.
- **Deferred from the 2026-10-15 `phase7-tps` review** (user: address in Phase 8 / Stage 2; numbers are the
  review's finding numbers):
  - (2, Phase 8) `csr_apply_test_kicks!` (`src/kick.jl`) copies `bins`/`kick` only when the test particles are on the
    CPU: a CPU bunch with GPU test particles passes CPU arrays to a GPU kernel; `z_t`/`alive_t` backends are not
    checked against `pz_t`; `pz_scale` is not passed through `device_scalar` (a `Float64` scale on Metal). Confirmed
    by reading, not run.
  - (4, Stage 2) The out-of-range error in `csr_apply_test_kicks!` is raised after the kernel has kicked the in-range
    particles; a caller that catches it and retries kicks them twice. Not documented. Confirmed by reading.
  - (8, Stage 2) Thread safety of TPS arithmetic. **Resolved 2026-10-16: not benign, a crash** (Section 9, Stage 2a).
    Nonlinear TPS operations segfault on several Julia threads (GTPSA C temporaries indexed by OpenMP thread number);
    `+`, `*` (all the test-kick kernel uses) do not. Fixed in CSR.jl by D45; GTPSA.jl report drafted.
- **Deferred from the 2026-10-16 Stage 2a review** (findings 1, 2, 4 fixed: D46, Section 9; numbers as reported):
  - (3) The D45 thread-safety fix is only exercised when the suite runs with more than one thread; no test asserts
    `Threads.nthreads() > 1` or starts threads itself. Fine while the `-t 4` run stays routine.
  - (5) `thread_safe_number` counts a `Dual` wrapping a TPS as safe (`Dual <: Real`); unsupported combination
    (`root_order` would `MethodError` there too). No action unless a clear error is wanted.
  - (6) `root_order(::TPS)` docstring/comments say "truncation order"; it returns the TPS's `mo` field, GTPSA's
    "Max order" (`GTPSA.jl/src/tps.jl` field list). If a lower truncation is set, extra Newton steps only cost time.
  - Stage 2b/2c: descriptor checks for test particles combined with a collective TPS kick, and for `track_csr!`
    (D46 exempts test particles).
- Float32 on GPUs limits strongly shielded kicks to ~3e-4 relative (cancellation); a mixed-precision table/convolve
  path would be needed for better.

**Upstream reports — drafted in `planning/upstream_reports.md`, not filed (the user files):** Bmad V1, V5, V7, V8, V2, V3, seam (items 6–8, 2026-10-15);
BeamTracking: the `R_to_gamma`/`R_to_v` ion bug only (SaganCavity items fixed in 0.8.7 or not reproducible, refreshed
2026-10-15). GTPSA.jl (2026-10-16): TPS on several threads segfaults (C temporaries by OpenMP thread number; cc
BeamTracking, whose `use_cpu_multithreading = true` crashes with TPS). GitHub searches (via WebFetch, titles, summarized: moderately reliable) found no existing issue
(2026-10-09, rechecked 2026-10-15). Bmad PR #2085 still open (2026-10-15); it moves but does not change any cited
site (line map in `upstream_reports.md`). V9 (small-angle path excess) drafted as Bmad item 5 (2026-10-14).

**Fixed 2026-10-15 (user: match BeamTracking v0.8.7; branch `sagan-tref`):** BeamTracking 0.8.7 (`8c302c6`,
2026-09-18) accumulates `t_ref` through a SaganCavity (`t_exit = t_enter + dt_ref`); CSR.jl's
`track_slice!(::SaganSlice)` (`sagan.jl`) reproduced the old reset (last slice: `t_exit = sagan_dt_ref(c, c.L)`), so
`track_csr!` with the cavity sliced (CSR on there) ended with `t_ref` short by the cavity's entrance `t_ref` (1.0007e-9
s after three 0.1 m bends; `diagnostics/sagan_tref_check.jl`). Unsliced cavities (CSR off) were tracked by BeamTracking
and correct. Coordinates were unaffected (the kernels use relative time: BeamTracking's `sagan_cavity` passes
`t_ref = 0`). Now every slice adds `sagan_dt_ref(c, s1) − sagan_dt_ref(c, s0)` (s1 = L at the last slice); after: 1e-24 s
from BeamTracking (round-off of the telescoped sum).

---

## 5. Code map

### Core (`src/`), no BeamTracking dependency — layout as BeamTracking.jl (since branch `reorg`, 2026-10-14)

Top level: types and host-side functions (launch kernels, allocate, check). `kernels/`: KernelAbstractions
`@kernel`s, `@inline` device functions, and the per-work-item physics the kernels call. `utils/`: helpers.

| File | Contents |
|---|---|
| `CSR.jl` | Module, exports (`CSRParams, FloorPosition, CSRGeometry, CSRElementInfo, CSRBins, CSRKickPoint, CSRStepResult, planar_survey, csr_bin_particles, csr_kick!, csr_kick_pz, csr_kick_coef, csr_n_step, dspline_len, track_csr!`), includes (types first: method signatures annotate them). |
| `types.jl` | `FloorPosition`, `CSRElementInfo`, `CSRGeometry`, `CSRKickPoint`, `CSRKick1`, `CSRBins`, `CSRStepResult`, `CSRParams` (with validation; docstring lists every compat difference). |
| `geometry.jl` | `CSRGeometry(L, g, centroid; ref_start, bmad_compat)`, `CSRGeometry(L, g, ref_exit, centroid)`, `CSRKickPoint(geom, ix, s_kick, γ; y_source, kick_factor)` (all with the D46 descriptor check), `primal` of `FloorPosition`/`CSRElementInfo`/`CSRGeometry`/`CSRKickPoint`. |
| `binning.jl` | `csr_bin_particles` (grid in the particles' float type; Int32 counter; serial path for non-atomic types; D46 check). |
| `tables.jl` | `csr_wake_table`, `csr_image_table` (CPU only). |
| `kick.jl` | `csr_convolve!`, `csr_image_convolve!`, `csr_apply_kicks!`, `apply_kicks_outside!`, `csr_apply_test_kicks!` (frozen-wake test particles, D33), `csr_kick!` (keyword `test_particles`; D46 check, D47 `pz` element-type check), `csr_kick_pz` (D46 check), `csr_binned_kick` (tables on `table_backend`, `to_backend`, layer weights), `csr_step_result`. |
| `adjoint.jl` | `csr_kick_adjoint`. |
| `param_adjoint.jl` | Geometry/`s_kick`/γ pullback (D30): `geometry_to_vector`/`geometry_from_vector`, `element_numbers`, `geometry_active_indices`, `LocalDual` slots, `ParamAdjointWork`, `node_adjoint!`, `nodes_adjoint!`, `middle_adjoint!`, `middle_excess` (local-geometry method), `hat_nodes`, `first_bin_adjoint!`, `csr_kick_param_adjoint`. |
| `reference.jl` | Serial reference (`csr_*_reference`), ground truth for the kernels. |
| `interface.jl` | Stubs + docstrings for `slice_element`, `csr_sliceable`, `track_csr!`. |
| `kernels/binning.jl` | `triangle_overlap`, `deposit_ix0`, `deposit_particle!`, `csr_deposit_kernel!`, `csr_edge_gradient_kernel!`. |
| `kernels/source_point.jl` | `ddz_calc_csr` (optional `dmid` offset of the middle-element sums, `middle_excess`), `s_source_calc` = `s_source_walk` (returns a `SourceOutcome`) with branch helpers `upstream_drift_source`, `downstream_drift_source`, `source_at`, `root_source`; `s_source_eval` (re-evaluate an outcome in another number type without searching); `PrimalSearch`, `primal_search`,
`s_source_calc(k, geom, kp, search)` (D31; the table kernels and image/wake quadratures take a trailing `search`); V4 literal, V7, V8, downstream search, infinite drifts with walk-back; `csr_kick1_like`. |
| `kernels/wake.jl` | `I_csr_value`, `I_csr_calc` (Bmad `I_csr`), `I_csr_integral`, `I_csr_quadrature`, `csr_wake_kernel!`, `csr_trapezoid_kernel!`, `csr_bin_quadrature_kernel!`. |
| `kernels/image_charges.jl` | `image_charge_kick`, `csr_image_kernel!`, hat integration (`image_kernel_at`, `image_simpson`, `image_segment`, `image_interval_moments`, each returning `(k, value)`, `csr_image_hat_kernel!`, `csr_image_hat_combine_kernel!`). |
| `kernels/convolve.jl`, `kernels/kick.jl` | `csr_convolve_kernel!`, `csr_image_convolve_kernel!`; `csr_apply_kicks_kernel!`. |
| `kernels/adjoint.jl` | `triangle_overlap_dzp`, `kick_cell`, `kick_adjoint_particle!`, five adjoint kernels. |
| `utils/math.jl` | `nint`, `modulo2`, `cube_root`, `cross3`, `dot3`. |
| `utils/backend.jl` | `csr_float_type`, `device_scalar`, `table_backend`, `to_backend`, `AtomicAdd`, `PlainAdd`, `supports_atomic_add`, `primal`, `thread_safe_number`, `cpu_workgroupsize` (D45), `descriptor_mismatch`, `check_descriptors` (D46). |
| `utils/spline.jl` | `spline_val/d1/d2`, `chord_spline`, `dspline_len` (Bmad, V1), `GL4_NODES/WEIGHTS`, `path_excess` (exact, V9). |
| `utils/floor.jl` | `x_axis`, `z_axis`, `advance_floor`, `advance_floor_patch`, `planar_survey` (D46 check), `centroid_floor`. |
| `utils/chord_coordinates.jl` | `s_ref_to_s_chord` (D40; Bmad's formula in compat), `chord_normal_offset`, `source_s`. |
| `utils/root_finding.jl` | `zbrent` (`super_zbrent`), `root_tangent` (Newton steps from the primal root, D35), `root_order`, `root_slope`, `RootTag`. |
| `utils/image_layers.jl` | `image_layer_weight` (D24), `image_u_min`, `image_hat_tol` (D20). |
| `utils/kick_setup.jl` | `csr_kick_coef` (V2), `csr_ref_energy`, `csr_n_step`. |

Every function has a docstring (the docs build enforces it with `checkdocs = :all`).

### Extensions (`ext/`)

| Extension | Contents |
|---|---|
| `CSRBeamTrackingBeamlinesExt` | `slicing.jl`: `slice_element` following Bmad `makeup_super_slave1` (fringe at ends, SciBmadStandard → Symplectic order, RungeKutta n_steps scaled, multipoles scaled, apertures at ends, alignment per slice, patch on first slice, RF voltage ∝ length, four-potential/`EMFieldParams` shifted, SaganCavity → `slice_sagan`; maps whole); `csr_sliceable`. `sagan.jl`: `SaganSlice` windows of the cavity's kernel sequence, `track_slice!`. `track.jl`: `track_csr!(bunch, ::Beamline / ::Branch)` → `track_csr_line!` (survey, centroid, `CSRGeometry`, step loop with half kicks at ends, `L_slice = i·ds − s0`, `csr_ref_energies`, `pz_scale` in SaganCavity; keywords `params`, `csr_on`, `centroid_start`, `sliced_centroid`, `diagnostics`, `test_bunch` (Phase 7 step 5: `check_bunches!`, `test_kick_particles`)). Uses BeamTracking internals listed in the files' imports. |
| `CSRChainRulesCoreExt` | `rrule` for `csr_kick_pz`. |
| `CSRGTPSAExt` | `primal(::TPS)` = scalar part, `root_order(::TPS)` = the TPS's max order (field `mo`), `modulo2` and `cube_root` for TPS, `descriptor_mismatch(::TPS{Float64,GTPSA.Dynamic})` (D46). |
| `CSRAbstractFFTsExt` | Empty stub. |

### Tests (`test/`)

| File | Covers |
|---|---|
| `runtests.jl` | Loads weakdeps, checks the extensions load, includes the files below. |
| `utils.jl` | `simpson`, `gaussian`, `dgaussian`, `steady_state_dpz_ds`, `circle_lattice`, `gaussian_line_bunch`. |
| `geometry_test.jl` | Survey conventions, modulo2, chord spline, `dspline_len` and `path_excess` vs quadrature, bend geometry. |
| `binning_test.jl` | Overlap sums, charge conservation, grid formula, Float32 grid, backend number types (mock no-Float64 backend), CSRParams validation. |
| `wake_test.jl` | zbrent, source point on a circle, exact vs small-angle path excess on a circle, first-bin integral, upstream infinite drift, V7. |
| `kick_test.jl` | Kick coefficient, csr_n_step, steady-state long bend vs analytic (< 2%), compat, half kick, dead particles, drift ≈ 0, errors. |
| `kernels_test.jl` | Kernels vs `reference.jl`: grid, wake table, source points exactly; charges/kicks ≤ 1e-12; 5 geometries × compat; blocking independence; Fortran nint; Float32 and BigFloat particles. |
| `shielding_test.jl` | Params; drift image kick = uniform-motion field (behind/ahead, downstream, past the end) and compat; V8; infinite drifts; hat table vs analytic; kernels == reference with images; blocking; steady-state bend free-space limit and shielding; averaged layer sum (weights, convergence). |
| `ad_test.jl` | rrule vs ForwardDiff vs finite differences (6 cases). |
| `ad_geometry_test.jl`, `ad_rrule_test.jl` | Forward mode w.r.t. geometry/`s_kick`/γ vs FD; rrules vs ForwardDiff (incl. `csr_kick_param_adjoint` vs ForwardDiff through `csr_bins_kick`, 8 cases) and ChainRulesTestUtils. |
| `tps_test.jl` | Phase 7: TPS `primal`/`root_order`; frozen-wake test particles (3000, threaded) vs ForwardDiff Jacobian/Hessian; Float64 test particles == bunch kicks; outside-range error. |
| `tps_collective_test.jl` | Stage 2a: TPS helpers; collective TPS in knobs (L, g, centroid, s_kick, γ, bunch) vs ForwardDiff Jacobian (per column), nested-Dual Hessian and order 3 (per block, since 10-16); free space, images, compat; `csr_kick!` with TPS `pz` == `csr_kick_pz` exactly; `pz` element-type error (D47); descriptor check with dynamic TPS (D46, last testset: leaves another descriptor current). Needs `kernel_test_bunch` from `kernels_test.jl`. |
| `slicing_test.jl` | Phase 2 gate: sliced == unsliced (CSR off) for all element kinds; controls. |
| `track_csr_test.jl` | Floor conventions, CSR off == track!, steady state via track_csr!, per-element selection, weights, patches/misalignment, errors, csr_ref_energy. |
| `lcavity_test.jl` | SaganCavity slices vs whole cavity; reference momentum steps; Branch; `pz_scale`. |
| `track_tps_test.jl` | Phase 7 step 5: `track_csr!` `test_bunch`. TPS order 2 vs ForwardDiff Jacobian and nested-Dual second derivatives (Beamline with a whole-tracked element; Branch with an lcavity); CSR changes the map; Float64 test particles follow bunch particles (both lattices); bunch unchanged by a test bunch; errors. Callable structs `TPSTestTrack`/`TPSTestJacobian` share Dual tags between the lattices (compile time). Needs `track_csr_test.jl`, `lcavity_test.jl` helpers. |

---

## 6. Physics and implementation notes

- Conventions (BeamTracking = Bmad): `pz = ΔP/P0`, `z > 0` ahead, coords N×6 `XI..PZI = 1..6`, `STATE_ALIVE = 0x1`.
- γ for the kick uses `R_to_E / massof` (BeamTracking's `R_to_gamma` is wrong for |Z| ≠ 1).
- Deposit atomics make multithreaded results differ in the last bits between runs; compare CSR-on runs with a tolerance.
- KA CPU backend: `@index(Global)` must be a standalone assignment (`@index(Global) + 1` fails); default workgroup on
  CPU covers the whole ndrange, so small ndranges need an explicit workgroup size (wake table uses 1).
- Atomix atomic add fails on ForwardDiff Duals; Duals use the serial deposit path.
- ForwardDiff ≥ 1.0 (checked 1.4.6): `==` compares partials, and `<`, `>`, `<=`, `>=`, `max`, `min` order Duals with
  equal values by their partials (`Dual(0, 1) > 0` is true). Every branch on a possibly-Dual value must compare
  `primal` values (else the branch depends on the seeded directions, and at ties differs from the Float64 run);
  `primal_max`/`primal_min` for value-producing max/min. `mod(Dual(0, 0), π)` has NaN partials: `modulo2` has a Dual
  method.
- GTPSA 1.5.6 descriptors (checked 2026-10-16 in `~/.julia/dev/GTPSA.jl/src`): `@vars(d)` returns **static**-descriptor
  TPS `TPS64{D}` (descriptor in the type; `ctors.jl:130-151`), `@vars(d, dynamic = true)` dynamic ones
  `TPS{Float64,GTPSA.Dynamic}`. Dynamic TPS made from a type or a plain number (`zero(T)`, `TPS64(5.0)`,
  `zeros(TPS64, n)`) take `GTPSA.desc_current` (`tps.jl:145-162`), which every `Descriptor(...)` call resets. MAD-NG
  refuses ops on two descriptors (`ensure(IS_COMPAT(...))` in `mad_tpsa_ops.c`; whether that throws or aborts not
  checked). `Descriptor(2, 2)` created twice returns the same descriptor (observed by spec-tester), so tests of
  "another descriptor" need another shape. A float array cannot store a TPS or `Dual` (no `convert(Float64, ::TPS)`).
- GPU (Metal): no Float64 and no 64-bit atomics. Float32 ranges carry Float64 internals (build grids elementwise).
  Float literals in kernels must take the argument's type. The table kernels need the CPU (`CSRKick1` immutable since D29, but
  non-isbits geometry).
- lcavity in SciBmad: RF element with `tracking_method = SaganCavity` in its own Beamline with `dE_ref` (a Branch
  splits it out). `dE_ref` is seen only by the first element of a Beamline, only used by SaganCavity. RFParams with
  `voltage = 0` count as inactive.
- SaganCavity slices (D22): only drifts/solenoids are split; the half ponderomotive and multipole kicks of each drift
  segment are point events at the segment ends (window rule s0 ≤ p < s1, ≤ for the last slice), so slices reproduce
  the whole cavity to round-off. Segment ends use the same expressions as the neighbouring events (`Lo + i·lc`).
- Image-layer series (D24): terms alternate, amplitude smooth with one sign change, tail ∝ 1/n (exact LW field at lag
  0 also ∝ 1/y). Plain truncation error ~|a_M|/2. Image source arc distance Ls ≈ (12ρ²y²)^{1/4} (γ → ∞ estimate);
  test lattices must be long enough that high layers' sources stay on the arc (use `circle_lattice(10.0, 0.1, 400)`).
- Shielding parameter Π = σ_z ρ^{1/2}/h^{3/2} (from memory, verify): 19 at 3 mm, 3.2 at 1 cm (strong shielding).

---

## 7. Findings in Bmad and BeamTracking

### 7a. Bmad (`bmad/space_charge/csr_and_space_charge_mod.f90`)

| ID | Where | Finding | Status in CSR.jl |
|---|---|---|---|
| V1 | `dspline_len` line 1796 | s⁴ term `9*c(3)**2` should be `/5`. Bmad's dL 1.71× quadrature for an asymmetric spline (end slopes 0.010, −0.012); exact when c₃ = 0. Write-up: `dspline_coeff.md`. | Superseded in default mode by `path_excess`; compat keeps Bmad. |
| V2 | Kick coefficient 769 (also 1141, 1609: LSC) | Derivation `ds r_e abs(Q) / (rel_mass e β² γ)`; Bmad `ds r_e abs(Q) / (rel_mass² e γ)`: extra `1/rel_mass` since commit `50b90dee33` (2020-03-24) replaced `r_e` by `classical_radius()` (= `r_e Q²/rel_mass`) and kept the explicit `rel_mass` (before it: `r_e/(rel_mass abs(Q))`, right mass, wrong charge for abs(Q) ≠ 1); no 1/β². Other `classical_radius` users (Touschek, envelope) have no extra `rel_mass`. | Fixed in default (D39); compat Bmad. Drafted as Bmad item 6 in `upstream_reports.md`. |
| V3 | `s_ref_to_s_chord` 1817–1842 | Treats `s_ref` as a straight distance along the entrance tangent and returns ρ × the angle seen from the bend centre: `ρ·atan(s/ρ)` on an on-axis bend instead of the chord coordinate `ρ[sin(s/ρ − θ/2) + sin(θ/2)]` (0.09967 vs 0.09996 at s = L, L/ρ = 0.1). Kick point upstream by `≈ 7L³/(24ρ²)` at the exit. | Fixed in default (D40); compat Bmad. Drafted as item 7. |
| Seam | `ddz_calc_csr` 1023–1086 | Same-element branch passes the secant slope as the angle of the L line (cross-element branches the angle), uses `2ds` for sources ahead (downstream: `2 L_plane`), and `L/(2γ²)` for `L(1 − β)` (infinite drifts: exact). `ddz` jumps at the kick element's ends (8.6e-9 m at s_kick = 0.17 in the `ad_rrule_test.jl` lattice, 1.1e-7 at 0.03): lags with two sources (old code: 163–197 of 201 lags in the overlap depend on the start element, `diagnostics/seam_uniqueness.jl`) or none (V4 branches). | Continuous in default (D41); compat Bmad. Drafted as item 8. |
| V4 | `s_source_calc` 951–959 | Round-off branches assign `ddz_calc_csr(...)` (a residual) to `s_source`. Matters for the image kick's `g_bend`. | Ported literally (D11). |
| V5 | `I_csr` first bin 1414–1426 | Holds `I_csr` constant over the rest of the bin: 2/3 of the integral in steady state, 3–4× too small in the entrance transient. Write-up: `upstream-bin.md`. | Quadrature (D15); compat flag. |
| V6 | `track1_bunch_csr` 330–331 | Energy interpolation weights reversed. | Fixed upstream (PR #2199); both modes follow it (D21). |
| V7 | `s_source_calc` 992 (`super_zbrent`; was "997", the `out_io` after the loop) | After `super_zbrent`, source geometry from the last evaluation, not the root. Catastrophic only when `ddz(0) == 0` exactly with CSR.jl's first-bin quadrature (I_csr 9.6e6 instead of −40). | Re-evaluated unless compat; regression test. |
| V8 | `ddz_calc_csr` 1077 | `floor_s%theta` = source direction + π for sources ahead (image kick only). Drift check: ahead sources 1.21, 8.99 (Bmad) vs −0.826, −0.111 analytic; true direction matches to 5 digits. Line from `ff57a4dda` (2016). | Fixed (D13); compat flag. Script `diagnostics/v8_drift.jl`. |
| V9 | `dspline_len` (all callers) | Small-angle path excess `½∫(x′ − dθ)²` drops the dθ⁴ terms: on a circle too large by Ls⁵/(1920ρ⁴) (1.8e-5, 1.9e-4, 1.9e-3 m at Ls = 3.2, 5.2, 8.2 m; ρ = 10 m). Acts as a lag shift; the image kernel near lag 0 at large y was ~14% low. | Exact `path_excess` in default mode; compat keeps Bmad. Drafted in `upstream_reports.md` (item 5). |

Other Bmad notes:
- At lag 0 the image kick on a drift is nonzero vs analytic 0 (y = 1 cm: 0.031 at γ = 200, 0.50 at γ = 50; at 1 mm
  lag 25 and 377): Bmad's approximate retardation `z ≈ L/(2γ²) + dL` (error `L/(8γ⁴)`), not V8. Default mode 0 to
  round-off since D41 (`diagnostics/drift_lag0_image.jl`).
- Seam: see the row above (D41).
- Constants: CODATA 2022 `r_e = 2.8179403205e-15 m` vs Bmad `2.8179403227e-15 m` (8e-10 relative).
- Bmad diagnostic file quirks: with images, kick and I_csr are divided by the last layer's `kick_factor` and
  `S_Source` is the last layer's; S-position has 6 decimals.

### 7b. BeamTracking (checked at `7d79938`; rechecked at 0.8.7 `65e343f` on 2026-10-15)
- Fixed in 0.8.7 (`8c302c6`, 2026-09-18): `universal!` for SaganCavity set `bunch.t_ref = dt_ref`, dropping the
  entrance `t_ref` (now `t_enter + dt_ref`); error messages interpolated an undefined `ele_name`; a stray `\` ended
  the FourPotentialParams check line. CSR.jl's `sagan.jl` followed the old `t_ref` reset until 2026-10-15 (Section 4,
  fixed).
- RFParams with `voltage = 0` count as inactive, so a zero-voltage SaganCavity errors ("without RFParams").
- Not reproducible with 0.8.7: "`bunch.p_over_q_ref` is not updated after a SaganCavity" (`track!` updates it:
  `diagnostics/sagan_tref_check.jl`; how it was concluded earlier is not recorded).
- `R_to_gamma` (`src/utils/energy.jl:6`) and `R_to_v` (`:9`) are wrong for |Z| ≠ 1 (He²⁺, R = 1: γ 1.00323 vs
  1.01286; v/c 0.16034 vs 0.15882; verified numerically). `R_to_v` feeds `rf_omega_calc`, which CSR.jl's lcavity code
  calls.
- Combined-function SBend with Bn1: bend–kick integrator with straight multipoles (warns), default `n_steps = 1`.
- Slicing changes BeamTracking's integration: default Symplectic takes `n_steps = 1` per element and each slice keeps 1
  step, so sliced and unsliced tracking differ (regression bend px 1.7e-6). Motivated D14.
- A particle's result depends on the bunch it is tracked in, in the last bit (2.8e-17 in px through a bend slice with
  pz ≠ 0; 191 of 400 particles differ from tracking alone, `diagnostics/track_tps_probe.jl`). BeamTracking uses
  explicit SIMD on the CPU by default (`src/kernel.jl:257`); with N = 397 only particle 397 matched single-particle
  tracking, consistent with SIMD lanes vs the scalar remainder (inference: which math function differs not checked).
  Not a bug; tests comparing test particles with bunch particles use a tolerance.

---

## 8. Validation: analytic checks and unverified items

Script `verify_analytic.jl` (Base Julia only): 33/33 checks pass (2026-10-01).

| Check | Relation | Result |
|---|---|---|
| C1 | Retardation `z = Ls − βL`; Bmad's `z ≈ L/(2γ²) + dL` has error `L/(8γ⁴)` | < 1% |
| C2 | Circle: `L = ρφ(1 − φ²/24)`, `dL = ρφ³/24 − ρφ⁵/1920` | < 1e-7 |
| C3 | Overtaking length `Ls = (24ρ²z)^{1/3}` (γ → ∞) | O(φ²) |
| C4 | Bmad `I_csr` on a circle → `−2·3^{−1/3} ρ^{−2/3} z^{−1/3}` for γφ ≫ 1 | 4e-6 |
| C5 | Bmad first-bin `I_int` (γ → ∞) `= −3^{2/3} ρ^{−2/3} z^{2/3}` | O(φ²) |
| C6 | First-bin `I_int` with the `ln(2γ²z/Ls)/γ²` term vs quadrature, γ = 30, 100 | ≤ 4e-4 |
| C7 | Bmad scheme converges to the design-doc §2.4 integral | 6e-3 → 6e-4, n_bin 100 → 1600 |
| C8 | `dspline_len` closed form vs quadrature of `(x′ − dθ)²/2` | exact with V1 |
| C9 | Upstream infinite-drift quadratic satisfies `Ls − βL = dz` | 1e-12 |
| C10 | Triangle overlap weights sum to 1 | 1e-15 |
| C11 | Reverse-mode adjoints (q̄, z̄, K̄) vs finite differences | 1e-10 |

Algebraic (by hand): C4 + Bmad prefactor ⇒ the steady-state formula with `2/3^{1/3}` and sign for electrons
(consistent with Bmad, not independent of it). `image_charge_kick_calc` has the Liénard–Wiechert structure
`coulomb1 = B·(N−β′)/(γ²L²κ³)`, `radiate = B·(N×((N−β′)×G))/(Lκ³)`, with `e/(4πε₀)` factored out — confirmed
numerically on 2026-10-14 against an independent exact-retardation LW field (`diagnostics/image_kernel_exact.jl`,
`image_large_y.jl`): CSR.jl's kernel equals the exact field to ≤ 3e-5 at y = 3 cm–1.9 m after V9.

### Not verified, and where to check (citations from memory unless marked; check before relying on them)

| Item | Where to verify |
|---|---|
| `I_csr` formula from first principles | D. Sagan & C. Mayes, "Coherent Synchrotron Radiation Simulations for Off-Axis Beams Using the Bmad Toolkit", IPAC2017, THPAB076 (URL in the Bmad source header: https://accelconf.web.cern.ch/ipac2017/papers/thpab076.pdf) |
| First-bin `I_int` log term analytically | Same paper; Bmad manual CSR section |
| Steady-state coefficient `2/3^{1/3}` independently | Derbenev, Rossbach, Saldin, Shiltsev, TESLA-FEL 95-05 (1995); Saldin, Schneidmiller, Yurkov, NIM A 398, 373 (1997); Murphy, Krinsky, Gluckstern, Part. Accel. 57, 9 (1997) |
| CSR impedance from retarded potentials | Stupakov & Zhou, PRAB 19, 044402 (2016) |
| Validity `σ_x ≪ (ρσ_z²)^{1/3}` | Derbenev et al. |
| Liénard–Wiechert field | Jackson, *Classical Electrodynamics*, 3rd ed., Ch. 14 |
| Shielding image geometry (`(−1)^n`, factor 2) and a shielding theory to compare against | Bmad manual CSR/space-charge chapter |
| Kick prefactor for ions (V2) | Sagan & Mayes; Bmad manual definition of macroparticle `charge`; maintainers |
| Enzyme + KernelAbstractions | Current docs |
| CSR vs LSC overlap | Bmad manual |

---

## 9. Results log, by phase

### Phase 0 — skeleton (10-01)
`Project.toml` with deps, weakdeps, 4 extension stubs, test target; `Pkg.test()` passes on Julia 1.12.7.

### Phase 1 — serial CPU reference port (10-01)
Line-by-line port of the Bmad CSR-only path (centroid chord/spline geometry, `dspline_len`, `s_ref_to_s_chord`,
triangle binning, `s_source_calc` with element walk + `super_zbrent` + upstream infinite drift, `I_csr`, causal
convolution as a direct sum, interpolated kick, kick coefficient) plus `planar_survey`. Results: circle source point
residual < 1e-10, geometry within 0.5%; steady-state kick within 2% of analytic (measured 0.28%, ρ = 10 m, σ_z = 1 mm,
n_bin 200); drift kick < 1e-6 of the bend scale.

### Phase 2 — BeamTracking integration (10-01/02)
`track_csr!`, `slice_element`, `csr_sliceable`; Bmad `makeup_super_slave1` slicing rules, each with a gate test
(sliced == unsliced ≤ 1e-15, CSR off) and a control. Floor conventions verified (field-free particle stays on a
straight floor line to 1e-14). 80 bends: steady state within 2%. Depends on BeamTracking internals (gate tests catch
changes). RungeKutta + `EMFieldParams` slicing added afterwards (`a5e0120`).

### Phase 4 — KernelAbstractions kernels (10-02)
Deposit (atomics), gradient, wake table (blocks of lags warm-started as Bmad; identical for any blocking), convolution,
kick. Kernels == reference: grid/wake/source points exactly, charges/kicks ≤ 1e-12. Timing (20001 particles, n_bin 200):
kernels 1.1 ms (1 thread) / 0.5 ms (4 threads); wake table dominates.

### lcavity mode (10-02)
`SaganSlice` windows of the cavity's kernel sequence; `track_csr!(bunch, ::Branch)`; γ linear between
E_ref − dE_ref and E_ref; Δpz × pc_lin(s)/P0c_step(s). One slice == whole cavity exactly; n slices ≤ 1e-18 traveling
wave, ≤ 1e-7 standing wave and multipoles (fixed by D22, Stage C). Cavity doubling the energy: 2 vs 8 cells agree to 5e-4 with the
rescale, 4% without.

### Type stability and latency (10-06)
JET/SnoopCompile: core type-stable; function barrier `track_csr_element!`; centroid bunch weight type matched. First
`track_csr!` 17–28 → 15 s (7 s BeamTracking's own first `track!`, ~8 s kernels specialized per slice; not reducible
from CSR.jl). No invalidations. No PrecompileTools workload (user: minimum precompile time).

### Phase 5 — shielding by image charges (10-06)
`image_charge_kick` (literal port), `csr_image_table`, `csr_image_convolve!`, layer loop; downstream source search and
downstream infinite drift (D12). Drift image kick = uniform-motion field within 5e-8 of the peak. V8 effect (σ = 1 mm,
h = 1 cm): max kick change 0.33 of max kick, net energy ≤ 6%. Cost per layer 2n_bin + 1 source solves.
(The "converged" 32-layer values reported then were truncation artifacts at small h; see D24 below.)

### Phase 3 — Bmad cross-check (10-06; directory `crosscheck/`, README there)
Tao generates the bunch; both codes track it; `bmad_compat = true`. Max |CSR.jl − Bmad| / max |Bmad|:

| Case | wake kick/m (free, shielded) | Δpz final (free, shielded) | I_csr from Bmad's orbit |
|---|---|---|---|
| pure bend, on axis | 1.5e-6, 2.1e-6 | 3.2e-6, 2.4e-6 | — |
| pure bend, centroid offset | 1.6e-6, 2.1e-6 | 3.2e-6, 2.4e-6 | — |
| regression bend (b1_gradient −10) | 9.9e-5, 1.1e-4 | 4.8e-5, 6.0e-5 | 2.8e-7 |
| Tao example (drifts + 2 combined bends) | 2.3e-5, 9.8e-5 | 1.8e-5, 1.0e-5 | 2.0e-6 |
| Tao example with patch | not meaningful (tracking) | — | 6.1e-7 |

CSR.jl reproduces Bmad's CSR to the precision of Bmad's diagnostic output; remaining end-to-end differences are
BeamTracking vs Bmad tracking. Default mode vs Bmad: final Δpz differs by 53% (single bend), 6% (Tao example), by
elimination V5.

**Rerun 2026-10-14** (`stage-c` after D22; BeamTracking 0.8.7, Beamlines 0.11.0, Bmad conda 20260918.0; result files
in `crosscheck/<case>/result.txt`, `geom_check.txt`). New section `shielded_default` (default mode with images: D20
hat kernel, D24 averaging, V8, V9). Max |CSR.jl − Bmad| / max |Bmad|:

| Case | CSR off, sliced (x, px) | wake kick/m compat (free, shielded) | Δpz compat (free, shielded) | Δpz default (free, shielded) | I_csr from Bmad's orbit |
|---|---|---|---|---|---|
| pure bend, on axis | 8.0e-15, 2.4e-15 | 3.6e-7, 4.9e-7 | 4.1e-8, 7.8e-8 | 0.48, 0.42 | — |
| pure bend, centroid offset | (as on axis) | 4.1e-7, 4.1e-7 | 4.1e-8, 7.8e-8 | 0.48, 0.42 | — |
| regression bend | 1.2e-8, 1.3e-7 | 1.1e-5, 9.3e-6 | 2.8e-6, 6.2e-6 | 0.49, 0.43 | 2.8e-7 |
| Tao example | 4.3e-8, 2.0e-7 | 9.6e-6, 2.1e-4 | 3.6e-6, 5.1e-6 | 0.046, 0.85 | 2.0e-6 |
| Tao example with patch | not meaningful (tracking) | — | — | — | 6.1e-7 |

- **Compat sections changed, but not because of CSR.jl:** the pre-session code (`468fc83`, run in a worktree against
  today's dependencies) gives the same compat numbers as `stage-c`. The change comes from BeamTracking: pure-bend
  tracking without CSR now agrees with Bmad to 8e-15 (was 1e-8), so the pure-bend Δpz improved 80× (3.2e-6 → 4.1e-8);
  regression bend wake 5.9e-6 → 1.1e-5 and Tao-example shielded wake 9.8e-5 → 2.1e-4 moved the other way, with
  Δpz still ≤ 6e-6. Reference floor geometry unchanged (≤ 5e-16).
- **Default mode, free space:** I_csr now differs from Bmad by up to 1.8% (pure bend, step 1; 1e-4 by step 10; Tao
  example 5.8%), was 1e-5. Cause: V9 at the bend entrance. With all sources on the upstream infinite drift,
  `dL = Lsz0 − Lz` is a small difference (2.5e-7 m) of lengths of order 1 cm, and Bmad's small-angle arc length to the
  kick point (`Lsz0`, spline slope ~0.06 in this 0.12 rad bend) is too long by 1.1e-8 m: far-source dL 4.6% too large
  in compat vs 1.9e-4 (finite-distance residual) in default, checked against the quadrature arc length. Δpz default vs
  Bmad unchanged (V5 dominates: 0.48–0.49 single bends, 0.046 Tao example).
- **Default mode, shielded:** 0.42–0.43 (single bends), 0.85 (Tao example) vs Bmad: D20, D24 (4 layers averaged),
  V8 and V9 together; for information only.

### Sliced centroid, D14 (10-06)
Regression bend vs Bmad: wake 9.9e-5 → 5.9e-6, Δpz 4.8e-5 → 3.1e-6 (shielded 1.1e-4 → 4.8e-6, 6.0e-5 → 3.4e-6).

### All-bin quadrature, D15 (10-06)
Kick error (max over |z| ≤ 2σ / peak), option A / B: steady state n_bin 50/100/200/400: 1.29e-2/4.89e-3/2.76e-3/1.82e-3
vs 1.29e-2/4.37e-3/1.96e-3/1.16e-3; transient s = 0.05 at n_bin 400: 0.050 vs 0.0026. Default vs Bmad Δpz: regression
bend 54.8% → 49.0%. `csr_kick!` 1.08 → 7.09 ms (1 thread).

### Phase 6 steps 0–2 — AD w.r.t. particles (10-09, `bb1dc13`)
`csr_kick_pz`; ForwardDiff through the kernels (serial deposit); `rrule` with explicit adjoint kernels. Constants under
differentiation: bin grid, `n_particle`, kick tables (∂/∂z of the min/max-z particle omits grid motion). rrule vs
ForwardDiff ~1e-15; ForwardDiff vs central FD: z ≲ 1.5e-9, q ≲ 1e-12.

### Stage B — open questions (10-09/10, `9030f17`)
D21 (V6 follows Bmad), D23 (Beamlines compat), D20 hat-integrated image kernel: per lag interval adaptive Simpson of
`(∫K, ∫K t)`, geometric split near u = 0 down to `0.01 y/γ`, tolerance `rtol·dz·max|K|`. Bugs found and fixed:
warm-started search on the upstream infinite drift did not walk back into the lattice; the point-table tolerance was
far too tight when dz ≫ y/γ (Simpson recursed to depth 40). Steady-state bend, 16 layers, h = 3 mm, n_bin 60…960:
point 2.68/3.41/4.88/0.365/0.355, hat 0.345/0.342/0.341/0.340/0.340 (16-layer values: not converged in layers).
Hat costs ~5× the source solves of point sampling.

### Phase 6 step 3a — forward-mode derivatives w.r.t. geometry, s_kick, γ (10-14, branch `phase6-geom_speed-deltas`)
- `FloorPosition{T}`, `CSRElementInfo{T}`, `CSRGeometry{T}`, `CSRKickPoint{T}`, `CSRKick1{T}` (immutable except
  `CSRKick1`, as before); promoting constructors; `number_type`, `table_type`; tables allocated in `T`;
  `csr_kick_pz` promotes over geometry, `s_kick`, γ, `kick_factor`, `pz_scale`.
- Source root: `zbrent` on primal values (Float64 results unchanged), then `root_tangent` (identity in core;
  `CSRForwardDiffExt`: `ds/dp = −∂ₚddz/∂ₛddz` with ∂ₛddz from a nested Dual under a private tag `RootTag` ordered
  inside any tag, so untagged `Dual{Nothing}` works). `k` is re-evaluated at the Dual root (default) or at zbrent's last
  point with the root's tangent (compat, V7). `ddz_calc_csr!` now stores `s_chord_source`.
- Bugs found on the way: ForwardDiff ≥ 1.0 `==` compares partials (`Dual(0, 1) == 0` is false). `advance_floor`'s
  `angle == 0` gave NaN for a Dual g at 0 (now `primal(angle) == 0` + second-order expansion); same fix in
  `I_csr!` (`s_chord_kick == 0`) and `y_source != 0`. First version kept `k` at the primal root in default mode:
  0.4% derivative error, found by FD (FD converged over h = 0.1…1e-5).
- Primal choices: `csr_n_step` rounds `primal(L)`; `image_u_min`, `image_hat_tol` primal.
- Checks (`test/ad_geometry_test.jl`, 106 tests): AD vs best-of-steps central FD. Free space ≤ 1e-5 (typ. 1e-8);
  images compat ≤ 1e-4; default images at kick level ≤ 1e-3 because the shielded kick's derivative is 7–25× smaller
  than its free-space part (cancellation amplifies FD noise of the hat quadrature); hat image table alone (rtol 1e-10)
  ≤ 2e-5 (measured ≤ 5e-6). Exact identity J[:,pz] = J[:,px]·(−px/(1+pz)) to ≤ 3e-10; residual of ddz at the Dual
  root has zero first-order partials; `advance_floor` at g = 0 vs analytic. FD outliers (centroid pz columns, L₁ at
  ~1e-4) traced to zbrent `abs_tol = 1e-8` noise in FD, not AD.
- Float64 path: 1141/1141 on 1 and 4 threads; `csr_kick!` 6.81 ms (free, was 6.94) and 70.5 ms (16 layers, was
  71.0), `diagnostics/bench_excess.jl`, 1 thread. Docs: `usage.md` "Derivatives" updated with an example; build clean.
- Not done: nested Duals (Hessians) in `root_tangent` (errors); GTPSA/TPS (Phase 7: needs Newton iterations per
  order); Duals on GPU backends.

### D28, D29 — ForwardDiff hard dependency, immutable `CSRKick1` (10-14, branch `phase6-geom_speed-deltas`)
- `CSRKick1{T}` immutable and isbits (Float64, Float32, Dual; tested). `with_fields(k; ...)` copies with fields
  replaced; `CSRKick1(geom, kp; ix_ele_source, dz_particles)`. Renamed (now non-mutating, returning `(k, value)`):
  `ddz_calc_csr`, `s_source_calc`, `I_csr_calc` (was `I_csr!`), `I_csr_quadrature`, `image_kernel_at`,
  `image_simpson`, `image_segment`, `image_interval_moments`. `zbrent` returns `(x, ok, x_last)`; compat (V7)
  re-evaluates `k` at `x_last` instead of relying on mutation. `ddz_calc_csr` is `@inline` (else ~3% slower).
- Diagnostics scripts updated (prototypes `image_proto.jl`, `image_layers.jl` hold `k` in a `Ref`); all rerun and
  reproduce their logged numbers.
- 1150/1150 on 1 and 4 threads (+9: isbits/`with_fields`). Timing, interleaved runs vs `8042ef1`, 1 thread:
  free space 7.29/7.19 vs 7.15/7.23 ms, 16 layers 74.7/76.6 vs 73.8/74.3 ms (noise ~3%; machine slower than the
  6.81 ms run above). No allocations in `s_source_calc`.

### Phase 6 step 3b — rrules w.r.t. geometry, s_kick, γ (10-14, branch `phase6-geom_speed-deltas`)
- `csr_kick_pz` rrule (`CSRChainRulesCoreExt`): particle part as before (`csr_kick_adjoint` now also returns the kick
  cotangent per bin `kbar`); geometry/`s_kick`/γ by `csr_kick_param_adjoint`: ForwardDiff gradient of `kbar ⋅ kick`
  through `csr_bins_kick` (split out of `csr_binned_kick`) with the bins fixed, on the CPU. Geometry cotangent is a
  `Tangent{CSRGeometry}` (no `bmad_compat` entry), built from `geometry_to_vector` order (4 + 26 n_ele numbers;
  `geometry_from_vector` the inverse). rrules for `CSRGeometry(L, g, centroid)` and `(L, g, ref_exit, centroid)`:
  transposed ForwardDiff Jacobian; keywords (`ref_start`, `kick_factor`, `pz_scale`) get no cotangents.
- Sparse seeding (`geometry_active_indices`): kick element (26), walked elements (11: floor0, theta_chord, L_chord,
  spline, dL_s), drift sums and `start`/last `floor1` when sources leave the lattice; range from source searches at
  the extreme lags. Assumes the source element is monotone in the lag. Pullback (n_bin 200, ρ = 10 m arc, 1 thread):
  dense 0.05 s (10 elements), 17 s (80); sparse 0.02 s (10), 2.6 s (80 and 400; 367 active numbers), kick 7 ms.
- Bugs found: (1) `mod` NaN partials at 0 → NaN derivatives for a kick at an element entrance (source on the boundary)
  and on-axis straight elements; (2) Dual comparisons at ties use partials → sparse and dense gradients differed by
  2.5% at a boundary root; single-direction AD and FD confirmed the sparse value. Fixed by primal comparisons in
  `source_point.jl`, `wake.jl`, `image_charges.jl`, `geometry.jl`, `chord_coordinates.jl`, the deposit
  (`primal_max`/`primal_min`) and the kick kernel. Float64 results unchanged (suite exact comparisons pass).
- Tests (`test/ad_rrule_test.jl`): rrule vs ForwardDiff ≤ 1e-12 (geometry vector, s_kick, γ; free space, images,
  compat); composed with the CSRGeometry rrules vs ForwardDiff over L (≤ 1e-6, cancellation), g, centroid, ref_exit
  (≤ 1e-9); sparse == dense in 5 cases (mid-lattice, entrance, upstream drift, downstream drift with images, compat);
  ChainRulesTestUtils `test_rrule` (z fixed via `rrule_f`; FD steps capped at 1e-6 because of curvature, rtol 1e-2
  because of the zbrent 1e-8 noise at those steps; `to_vec`/`test_approx` for `CSRGeometry` in the test file).
  ChainRulesTestUtils in `[extras]` (compat 1.13). 1355/1355 on 1 and 4 threads; docs updated, build clean.

### Cheaper geometry pullback, D30 (10-14, branch `geom-pullback`)
- Before: profile of the 80-element free-space pullback (3.4 s): 71% source search (zbrent 52%, element walk 19%) in
  Dual arithmetic whose partials were discarded; 29% root evaluation + `root_tangent`, dominated by `path_excess` over
  the elements between source and kick element. Cost ≈ one kick per seeded number (367 → 367×).
- `csr_kick_param_adjoint` rewritten (`src/param_adjoint.jl`): φ = k̄⋅kick = coef·(Σ Ī_int I_int + Σ W image); table
  cotangents from the transposed convolutions; each entry a fixed linear combination of evaluations at fixed lags
  (bin quadrature nodes, trapezoid, image lags, or the accepted adaptive-Simpson intervals of the hat quadrature, via a
  new `leaf` callback in `image_simpson`: Boole's rule on 5 nodes). Per evaluation: `s_source_walk` in Float64 →
  `SourceOutcome`; `s_source_eval` once in `LocalDual` (33 partials: s_kick, γ, 16 kick-element numbers, 9
  source-element numbers, start or last `floor1`, and two offsets `dmid` of the middle-element sums, at the root of the
  implicit derivative and at the evaluated source (they differ in compat, V7)); `middle_adjoint!` passes the offsets'
  cotangents to the middle elements (∂path_excess by a 6-wide ForwardDiff gradient; `dL_s + L_chord` on the drifts).
  On the local geometry the middle elements carry no partials, so `middle_excess` is linearized in θ_L
  (`M(θ₀) + M′(θ₀)(θ_L − θ₀)`, exact to first order). `I_int[1]` (quadrature nodes move with the parameters):
  ForwardDiff over `geometry_active_indices(...; lag = dz, n_images = 0)`. Coefficient: separate small gradient.
  Removed: `sparse` keyword (sparse/dense seeding through the whole table).
- Refactor of `s_source_calc` into walk + branch helpers: Float64 results unchanged (suite exact comparisons pass).
- Check vs ForwardDiff through `csr_bins_kick` (all numbers seeded), 8 cases (mid-lattice, trapezoid, entrance,
  upstream drift, images past the lattice end, images mid-lattice, compat free, compat images): max error relative to
  the largest gradient component ≤ 5e-12 (free space ≤ 2e-13); test bound 1e-10. The s_kick component in "images
  mid-lattice" is 5e-10 relative to itself, because it is 3000× smaller than the largest component (absolute 3e-20).
- Timing (`diagnostics/bench_pullback.jl`, ρ = 10 m arc of 0.1 m elements, kick in the last element, 1 thread),
  kick / pullback (ratio), before → after:

  | Elements | free space, n_bin 200 | 4 layers, n_bin 120 |
  |---|---|---|
  | 10 | 0.9 ms / 27.7 → 13.3 ms (32× → 14×) | 5.3 ms / 424 → 72 ms (81× → 9×) |
  | 80 | 9.4 ms / 3450 → 50 ms (367× → 5×) | 24 ms / 8062 → 163 ms (329× → 7×) |
  | 400 | 9.7 ms / — → 66 ms (7×) | 25 ms / — → 198 ms (8×) |

  Suite: 1359/1359 on 1 thread; 1358/1359 on 4 threads (the pre-existing `test_rrule` FD flake, Section 4). Primal
  tables bitwise identical to `90e0b77` (hash over 20 cases, 1 and 4 threads). Docs build clean (new warning:
  `internals.md` HTML over Documenter's size warning threshold).
  4 threads, 80 elements: 36 ms (free), 119 ms (4 layers), before the middle-sum linearization (13× then). Remaining
  cost: ~60% the local-Dual evaluations, ~13% the first-bin fallback (10-element profile).

### Phase 7 steps 1–3 (10-14 evening, branch `phase7-tps`)
- **Step 1 (D35, `a5f28e6`).** `root_tangent(f, s0, f0, dfds)`: `root_order(f0)` Newton steps `s ← s − (f(s) − r0)/f′`
  from the primal root, `r0 = primal(f0)` (primal root unchanged), constant primal slope (`root_slope` by a `RootTag`
  Dual wrapping the parameters' Duals, or `primal_slope` on a `PrimalSearch`'s geometry). Dual tables bitwise
  identical to before (16-case snapshot). Nested-Dual Hessians (previously an error) in (g1, g2, s_kick, γ), vs central
  differences of ForwardDiff gradients (best of h = 1e-3…1e-5): free space ≤ 8e-8; compat images ≤ 7e-6; default images
  at kick level 5e-4–1e-3 (adaptive-quadrature FD noise, as at first order); hat image table alone with rtol 1e-10
  ≤ 1.3e-6. Symmetric to ≤ 3e-11.
- **Steps 2–3 (D33, D34).** SciBmad convention checked: BeamTracking decides branches on the scalar part (`vifelse`,
  GTPSA compares scalar parts); no interpolated tables or smoothing anywhere. `CSRGTPSAExt`: `primal`, `root_order`.
  Kick kernel picks the cell from `primal(z)`. `csr_apply_test_kicks!` (test particles on their own backend; error if
  a live one is outside the bin range). TPS order 1/2 == ForwardDiff Jacobian/Hessian of the same Float64 function
  (rtol 1e-14/1e-13; 0 difference in a prototype). KernelAbstractions 0.9.45 CPU: ndrange ≤ 1024 is one workgroup,
  larger ones split into 1024-item groups, so the 3000-particle test runs TPS arithmetic on several threads.
- Suite 1446/1446 on 1 and 4 threads; docs build clean (TPS section in `usage.md` as a non-executed block: the docs
  environment does not load GTPSA; the example was run by hand).

### Review of `phase7-tps` and fixes (10-15, branch `phase7-review-fixes`)
- Review by the project subagent `sonnet-reviewer` (effort high), each finding checked by reading the code. No defect
  in the Newton scheme (error O(δ^k) → O(δ^(k+1)) per step with the constant primal slope; `root_order` correct for
  Float64, nested Duals of any tags, TPS `mo`) and no branch on non-primal values in the changed code. Known scope
  (finding 1): the TPS root path is unreachable until Stage 2 (`T<:Real` types, no GTPSA `mod` for `modulo2`).
- Fixed: (3) the out-of-range message printed `[zc[1] − dz/2, zc[end] + dz/2]`; the kernel accepts
  `[zc[1] − dz/100, zc[end])`, now in the message and docstring, with an edge test. (5) `root_order(x)` recursed to a
  `StackOverflowError` for non-Real numbers (e.g. `Complex`): now `root_order(x::Number)`, a `MethodError`. (6)
  `tps_test.jl`: Jacobian compared per column (a matrix norm is dominated by column 6, ∂pz/∂pz = 1); cubic term added
  to the test particles' z and all third derivatives (order 3 = truncation order) compared with nested ForwardDiff;
  dead test particle checked with `isequal` (GTPSA `==` compares scalar parts only).
- Added (7): `ad_geometry_test.jl` "residual stationary to second order": nested-Dual d²/dt² of `ddz` at the root along
  a direction moving g, s_kick, γ, relative to the same at the primal root: ≤ 1e-14 (bound 1e-10). Mutation check
  (`diagnostics/root_newton_mutation.jl`): with one Newton step (`root_order(Dual) = 1`) the ratio is 1.2–7, so the test detects a missing step.
- Deferred: findings 2, 4, 8 (Section 4).
- Checks (`4a6140f`): full suite 1484/1484 on 4 threads; on 1 thread 1483/1484 (the D32 flake), then 1484/1484 on
  rerun; docs build clean (existing warnings only). Then `2748fb9` fixed the flake (Section 4): `ad_rrule_test.jl`
  206/206 under 10 seeds (1 thread) and 3 seeds (4 threads); full suite not rerun after `2748fb9` (test file only).

### Reruns after D39–D41 and D42 (10-15, branch `reruns-d41`)
- **`image_kernel_exact.jl`:** all ratios 0.99997–1.00003 at y = 3 mm–1.92 m except lag 0 at y = 3 mm (0.945) and
  3 cm (1.0002), both near a sign change (Section 4). Lag −3 mm at y = 3 mm: exact (was −1.3e-4 vs −1.11e-3).
- **Bmad cross-check** (this machine: Bmad conda 20260918.0 at `~/miniconda3/envs/bmad`, `TAO` set). Compat sections:
  changes only in the 3rd digit of round-off-level numbers (CSR-off z 1.970e-15 → 1.972e-15, y rel. diff 2.34e-8 →
  2.35e-8, one Tao-example shielded kick 7.33e-6 → 7.34e-6). The old code `f3c69bf` on this machine gives the new
  numbers exactly, so the change is the machine (the 10-14 rerun was on the other one), not D39–D41. `geom_check.txt`
  unchanged.
- **Default sections, before → after D39–D42** (Δpz free, shielded; worst I_csr): single bends 0.48–0.49 → 0.48–0.50,
  0.42–0.43 → 0.42–0.43; I_csr 1.8e-2 → 2.4e-2–2.7e-2 (growing towards the bend exit, with s_source 5e-4: consistent
  with V3 moving the kick point, not checked term by term). Tao example 0.046 → 0.047, 0.85 → 0.85; I_csr 5.8e-2 →
  0.13.
- **Found:** default mode after D41 gave a kick on the Tao example's first drift (all sources on the upstream infinite
  drift) of 2.81e-11 /m, constant over steps 1–6, 2.2e-7 of the peak bend kick (Bmad 0; compat round-off), visible as
  kick ratios 1.26e297 (Bmad's 0 normalized by `floatmin`). Cause: the first-bin closed form
  `−kf((g Ls/2)² − log(2γ² zz/Ls)/γ²)` vanishes at g = 0 only for zz = Ls/(2γ²); D41 made zz exact, Ls/(γ²(1+β)), so
  the log gave 1/(4γ⁴) (`diagnostics/first_bin_straight.jl`: I_int[1] = 1/(4γ⁴) to 6e-5 in both branches, γ = 82, 1e3).
  Fixed by D42 (`wake.jl`): I_int[1] ≤ 2e-9 of 1/(4γ⁴); drift kick ≤ 1e-66 (Tao example), patch case ≤ 4e-19 of the
  peak (`diagnostics/drift_kick_default.jl`). The patch case's remaining kick ratio 9.7e12 is round-off (8.6e-23 /m at
  steps 3 and 6, also in compat in the non-patch case) over Bmad's ~1e-35.
- Test (`wake_test.jl`, "First-bin integral vanishes on a straight line (D42)"): 8 tests, 4 fail without the fix.
  Suite 1616/1616 on 4 threads (5m27s) and 1 thread (4m43s). Docs build clean (existing warnings); D42 row in
  `decisions.md`, item 9 in the `CSRParams` docstring. Cross-check result files rerun with D42.

### s_kick rrule residual (10-15, branch `s-kick-residual`)
- `v3_rrule_scan.jl` reproduces it: rrule vs ForwardDiff for s_kick 4.0e-11 (s = 0.19), 4.0e-10 (s = 0.23), 1 or 2
  image layers; γ ≤ 2.4e-14. `diagnostics/s_kick_residual.jl [s]`:
  - Free space alone: 4.4e-16 (s = 0.23), 1.3e-13 (0.19). The residual is in the image part.
  - Per image hat node (layer 1, 683 nodes, s = 0.23): node_adjoint! vs ForwardDiff of `image_kernel_at` differ by
    ≤ 2.7e-10 relative, largest on upstream-drift nodes near u = 4.5e-3 whose s_kick derivatives are ~1e6–1e7;
    weighted, they cancel to a total of 3.8e-5 (sum of the kick derivative).
  - ForwardDiff with its seed rescaled by c (d/dt f(s + c t)/c, exact in exact arithmetic), relative to the rrule:
    s = 0.23: −4.0e-10 to +4.7e-10 over c ∈ {1, 2, 3, 0.7, 1/3, 5, 0.1}; s = 0.19: −2.7e-11 to +5.2e-11; s = 0.13:
    −1.5e-11 to +5.7e-11. ForwardDiff's own round-off spread covers the rrule–ForwardDiff difference at each s.
- Conclusion: round-off in the image derivative (as in Section 4, issue (1): `1 − N·β′ ≈ 1/(2γ²)` cubed, root slope
  `O(1/γ²)`; attribution inferred, not measured term by term), amplified by the cancellation over nodes. Not a pullback
  error. A BigFloat reference was not usable: its root search stops at a different point (`zbrent` tolerance 1e-8 m),
  so it is not the same discretized function. No code change.
- **D43 (user's choice): image kernel without cancellation.** `image_charge_kick` computes `1 − N·β′`, `N − β′` and
  `B·(N − β′)` from `d = N − t` (`|N| = 1` exactly here: `|Δr|/L − 1` = 0 at all nodes), in both modes.
  `diagnostics/image_kernel_roundoff.jl` (Float64 vs BigFloat on the same inputs, layer-1 hat nodes, s = 0.23):
  `1 − N·β′` round-off 2e-11 → 4e-17, drift Coulomb term 5e-11 → ≤ 1e-12 (before the `B·(N − β′)` rewrite) and lower
  after; the worst node overall (1.3e-10 old and new) is a bend node where radiate and Coulomb cancel (−30, +29).
  - rrule vs ForwardDiff for s_kick (`v3_rrule_scan.jl`, 34 positions): ≤ 5.2e-10 → ≤ 3.6e-13; γ ≤ 8.9e-14.
    ForwardDiff's seed-scaling spread at s = 0.23: ≤ 4.7e-10 → ≤ 2.2e-13.
  - `test_rrule` FD flake (`d32_flake.jl`, 20000 ȳ, Γ = 1e5): compat 1.5e-4 → 0. Free space (unchanged code):
    2e-4, geometry columns of element 2 (`theta_chord`, `spline[1]`, `floor0.r[1]`) and element 1 (`floor0.r[3]`); FD column error ~1e-3, source-root
    noise). The first 4-thread suite run after D43 hit it (1614/1616, element 2 `theta_chord` and `spline[1]`).
  - Test (`ad_rrule_test.jl`, "s_kick with images, round-off (D43)"): s = 0.19, 0.23, rtol 1e-11; fails 2/2 with
    the old formula. Suite: 1618/1618 on 4 threads (4m35s); 1 thread 1616/1616 (4m46s, run before the new test was
    added). Cross-check rerun: all five `result.txt` and the `.dat` files identical to before. Docs build
    clean (existing warnings); D43 row in `decisions.md`.

### Free-space test_rrule flake, D44 (10-15, branch `fd-flake-d44`)
- `diagnostics/flake_geom_steps.jl [compat]` (FD error per geometry column vs `max_range`, test_rrule wrapper of
  `ad_rrule_test.jl`): five columns, element 2 `floor0.r[1]`, `theta_chord`, `spline[1]`, `spline[2]` and element 1
  `floor0.r[3]`, had FD error 3e-4–1.3e-3 at every `max_range` from 1e-7 to 1e-3, and the same with zbrent `abs_tol`
  1e-13 (temporary edit). So neither step size (D32's case) nor the root tolerance.
- One output row vs h (element 2 `theta_chord`): `F(x₀+h) − F₀ − J h` is O(h²) for some h and −4.45e-7·h (0.4% of
  the slope) for others, switching with round-off (h = ±1e-10 differ). The switch is the first-bin quadrature's first
  node (`I_csr_integral(zz, z)`, lag `cbrt(zz)^3`): its source search returned element 1 at `s = L_chord` (AD's
  branch) or element 2 at `s ≈ 0`. One point in a consistent geometry; perturbing one raw number of the kick element
  makes them differ to first order. (Real derivatives go through `CSRGeometry(L, g, centroid)`, consistent.)
- Fix (D44, user's choice): `I_csr_calc` keeps the entrance source from `ddz_calc_csr` and passes it as the first node
  (`I_csr_integral(...; k0)`). After: the five columns ≤ 2.2e-8 at max_range 1e-6 (all free-space columns ≤ 1.8e-6;
  compat ≤ 5.8e-6, except element 2 `ref_floor0.r[1]`, where AD gives 6e-19 and FD exactly 0);
  `d32_flake.jl free 1e5`: 0/20000 (was 4/20000). Primal kicks bitwise unchanged (4 kick points incl. the entrance).
- Test (`wake_test.jl`, "First-bin quadrature: one branch for each raw geometry number"): FD of I_int[1] w.r.t.
  element 2's `theta_chord` at h = 1e-9…1e-6 vs ForwardDiff, rtol 1e-5; fails 3/4 without the fix (1–2%). The
  `ad_rrule_test.jl` comment on FD accuracy updated (the ~1e-3 "source-root noise" was this).
- Suite 1622/1622 on 4 threads (5m11s) and 1 thread (4m52s); docs build clean (existing warnings). Cross-check rerun:
  identical except `csr_beam_tracking_patch` `free_space_default`, element 3 steps 1 and 5, kick ratio 0.815 → 0.584
  and 0.0137 → 0.0154: kicks there are ~1e-33 /m (1e-29 of the peak, `drift_kick_default.jl`), round-off on both sides.

### test_rrule tolerance tightening (10-15/16, branches `tighten-rrule-rtol`, `tighten-rrule-tol-atol`)
- Started by the user (`cddf927`): rtol 1e-2 → 1e-5 (free), 3e-3 (compat) from `d32_flake.jl` rates, and
  `test_rrule_seeds.jl` (the real `test_rrule` over RNG seeds; arg `verbose` added 10-16 prints the failing check).
- **`d32_flake.jl` was too lenient.** It passed the geometry block when the norm over all its entries passed.
  `test_approx` (ChainRulesTestUtils, `check_result.jl`) compares numeric vectors (pz, q) by norm but a `Tangent`
  field by field: each scalar field alone, `SVector` fields (r, spline) by norm; `_can_pass_early` ignores the
  kwargs (default rtol). Free space at rtol 1e-7: old estimate 0, real test 42/100 seeds (first failure element 2
  `ref_floor0.theta`, AD 7.2585845e-4 vs FD 7.2585871e-4). `d32_flake.jl` now groups as `test_approx` (`tangent`,
  default; `block` the old rule) and takes atol; with it free space 0.42 at 1e-7, 0.75 at 1e-8 (seeds: 42/100,
  72/100).
- Failure rates, 20000 ȳ, Γ = 1e5 (rtol, atol → P): free (1e-5, 1e-10) 0; (3e-6, 1e-10) 3e-4; (1e-6, 1e-10) 1.5e-3;
  (1e-6, 1e-9) 0; (1e-6, 1e-8) 0. Compat (3e-3, 1e-10) 5.5e-4 (the committed rtol; all 11 failures element 2
  `L_chord`, FD column error only 2.6e-6: ȳ⋅column cancels and the FD absolute error ~1e-8 exceeds both bounds, so
  no rtol fixes it); (3e-3, 1e-9) 5e-4; (3e-3, 1e-8) 0; (1e-3, 1e-8) 5e-5; (1e-4, 1e-8) 2e-3; (1e-4, 1e-7) 5.5e-4.
  Seeds (100 each, atol 1e-10): free 0 at ≥ 1e-6; compat 0 at ≥ 1e-3, 1 at 3e-4 and 1e-4.
- **Chosen (user):** free (1e-6, 1e-9), compat (3e-3, 1e-8). 200 seeds each: 0 failures.
- **atol/‖cotangent‖** (per `test_approx` group, AD value, over the 20000 ȳ; all-zero groups excluded, 33 free / 32
  compat): free median 6.3e-8, atol binds (atol > rtol·‖a‖) in 16% of comparisons: always for element 2 `g` (median
  ratio 8.5e-5, |∂kick| ≤ 1.1e-6) and `ref_floor0.theta` (5.1e-6), sometimes for element 2 `L_chord` (26%,
  5e-7), element 1 `theta_chord` (16%), `start.theta` (12%); max ratio 0.63. Compat median 1.2e-7, binds in 3e-4 of
  comparisons (element 2 `g`, `L_chord`: 0.2% each), max 0.19. So in free space the two small cotangents are checked
  to ~1e-4 and ~5e-6 relative, not 1e-6.
- `ad_rrule_test.jl` 208/208 on 1 and 4 threads; suite 1622/1622 on 4 threads (3m59s) and 1 thread (4m03s).

### Phase 7 Stage 2a review and fixes, D46, D47 (10-16, branch `stage2a-fixes`)
- **Review** (Claude, reading `src/utils/backend.jl`, `ext/CSRGTPSAExt`, `src/utils/math.jl`, `src/utils/root_finding.jl`,
  `src/kernels/source_point.jl`, `src/utils/chord_coordinates.jl`, `src/types.jl`, `src/tables.jl`, `src/binning.jl`,
  `src/kick.jl`, `src/kernels/kick.jl`, `test/tps_collective_test.jl`, `docs/src/usage.md`; no Bash, so no diff and no
  runs). Checked and fine: TPS `modulo2`/`cube_root`, `primal_search` for non-`Real` types, `root_tangent` step count,
  primal branches in the walk, every launch that can see a TPS runs as one work item or one workgroup (wake, bin
  quadrature, image, hat, trapezoid, hat-combine, edge gradient, both convolutions, apply-kicks), serial TPS deposit,
  adjoint kernels `Real` only. Findings in Section 1, item 11; deferred ones in Section 4.
- **Spec and agents.** `specs/stage2a-fixes.md` (one stage), implemented by `spec-implementer` in three runs (initial;
  dynamic-TPS descriptor test after the static/dynamic discovery; D47), accepted by `spec-tester`. Spec changes (both
  approved): (1) the descriptor test uses `@vars(d, dynamic = true)`, and docs/docstrings say only dynamic TPS are
  checked; (2) D47 added, Section 2's "CSRKickPoint MethodError" item removed.
- **MethodError diagnosis** (spec-tester, scratch only): `CSRKickPoint(geom0, 2, 0.17, 200.0 + x[1])` works (static and
  dynamic TPS, Duals; on HEAD and the working tree). `csr_kick!` with a `Vector{Float64}` `pz` and TPS `s_kick`, `gamma`
  or `q` fails at `src/kernels/kick.jl:21` (`pz[i] += …`: `convert(Float64, ::TPS)`), same with Duals and dynamic TPS;
  `csr_kick_pz` and a TPS `pz` work. GTPSA's promotion is correct (`GTPSA.jl/src/tps.jl:226`). → D47.
- **Changes** (8 files): `src/utils/backend.jl` (`descriptor_mismatch`, `check_descriptors`), `ext/CSRGTPSAExt`
  (dynamic-TPS method), `src/utils/floor.jl`, `src/geometry.jl`, `src/binning.jl`, `src/kick.jl` (checks as the first
  statement; D47 in `csr_kick!`; docstrings), `docs/src/usage.md` (one paragraph), `test/tps_collective_test.jl`.
- **Tests.** Hessian blocks: `err ≤ 1e-10·ref + 1e-14·max|H|` per knob pair (21), order 3 `≤ 1e-9·ref + 1e-13·max`
  per block (4), replacing the overall-max bounds. Worst err/bound per case: order 2 2.2e-3, 4.1e-4, 1.77e-2, 3.1e-3
  (free/compat, images/compat); order 3 4.3e-4, 1.92e-3; smallest nonzero ref/max 3.7e-10–1.06e-8; no zero blocks.
  Mutation `root_order(::TPS) = 1`: order 2 15/21 blocks fail in each case (err/bound 3e8–6e11); order 3 4/4 fail,
  weakest block (2,2,2) by 3.36× (no images) and 98.8× (images). New testsets: `csr_kick!` with TPS `pz` ==
  `csr_kick_pz` (exact, 4 tests); `pz` element type (D47, 8 tests: TPS `s_kick`, TPS `q`, Dual γ error; Float32 and
  Float64 `pz` run); "Descriptor check" (D46, 24 tests, last). Tester's extra scratch checks (54, `accept/extras.jl`):
  every named argument triggers, check order, fallbacks (static, complex TPS, Dual, Float32), BigFloat and Dual `pz`.
- Suite 1812/1812 on 4 threads (5m22s) and 1 thread (5m33s), Julia 1.13.2. Docs build: the two pre-existing warnings
  (`internals.md` size 175.5 KiB, was 173.7; no deployment environment).
- Not done: `decisions.md` ("Physics decisions" page) rows for D46, D47 (not physics; not requested).

### Phase 7 Stage 2a — collective TPS core, D45 (10-16, branch `stage2-tps`)
- **Recon.** Four changes suffice for a TPS kick in knobs: `<:Real` → `<:Number` in `FloorPosition`, `CSRElementInfo`,
  `CSRGeometry`, `CSRKickPoint`, `CSRKick1`; `primal_search` also for non-`Real` table types (TPS cannot be wrapped
  in the `RootTag` Dual of `root_slope`, so the primal search's `primal_slope` is used); TPS `modulo2` (shifts the
  scalar part); `cube_root` (GTPSA has no `mod`, `cbrt`, `copysign`, `trunc`/`round` for TPS; `atan(y, x)`, `log`,
  `sqrt`, comparisons on the scalar part exist). Deposit already serial for TPS; bin grid primal.
- **Threads (finding 8).** Segfault 3/3 on 4 and 8 threads, none on 1. `diagnostics/gtpsa_threads.jl` (pure GTPSA,
  4 threads): `/`, `c/t`, `sqrt`, `log`, `sin`, `exp`, `atan(y, x)`, `^(1/3)` segfault; `+`, `*`, `abs` run. Cause in
  MAD-NG `mad_tpsa_impl.h:168-186` (master `9f591769c`): temporaries at `d->t[tid*DESC_MAX_TMP + d->ti[tid]++]`,
  `tid = omp_get_thread_num()` = 0 on every Julia thread. `libgtpsa.dylib` (GTPSA_jll 1.7.1+0) links libgomp; descriptor
  `nth = 10` = `hw.ncpu`. BeamTracking 0.8.7 `track!` with 2000 TPS particles on 4 threads: segfault with
  `use_cpu_multithreading = true`, runs with the default `false` (`diagnostics/bt_tps_threads.jl`); so BeamTracking is
  safe only because CPU threading is opt-in. The frozen-wake test-kick kernel uses only `+`, `*`: safe by accident.
- **D45.** `thread_safe_number(T...)` (`T <: Real`), `cpu_workgroupsize(n, T...)` (`n` = one workgroup, which KA's
  CPU `__run` executes on the calling thread); applied to the edge-gradient, convolution, image convolution, kick,
  trapezoid and hat-combine launches; table kernels forced to one block. Adjoint kernels unchanged (Real only).
- **Checks** (`test/tps_collective_test.jl`, 68 tests; `diagnostics/stage2_tps_margins.jl`): knobs element 2 L and g,
  centroid x (row 3), s_kick, γ, bunch knob (q × (1 + v), z × (1 + 0.2v)); 400 particles, n_bin 40. vs ForwardDiff,
  relative to the largest entry: scalar part ≤ 7.5e-15; Jacobian columns ≤ 2.8e-13 (bound 1e-11); Hessian (nested
  Duals, all outputs) ≤ 4.9e-13 (bound 1e-11); order 3 in (g, γ) ≤ 1.7e-12 (bound 1e-10); free space, 2 image layers,
  compat. Mutation `root_order(::TPS) = 1`: Hessian errors 1.6e-2–4.3, order 3 0.07–0.42 (detected). 4-thread results
  bitwise equal to 1 thread. Timing (probe, 3 knobs, order 2, 400 particles): 8 ms free space, 0.24 s with 4 layers.
  (Hessian/order-3 bounds per block since 10-16, above.)
- Docs: `usage.md` GTPSA section (collective TPS example, not executed; run by hand in
  `diagnostics/stage2_docs_example.jl`), `csr_kick_pz` docstring. Suite 1690/1690 on 4 threads (5m26s) and 1 thread
  (5m41s), Julia 1.13.2. Docs build clean (existing warnings).
- Docstrings/docs reworded (user): `(docs, "Physics decisions", <label>)` instead of `planning/...` files;
  `decisions.md` "Where the details are" points to Validation. Comments: `planning/PLAN.md` → `PLAN.md`.

### SaganCavity slice `t_ref` (10-15, branch `sagan-tref`)
- `track_slice!(::SaganSlice)`: `t_exit = t_enter − sagan_dt_ref(c, s0) + sagan_dt_ref(c, last ? L : s1)` (Section 4).
- Tests (`lcavity_test.jl`): `lc_bunch` starts at `t_ref = 1e-6` (the slices-vs-whole test started at 0 and could not
  see the reset); Branch test: CSR off `t_ref` equal, cavity sliced `t_ref ≈` element-by-element `track!` (rtol 1e-14).
  With the old `sagan.jl`: 46 failures; with the fix 229/229.
- Suite 1608/1608 on 4 threads (3m53s) and 1 thread (4m13s).

### Upstream refresh (10-15, branch `upstream-refresh`)
- Bmad: fetched; `origin/main` still `7c9df4059`, all cited lines re-read and matching. PR #2085 fetched
  (`refs/pull/2085/head` = `3dd783a67`, 2026-07-04), still open per its GitHub page; every cited site unchanged in it,
  line map added to `upstream_reports.md`. In the PR `s_ref_to_s_chord` becomes a subroutine over wiggler sub-chords
  (bend branch, V3, unchanged); V4 branches and V7 unchanged.
- BeamTracking: `origin/main` = v0.8.7 `65e343f` (no newer commits). `unpack_bl.jl` changed since `fe92aac` (the
  draft's reference); `8c302c6` fixed the SaganCavity `t_ref` reset and `$ele_name`; `p_over_q_ref` is updated after a
  SaganCavity (`sagan_tref_check.jl`). Report reduced to `R_to_gamma`/`R_to_v` (lines 6, 9 and `utils_bl.jl:205`
  unchanged). Consequence for CSR.jl: Section 4 (open).
- Docs workflow: `actions/checkout@v4 → v7`, `julia-actions/setup-julia@v2 → v3`, `julia-actions/cache@v2 → v3`,
  `actions: write` and `workflow_dispatch` added, as in Documenter 1.19's hosting page (WebFetch summary). Latest tags
  by `git ls-remote`: checkout v7.0.1, setup-julia v3.0.2, cache v3.5.0. Release notes (WebFetch summaries): setup-julia
  v3 needs Node 24 and `version: min` changed meaning (we use `'1'`); cache v3 saves on job failure; checkout v7 blocks
  fork-PR checkout under `pull_request_target`/`workflow_run` (not used here). Not run on GitHub. The docs environment
  resolves registered BeamTracking 0.8.7 and Beamlines 0.11.0 (local docs Manifest; Manifests are gitignored).

### Phase 7 step 5: `track_csr!` with a test bunch (10-15, branch `phase7-step5`)
- **API.** `track_csr!(bunch, bl_or_branch; ..., test_bunch = nothing)`: a `Bunch` of any number type tracked through
  the same elements/slices (BeamTracking `track!` / `track_slice!`, incl. SaganCavity slices) and kicked through
  `csr_kick!(...; test_particles)` with the bunch's bins, kick and `pz_scale`; no deposit, no effect on the centroid.
  `check_bunches!`: BeamTracking's bunch check on both, then same species and `p_over_q_ref` (error). Test particles
  outside the bin range at a kick: the existing `csr_apply_test_kicks!` error.
- **Probes** (`track_tps_probe.jl`, `track_tps_lcavity_probe.jl`; 1 test particle, TPS order 2 vs ForwardDiff of the
  Float64 tracking): Beamline (drift, bends, quad) Jacobian ≤ 9e-16 abs (columns ≤ 9e-16 rel), Hessians ≤ 2e-15 abs
  (max |H| 2.3); Branch with bends + lcavity (dE_ref = 1e8, 4 cells): columns ≤ 1.8e-15 rel, Hessians ≤ 2.4e-12 abs
  (max |H| 630). Scalar parts vs Float64: 6e-17, 2e-17. TPS through `RefState{TPS}` in `track_slice!(::SaganSlice)`
  works without changes.
- **Float64 test particles vs bunch particles:** 3.5e-17, not bitwise. Traced to BeamTracking (Section 7b, SIMD): the
  first difference is px after a bend slice, before any CSR effect on px; re-tracking the snapshot slice reproduces
  it, the CSR kick kernel is identical for both.
- **Compile time dominates** (`track_tps_timing.jl`, `track_tps_compile.jl`): warm runs 7 ms for every number type;
  first calls 7–11 s (Float64, TPS, Dual), 55 s for nested Duals; BeamTracking `track!` alone 37 s per new
  nested-Dual type; each Hessian closure is a new tag (54 s again). Tests therefore use callable structs (one tag
  type for both lattices) and one nested pass (Jacobian of the Jacobian) per lattice: file 98 s → 50 s with both
  lattices' second derivatives. Finite differences are no reference at second order: with h = 1e-6 they cross bin
  centres (the kick is piecewise linear in z), entries off by up to 5e5 (tried, failed).
- **Mutation checks** (`track_tps_mutation.jl`): test kicks skipped → 4 failures (CSR-changes-map, both
  follow-the-bunch, out-of-range error); `pz_scale` ignored for test particles → 1 failure (lcavity follow-the-bunch).
- Docs: `track_csr!` docstrings, `usage.md` (keyword table, GTPSA section example, not executed; run by hand in
  `track_tps_docs_example.jl`), `validation.md` row. Docs build clean (existing warnings only).
- **Threads** (`track_tps_thread_spread.jl`, 4 threads, 8 runs): the TPS and ForwardDiff runs track separate bunches,
  and the atomic deposit order changes the frozen wake: lcavity Branch (4001 particles) Jacobian columns 1.9e-12,
  second derivatives 6.3e-15 of max |H|; Beamline (400 particles, one KA workgroup) 0. First full 4-thread run failed
  3 Jacobian checks at rtol 1e-12, then one Hessian check; bounds now Jacobian rtol 1e-10, Hessian atol
  1e-12·max|H| (rtol 1e-12).
- Suite 1602/1602 on 4 threads (4m12s) and 1 thread (4m23s); new file 5/5 runs on 4 threads (50–58 s each). Base
  `12ec910` on 1 thread: 1565/1565, 3m40s (so +43 s).

### V2, V3 and the seam, D39–D41 (10-15, branch `v2-v3-physics`)
- **V2 (D39).** `csr_kick_coef(species, γ, ds; bmad_compat)`, flag passed from `csr_bins_kick`, the reference and
  the pullback. Electrons change by 1/β² only (2.5e-5 at γ = 200). Tests: proton/electron = 1/rel_mass, compat =
  default·β²/rel_mass, positron == electron.
- **V3 (D40).** `s_ref_to_s_chord(s, e; bmad_compat)`, `chord_normal_offset`; 0 and `L_chord` exact at the ends.
  On an on-axis bend (ρ = 5, L = 0.5) the result equals the arc point's chord projection to the cubic spline's accuracy
  (≤ 1e-7 m; the spline deviates from the arc by ~ρφ⁴/384 = 1.3e-6 m), Bmad's formula is off by up to 1.4e-3 m. Kick
  point in the reference normal plane to 1e-15 with an off-axis centroid; first/second derivatives in s and the
  derivative in g vs central differences. The pullback seeds every kick-element number the new formula uses
  (`KICK_OFFSETS`).
- **Seam (D41).** After V3, `ad_rrule_test.jl` "images" failed (rrule vs ForwardDiff up to 5e-5). Old code at other kick
  positions: same (2.4e-5 at s = 0.13; `v3_rrule_scan.jl`). Traced (`v3_image_entry_adjoint.jl`, `v3_node_adjoint.jl`,
  `v3_warm_start.jl`) to hat interval m = 0: one hat node (u = 1.553e-4, layer 1) had sources in element 1 (end) and
  element 2 (start), image kernel 43 vs 3209 (the bend's curvature begins there); the forward table's warm start found
  one, the pullback's the other. Cause: same-element `ddz` passed the secant slope as the L-line angle (`path_excess`
  is stationary at the true angle, so the jump is ~(α³/3)² Δs/2 ≈ 8e-9 m). Fixed as D41. After: `ddz` jumps ≤ 1e-16 at
  the kick element's boundaries, infinite-drift sources at the lattice ends ≤ 1e-9 from the end for the end lag,
  start-independent sources (new test in `wake_test.jl`; with the old default code 163–197 of 201 lags near the
  boundary depended on the start element, `seam_uniqueness.jl`). rrule vs ForwardDiff over 34 kick positions: γ
  ≤ 2.3e-13; s_kick ≤ 5e-10 (unchanged, Section 4). Drift image kick at lag 0: 0 to round-off (compat 0.031 at
  γ = 200). `wake_test.jl` V7 test: `zz` now from `ddz_calc_csr`.
- Suite 1565/1565 on 1 and 4 threads; docs build clean (existing warnings). `bench_excess.jl`, 1 thread, interleaved
  with `da6650d`: free space 7.26–7.30 vs 7.22–7.25 ms, 16 layers 74.6–74.9 vs 73.4–73.9 ms (+0.5%, +1.5%).
- Not rerun: the Bmad cross-check default-mode columns (Section 9, Phase 3) and `image_kernel_exact.jl`; compat is
  unchanged by construction (compat tests pass). V2, V3, seam drafted as Bmad items 6–8 in `upstream_reports.md` (10-15).

### Forward mode with primal source search, D31 (10-14 evening, branch `fwd-primal-search`)
- `PrimalSearch(primal(geom), primal(kp))` built once per table call when the table type is a `Dual`
  (`primal_search`); `s_source_calc(k, geom, kp, search)` walks in `Float64` (`s_source_walk`) and evaluates once
  (`s_source_eval`). Threaded as a trailing `search` argument through `csr_wake_kernel!`, `csr_bin_quadrature_kernel!`,
  `I_csr_calc`/`I_csr_integral`/`I_csr_quadrature`, `csr_image_kernel!`, `csr_image_hat_kernel!`,
  `image_interval_moments`/`image_segment`/`image_simpson`/`image_kernel_at` (default `nothing` = old path).
  `image_hat_tol` on the primal copies. `first_bin_adjoint!` (pullback) uses it too.
- Check: `Dual` wake/image tables (values and all partials, 16 cases: default/compat, mid-lattice, entrance kick,
  upstream drift, images past the end, both table modes) bitwise identical to `5461eaa`. New test: primal search ==
  search in Duals (sources in the lattice, ahead, both infinite drifts).
- Timing (`diagnostics/bench_forward.jl`, ρ = 10 m arc of 0.1 m elements, kick in the last element, 1 thread), one
  pass with 12 partials, before → after (× the Float64 kick):

  | Elements | free space, n_bin 200 | 4 layers, n_bin 120 |
  |---|---|---|
  | 10 | 2.6 → 1.8 ms (6.8× → 4.6×) | 37.2 → 17.9 ms (7.6× → 3.7×) |
  | 80 | 105.5 → 37.9 ms (12.8× → 4.6×) | 256.9 → 95.2 ms (11.5× → 4.3×) |
  | 400 | — → 38.0 ms (4.5×) | — → 95.6 ms (4.2×) |

  1 partial: 1.7–2.5× → 1.6–1.9×. Pullback (same session, interleaved): 10 elements free space 10.4 → 9.1 ms; other
  cases within noise. Suite 1424/1424 on 1 and 4 threads (Julia 1.13.1); docs build clean.

### Stage C (10-14, branch `stage-c`)
- **Step 0.** Compat `Beamlines = "0.10.6, 0.11"`, `BeamTracking = "0.8.7"` (registered 0.8.5–0.8.7 contain
  `RungeKutta`; only 0.8.7 tested). 857/857 against dev BeamTracking 0.8.7, Beamlines 0.11.0.
- **GPU (Metal, M3, Metal.jl 1.11.1).** Float32 grid, Int32 counters, typed literals; then `backend.jl` (tables on the
  CPU in Float64, copied in `csr_float_type`). Full `csr_kick!` on `MtlArray{Float32}` vs CPU Float32: 2.7e-6 (free
  space), 3e-4 (32 layers, h = 3 mm; Float32 cancellation, the result is ~3e-3 of its terms). Metal 17 ms vs CPU 9 ms per
  free-space kick (tables on the CPU). Particle-side alone (bin + convolve + kick, 1M particles): Metal 6.9 ms vs CPU
  125 ms (4 threads; CPU Float32 atomics not investigated).
- **D24 investigation.** 8 m arc artifact: 1 cm terms constant (−7.78e-11) once sources leave the arc (n ≳ 185).
  40 m arc, 1024 layers, reference avg(8) (before V9): 1 + Σa = 2.8869e-3 (3 mm), 2.6444e-3 (1 cm), 4.1094e-2 (3 cm).
  Methods from the first M layers (relative error vs reference):

  | M | h | plain | avg(4) | avg(8) | Wynn ε |
  |---|---|---|---|---|---|
  | 32 | 3 mm | 13.8 | 3.7e-4 | 7.7e-7 | 1.1e-6 |
  | 64 | 3 mm | 0.45 | 4.0e-6 | 5.6e-9 | 5.5e-8 |
  | 32 | 1 cm | 3.2e-3 | 7.0e-6 | 7.2e-8 | 8.8e-8 |

  Per-bin kick (max over |z| ≤ 3σ): avg(8) at M = 32: 6.6e-7 (3 mm), 1.4e-6 (1 cm); plain at 16 layers 170× (3 mm),
  6× (1 cm). Data: `diagnostics/layer_terms_<h>_240_400.txt`.
- **D24 implementation.** `image_sum_order = 8`; layer weights `2^-k Σ_{j ≥ n−M+k} C(k, j)` (k = min(order, M),
  Σw = M − k/2) on `kick_factor` in `csr_binned_kick` and `csr_kick_reference!`; AD unchanged. The h = 3 mm
  coarse-vs-fine-bin test pins `image_sum_order = 0`.
- **V9, exact path excess.** Diagnosis (`image_large_y.jl`): CSR.jl's kernel = exact LW field at its own source
  position; source misplaced by the dropped Ls⁵/(1920ρ⁴). `path_excess = ∫ x′²/(1+√(1+x′²)) ds + 2 sin²(dθ/2) Δs −
  sin(dθ) Δx` (from L = ∫cos(θ − θ_L) ds_arc), Gauss–Legendre 4 points. After: kernel vs exact ≤ 3e-5 (y = 3 cm–1.9 m),
  circle excess ≤ 1e-8 relative (compat 4e-5–6.5e-4 = Ls²/(80ρ²)), image kernel at y = 3 mm, lag 6 mm
  1.0010 → 0.99999 of exact.
  Cost: `csr_kick!` 6.68 → 6.79 ms (free space), 70.4 → 70.6 ms (16 layers), 1 thread (`diagnostics/bench_excess.jl`).
  Default free-space I_csr changes up to ~7e-4 relative at long lags.
- **D22, exact SaganCavity split.** `sagan_segment!` in `sagan.jl`: a drift segment [a, b] applies its half kicks
  (ponderomotive for standing waves, multipoles with radiation) at a and b as point events with the full segment
  strength, and only the drift/solenoid is split. Sliced vs whole (max |Δv|, 3/4/7/20 slices): standing wave 2e-7 →
  ≤ 7e-19; n_cells = 0 and multipoles with radiation damping ≤ 9e-19; with a solenoid ≤ 2e-16 (BeamTracking's own
  Solenoid split once: 6e-17, the exact solenoid map's round-off). `lcavity_test.jl` bounds 1e-17 (1e-15 with a
  solenoid), new cases added. 1035/1035.

---

## 10. Roadmap, limitations, risks

### Phases
| Phase | Content | Status |
|---|---|---|
| 0 | Package skeleton | done |
| 1 | Serial CPU reference port (Float64) | done |
| 2 | BeamTracking integration (route a) | done |
| 3 | Bmad cross-check | done (rerun after D39–D42, 2026-10-15) |
| 4 | KernelAbstractions kernels | done (GPU: particle side + CPU tables, Stage C) |
| 5 | Shielding (image charges) | done (+ D20, D24, V9) |
| 6 | AD: steps 0–2 particles (done); step 3 geometry/γ via implicit differentiation of the source root | done (3a forward, 3b rrules, D30 cheaper pullback) |
| 7 | TPS: generic root (done), frozen-wake test particles (done), `track_csr!` with a TPS test bunch (done), full collective TPS (Stage 2, D37: 2a done + review fixes D46, D47; 2b, 2c open) (D33–D38); tests: first order = Jacobian, second = ForwardDiff Hessian | in progress |
| 8 | FFT convolution (`CSRAbstractFFTsExt`) for large n_bin; benchmarks; GPU wake table (isbits geometry via Adapt; work struct immutable since D29; errors/recursion in the source search); issue for a BeamTracking collective-step hook (route b); test-particle backend handling (review finding 2, Section 4) | not started |

Follow-ups not requested: per-element `ds_step` (Bmad `csr_ds_step`); caching slices per element (rebuilt every step);
Metal tests in the suite (conditional on Metal.jl).

### v1 limitations
- Planar geometry only: bends with `tilt_ref ∈ {0, π}`; vertical bends rejected.
- Patches with out-of-plane offsets rejected (Bmad rejects `floor_shift`).
- No LSC or transverse space charge; no wigglers.
- Requires live particles ≥ `n_bin` (Bmad behavior).
- Wake and image tables on the CPU only; GPU kernels in Float32 on Metal.
- Dynamic-descriptor TPS inputs must all be in GTPSA's current descriptor (D46).

### Risks
| Risk | Mitigation |
|---|---|
| Slicing does not reproduce unsliced tracking | Phase 2 gate tests |
| Per-slice `track!` overhead | Measure in phase 8; route (b) upstream if dominant |
| Dependence on BeamTracking internals | Gate tests; compat pinned at 0.8.7 |
| Dependence on GTPSA internals (`getdesc`, `desc_current`, D46) | "Descriptor check" testset; compat `GTPSA = "1.5.6"` |
| Steady-state coefficient not independently verified | Check a primary source (Section 8) |
| `|q|` placement for ions (V2) | Fixed by derivation (D39); coefficient ratios tested, no ion tracking test |
| Float32 on GPUs in strong shielding | Documented ~3e-4; mixed precision if needed |

---

## 11. Reference sources and planning files

### Sources (local checkouts under `~`)
- **Bmad** `~/bmad-ecosystem`: `bmad/space_charge/csr_and_space_charge_mod.f90` (line numbers as of 2026-10-01:
  `track1_bunch_csr` 137–404, `csr_bin_particles` 434–678, `csr_bin_kicks` 701–835 incl. images 813–826,
  `s_source_calc` 857–1003, `ddz_calc_csr` 1023–1086, `I_csr` 1368–1432, `image_charge_kick_calc` 1451–1494,
  `csr_and_sc_apply_kicks` 1513–1652, `dspline_len` 1770–1799, `s_ref_to_s_chord` 1817–1842); slicing
  `bmad/code/create_element_slice.f90`, `bmad/modules/bookkeeper_mod.f90` (`makeup_super_slave1` 1172–1509);
  reference cases `regression_tests/csr_and_space_charge_test/` (sbend L = 0.1, 42 MeV, 77 pC, σ_z = 0.899377 mm,
  n_bin 40; shielded n = 4, h = 0.01), `bmad-doc/tao_examples/csr_beam_tracking/`.
- **BeamTracking.jl**: `src/types.jl`, `src/kernel.jl` (`KernelChain`, `RefState`, `launch!`), `src/tracking_methods.jl`,
  `src/kernels/sagan_cavity.jl`, `src/kernels/runge_kutta.jl`, `src/utils/energy.jl`, `ext/BeamTrackingBeamlinesExt/`.
- **Beamlines.jl**: `src/beamline.jl` (reference energy, `dE_ref`), `src/branch.jl`, `src/misc.jl` (`EMFieldParams`),
  `src/bend.jl`.
- **GTPSA.jl** 1.5.6: `src/tps.jl` (TPS struct and fields, descriptor modes, promotion), `src/ctors.jl` (`@vars`),
  `src/descriptor.jl`, `src/utils.jl` (`getdesc`), `src/global.jl` (`desc_current`), `src/operators.jl` (`zero`).
  MAD-NG `src/mad_tpsa_ops.c` (descriptor compatibility `ensure`), `mad_tpsa_impl.h` (thread temporaries, D45).
- **Others**: AtomicAndPhysicalConstants.jl (`chargeof`, `massof`; CODATA 2022), SpaceCharge.jl (3D solver, not a
  dependency), SciBmad.jl. KernelAbstractions: `supports_float64`, `supports_atomics` (checked in 0.9.35–0.9.45).

### Planning files (Google Drive planning folder, Section 2; `planning/` in git until `8359698`)
| File | Contents |
|---|---|
| `PLAN.md` | This document. |
| `csr_gpu_ad_design.md` | User's original design notes (physics, numerics, KA sketches, AD adjoints); partly superseded; its KA sketches were untested. |
| `verify_analytic.jl` | Section 8 checks (Base Julia). |
| `dspline_coeff.md`, `upstream-bin.md` | Write-ups for V1 and V5. |
| `upstream_reports.md` | Drafted upstream reports (Section 4). |
| `diagnostics/` | Scripts (run with the test env; they include `~/.julia/dev/CSR.jl/test/utils.jl`): `v8_drift.jl`, `first_bin_vs_quadrature.jl`, `transient_*.jl`, `steady_state_convergence.jl`, `image_proto.jl`, `image_layers.jl`, `layer_terms.jl [n_layers] [n_ele] [h,...]`, `layer_shape.jl`, `layer_sum_methods.jl`, `layer_kick_profile.jl`, `image_kernel_exact.jl`, `image_large_y.jl`, `bench_excess.jl`, `bench_pullback.jl`, `bench_forward.jl`, `d32_flake.jl [compat\|free] [Γ] [rtol] [tangent\|block] [atol]`, `test_rrule_seeds.jl [free\|compat] [rtol] [n_seeds] [atol] [verbose]`, `d32_steps.jl`, `root_newton_mutation.jl`, `track_tps_probe.jl`, `track_tps_lcavity_probe.jl`, `track_tps_timing.jl`, `track_tps_compile.jl`, `track_tps_mutation.jl [none\|scale]`, `track_tps_docs_example.jl`, `track_tps_thread_spread.jl [n_runs]`, `sagan_tref_check.jl`, `v3_rrule_images.jl`, `v3_rrule_scan.jl`, `v3_hat_node_search.jl`, `v3_image_entry_adjoint.jl`, `v3_node_adjoint.jl`, `v3_warm_start.jl [s_kick] [layer]`, `seam_uniqueness.jl`, `drift_lag0_image.jl`, `drift_kick_default.jl [case] [ix_last]`, `first_bin_straight.jl`, `s_kick_residual.jl [s]`, `image_kernel_roundoff.jl [s]`, `flake_geom_steps.jl [compat]`; data `layer_terms_*.txt` (the `_400` files predate V9). Not yet copied there (session scratchpad, 2026-10-16, deleted when the session ends): the Stage 2a-fixes scripts `diag/r1–r8.jl` (MethodError matrix), `accept/margins.jl` (`MUTATE=1`), `accept/extras.jl`. |
| `../crosscheck/` | Bmad/Tao cross-check (README there). |
| `../docs/` | Documenter docs (`julia --project=docs docs/make.jl`; `docs/README.md`); pages Home, Usage, Physics model, Physics decisions, Validation, API, Internals. Deployed by `.github/workflows/Documentation.yml` from `main`. Keep "Physics decisions" in sync with Section 3 when physics changes. |
| `../specs/` (repo) | Specs for `spec-implementer`/`spec-tester` (Section 2): `stage2a-fixes.md` (D46, D47, per-block tolerances; with its "Spec changes" log). |
