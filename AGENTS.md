 # Working on CSR.jl

CSR-specific guide for contributors and AI coding agents: the package layout, how CSR couples to 
BeamTracking.jl, the rules for keeping new code compatible with BeamTracking, the physics 
conventions, how to verify changes and the current status. **The rules for how to work (mindset, 
verification standard, controls) are in `../SciBmad.jl/AGENTS.md`**, and the tracking-specific ones 
in `../BeamTracking.jl/AGENTS.md`. The SciBmad rules apply here strictly: nothing in this file 
overrides them, it only adds to them. Read all three before changing code, and keep this file up 
to date: when a session settles something (a convention, a pitfall, a measured result, an open 
question), add it here.

CSR.jl computes one-dimensional (line-charge) coherent synchrotron radiation kicks, ported from 
Bmad's `csr_method = one_dim`, with optional shielding by parallel plates (image charges). The core 
(package `CSR`) works on plain particle arrays with KernelAbstractions kernels. The extension 
`CSRBeamTrackingExt` makes CSR an effect of BeamTracking's 
`track!(bunch, bl; effects = (CSREffect(params),))`. CSR.jl is under development: the coupling to 
BeamTracking relies on a hook that exists only on local BeamTracking branches (see "BeamTracking 
coupling" and "Status").

## Additional rules for CSR.jl

- **Line length:** code lines ≤ 100 characters, 80 preferred; exceed 80 only where it avoids a wrap 
  (whole CSR package; in Beamlines/BeamTracking, new or changed lines only; long string literals 
  excepted).
- **Ask, don't guess:** missing context and physics choices are the user's decisions. Record each 
  decision with its date and reason (`~/.julia/dev/CSR-scratch/decisions.md`; physics ones also in 
  `docs/src/decisions.md`).
- **Reports of code changes:** what was wrong, what changed, whether and how it was tested, what 
  the user must do next.

## Environment

- Julia: `~/.juliaup/bin/julia` (1.13.4 as of 2026-11-18). Package compat is `julia = "1.10"`; 
  CSR itself has no CI test workflow, only `.github/workflows/Documentation.yml`.
- Development checkouts needed (see "Status" for the branches): `~/.julia/dev/Beamlines.jl`, 
  `~/.julia/dev/BeamTracking.jl`, `~/.julia/dev/AtomicAndPhysicalConstants.jl`.
- Scratch directory: `~/.julia/dev/CSR-scratch` (outside the repository) holds the test 
  environment, scripts, logs, serialized results and the development notes. Nothing of it goes 
  into the repository. Never commit a `[sources]` section.
- Test environment: `~/.julia/dev/CSR-scratch/env-test`, built by its `setup.jl` (develops CSR.jl, 
  Beamlines.jl, BeamTracking.jl and APC by path, adds Test, ForwardDiff, GTPSA, StaticArrays). Run 
  `setup.jl` again after changing CSR's dependencies or the checkouts' versions. As of 2026-11-18 it 
  does not resolve: the Beamlines checkout is 0.13.0, CSR's compat is 0.12.
- CSR suite: `julia --project=$HOME/.julia/dev/CSR-scratch/env-test test/runtests.jl` from 
  `CSR.jl` (about a minute). Run it with 
  `-t 1` and with several threads (the user's shell sets `JULIA_NUM_THREADS=8`): the kernels and 
  the binning (Atomix atomics) run multithreaded on the CPU.
- Test dependencies (`[targets]`): Test, BeamTracking, ForwardDiff, GTPSA. `test/runtests.jl` 
  includes every `*_test.jl` file in one `@testset "CSR.jl"`; a new test file must be added there.
- Docs: `julia --project=docs docs/make.jl` after developing the three checkouts into `docs` (see 
  `docs/README.md`). `checkdocs = :all`: every function needs a docstring, and `api.md`/`internals.md` 
  pick them up by file through `@autodocs`; a new file must be added to a `Pages` list there.
- Development notes, in `~/.julia/dev/CSR-scratch` (moved from `CSR.jl/.modwork/` on 
  2026-11-18): handoff `HANDOFF.md`, `spec.md` Section 9, `decisions.md` 1–40, regression scripts, 
  environments and logs. The notes still say `.modwork/` for paths; the absolute paths in its 
  scripts point to the new place. Tracked: `planning/`, `specs/`, `diagnostics/`, 
  `crosscheck/` (the last two still call the removed `track_csr!`).

## Package layout

### Core (`src/`), no tracking package needed

- `CSR.jl`: module, exports, includes. Dependencies: StaticArrays, AtomicAndPhysicalConstants, 
  KernelAbstractions, Atomix, Beamlines (for `scalarize` and conversions).
- `types.jl`: `FloorPosition`, `CSRElementInfo`, `CSRGeometry`, `CSRKickPoint`, `CSRKick1`, 
  `CSRBins`, `CSRStepResult`, `CSRParams` (`@kwdef`: `n_bin`, `particle_bin_span`, 
  `n_shield_images`, `beam_chamber_height`, `image_sum_order`, `all_bin_quadrature`, 
  `bmad_compat`), and the tracking-facing `CSREffect`, `CSRRecord`, `CSRReplay`.
- `geometry.jl` (`planar_survey`, `CSRGeometry` constructors), `binning.jl` (`csr_bin_particles`), 
  `tables.jl` (kick tables), `kick.jl` (`csr_kick!`, `csr_kick_pz`, `csr_kick_coef`, input checks, 
  test-particle kicks), `reference.jl` (serial reference implementation the kernels are tested 
  against).
- `kernels/`: KernelAbstractions kernels and the per-work-item physics they call (`binning.jl`, 
  `source_point.jl`, `wake.jl`, `image_charges.jl`, `convolve.jl`, `kick.jl`), as in BeamTracking.
- `utils/`: `math.jl`, `backend.jl` (`csr_float_type`, `device_scalar`), `spline.jl`, `floor.jl`, 
  `chord_coordinates.jl`, `root_finding.jl`, `image_layers.jl`, `kick_setup.jl`.

### One kick (`csr_kick!`)

`csr_kick!(pz, z, q, alive, geom, ix, ds, s_kick, gamma, species, params; binned, …)` bins the live 
particles, builds the kick table for the kick point (retarded source search along the centroid 
geometry; free space and/or image charges), convolves, and adds the kick to `pz` interpolated 
between bin centres. `z` is not changed. It returns a `CSRStepResult`. The kick tables are always 
computed on the CPU in `Float64` and copied to the device, as `Float32` where 
`KernelAbstractions.supports_float64` is false (`csr_float_type`).

### Extension (`ext/CSRBeamTrackingExt/`)

- `CSRBeamTrackingExt.jl`: imports from CSR, BeamTracking (`Bunch`, `quat_rotate`, 
  `rot_quaternion`, `STATE_ALIVE`, `ZI`, `PZI`) and Beamlines (`deval`, `isactive`, `scalarize`, 
  `R_to_E`, `SaganCavity`).
- `effect.jl`: `effect_setup`/`effect_kick!` methods for `CSREffect` and `CSRReplay`, and their 
  helpers (`csr_ref_curvature`, `csr_ref_exit`, `macro_weights`, `bunch_centroid`, 
  `centroid_bunch`, `csr_kicked`).
- `effect_setup(::CSREffect, bunch, bl; kwargs...)`: reference survey of `bl.line` (with patches), 
  centroid orbit (one particle tracked element by element with `track!(cb, ele; context, 
  notify = false, kwargs...)`, without CSR, in the elements' slices), `CSRGeometry`, macroparticle 
  charges `weight × |q|`. Empties `e.record`. Returns `(; geom, q, first_bend)`.
- `effect_kick!(::CSREffect, st, bunch, ele, sp)`: one `csr_kick!` at the slice entrance covering 
  `sp.L_slice`, γ from `bunch.p_over_q_ref` there; then `e.diagnostics` and `e.record`.
- Kicked (`csr_kicked`, decision 36): `L_slice > 0`, the element is a bend or after the first bend 
  of the line, its tracking method is not `SaganCavity`, and `e.on(ele)` is true.

## BeamTracking coupling

### The hook (BeamTracking, `src/collective_effects.jl`, on the `collective/*` branches)

- `track!(bunch, bl::Beamline; effects::Tuple = (), kwargs...)` calls 
  `effect_setup(effect, bunch, bl; kwargs...)` once per effect after the reference is resolved, 
  with the call's keywords except `notify` and `effects` (`scalar_params`, 
  `ramp_particle_energy_without_rf`, `ramp_update_each_particle`, `rf_on`, `batch_start`, any other 
  keyword). Methods must accept `; kwargs...`.
- Then `_track!` calls `apply_effects!` → `effect_kick!(effect, state, bunch, ele, sp)` at the 
  entrance of every slice of every element, before `universal!`, in tuple order. 
  `SlicePoint(ix, k, n_slices, s0, L_slice, L, s)`: element index in `bl.line`, slice number, number 
  of slices, slice entrance from the element entrance, slice length, element length, slice 
  entrance from the Beamline entrance (`bunch.s`).
- Effects run only in `track!(bunch, bl; effects)`: not in `track!(bunch, ele)` and not in prepared 
  chains (`prepare`, `track!(bunch, ::Vector{PreKernelChain})`). SciBmad's `track` uses `prepare`, so 
  CSR does not run through it.
- Tests: BeamTracking `test/BeamlinesExt/beamlines_effects_test.jl` (in `BeamlinesExt_test.jl`).
- No exports, no default methods. Names approved by the user (decision 35): `effects`, 
  `effect_setup`, `effect_kick!`, `SlicePoint`.

### Guidelines for compatible development

- **Keep the hook minimal** (decision 32): a generic collective-effect interface is being prepared 
  upstream (content unknown), and the hook must be replaceable by it. Put CSR logic in CSR, not in 
  BeamTracking. A change to the hook needs the user's decision first.
- **The hook is in BeamTracking's dynamic layer.** A change there follows BeamTracking's rules: 
  `effects`/`states` cross the barrier as tuples (types known), `isempty(effects)` must stay a 
  compile-time constant so that `track!` without effects is unchanged, and no `Val{runtime_bool}()`. 
  Check the BeamTracking bitwise snapshot (BeamTracking `AGENTS.md`, "Verification") and an ESR 
  one-particle `track!` A/B (alternated fresh processes) with and without the change.
- **Effects work host-side, between kernel launches.** BeamTracking runs each slice as one fused 
  per-particle kernel; a whole-bunch histogram can only be built between launches (decision D1). 
  Don't put CSR code into BeamTracking kernels.
- **GPU synchronization:** on BeamTracking `main`, `launch!` no longer waits for the kernel; 
  `track!` synchronizes once at its end (`synchronize_launches`). The hook branch predates this and 
  synchronizes per launch. When the hook is moved onto that code, an effect that reads coordinates 
  on a GPU must wait for the launches first. Not implemented; CSR tracking is CPU only for now.
- **Use only what BeamTracking/Beamlines define**, through the imports listed above. Look up every 
  name before use. Read element parameters with `deval(ele.<Group>, context)` and `isactive`, once per 
  element in `effect_setup` (`ele.X` costs ~40 ns per access; not per particle). Known gap: the 
  survey reads `ele.BendParams` even where `IgnoreParams` drops it (unverified review note).
- **Coordinates and conventions are BeamTracking's:** `bunch.coords.v` columns via `ZI`, `PZI`; 
  `z > 0` ahead of the reference particle; `pz = ΔP/P₀`; lost particles have `state != STATE_ALIVE` 
  and are excluded from binning and kicks; `weight = nothing` means 1 (as BeamTracking's IBS). Floor 
  conventions match BeamTracking's (tested in `effect_test.jl`).
- **Reference energy:** γ comes from `bunch.p_over_q_ref` at the slice entrance (`R_to_E`). A 
  `TimeDependentParam` reference with CSR is untested (open item).
- **Number types:** CSR accepts ForwardDiff `Dual` and GTPSA `TPS` coordinates and parameters, 
  `scalarize`s them at its entry points (Beamlines' recursive `scalarize`) and computes the kick in 
  real numbers. The kick changes only the value of `pz`; partials and higher TPS coefficients are 
  unchanged (decision 24: no derivatives through the kick). Keep this when adding entry points.
- **The centroid is tracked as the bunch is:** same keywords as the `track!` call (decision 40), 
  same slices. Any new reference-like tracking inside an effect must forward the keywords the same 
  way.
- **Version coupling:** CSR 0.1 compat is BeamTracking 0.10, Beamlines 0.12, APC 0.12. Registered 
  versions that meet it lack the hook and the recursive `scalarize`. Change the compat entries 
  together with the branches they need, and update README and `docs/src/index.md`, which name the 
  BeamTracking branch.
- **Not supported, by decision:** `Branch` tracking (37), CSR inside cavities (`SaganCavity` 
  elements are skipped, 30), bends out of the horizontal plane (error).
- **Error messages:** `"CSREffect: <what>"` / `"CSRReplay: <what>"`, naming the element index.

## Physics conventions

- The default fixes demonstrated errors or approximations of Bmad's implementation; 
  `bmad_compat = true` reproduces Bmad (D5). Where Bmad's intent is unclear, its code is ported 
  literally. Every difference is listed in `docs/src/decisions.md` (labels `D…` project decisions, 
  `V…` findings in Bmad's `csr_and_space_charge_mod.f90`, line numbers for bmad-ecosystem 
  `00b4557af`) with its measured effect. A new physics difference goes there, with a measurement, and 
  behind `bmad_compat`.
- Kick placement (decisions 29, 31): one kick at the entrance of each slice, `ds = L/n_slices`, 
  `kick_factor = 1`; first order in the slice length. The step length is set per element with 
  `n_slices`.
- The model, retardation, free-space and image kicks: `docs/src/physics.md`. Limits: planar 
  geometry, no space charge, at least `n_bin` live particles.
- Bmad's Fortran source is the reference for Bmad's conventions, not for correctness (SciBmad 
  `AGENTS.md`, "References are not ground truth").

## Verification

- **CSR suite** at `-t 1` and with several threads; all pass before a change is reported.
- **Kernels vs the serial reference** (`reference.jl`, `kernels_test.jl`): a kernel change must keep 
  them equal (tables and source points exactly, kicks to 1e-12).
- **Bitwise checks** for changes that should not change results: serialize the `pz` (and 
  `CSRStepResult`s) of fixed cases before and after, compare with `isequal`. Multithreaded binning 
  sums atomically in a varying order, so bitwise comparisons are made at `-t 1`.
- **Regression vs the old driver** (commit 44a4e72): `~/.julia/dev/CSR-scratch/s8/check.jl` (exit 1 on 
  failure; tolerances in `s8/tolerances.tsv` there, set by the user, decision 38).
- **Equivalence and analytic checks** are the strongest: steady-state kick in a long bend vs the 
  analytic wake (< 2%), no kick equals plain `track!`, one effect kick equals `csr_kick!`, linearity 
  in charge, record/replay. Each new test needs a negative control that fails by a large margin.
- **Hook changes:** also BeamTracking's suite (`BEAMTRACKING_TEST_GROUP=core` at least) and a 
  bitwise snapshot (BeamTracking `AGENTS.md`, "Verification"; stage 10 added a kick effect and 
  `CSREffect` cases). The snapshot scripts are scratch files, not kept: recreate them.
- **Bmad/Tao cross-check:** `crosscheck/` (not ported to the effect call; its numbers in 
  `docs/src/validation.md` are from the old `track_csr!` driver and are marked so).

## Status (2026-11-18)

Stages S0–S2f, S6–S9 are merged into CSR `main` (2353ed6, PR #1). Stage 10 (decision 40, the 
setup keywords) is on branch `collective/stage-10-effect-kwargs`, uncommitted (`docs/src/usage.md`, 
`ext/CSRBeamTrackingExt/effect.jl`, `src/types.jl`, `test/effect_test.jl`). Per the handoff 
(2026-11-16, not re-run since): CSR 629/629 at `-t 1` and `-t 8`, BeamTracking 1448/1448.

Branches CSR needs (no upstream, not pushed):
- BeamTracking `collective/stage-09-newcallbacks` (2cefb48: upstream `newcallbacks` 65cfa90, the 
  hook, `SlicePoint.s`, the setup keywords). `collective/stage-10-effect-kwargs` (f96dce1) does not 
  have the keywords commit.
- Beamlines `collective/stage-01b-restore-scalarize-test` (5886462: v0.12.1 + recursive 
  `scalarize`).
- AtomicAndPhysicalConstants `main` (0.12).

As of 2026-11-18 the BeamTracking and Beamlines working trees are on `main` (BeamTracking 8b0afbc, 
`newcallbacks` merged upstream, no hook; Beamlines 0.13.0, outside CSR's compat). The CSR tests 
need the branches above checked out. Moving the hook onto BeamTracking `main` and CSR onto 
Beamlines 0.13 is not done and needs the user's decision.

Open items:
- Code review of the hook (S6) and the CSR effect (S7): only partly done (it found decision 40).
- `bmad_compat = true` with images: isolated large kicks at the low-charge head bins (also in 
  44a4e72); cause not isolated.
- Time-dependent `p_over_q_ref` with CSR: untested.
- Old BeamTracking-level tests not ported (`test/MOVED.md`): lcavity, `Branch`, TPS maps of a test 
  bunch.
- `Documentation.yml` cannot build on GitHub (local branches, unregistered dependencies).
- `crosscheck/`, `diagnostics/` still call `track_csr!`.
- `docs/src/physics.md` says patches with vertical offsets are rejected; `advance_floor_patch` may 
  drop the y parts instead (not verified).
- Later: GPU tracking with `CSREffect` (`gpu-porter` assessment first), 3D CSR, package 
  replacements (decision 19).
