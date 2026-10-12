# Differentiable, GPU-native 1D CSR for the SciBmad ecosystem

Design notes: source material, analytic CSR, numerical CSR, KernelAbstractions integration.

## Status and trust levels

- **Read directly:** the Bmad Fortran CSR module (an *older* copy from the Cornell SVN `Defunct` path, not current `bmad-ecosystem` main), the `bmad-ecosystem`, `SciBmad.jl` and `SpaceCharge.jl` README pages, and `SpaceCharge.jl/Project.toml`.
- **Not read:** any source in `BeamTracking.jl`, `Beamlines.jl`, `GTPSA.jl`, or `SpaceCharge.jl/src` (GitHub blocked the `src` page for automated access).
- **Tags used below:**
  - `[CODE]` = seen in the Bmad Fortran I read.
  - `[MEMORY]` = from memory, not a primary source. Verify.
  - `[DERIVED]` = my own derivation. Cross-check.
  - `[UNTESTED]` = code written here, never compiled or run.
- No function or type names from SciBmad packages are given, because I have not seen them. Find them in the source.

---

## 1. Repositories to have source code for

Goal: the numerical conventions (signs, units, coordinates, charge, species) must match across all of them, or the kicks will be wrong without any error message.

### 1.1 Physics reference (the algorithm you are re-implementing)

| Repository / document | What to extract |
|---|---|
| `bmad-sim/bmad-ecosystem` (Fortran) | Current location of the CSR module (bmad-ecosystem/bmad/space_charge/csr_and_space_charge_mod.f90). Routines to read: `track1_bunch_csr`, `csr_bin_particles`, `csr_bin_kicks`, `s_source_calc`, `I_csr`, `image_charge_kick_calc`, `csr_and_sc_apply_kicks`. Also the regression tests (reference numbers to benchmark against). |
| Bmad manual, space charge / CSR chapter | Parameter meanings (`n_bin`, `particle_bin_span`, `ds_track_step`, `n_shield_images`, `beam_chamber_height`) and the stated validity limits. The section was numbered 21.4 in the April 2026 revision `[MEMORY]`. |
| Sagan & Mayes, IPAC2017 THPAB076 | Cited in the Bmad file header as the method paper. I have not read it. Check it for the exact kick formula and the off-axis (centroid-orbit) treatment. |
| `ChristopherMayes/OpenCSR` | Named in a Bmad source comment as the origin of the experimental 3D steady-state CSR. Relevant only if you later add a 3D model. |

### 1.2 SciBmad ecosystem (what you must plug into)

| Repository | What to extract |
|---|---|
| `bmad-sim/BeamTracking.jl` | (a) Coordinate convention: which of `z`/`pz` is stored, sign of `z`, whether the energy variable is relative to a reference. (b) The bunch/particle container and how it is laid out in memory (struct-of-arrays?). (c) How element tracking launches kernels and where a collective kick could be inserted between steps. (d) How lost or invalid particles are flagged. (e) How reference energy, species and charge are stored. |
| `bmad-sim/Beamlines.jl` | Bend parameters (strength, length, edge angles), element placement and orientation, and whatever gives the global geometry. You need these to build the centroid path geometry (Section 3.2). |
| `bmad-sim/SciBmad.jl` | Top-level integration, examples, docs. Check how a user would request an effect on an element or beamline. Paper: Signorelli et al., IPAC'26 THP5325 (linked from the README). |
| `bmad-sim/SpaceCharge.jl` | The mesh/grid struct, deposit, field solve and interpolate routines, GPU examples, tests, benchmarks. MIT license, v1.2.0 per `Project.toml`; deps are AbstractFFTs, Atomix, FFTW, KernelAbstractions. Questions to answer from the source: Is the deposit atomic-based? Does it define any AD rules? Which shape function does it use? What are its units for charge and field? |
| `bmad-sim/GTPSA.jl` | How TPS values are constructed (descriptor handling), what `zero`/`one`/`promote` do for a TPS, whether TPS values can live in plain Julia arrays used by kernels, and whether anything runs on GPU. `[MEMORY]`: it wraps a C library, so I expect CPU only. |
| `bmad-sim/AtomicAndPhysicalConstants.jl` | Classical radius, species mass and charge, and their units. This feeds the kick prefactor (Section 2.5). |
| `SciBmad-Distribution` README | Lists `TPSAInterface` as part of the core bundle. Look at it for how generic code is written over TPS and plain numbers. |

### 1.3 Generic dependencies

URLs are from memory. Verify them.

- `JuliaGPU/KernelAbstractions.jl` (docs: kernel language, `@atomic` via Atomix, backends).
- `JuliaGPU/Atomix.jl`.
- `JuliaDiff/ChainRulesCore.jl` (custom reverse-mode rules).
- `JuliaDiff/ForwardDiff.jl` (dual numbers through generic kernels).
- `EnzymeAD/Enzyme.jl` (reverse and forward AD of kernels; check its current KernelAbstractions support).
- `JuliaMath/AbstractFFTs.jl` and your GPU FFT provider (for example the CUDA.jl FFT bindings).
- `JuliaGPU/Adapt.jl` (moving structs to the device).

### 1.4 Consistency checklist (resolve before writing kernels)

1. Sign of `z`: Bmad uses `z > 0` for particles ahead of the bunch center `[CODE]`. Confirm that BeamTracking uses the same convention. If not, flip the sign in the wake table, not in the kernels.
2. Meaning of the energy variable (`vec(6)` in Bmad is a relative momentum deviation `[CODE]`) versus BeamTracking's.
3. How `z` must be updated after an energy kick. Bmad rescales `z` by `β/β0` in the space-charge branch `[CODE]`.
4. Units of charge per macroparticle (Coulombs in Bmad `[CODE]`) and the species/charge-state handling.
5. Whether the reference energy changes along the element (RF). Bmad interpolates energy linearly across a cavity `[CODE]`.
6. How many particles per bin you will have. Bmad aborts if live particles < `n_bin` `[CODE]`.

---

## 2. CSR calculation, analytic

### 2.1 Scope and assumptions

The model is the **1D line-charge model**:

- The bunch is a line charge with longitudinal density `λ(z)`.
- The wake is longitudinal only. Transverse CSR forces are ignored (this is what the Bmad 1D routine applies, `[CODE]`: only the energy variable is kicked).
- The geometry is a planar trajectory. In Bmad it is the **bunch centroid** orbit, not the design orbit.
- Free space, or parallel plates via image charges.

Validity criterion (Derbenev) `[MEMORY]`: the transverse size must satisfy `σ_x ≪ (ρ σ_z²)^{1/3}`. Verify the exact form before using it as a test.

### 2.2 Pairwise force from retarded fields

For a source charge `e` and a witness particle, the retarded condition and the Liénard–Wiechert field are:

$$ c\,(t - t') = L = |\mathbf r_k(t) - \mathbf r_s(t')| ,\qquad \mathbf n = \frac{\mathbf r_k - \mathbf r_s}{L},\qquad \kappa = 1 - \mathbf n\cdot\boldsymbol\beta_s $$

$$ \mathbf E = \frac{e}{4\pi\varepsilon_0}\left[ \frac{(\mathbf n - \boldsymbol\beta_s)}{\gamma^2\,\kappa^3 L^2} + \frac{\mathbf n \times \big((\mathbf n - \boldsymbol\beta_s)\times \dot{\boldsymbol\beta}_s\big)}{c\,\kappa^3 L} \right] $$

The first term is the velocity (Coulomb) field and the second is the radiation field. The energy change of the witness per unit path is the field projected on its velocity:

$$ \frac{dE_k}{ds_k} = e\,\hat{\boldsymbol\beta}_k\cdot \mathbf E $$

The Bmad image-charge routine evaluates exactly this structure (`radiate` plus `coulomb1`, with the acceleration vector `β² g` times a unit vector) `[CODE]`, which is a useful cross-check for your own implementation. All of this is standard Liénard–Wiechert; the assembly above is `[DERIVED]`.

### 2.3 Retardation geometry (the key relation)

Let `s_k` be the witness arc position and `s'` the retarded source arc position. The source moves `βL` along its path during the light travel time. If `z` is the separation at equal time (witness ahead of source), then:

$$ z = L_s - \beta L ,\qquad L_s \equiv s_k - s' \ \text{(path length)} $$

Using `β ≈ 1 − 1/(2γ²)` and `dL ≡ L_s − L`:

$$ z \;\approx\; \frac{L}{2\gamma^2} + dL $$

This matches the residual Bmad drives to zero in `s_source_calc` (`ddz_this = L/(2γ²) + dL − dz`) `[CODE]`. The same equation is the root-finding problem in the numerical method.

### 2.4 Steady state in a circular bend (closed form)

For a circular arc of radius `ρ`, source behind the witness by angle `φ` (small), ultra-relativistic:

$$ L \approx \rho\varphi\Big(1 - \frac{\varphi^2}{24}\Big),\qquad dL \approx \frac{\rho\varphi^3}{24},\qquad z \approx \frac{\rho\varphi^3}{24} + \frac{\rho\varphi}{2\gamma^2} $$

For `z ≫ ρ/γ³` the `γ` term drops, giving the overtaking length `[DERIVED]`:

$$ L_s = (24\,\rho^2 z)^{1/3} $$

The steady-state free-space wake of a line charge is (Derbenev et al. 1995; Saldin et al. 1997; Murphy–Krinsky–Gluckstern 1995; all `[MEMORY]`, I have not re-read them):

$$ \frac{dE}{ds}(z) \;=\; -\frac{2}{3^{1/3}}\,\frac{N e^2}{4\pi\varepsilon_0}\,\rho^{-2/3} \int_{-\infty}^{z} \frac{\lambda'(z')}{(z-z')^{1/3}}\,dz' $$

with `λ` normalized to 1, `N` the number of electrons, and `z > 0` ahead. **Verify the coefficient `2/3^{1/3}` and the sign convention against a primary source** (Stupakov & Zhou, PRAB 19, 044402 (2016), derives CSR impedance from retarded potentials; it was in my search results but I have not checked their coefficient).

Sign plausibility check `[DERIVED]`: for a bump-shaped `λ`, the tail and center (`λ' > 0`) lose energy and the head (`λ' < 0`) gains. That is the expected qualitative behavior.

### 2.5 Other analytic ingredients

- **Transients at bend entrance and exit.** Closed forms exist (Saldin et al. 1997, `[MEMORY]`). You do not need them: the numerical geometry in Section 3 includes upstream drifts and bends exactly, because the source point is searched along the real centroid path. Bmad treats the start of the lattice as an infinite upstream drift `[CODE]`.
- **Shielding by parallel plates.** Image charges at vertical offsets `n·h` with sign `(−1)^n`. Bmad sums `n_shield_images` images, with an extra factor 2 for images above and below, and half weight at the ends of the element `[CODE]`. Check the manual for the exact geometry (beam on the midplane?).
- **Kick prefactor.** The Bmad code uses:

  `coef = ds · r_c / (rel_mass · e · |q| · γ)`

  multiplying the slice charge (Coulombs) times the kernel `[CODE]`. My dimensional analysis `[DERIVED]` gives the witness charge `|q|` in the **numerator**: `δ = |q| · r_c · Q · I · ds / (e · rel_mass · γ)`. The two agree for `|q| = 1`. **Check which is right for ions before generalizing.** Unit-test the prefactor against the analytic result in Section 2.4 for an electron bunch.
- **Straight sections.** With zero curvature, `dL = 0` and the angles vanish, so the 1D CSR term vanishes. This is a free sanity test.

### 2.6 What the analytic model cannot give you

- Non-steady-state, arbitrary lattices, off-axis centroids: no closed form. Use the numerical method (Section 3).
- 3D/transverse effects: out of scope. Use a 3D method (OpenCSR-style, or SpaceCharge.jl-type mesh solves for space charge).

---

## 3. CSR calculation, numerical

Structure follows the Bmad 1D algorithm `[CODE]`, re-organized so every stage is a batched, branch-light kernel with a clean derivative.

### 3.1 Pipeline per tracking step

1. **Geometry pre-pass (once per beamline, before the particle loop).** Track a single centroid particle through the lattice; build per-element path geometry (Section 3.2). Bmad does the same: it requires a precomputed centroid orbit `[CODE]`.
2. **Wake tables (once per step, depends on geometry and γ only).** `K[m]` for lags `m = 1..N_bin`, from a batched root-find (Section 3.4).
3. **Bin** particle charge onto a 1D `z` grid, giving `ρ_j` (Section 3.3).
4. **Differentiate** `ρ` to get `g_j` (density gradient at bin edges).
5. **Convolve:** `k_i = c · Σ_{j≤i} K_{i−j+1} · g_j`.
6. **Interpolate** `k` back to particles and add to the energy variable. Update `β`/`z` if your coordinates require it (Section 1.4, item 3).
7. **Repeat** for `n_step = max(1, nint(L/ds))` steps per element, with half-weight kicks at the element ends `[CODE]`.

Because the wake tables depend only on the centroid geometry and `γ`, they can be computed for all steps up front in one batched kernel launch. Only steps 3 to 6 touch particles.

### 3.2 Centroid path geometry

Bmad stores, for each element: the centroid floor position and angle at entrance and exit, a chord (angle, length), and a cubic spline of the centroid transverse offset relative to the chord `[CODE]`. Path-length excess `dL` is integrated analytically from the spline (`dspline_len`) `[CODE]`.

Recommendations:

- Keep this representation. It gives smooth `dL` and angles, and a path-length quantity that is differentiable in the centroid orbit.
- Store it in flat arrays (struct of arrays): per element `θ_chord`, `L_chord`, spline coefficients, floor positions. Index by element number so a GPU thread can walk upstream.
- Known Bmad limitations to inherit or fix `[CODE]`: a spline for the centroid is inaccurate in wigglers (Bmad warns), `floor_shift` elements are rejected, `match` elements are special-cased, and negative-step elements need care.
- For differentiability, the centroid orbit must be tracked with the same generic number type as the rest of the calculation (so that parameter derivatives flow into the geometry).

### 3.3 Binning

- Grid: `N_bin` bins over the live-particle `z` range with empty padding bins at both ends (Bmad leaves the first and last bins empty `[CODE]`).
- Bmad spreads each particle as a triangle of width `particle_bin_span · Δz` for noise reduction `[CODE]`. Reproduce this, or use a B-spline shape function of your choice (Section 3.8 explains why this choice limits the derivative order you can trust).
- Gradient: Bmad uses edge differences of charge divided by `Δz²` `[CODE]`:

  `g_j = (Q_j − Q_{j−1}) / Δz²`

- Shot noise: `λ'` amplifies noise. Keep enough particles per bin, and use smoothing (the shape function above) rather than post-filtering, so the derivative stays analytic.

### 3.4 Wake table via batched root-finding

For each step (kick point) and each lag `m`:

1. Target separation: `z_m = m · Δz`.
2. Solve for the source arc position `s_s` such that
   `F(s_s) = L/(2γ²) + dL − z_m = 0` (Section 2.3).
3. Evaluate the pair integrand at the solution. In Bmad this is `I_csr = −2·κ_f·(dL/z + γ²·θ_sl·θ_lk/(1 + γ²·θ_sl²)) / L`, where `θ_sl` is the angle between source velocity and the source-to-kick line, `θ_lk` the angle between that line and the kicked particle's velocity, and `κ_f` the kick factor `[CODE]`. I have not re-derived this expression from Section 2.2. Do that, or cross-check against the paper.
4. Integrate across the bin to get `K_m`. Bmad uses the trapezoid on `I_csr` for `m > 1`, and a separate analytic expression for the first bin, `−κ_f·((g·L_s/2)² − ln(2γ² z/L_s)/γ²)`, to handle the near-field log singularity `[CODE]`.

Implementation notes `[DERIVED]` unless stated:

- Each `(step, m)` is an independent thread. Use a fixed number of Newton iterations (no data-dependent loop exit), starting from the previous lag's root. Bmad uses a bracketing zero-finder (`super_zbrent`) and walks element by element `[CODE]`. On a GPU, precompute the element containing each root by bracketing over the element table, then run Newton inside that element.
- `F` is not always monotonic in `s_s` (Bmad notes patch elements can break monotonicity `[CODE]`). Detect `F_s ≈ 0` and flag it instead of silently returning garbage.
- If the source is upstream of the lattice start, Bmad treats the upstream region as an infinite drift and solves a quadratic `[CODE]`. Reproduce it in closed form (no iteration).
- For a tighter first-bin treatment you can subtract the known singular part analytically and integrate the remainder numerically.

### 3.5 Convolution

- Direct sum is a lower-triangular Toeplitz mat-vec, `O(N_bin²)` per step. For `N_bin` of order 10²–10³ this is cheap on a GPU and works for any element type (floats, duals, TPS).
- FFT convolution (`O(N log N)`) is worthwhile only for large `N_bin` and plain floats. FFT libraries will not accept dual or TPS element types. Dispatch on element type.

### 3.6 Applying the kick

- Interpolate `k` between neighboring bin centers (Bmad does linear interpolation, `[CODE]`), and add to the energy variable.
- Keep the half-weight convention at element ends so consecutive elements do not double-count.
- Shielding: loop over `n_shield_images`, add image kicks with alternating sign (Section 2.5) `[CODE]`.

### 3.7 What is deliberately left out

Longitudinal space charge (LSC), transverse space charge, and 3D CSR. Bmad's `slice` space-charge branch computes LSC from the same bins `[CODE]`, and SpaceCharge.jl handles 3D space charge. Decide explicitly how the 1D CSR near-field term and the space-charge solver interact, to avoid double counting. I have not verified how Bmad's CSR and LSC parts overlap. Check the manual.

### 3.8 Derivatives

#### Reverse mode (gradients of ensemble observables)

Notation: `w_i(z_p)` are shape-function weights, `Q_j = Σ_p q_p w_j(z_p)`, `g = D Q / Δz²`, `k = c·T(K)·g` with `T(K)` the lower-triangular Toeplitz matrix, and `δ_p ← δ_p + Σ_i w_i(z_p) k_i`. Given the output cotangent `δ̄_p`, the adjoints are `[DERIVED]`:

$$ \bar k_i = \sum_p w_i(z_p)\,\bar\delta_p \qquad\text{(scatter; reuse the deposit kernel)} $$

$$ \bar z_p \mathrel{+}= \bar\delta_p \sum_i w_i'(z_p)\,k_i \qquad\text{(gather)} $$

$$ \bar g_j = c\sum_{i\ge j} K_{i-j+1}\,\bar k_i \qquad\text{(correlation, reuse the convolution kernel with reversed indexing)} $$

$$ \bar Q_j = (\bar g_j - \bar g_{j+1})/\Delta z^2 $$

$$ \bar q_p = \sum_j w_j(z_p)\,\bar Q_j ,\qquad \bar z_p \mathrel{+}= q_p\sum_j w_j'(z_p)\,\bar Q_j $$

$$ \bar K_m = c\sum_{j} \bar k_{j+m-1}\, g_j \qquad\text{(cotangent of the wake table)} $$

- Do not differentiate through the atomic scatter-add. Write these adjoints as kernels and register them as a custom rule on the whole `csr_step` function (via ChainRulesCore, or the Enzyme equivalent; check what Enzyme currently supports for KernelAbstractions kernels).
- **Root-finding:** differentiate the table entries with the implicit function theorem, not through the Newton iterations:

  `ds_s/dp = −(∂F/∂p) / (∂F/∂s_s)`, where `p` collects geometry parameters and `γ`.

  Requires `∂F/∂s_s ≠ 0`.
- Treat `z_min`, `z_max`, `Δz` (computed from min/max of the live particles) as constants during differentiation. The true derivative contains terms from their dependence on the particles. I expect those to be of the order of the discretization error, but I have not tested this.
- Lost-particle masks and `nint` step counts are not differentiable; keep them outside the differentiated path.

#### Forward mode and TPS (high-order maps)

- Convolution, differencing and interpolation are linear in `Q` and `k`, so the TPS-valued density is handled by applying the same operations coefficient by coefficient `[DERIVED]`. No TPS-aware FFT is needed if you use the direct sum.
- The deposit weights `w_i(z_p)` are nonlinear in `z_p`. With TPS-valued `z_p` you expand `w` about the numeric value, which needs derivatives of the shape function. A degree-`d` B-spline is `C^{d−1}`, so derivatives up to order `n` stay continuous only if `d ≥ n+1` `[MEMORY of B-spline smoothness]`. Linear (CIC) weights give at best first-order derivatives, and the Bmad triangle + linear-interpolation combination is only `C^0`/`C^1` `[DERIVED]`. For maps beyond first order, use a higher-degree shape and a matching interpolation.
- Wake table with TPS entries: either use the implicit-function derivatives above, or run Newton in TPS arithmetic. Newton on power series gains correct orders with each iteration `[MEMORY]`, so a few extra iterations suffice. Verify against finite differences.
- Cost: `N_particles × N_monomials` per quantity, CPU only if GTPSA is CPU-bound (Section 1.2).

### 3.9 Validation plan

1. **Straight drift:** CSR kick must be zero (Section 2.5).
2. **Long circular bend, Gaussian bunch:** compare with the steady-state formula in Section 2.4, away from the entrance transient. Use it to fix the coefficient and prefactor.
3. **Bmad cross-check:** same lattice and bunch through Bmad `csr_method = one_dim`; compare wake `W(z)` per step and final phase space.
4. **Derivative consistency:** reverse-mode gradient = ForwardDiff = finite difference; TPS first-order coefficients = Jacobian; TPS second order = ForwardDiff Hessian on a small ensemble.
5. **Backend equivalence:** KernelAbstractions CPU backend versus GPU backend, with tolerance for atomic-ordering differences.
6. **Convergence:** vary `N_bin`, `ds`, particle count; confirm convergence and the noise floor.

---

## 4. Integration with KernelAbstractions.jl

**Status of the code below: `[UNTESTED]` sketches.** The macros and calls used (`@kernel`, `@index`, `@Const`, `get_backend`, `Atomix.@atomic`, `KernelAbstractions.zeros`, `KernelAbstractions.synchronize`, the `kernel(backend, workgroupsize)(args...; ndrange=...)` launch form) are what I recall from KernelAbstractions 0.9 (the version range in SpaceCharge.jl's `[compat]`). **Check the current KernelAbstractions and Atomix docs before copying.** The helper names `residual_and_derivative`, `pair_integrand`, `bin_integral` and the `geom` layout are placeholders for code you must write.

### 4.1 Principles

- **One kernel per stage:** deposit, gradient, wake table, convolution, interpolate/apply. Launch code lives outside the kernels. Order on a single backend queue is preserved; call `synchronize` once at the end of a step (or when the host needs results).
- **Generic element type:** take `T = eltype(array)`. Use `zero(x)`-style constructors from an existing value, not hard-coded `Float64`. This lets the same kernels run with `Float32`/`Float64`, dual numbers, and (on the CPU backend) TPS.
  - For TPS, a type-only `zero(T)` may be unavailable because a TPS needs a descriptor (check GTPSA.jl docs). Prefer `zero(some_instance)`.
- **No allocation, no dynamic dispatch, no `Vector` creation inside kernels.** Pass flat arrays and `isbits` structs.
- **Structs on device:** make the geometry struct `isbits` or adapt it (Adapt.jl's `@adapt_structure`, check the docs) so arrays convert to device arrays when passed to a kernel.
- **Backend from data:** `backend = get_backend(z)`. Never hard-code CUDA; the same code then runs on the CPU backend for tests.
- **Dispatch on element type and backend** for the convolution: direct sum for duals/TPS, optional FFT for plain floats.

### 4.2 Kernel sketches

Bin centers are used for both deposit and interpolation (as Bmad interpolates between bin centers). Shape function shown is linear for brevity. Replace the weight computation with a smoother shape function for high-order derivatives (Section 3.8). The grid is assumed padded so `i+1` stays in bounds. Add a guard if you cannot guarantee that.

```julia
using KernelAbstractions
import Atomix

# 1) Deposit: Q_j = sum_p q_p * w_j(z_p)      [UNTESTED]
@kernel function deposit_kernel!(Q, @Const(z), @Const(q), z0, inv_dz)
    p = @index(Global)
    @inbounds begin
        u = (z[p] - z0) * inv_dz - 1//2        # fractional bin-center coordinate
        i = floor(Int, u) + 1                  # left neighbor bin (1-based)
        f = u - (i - 1)                        # weight toward bin i+1
        Atomix.@atomic Q[i]     += q[p] * (1 - f)
        Atomix.@atomic Q[i + 1] += q[p] * f
    end
end

# 2) Edge gradient: g_j = (Q_j - Q_{j-1}) / dz^2   [UNTESTED]
@kernel function grad_kernel!(g, @Const(Q), inv_dz2)
    j = @index(Global)
    @inbounds g[j] = (j == 1 ? Q[1] : Q[j] - Q[j - 1]) * inv_dz2
end

# 3) Direct convolution: k_i = c * sum_{j<=i} K[i-j+1] * g[j]   [UNTESTED]
@kernel function conv_kernel!(kick, @Const(K), @Const(g), c)
    i = @index(Global)
    acc = zero(eltype(kick))
    for j in 1:i
        @inbounds acc += K[i - j + 1] * g[j]
    end
    @inbounds kick[i] = c * acc
end

# 3b) Adjoint of (3) w.r.t. g: gbar_j = c * sum_{i>=j} K[i-j+1] * kbar[i]   [UNTESTED]
@kernel function corr_kernel!(gbar, @Const(K), @Const(kbar), c, n)
    j = @index(Global)
    acc = zero(eltype(gbar))
    for i in j:n
        @inbounds acc += K[i - j + 1] * kbar[i]
    end
    @inbounds gbar[j] = c * acc
end

# 4) Interpolate kick back to particles and apply   [UNTESTED]
@kernel function apply_kick_kernel!(delta, @Const(z), @Const(kick), z0, inv_dz)
    p = @index(Global)
    @inbounds begin
        u = (z[p] - z0) * inv_dz - 1//2
        i = floor(Int, u) + 1
        f = u - (i - 1)
        delta[p] += (1 - f) * kick[i] + f * kick[i + 1]
    end
end
```

Wake-table kernel skeleton (one thread per lag and step; the physics helpers are yours to write):

```julia
# [UNTESTED, structure only]
@kernel function wake_table_kernel!(K, @Const(geom), gamma, dz, niter)
    m, step = @index(Global, NTuple)
    z_lo = (m - 1) * dz
    z_hi = m * dz
    # fixed-iteration Newton for the retarded source point (no early exit)
    # s = initial_guess(geom, step, m)
    # for _ in 1:niter
    #     F, dF = residual_and_derivative(geom, gamma, s, z)   # Section 2.3
    #     s -= F / dF
    # end
    # K[m, step] = bin_integral(pair_integrand, geom, gamma, s, z_lo, z_hi)
end
```

### 4.3 Orchestration

```julia
# [UNTESTED]
function csr_step!(delta, z, q, K, c, z0, dz, nbin)
    backend = get_backend(z)
    T = eltype(z)
    Q    = KernelAbstractions.zeros(backend, T, nbin)
    g    = KernelAbstractions.zeros(backend, T, nbin)
    kick = KernelAbstractions.zeros(backend, T, nbin)

    deposit_kernel!(backend, 256)(Q, z, q, z0, inv(dz); ndrange = length(z))
    grad_kernel!(backend, 256)(g, Q, inv(dz)^2; ndrange = nbin)
    conv_kernel!(backend, 256)(kick, K, g, c; ndrange = nbin)
    apply_kick_kernel!(backend, 256)(delta, z, kick, z0, inv(dz); ndrange = length(z))

    KernelAbstractions.synchronize(backend)
    return delta
end
```

### 4.4 Known pitfalls

- **Atomics are not deterministic.** Floating-point atomic adds give run-to-run bitwise differences. If you need reproducibility, sort particles by bin index and use segmented reductions, or per-workgroup local histograms (`@localmem`, `@synchronize`; check the docs).
- **Atomics and non-`isbits` element types.** TPS values are not `isbits`, so atomic accumulation does not apply. On the CPU TPS path use a serial or per-thread-buffer deposit.
- **Atomic scatter and AD.** Do not rely on an AD tool to differentiate through the atomic add. Use the explicit adjoints from Section 3.8.
- **Branching and early exits** in the root-finder cost performance and hamper AD. Use fixed iteration counts and implicit differentiation.
- **Small kernels.** `N_bin` threads for the convolution is small for a GPU. If the step is launch-bound, batch several steps or several bunches (leading batch dimension) into one launch. Consider this together with SciBmad's batch-parameter evaluation (look at how it batches).

### 4.5 Differentiation modes through KernelAbstractions

| Mode | Element type | Backend | Notes |
|---|---|---|---|
| Plain tracking | `Float32`/`Float64` | CPU or GPU | Direct sum or FFT |
| Forward-mode AD (gradients) | ForwardDiff duals | CPU, probably GPU | Duals are `isbits`. Verify on your GPU backend. Use the direct sum (no FFT). |
| Reverse-mode AD | `Float64` + custom rule | CPU or GPU | Explicit adjoint kernels (Section 3.8). Check whether Enzyme can differentiate your KA kernels directly. |
| TPS maps | GTPSA TPS | CPU backend only (assumed) | Serial/per-thread deposit, direct sum, smooth shape function |

### 4.6 Integration with SpaceCharge.jl

1. Read its mesh struct and deposit/interpolate code first. Align conventions: units of charge, which axis is longitudinal, bin-center versus node-based grids, and the argument order of its kernels.
2. Reuse its backend handling pattern (KernelAbstractions + Atomix, per `Project.toml`) so a bunch lives on one device and neither package copies data.
3. Do not make CSR depend on SpaceCharge.jl unless you reuse its code. Prefer a Julia package extension if you only need interoperability. `[MEMORY]`: extensions need Julia ≥ 1.9, which matches its `compat`.
4. Proposal (my naming, not existing API): a small shared interface, such as one function that applies a collective effect for one step given a bunch, geometry, and step length, with SpaceCharge and CSR each implementing it. Then BeamTracking only needs a single hook. Ask the maintainers whether such a hook already exists or is planned. SciBmad's README lists both space charge and CSR under future releases.
5. Differentiability of SpaceCharge.jl: its `Project.toml` lists no AD packages. Find out whether its deposit/solve is differentiable (and with which AD) before promising end-to-end gradients through both effects.

---

## 5. Open items I could not verify

1. Where the current CSR module lives on `bmad-ecosystem` main, and whether the algorithm changed since the SVN copy I read.
2. The exact kick formula and coefficient in the Sagan & Mayes paper and in the steady-state formula (Section 2.4).
3. Whether `|q|` belongs in the numerator or denominator of the kick prefactor for ions (Section 2.5).
4. BeamTracking's coordinate conventions and available hook for collective kicks.
5. Whether GTPSA types can be used on any GPU backend (I assume not).
6. SpaceCharge.jl internals: shape function, differentiability, API.
7. Where SciBmad's intrabeam scattering (listed as a 0.5 feature) lives. It is the closest existing precedent for a collective effect.
8. Current Enzyme support for KernelAbstractions kernels.
9. Overlap or double counting between 1D CSR near-field terms and space-charge solvers.
