# Physics model

CSR.jl follows Bmad's one-dimensional CSR model (`csr_method = one_dim`): the bunch is a line charge along the
centroid orbit, and each particle receives a longitudinal kick from the radiation and space-charge fields of the
charge behind (and, with image charges, ahead of) it, evaluated with retardation. This page states the model as
implemented; the places where CSR.jl deliberately differs from Bmad are collected on [Physics decisions](@ref).

Conventions are those of BeamTracking and Bmad: `z > 0` is ahead of the reference particle, `pz = ΔP/P₀`, and floor
coordinates `r = (X, Y, Z)` with orientation angle θ, direction of motion `(sin θ, 0, cos θ)`; a bend with `g > 0`
curves towards `-x` and θ decreases.

## Steps

`track_csr!` divides every element into `n_step = max(1, round(L / ds_step))` slices. The bunch is tracked
slice by slice with BeamTracking; at each slice boundary (and at the element ends, with half weight) one kick
[`csr_kick!`](@ref) is applied:

1. **Geometry.** The centroid orbit (tracked without CSR) defines, per element, the chord from the entrance to
   the exit centroid position and a cubic spline `x(t)` of the centroid offset from the chord
   ([`CSRGeometry`](@ref), Bmad `csr_ele_info_struct`).
2. **Binning.** Live particles are deposited on `n_bin` bins with a triangular shape of base
   `particle_bin_span · dz`; the grid spans the live particles with empty padding bins,
   `dz = 1.0000001 (z_max − z_min) / (n_bin − 3 − particle_bin_span)` ([`csr_bin_particles`](@ref)).
3. **Kick table.** For every lag `i·dz`, the retarded source point is found and the kick integrand evaluated
   (free space), or the image-charge field (shielding).
4. **Convolution** of the table with the charge-density gradient (free space) or the bin charge (images).
5. **Kick.** `pz` changes by the kick interpolated linearly between bin centres; `z` is not changed.

## Retardation

A source particle a distance `dz` behind the kicked particle (at equal time) radiated from a point a path length
`L_s` behind the kick point, at straight-line distance `L`. The retardation condition `L_s − βL = dz` is solved
in Bmad's approximate form

```math
\frac{L}{2\gamma^2} + \Delta L = dz, \qquad \Delta L = L_s - L ,
```

by a bracketed root search along the elements (Bmad `s_source_calc`, `ddz_calc_csr`), with closed forms on infinite
drifts before the lattice start and (image charges) after its end. The path-length excess `ΔL` is a sum over the
elements between source and kick of the excess of the centroid path over its projection on the source–kick line;
for an element whose chord makes the angle `dθ` with that line,

```math
\Delta L_\text{ele} = \int \left(\sqrt{1 + x'^2} - \cos d\theta - x' \sin d\theta\right) dt
                   = \int \frac{x'^2}{1 + \sqrt{1 + x'^2}}\, dt + 2 \sin^2\frac{d\theta}{2}\,\Delta t - \sin d\theta\, \Delta x
```

(exact; `path_excess`). Bmad uses the small-angle form `½∫(x′ − dθ)² dt` (`dspline_len`), see
[Physics decisions](@ref).

## Free-space kick

The kick integrand for lag `z` is (Bmad `I_csr`)

```math
I_\text{csr}(z) = -\frac{2}{L}\left[\frac{\Delta L}{z} + \frac{\gamma^2 \theta_{sl}\theta_{lk}}{1 + \gamma^2\theta_{sl}^2}\right],
```

with `θ_sl` (`θ_lk`) the angle between the source (kicked particle) velocity and the source–kick line. On a circle
of radius ρ in steady state, `I_csr → −2·3^{−1/3} ρ^{−2/3} z^{−1/3}`. The kick on bin `i` is

```math
k_i = c \sum_{m=1}^{i} I_\text{int}[m]\; g_{i+1-m}, \qquad g_j = \frac{Q_j - Q_{j-1}}{dz^2},
\qquad c = \frac{ds\; r_e}{e\,\gamma}\ \text{(electrons)},
```

where `Q_j` is the bin charge, `I_int[m]` the integral of `I_csr` over bin `m` (the first bin from Bmad's closed form
with its logarithmic near-field term, plus quadrature beyond the kick element; the other bins by quadrature or
trapezoid, see [Physics decisions](@ref)), and `ds` the step length scaled by the chord-to-arc ratio of the element.
For other species the coefficient is Bmad's `ds r_c / (m/m_e · e |q| γ)` ([`csr_kick_coef`](@ref)); its form for
ions is unresolved (V2).

## Shielding by image charges

Parallel plates a distance `h` apart, the beam on the midplane, are represented by layers of image charges at
heights `y = n h`, `n = 1, 2, …`, with sign `(−1)ⁿ` and a factor 2 for the images above and below
(`n_shield_images` layers). For each layer the longitudinal Liénard–Wiechert field at the kick point of a source at
height `y` is evaluated at the retarded source point (Bmad `image_charge_kick_calc`: Coulomb term
`(N − β′)/(γ²L²κ³)` and radiation term `N×((N − β′)×G)/(Lκ³)`, `κ = 1 − N·β′`), for lags `−n_bin … n_bin`
(sources behind and ahead). The kick is `k_i += c Σ_j image(i − j) Q_j`.

Two numerical aspects matter here:

- The image kernel varies on scales far below a bin (`~y/γ` on a drift). CSR.jl integrates it against the hat
  function of the linearly interpolated charge density (adaptive quadrature) instead of sampling it at the lags.
- The layer series alternates in sign and its terms decay only as `~1/n` (the field of a distant image falls as
  `1/y`), so a plain truncated sum converges slowly. CSR.jl averages the last partial sums with binomial weights
  (`image_sum_order`).

## Reference energy

γ for the kick is the reference particle's. It is constant inside elements, except in SaganCavity elements with
`dE_ref` (lcavities), where it ramps linearly from `E_ref − dE_ref` to `E_ref` ([`CSR.csr_ref_energy`](@ref), Bmad). The
particles' reference momentum there steps at each RF cell (BeamTracking), so the applied `Δpz` is rescaled from the
linearly ramped reference momentum to the stepped one (`pz_scale`).

## Limits of the model

- Planar geometry: bends with `tilt_ref ∈ {0, π}` only; patches with vertical offsets are rejected.
- No longitudinal or transverse space charge; no wigglers.
- At least `n_bin` live particles are needed (as Bmad).
- The 1D model needs `σ_x ≪ (ρ σ_z²)^{1/3}` (Derbenev et al.; from memory, not checked here).
