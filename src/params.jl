"""
    CSRParams(; n_bin = 40, particle_bin_span = 2, ds_step = 0.0, n_shield_images = 0, beam_chamber_height = 0.0,
              image_sum_order = 8, all_bin_quadrature = true, bmad_compat = false)

Parameters for the 1D CSR calculation. Names follow Bmad's `space_charge_com` components.

## Properties
- `n_bin`: Number of longitudinal bins. Must satisfy `n_bin - 2 - (particle_bin_span + 1) ≥ 1`.
- `particle_bin_span`: Width of each particle's triangular charge distribution, in units of the bin width.
- `ds_step`: Requested tracking step length [m]. The actual step is `L / max(1, round(L / ds_step))`.
- `n_shield_images`: Number of image-charge layers for shielding by parallel plates (0 = free space).
    Layer `n` is a pair of images at heights `±n * beam_chamber_height` with sign `(-1)^n`, as in Bmad.
- `beam_chamber_height`: Height of the beam chamber [m], i.e. the distance between the plates. Must be positive
    when `n_shield_images > 0`.
- `image_sum_order`: Order `k` of the averaging applied to the image-layer sum (0 = plain sum, at most 60). The layer
    terms alternate in sign and decay only as ~1/n, so the plain sum converges slowly. With `M = n_shield_images`, the
    kick is the binomially weighted average `2^-k Σ_j C(k, j) S_{M-k+j}` of the partial sums `S_m` (free space plus
    layers 1..m), i.e. layer `n` is weighted by `2^-k Σ_{j ≥ n-M+k} C(k, j)` (1 for `n ≤ M - k`); `k` is capped at `M`.
    Default 8: with `M ≥ 32` the steady-state kick agrees with a 1024-layer reference to ~1e-6
    (`planning/PLAN_continued.md`, D24). Use `M` of 32–64 when the chamber height is comparable to or smaller than the
    bunch length. Ignored with `bmad_compat = true` (Bmad: plain sum).
- `all_bin_quadrature`: Integrate the kick integrand over every bin by quadrature (Simpson in `u^(1/3)`, 8 nodes per
    bin), instead of the trapezoid `(I_csr[i] + I_csr[i-1]) dz / 2` for bins `i ≥ 2`. Default `true`
    (`planning/upstream-bin.md`, option B). `false` gives the previous default (first bin only, option A). Ignored with
    `bmad_compat = true` (Bmad: trapezoid).
- `bmad_compat`: If `true`, reproduce Bmad expressions that CSR.jl replaces, for like-for-like comparison with
    Bmad output. Default `false`.
    1. `dspline_len` s⁴ term: Bmad uses `9c₃²`, the integral gives `9c₃²/5` (`planning/dspline_coeff.md`).
    2. First-bin kick integral when the source lies outside the kick element: Bmad holds `I_csr` constant
       over the rest of the bin (2/3 of the integral in steady state); CSR.jl integrates by quadrature
       (`planning/upstream-bin.md`).
    3. Image charges, source ahead of the kick point: Bmad gives the source the reverse direction of motion,
       and does not look for sources beyond the end of the kick element (`planning/PLAN.md`, V8).
    4. Image charges: Bmad samples the image kernel at the bin centres and sums the layers plainly; CSR.jl integrates
       the kernel against the bin hat function and averages the layer sum (`image_sum_order`).
"""
Base.@kwdef struct CSRParams
  n_bin::Int = 40
  particle_bin_span::Int = 2
  ds_step::Float64 = 0.0
  n_shield_images::Int = 0
  beam_chamber_height::Float64 = 0.0
  image_sum_order::Int = 8
  all_bin_quadrature::Bool = true
  bmad_compat::Bool = false
  function CSRParams(n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height, image_sum_order,
                     all_bin_quadrature, bmad_compat)
    n_bin_eff = n_bin - 2 - (particle_bin_span + 1)
    n_bin_eff >= 1 || error("CSRParams: n_bin (= $n_bin) must be greater than 3 + particle_bin_span (= $particle_bin_span)")
    particle_bin_span >= 1 || error("CSRParams: particle_bin_span must be at least 1")
    ds_step >= 0 || error("CSRParams: ds_step must be non-negative")
    n_shield_images >= 0 || error("CSRParams: n_shield_images must be non-negative")
    n_shield_images == 0 || beam_chamber_height > 0 ||
      error("CSRParams: beam_chamber_height must be positive when n_shield_images > 0")
    0 <= image_sum_order <= 60 || error("CSRParams: image_sum_order must be in 0:60")
    return new(n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height, image_sum_order,
               all_bin_quadrature, bmad_compat)
  end
end

# Weight of image layer `ns` in the averaged layer sum (see `image_sum_order`); 1 with `bmad_compat`.
function image_layer_weight(ns, params::CSRParams)
  params.bmad_compat && return 1.0
  M = params.n_shield_images
  k = min(params.image_sum_order, M)
  ns <= M - k && return 1.0
  return sum(binomial(k, j) for j in ns-(M-k):k) / 2^k
end
