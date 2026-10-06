"""
    CSRParams(; n_bin = 40, particle_bin_span = 2, ds_step = 0.0, n_shield_images = 0, beam_chamber_height = 0.0,
              bmad_compat = false)

Parameters for the 1D CSR calculation. Names follow Bmad's `space_charge_com` components.

## Properties
- `n_bin`: Number of longitudinal bins. Must satisfy `n_bin - 2 - (particle_bin_span + 1) ≥ 1`.
- `particle_bin_span`: Width of each particle's triangular charge distribution, in units of the bin width.
- `ds_step`: Requested tracking step length [m]. The actual step is `L / max(1, round(L / ds_step))`.
- `n_shield_images`: Number of image-charge layers for shielding by parallel plates (0 = free space).
    Layer `n` is a pair of images at heights `±n * beam_chamber_height` with sign `(-1)^n`, as in Bmad.
- `beam_chamber_height`: Height of the beam chamber [m], i.e. the distance between the plates. Must be positive
    when `n_shield_images > 0`.
- `bmad_compat`: If `true`, reproduce Bmad expressions that CSR.jl replaces, for like-for-like comparison with
    Bmad output. Default `false`.
    1. `dspline_len` s⁴ term: Bmad uses `9c₃²`, the integral gives `9c₃²/5` (`planning/dspline_coeff.md`).
    2. First-bin kick integral when the source lies outside the kick element: Bmad holds `I_csr` constant
       over the rest of the bin (2/3 of the integral in steady state); CSR.jl integrates by quadrature
       (`planning/upstream-bin.md`).
    3. Image charges, source ahead of the kick point: Bmad gives the source the reverse direction of motion,
       and does not look for sources beyond the end of the kick element (`planning/PLAN.md`, V8).
"""
Base.@kwdef struct CSRParams
  n_bin::Int = 40
  particle_bin_span::Int = 2
  ds_step::Float64 = 0.0
  n_shield_images::Int = 0
  beam_chamber_height::Float64 = 0.0
  bmad_compat::Bool = false
  function CSRParams(n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height, bmad_compat)
    n_bin_eff = n_bin - 2 - (particle_bin_span + 1)
    n_bin_eff >= 1 || error("CSRParams: n_bin (= $n_bin) must be greater than 3 + particle_bin_span (= $particle_bin_span)")
    particle_bin_span >= 1 || error("CSRParams: particle_bin_span must be at least 1")
    ds_step >= 0 || error("CSRParams: ds_step must be non-negative")
    n_shield_images >= 0 || error("CSRParams: n_shield_images must be non-negative")
    n_shield_images == 0 || beam_chamber_height > 0 ||
      error("CSRParams: beam_chamber_height must be positive when n_shield_images > 0")
    return new(n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height, bmad_compat)
  end
end
