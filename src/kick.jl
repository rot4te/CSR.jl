# One CSR kick (host side): bin, build the kick table, convolve, apply.
# Port of the per-step part of Bmad's track1_bunch_csr and csr_bin_kicks (CSR only, with image charges).

"""
    csr_convolve!(kick, I_int, edge_grad, coef)

`kick[i] = coef * Σ_{k=1}^{i} I_int[k] * edge_grad[i+1-k]`. One work item per bin, on the backend of `kick`.
"""
function csr_convolve!(kick::AbstractVector, I_int::AbstractVector, edge_grad::AbstractVector, coef)
  backend = get_backend(kick)
  csr_convolve_kernel!(backend)(kick, I_int, edge_grad, coef; ndrange = length(kick))
  KernelAbstractions.synchronize(backend)
  return kick
end

"""
    csr_image_convolve!(kick, image, charge, coef)

`kick[i] += coef * Σ_{j=1}^{n_bin} image(i - j) * charge[j]`, with the image kick for lag `l` stored at
`image[l + n_bin + 1]` (`csr_image_table`). Bmad's image-charge sum (all lags, bin charge rather than its
gradient). One work item per bin.
"""
function csr_image_convolve!(kick::AbstractVector, image::AbstractVector, charge::AbstractVector, coef)
  length(image) == 2length(kick) + 1 || error("csr_image_convolve!: image must have 2n_bin + 1 entries")
  backend = get_backend(kick)
  csr_image_convolve_kernel!(backend)(kick, image, charge, coef; ndrange = length(kick))
  KernelAbstractions.synchronize(backend)
  return kick
end

"""
    csr_apply_kicks!(pz, z, alive, bins, kick; pz_scale = 1)

Add the kick, linearly interpolated between bin centers and multiplied by `pz_scale`, to `pz` of each live particle.
As in Bmad's CSR-only branch, `z` is not rescaled for the change in velocity.
One work item per particle, on the backend of `pz`.
"""
function csr_apply_kicks!(pz::AbstractVector, z::AbstractVector, alive::AbstractVector{Bool}, bins::CSRBins, kick::AbstractVector;
                          pz_scale = 1)
  backend = get_backend(pz)
  n_outside = KernelAbstractions.zeros(backend, Int32, 1)   # Int32: no 64-bit atomics on some GPUs
  csr_apply_kicks_kernel!(backend)(pz, n_outside, z, alive, bins.z_center, kick, bins.dz, pz_scale; ndrange = length(pz))
  KernelAbstractions.synchronize(backend)
  Array(n_outside)[1] == 0 || error("csr_apply_kicks!: particle outside the bin range (internal error)")
  return pz
end

"""
    csr_kick!(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0,
              pz_scale = 1.0) -> CSRStepResult

Apply one 1D CSR kick at reference position `s_kick` [m] from the entrance of element `ix_ele_kick`.

- `pz`, `z`: particle coordinates (`pz` is modified).
- `q`: macroparticle charges [C]. `alive`: live-particle mask.
- `gamma`: Lorentz factor of the reference particle at the kick point.
- `kick_factor`: `1/2` at the first and last step of an element, `1` otherwise.
- `pz_scale`: multiplies the Δpz applied to the particles (not the returned diagnostics). Used where the particles'
  reference momentum differs from the one the kick is computed for (inside a SaganCavity, whose reference momentum
  steps at each cell while γ ramps linearly).

The kick covers one step, `ds = L / n_step` (see `csr_n_step`) scaled by `L_chord / L`, as in Bmad.

With `params.n_shield_images > 0` the image-charge kicks of the shielding plates are added (Bmad's image part of
`csr_bin_kicks`), with the layer sum averaged per `params.image_sum_order`. `kick_per_meter` includes them;
`I_csr_per_meter` and `s_source` are for free space only.

Runs as KernelAbstractions kernels on the backend of `pz` (deposit, edge gradient, convolutions, kick). The wake and
image tables are always computed on the CPU in `Float64` and copied to a GPU backend in `csr_float_type(backend)`
(`Float32` where the backend has no `Float64`, e.g. Metal); scalars passed to the kernels are converted likewise.
`csr_kick_reference!` is the serial phase 1 version.
"""
function csr_kick!(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                   geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                   params::CSRParams; kick_factor = 1.0, pz_scale = 1.0)
  st = csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor)
  csr_apply_kicks!(pz, z, alive, st.bins, st.kick; pz_scale = device_scalar(get_backend(pz), pz_scale))
  return csr_step_result(st)
end

"""
    csr_kick_pz(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0,
                pz_scale = 1.0) -> pz_new

Non-mutating `csr_kick!`: returns `pz` after the kick in a new vector (element type promoted with `z`, `q`, the
geometry's number type, `s_kick`, `gamma`, `kick_factor` and `pz_scale`), leaving `pz` unchanged.

Differentiable with ForwardDiff `Dual`s in any of these: the particles (`pz`, `z`, `q`), the geometry (built by
`CSRGeometry` from `Dual` lengths, curvatures or centroid), `s_kick`, `gamma`, `kick_factor`, `pz_scale`. The kick
tables then carry derivatives; each retarded source point gets its derivatives by implicit differentiation of its
root (`root_tangent`). Held constant: the bin grid (it depends on the extreme live particles) and the discrete
choices (number of steps `csr_n_step`, the element a source lies in, the adaptive quadrature's subdivision).
With ChainRulesCore loaded an `rrule` gives the pullback with respect to `pz`, `z` and `q` by explicit adjoint
kernels; there, derivatives with respect to `geom`, `s_kick`, `gamma` and the keywords are not provided.
"""
function csr_kick_pz(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                     geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                     params::CSRParams; kick_factor = 1.0, pz_scale = 1.0)
  pz_new = similar(pz, promote_type(eltype(pz), eltype(z), eltype(q), number_type(geom), typeof(s_kick),
                                    typeof(gamma), typeof(kick_factor), typeof(pz_scale)))
  copyto!(pz_new, pz)
  csr_kick!(pz_new, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor, pz_scale)
  return pz_new
end

"""
    csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0)

Bin the particles and compute the binned kick (all of csr_kick! except applying it). Returns the intermediate
quantities the adjoint needs: bins, kick, free-space tables (I_csr, I_int, s_source), image tables per layer,
the kick coefficient and the step length.
"""
function csr_binned_kick(z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, geom::CSRGeometry,
                         ix_ele_kick::Integer, s_kick, gamma, species::Species, params::CSRParams; kick_factor = 1.0)
  n_bin = params.n_bin
  params.bmad_compat == geom.bmad_compat || error("csr_kick!: params.bmad_compat and geom.bmad_compat differ")
  count(alive) >= n_bin || error("csr_kick!: number of live particles ($(count(alive))) is less than n_bin ($n_bin)")
  backend = get_backend(z)
  tb = table_backend(backend)

  e = geom.ele[ix_ele_kick]
  _, ds_step = csr_n_step(e.L, params)
  actual_track_step = ds_step * (e.L_chord / e.L)

  bins = csr_bin_particles(z, q, alive, params)
  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)
  dz_t = tb === backend ? bins.dz : Float64(bins.dz)   # tables off the particle backend: Float64 lags
  I_csr, I_int, s_source = to_backend.(Ref(backend),
    csr_wake_table(geom, kp, dz_t, n_bin, tb; all_bin_quadrature = params.all_bin_quadrature && !params.bmad_compat))

  coef = device_scalar(backend, csr_kick_coef(species, gamma, actual_track_step))
  edge = bins.edge_dcharge_density_dz
  kick = KernelAbstractions.zeros(backend, promote_type(typeof(coef), eltype(I_int), eltype(edge)), n_bin)
  csr_convolve!(kick, I_int, edge, coef)
  # Shielding: image layer ns at height ns*h with sign (-1)^ns; the factor 2 counts the images above and below.
  # image_layer_weight averages the slowly converging layer sum (params.image_sum_order).
  images = map(1:params.n_shield_images) do ns
    kp_ns = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source = ns * params.beam_chamber_height,
                         kick_factor = (-1)^ns * 2 * image_layer_weight(ns, params) * kick_factor)
    image, _ = csr_image_table(geom, kp_ns, dz_t, n_bin, tb; bin_integrate = !params.bmad_compat)
    image = to_backend(backend, image)
    csr_image_convolve!(kick, image, bins.charge, coef)
    image
  end
  return (; bins, kick, I_csr, I_int, s_source, images, coef,
          ds_kick = device_scalar(backend, kick_factor * actual_track_step))
end

"""
    csr_step_result(st) -> CSRStepResult

Per-meter diagnostics from the intermediates of `csr_binned_kick`.
"""
csr_step_result(st) = CSRStepResult(st.bins.z_center, st.bins.charge ./ st.bins.dz, st.kick ./ st.ds_kick,
                                    st.I_csr ./ st.ds_kick, st.s_source)
