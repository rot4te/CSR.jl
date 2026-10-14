# One CSR kick: bin, build the kick table, convolve, apply.
# Port of the per-step part of Bmad's track1_bunch_csr and csr_bin_kicks (CSR only, with image charges).

"""
    csr_kick_coef(species, gamma, ds)

Prefactor of the CSR kick for a step of length `ds`: `ds * r_c / (rel_mass * e * |q| * γ)` with
`r_c = r_e q² / rel_mass` and `rel_mass = m / m_e`, as in Bmad. Multiplies (bin charge [C]) × (kernel).

For non-electrons the placement of `|q|` and `rel_mass` is unresolved (`planning/PLAN.md`, V2).
"""
function csr_kick_coef(species::Species, gamma, ds)
  rel_mass = massof(species) / M_ELECTRON
  q = chargeof(species)
  q == 0 && error("csr_kick_coef: species has zero charge")
  r_c = R_ELECTRON * q^2 / rel_mass
  return ds * r_c / (rel_mass * E_CHARGE * abs(q) * gamma)
end

"""
    CSRStepResult

Diagnostics for one kick, per bin (mirrors Bmad's `diagnostic_output_file` columns):
`z_center`, `charge_per_meter` [C/m], `kick_per_meter` (Δpz per meter), `I_csr_per_meter`,
`s_source` (reference s of the source point). Arrays are on the backend of the particle coordinates.
"""
struct CSRStepResult{Z<:AbstractVector,C<:AbstractVector,K<:AbstractVector,I<:AbstractVector,S<:AbstractVector}
  z_center::Z
  charge_per_meter::C
  kick_per_meter::K
  I_csr_per_meter::I
  s_source::S
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

Non-mutating `csr_kick!`: returns `pz` after the kick in a new vector (element type promoted with `z` and `q`),
leaving `pz` unchanged. Differentiable with respect to `pz`, `z` and `q`: ForwardDiff Duals pass through, and with
ChainRulesCore loaded an `rrule` gives the pullback by explicit adjoint kernels. In both, the bin grid and the kick
table are held constant (the grid depends on the extreme live particles, the table on the geometry and `gamma`);
derivatives with respect to `geom`, `s_kick`, `gamma` and the keywords are not provided.
"""
function csr_kick_pz(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                     geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                     params::CSRParams; kick_factor = 1.0, pz_scale = 1.0)
  pz_new = similar(pz, promote_type(eltype(pz), eltype(z), eltype(q)))
  copyto!(pz_new, pz)
  csr_kick!(pz_new, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor, pz_scale)
  return pz_new
end

# Bin the particles and compute the binned kick (all of csr_kick! except applying it). Returns the intermediate
# quantities the adjoint needs: bins, kick, free-space tables (I_csr, I_int, s_source), image tables per layer,
# the kick coefficient and the step length.
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

csr_step_result(st) = CSRStepResult(st.bins.z_center, st.bins.charge ./ st.bins.dz, st.kick ./ st.ds_kick,
                                    st.I_csr ./ st.ds_kick, st.s_source)

"""
    csr_ref_energy(E_start, E_end, s, L)

Reference total energy at `s` in an element of length `L` with entrance/exit reference energies
`E_start`/`E_end`, linearly interpolated as in Bmad's `track1_bunch_csr` ("linear energy gain in a cavity").

Bmad weighted the two ends the other way round until bmad-ecosystem PR #2199 (2026-10-06; `planning/PLAN.md`, V6);
CSR.jl follows the fixed version in both modes.
"""
function csr_ref_energy(E_start, E_end, s, L)
  f1 = s / L
  return (1 - f1) * E_start + f1 * E_end
end

"""
    csr_n_step(L, params) -> (n_step, ds_step)

Number of CSR steps in an element of length `L` and the true step length, as in Bmad:
`n_step = max(1, round(L / ds_step))`, `ds_step = L / n_step`.
"""
function csr_n_step(L, params::CSRParams)
  params.ds_step > 0 || error("csr_n_step: params.ds_step must be set")
  n_step = max(1, round(Int, L / params.ds_step, RoundNearestTiesAway))
  return n_step, L / n_step
end
