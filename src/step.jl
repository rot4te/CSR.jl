# One CSR kick: bin, build the kick table, convolve, apply.
# Port of the per-step part of Bmad's track1_bunch_csr and csr_bin_kicks (free space, CSR only).

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
    csr_kick!(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0)
        -> CSRStepResult

Apply one 1D CSR kick at reference position `s_kick` [m] from the entrance of element `ix_ele_kick`.

- `pz`, `z`: particle coordinates (`pz` is modified).
- `q`: macroparticle charges [C]. `alive`: live-particle mask.
- `gamma`: Lorentz factor of the reference particle at the kick point.
- `kick_factor`: `1/2` at the first and last step of an element, `1` otherwise.

The kick covers one step, `ds = L / n_step` (see `csr_n_step`) scaled by `L_chord / L`, as in Bmad.

Runs as KernelAbstractions kernels on the backend of `pz` (deposit, edge gradient, wake table, convolution,
kick). Only the CPU backend is supported (the wake table needs it). `csr_kick_reference!` is the serial
phase 1 version.
"""
function csr_kick!(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                   geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                   params::CSRParams; kick_factor = 1.0)
  n_bin = params.n_bin
  params.bmad_compat == geom.bmad_compat || error("csr_kick!: params.bmad_compat and geom.bmad_compat differ")
  count(alive) >= n_bin || error("csr_kick!: number of live particles ($(count(alive))) is less than n_bin ($n_bin)")
  backend = get_backend(pz)

  e = geom.ele[ix_ele_kick]
  _, ds_step = csr_n_step(e.L, params)
  actual_track_step = ds_step * (e.L_chord / e.L)

  bins = csr_bin_particles(z, q, alive, params)
  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)
  I_csr, I_int, s_source = csr_wake_table(geom, kp, bins.dz, n_bin, backend)

  coef = csr_kick_coef(species, gamma, actual_track_step)
  edge = bins.edge_dcharge_density_dz
  kick = KernelAbstractions.zeros(backend, promote_type(typeof(coef), eltype(I_int), eltype(edge)), n_bin)
  csr_convolve!(kick, I_int, edge, coef)
  csr_apply_kicks!(pz, z, alive, bins, kick)

  ds_kick = kick_factor * actual_track_step
  return CSRStepResult(bins.z_center, bins.charge ./ bins.dz, kick ./ ds_kick, I_csr ./ ds_kick, s_source)
end

"""
    csr_ref_energy(E_start, E_end, s, L; bmad_compat = false)

Reference total energy at `s` in an element of length `L` with entrance/exit reference energies
`E_start`/`E_end`, linearly interpolated as in Bmad's `track1_bunch_csr` ("linear energy gain in a cavity").

Bmad's code (csr_and_space_charge_mod.f90:330–331) weights the two ends the other way round, giving the exit
energy at `s = 0` (`planning/PLAN.md`, V6); `bmad_compat = true` reproduces that.
"""
function csr_ref_energy(E_start, E_end, s, L; bmad_compat::Bool = false)
  f1 = s / L
  return bmad_compat ? f1 * E_start + (1 - f1) * E_end : (1 - f1) * E_start + f1 * E_end
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
