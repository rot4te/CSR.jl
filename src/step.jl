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
`s_source` (reference s of the source point).
"""
struct CSRStepResult
  z_center::Vector{Float64}
  charge_per_meter::Vector{Float64}
  kick_per_meter::Vector{Float64}
  I_csr_per_meter::Vector{Float64}
  s_source::Vector{Float64}
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
"""
function csr_kick!(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                   geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                   params::CSRParams; kick_factor = 1.0)
  n_bin = params.n_bin
  params.bmad_compat == geom.bmad_compat || error("csr_kick!: params.bmad_compat and geom.bmad_compat differ")
  count(alive) >= n_bin || error("csr_kick!: number of live particles ($(count(alive))) is less than n_bin ($n_bin)")

  e = geom.ele[ix_ele_kick]
  _, ds_step = csr_n_step(e.L, params)
  actual_track_step = ds_step * (e.L_chord / e.L)

  bins = csr_bin_particles(z, q, alive, params)
  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)

  # Kick table for lags i*dz, i = 1:n_bin. Each search starts in the element where the previous one ended.
  I_int = zeros(n_bin)
  I_csr = zeros(n_bin)
  s_source = zeros(n_bin)
  k = CSRKick1()
  k.ix_ele_source = ix_ele_kick
  for i in 1:n_bin
    k.dz_particles = i * bins.dz
    s_chord = s_source_calc!(k, geom, kp)
    I_csr!(k, i, bins.dz, geom, kp, i == 1 ? 0.0 : I_csr[i-1])
    I_csr[i] = k.I_csr
    I_int[i] = k.I_int_csr
    s_source[i] = k.ix_ele_source == 0 ? s_chord : geom.ele[k.ix_ele_source].s_start + s_chord
  end

  coef = csr_kick_coef(species, gamma, actual_track_step)
  kick = zeros(n_bin)
  csr_convolve!(kick, I_int, bins.edge_dcharge_density_dz, coef)
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
