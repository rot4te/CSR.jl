# Per-step constants: kick coefficient, reference-energy ramp, number of steps.

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
