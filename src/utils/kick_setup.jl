# Per-step constants: kick coefficient and reference-energy ramp.

"""
    csr_kick_coef(species, gamma, ds; bmad_compat = false)

Prefactor of the CSR kick for a step of length `ds`. Multiplies (bin
charge [C], unsigned) × (kernel).

Default: `Δpz = ΔE / (β² E)` with `ΔE = |q| e E_z ds` and `E_z =
(1/4πε₀)` × (kernel ⋆ charge), which gives `ds * r_e * |q| /
(rel_mass * e * β² * γ)` (`q` in units of `e`, `rel_mass = m /
m_e`). `bmad_compat = true`: Bmad's `ds * r_c / (rel_mass * e * |q| *
γ)` with `r_c = r_e q² / rel_mass`, i.e. `β²/rel_mass` times the
default (docs, "Physics decisions", V2). Errors unless `gamma > 1`
(primal value; NaN fails).
"""
function csr_kick_coef(species::Species, gamma, ds;
                       bmad_compat::Bool = false)
  Beamlines.scalarize_all(gamma) > 1 || error("csr_kick_coef: gamma must be > 1 (got $(Beamlines.scalarize_all(gamma)))")
  rel_mass = massof(species) / M_ELECTRON
  q = chargeof(species)
  q == 0 && error("csr_kick_coef: species has zero charge")
  if bmad_compat
    r_c = R_ELECTRON * q^2 / rel_mass
    return ds * r_c / (rel_mass * E_CHARGE * abs(q) * gamma)
  end
  beta2 = 1 - 1 / gamma^2
  return ds * R_ELECTRON * abs(q) / (rel_mass * E_CHARGE * beta2 *
                                     gamma)
end

"""
    csr_ref_energy(E_start, E_end, s, L)

Reference total energy at `s` in an element of length `L` with
entrance/exit reference energies `E_start`/`E_end`, linearly
interpolated as in Bmad's `track1_bunch_csr` ("linear energy gain in a
cavity").

Bmad weighted the two ends the other way round until bmad-ecosystem PR
#2199 (2026-10-06; docs, "Physics decisions", D21); CSR.jl follows the
fixed version in both modes.
"""
function csr_ref_energy(E_start, E_end, s, L)
  f1 = s / L
  return (1 - f1) * E_start + f1 * E_end
end
