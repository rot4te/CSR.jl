# D42: first-bin integral I_int[1] on a straight line, both branches of I_csr_calc (source in the kick element:
# dz < s_chord(1 − β); source upstream: dz larger), default and compat. Exact retardation: I_int[1] should vanish.
using CSR
geom(bc) = CSRGeometry(fill(0.5, 4), zeros(4), zeros(5, 6); bmad_compat = bc)
for bc in (false, true), γ in (82.0, 1e3), dz in (1e-7, 1e-4)
  g = geom(bc)
  kp = CSR.CSRKickPoint(g, 3, 0.25, γ)
  _, I_int, _ = CSR.csr_wake_table(g, kp, dz, 4, CSR.CPU())
  println("bmad_compat = $bc  γ = $γ  dz = $dz  I_int[1] = $(I_int[1])  1/(4γ⁴) = $(1 / (4γ^4))")
end
