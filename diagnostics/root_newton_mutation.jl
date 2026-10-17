# Mutation check of the second-order source-root test (ad_geometry_test.jl "residual stationary to second order"):
# d²/dt² of ddz at the nested-Dual root vs at the primal root, with root_tangent's two Newton steps and with one
# (root_order(Dual) forced to 1). Usage (test env): julia --project=<env> root_newton_mutation.jl
# Results (2026-10-15): two steps ratio ≤ ~1e-14, one step 1.2–7 (test bound 1e-10). PLAN.md Section 9.
using CSR, ForwardDiff, StaticArrays, Printf

L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]

function resid(t, fixed, bc, dz)
  G = CSRGeometry(L3, g3 .+ t .* [0.1, 1.0, 0.0], c3; bmad_compat = bc)
  kp = CSRKickPoint(G, 2, 0.17 + 0.05t, 200.0 + 50t)
  k, s = CSR.s_source_calc(CSR.CSRKick1(G, kp; ix_ele_source = 2, dz_particles = dz), G, kp)
  k2 = CSR.CSRKick1(G, kp; ix_ele_source = k.ix_ele_source, dz_particles = dz)
  return last(CSR.ddz_calc_csr(k2, fixed ? CSR.primal(s) : s, G, kp))
end
d2(fixed, bc, dz) = ForwardDiff.derivative(t -> ForwardDiff.derivative(u -> resid(u, fixed, bc, dz), t), 0.0)
ratios(label) = for bc in (false, true), dz in (2e-5, 3e-4, 1e-3)
  @printf("%s bmad_compat=%-5s dz=%.0e  |d2 root / d2 primal root| = %.2e\n", label, bc, dz,
          abs(d2(false, bc, dz) / d2(true, bc, dz)))
end

ratios("2 steps")
@eval CSR root_order(::Type{<:ForwardDiff.Dual{T,V}}) where {T,V} = 1   # exact signature, else the original wins
ratios("1 step ")
