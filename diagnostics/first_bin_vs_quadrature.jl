# First-bin kick integral I_int(1): CSR.jl default vs bmad_compat vs high-resolution quadrature.
# Steady-state circle: rho = 10 m, 80 bends of 0.1 m, gamma = 1e4, kick mid-way through the last element.
# Run from the CSR.jl root: julia --project=. planning/diagnostics/first_bin_vs_quadrature.jl
using CSR, Printf
using CSR: CSRKickPoint, CSRKick1, s_source_calc, I_csr_calc
include(joinpath(@__DIR__, "..", "test", "utils.jl"))
rho = 10.0; L_ele = 0.1; n_ele = 80; gamma = 1e4
geoms = (CSRGeometry(circle_lattice(rho, L_ele, n_ele)...),
         CSRGeometry(circle_lattice(rho, L_ele, n_ele)...; bmad_compat = true))
kp = CSRKickPoint(geoms[1], n_ele, L_ele/2, gamma)
function Ifun(dz)
  k, _ = s_source_calc(CSRKick1(; ix_ele_source = n_ele, dz_particles = dz), geoms[1], kp)
  I_csr_calc(k, 2, dz, geoms[1], kp, 0.0).I_csr
end
function I_int1(g, dz)
  kpg = CSRKickPoint(g, n_ele, L_ele/2, gamma)
  k, _ = s_source_calc(CSRKick1(; ix_ele_source = n_ele, dz_particles = dz), g, kpg)
  k = I_csr_calc(k, 1, dz, g, kpg, 0.0)
  return k.I_int_csr, k.ix_ele_source == n_ele
end
for dz in (2e-8, 1e-6, 6e-5, 1e-3)
  # reference: ∫0^dz I(u) du with u = t^3 (I ~ u^(-1/3) => smooth in t)
  ex = simpson(t -> t == 0 ? 0.0 : Ifun(t^3) * 3t^2, 0.0, cbrt(dz), 200)
  (a, same), (b, _) = I_int1(geoms[1], dz), I_int1(geoms[2], dz)
  @printf("dz=%8.1e  source in kick ele=%5s  CSR.jl/quad=%.4f  bmad_compat/quad=%.4f  steady-state ∫=%11.4e\n",
          dz, same, a/ex, b/ex, -3^(2/3)*rho^(-2/3)*dz^(2/3))
end
