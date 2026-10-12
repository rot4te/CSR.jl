using CSR, Printf, AtomicAndPhysicalConstants
using CSR: CSRKickPoint, CSRKick1, s_source_calc!, I_csr!, I_csr_value, I_csr_integral
include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))
e = Species("electron")
gam = 42e6 / massof(e); pc = sqrt(42e6^2 - massof(e)^2); rho2 = pc / C_LIGHT / 0.17
sig2 = 0.000899377
z2, q2, a2 = gaussian_line_bunch(sig2, 77e-12, 20001)
p = CSRParams(n_bin=40, particle_bin_span=2, ds_step=0.01)
bins = csr_bin_particles(z2, q2, a2, p); dz = bins.dz
for s in (0.01, 0.05, 0.1), bc in (false, true)
  g = CSRGeometry([0.1], [1/rho2], zeros(2,6); bmad_compat=bc)
  kp = CSRKickPoint(g, 1, s, gam)
  k = CSRKick1(); k.ix_ele_source = 1; k.dz_particles = dz
  s_source_calc!(k, g, kp); I_csr!(k, 1, dz, g, kp, 0.0)
  @printf("s=%.2f compat=%5s  ix_src=%d  I_csr(dz)=%10.3e  I_int1=%10.3e\n", s, bc, k.ix_ele_source, k.I_csr, k.I_int_csr)
end
# I(u) profile across first bin for s=0.05
g = CSRGeometry([0.1], [1/rho2], zeros(2,6)); kp = CSRKickPoint(g, 1, 0.05, gam)
L = kp.s_chord_kick; spl = g.ele[1].spline
dL = CSR.dspline_len(0.0, L, spl, spl[2]+spl[3]*L+spl[4]*L^2); zz = L/(2gam^2) + dL
@printf("\ns=0.05: zz=%.3e  dz=%.3e\n", zz, dz)
for u in zz .* (1.0001, 1.01, 1.1, 2, 5) ∪ dz .* (0.01, 0.1, 0.5, 1.0)
  k = CSRKick1(); k.ix_ele_source = 1; k.dz_particles = u; s_source_calc!(k, g, kp)
  @printf("  u=%10.3e  ix_src=%d  dL=%10.3e  L=%9.3e  th_sl=%9.2e th_lk=%9.2e  I=%11.4e\n", u, k.ix_ele_source, k.dL, k.L, k.theta_sl, k.theta_lk, I_csr_value(k, kp))
end
for n in (16, 64, 256, 1024)
  @printf("  quadrature n=%5d: %12.5e\n", n, I_csr_integral(zz, dz, g, kp; n))
end
