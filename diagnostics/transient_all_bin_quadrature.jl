using CSR, Printf, AtomicAndPhysicalConstants
using CSR: CSRKickPoint, CSRKick1, s_source_calc, I_csr_calc, with_fields, I_csr_integral, csr_convolve!, csr_kick_coef
include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))
e = Species("electron")
gam = 42e6 / massof(e); pc = sqrt(42e6^2 - massof(e)^2); rho2 = pc / C_LIGHT / 0.17
sig2 = 0.000899377
z2, q2, a2 = gaussian_line_bunch(sig2, 77e-12, 40001)
zt = range(-2sig2, 2sig2; length = 41)
interp(zc, kk, x) = (i = searchsortedlast(zc, x); f = (x - zc[i]) / (zc[2]-zc[1]); (1-f)*kk[i] + f*kk[i+1])
# mode: :bmad, :csrjl (first-bin quadrature + trapezoid), :allquad (every bin by quadrature)
function wake(nb, s, mode)
  bc = mode == :bmad
  g = CSRGeometry([0.1], [1/rho2], zeros(2,6); bmad_compat = bc)
  p = CSRParams(n_bin=nb, particle_bin_span=2, ds_step=0.01, bmad_compat=bc)
  bins = csr_bin_particles(z2, q2, a2, p); dz = bins.dz
  kp = CSRKickPoint(g, 1, s, gam)
  Iint = zeros(nb); Ic = zeros(nb); k = CSRKick1(; ix_ele_source = 1)
  for i in 1:nb
    k, _ = s_source_calc(with_fields(k; dz_particles = i*dz), g, kp); k = I_csr_calc(k, i, dz, g, kp, i==1 ? 0.0 : Ic[i-1])
    Ic[i] = k.I_csr; Iint[i] = k.I_int_csr
    if mode == :allquad && i > 1
      Iint[i] = I_csr_integral((i-1)*dz, i*dz, g, kp; n = 8)
    end
  end
  kick = zeros(nb); csr_convolve!(kick, Iint, bins.edge_dcharge_density_dz, 1.0)
  return bins.z_center, kick
end
werr(zk, Wr, sc) = maximum(abs.(interp.(Ref(zk[1]), Ref(zk[2]), zt) .- Wr)) / sc
for s in (0.05, 0.1)
  zc, kr = wake(1600, s, :allquad); Wr = interp.(Ref(zc), Ref(kr), zt); sc = maximum(abs, Wr)
  @printf("s=%.2f  reference: all-bin quadrature, n_bin=1600\n", s)
  for nb in (40, 100, 400, 1600)
    errs = [werr(wake(nb, s, m), Wr, sc) for m in (:allquad, :csrjl, :bmad)]
    @printf("  n_bin=%4d  all-bin quad err=%8.2e  CSR.jl(first-bin quad) err=%8.2e  bmad err=%8.2e\n", nb, errs...)
  end
end
