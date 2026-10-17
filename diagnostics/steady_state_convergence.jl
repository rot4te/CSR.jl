using CSR, Printf, AtomicAndPhysicalConstants
include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))
e = Species("electron")
sigma = 1e-3; Q = 1e-9
z, q, alive = gaussian_line_bunch(sigma, Q, 40001)
rho = 10.0; L_ele = 0.1; n_ele = 80; gamma = 1e4
W(x) = steady_state_dpz_ds(x, y -> dgaussian(y, sigma), rho, Q/E_CHARGE, gamma; u_max = (12sigma)^(2/3))
println("Steady state (rho=10 m, sigma_z=1 mm, gamma=1e4): max |kick - analytic| / max|analytic| over |z|<=2 sigma")
for nb in (50, 100, 200, 400, 800)
  out = Float64[]
  for bc in (false, true)
    p = CSRParams(n_bin = nb, particle_bin_span = 2, ds_step = L_ele, bmad_compat = bc)
    g = CSRGeometry(circle_lattice(rho, L_ele, n_ele)...; bmad_compat = bc)
    r = csr_kick!(zeros(length(z)), z, q, alive, g, n_ele, L_ele/2, gamma, e, p)
    core = findall(x -> abs(x) <= 2sigma, r.z_center)
    Wc = W.(r.z_center[core])
    push!(out, maximum(abs.(r.kick_per_meter[core] .- Wc)) / maximum(abs, Wc))
  end
  @printf("  n_bin=%4d  dz/sigma=%6.4f  CSR.jl err=%8.2e  bmad_compat err=%8.2e\n", nb, 12/(nb-5), out[1], out[2])
end
# Bmad regression-test-like case: 42 MeV e-, sbend L=0.1, b_field=0.17 T, n_bin=40, span 2, ds=0.01, sigma_z=0.899 mm
gam = 42e6 / massof(e); pc = sqrt(42e6^2 - massof(e)^2)
rho2 = pc / C_LIGHT / 0.17
sig2 = 0.000899377
z2, q2, a2 = gaussian_line_bunch(sig2, 77e-12, 20001)
println(@sprintf("\nRegression-test-like single bend: rho=%.4f m, gamma=%.2f, n_bin=40", rho2, gam))
for s in (0.01, 0.05, 0.1)
  r = [csr_kick!(zeros(length(z2)), z2, q2, a2, CSRGeometry([0.1], [1/rho2], zeros(2,6); bmad_compat=bc), 1, s, gam, e, CSRParams(n_bin=40, particle_bin_span=2, ds_step=0.01, bmad_compat=bc)) for bc in (false, true)]
  imid = argmin(abs.(r[1].z_center)); ipk = argmax(abs.(r[1].kick_per_meter))
  @printf("  s=%.2f  center: CSR.jl %10.3e  bmad %10.3e  (bmad/CSR.jl = %.4f);  |peak| ratio %.4f\n", s, r[1].kick_per_meter[imid], r[2].kick_per_meter[imid], r[2].kick_per_meter[imid]/r[1].kick_per_meter[imid], maximum(abs, r[2].kick_per_meter)/maximum(abs, r[1].kick_per_meter))
end
