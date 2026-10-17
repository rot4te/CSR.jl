@testset "CSR kick" begin
  electron = Species("electron")
  sigma = 1e-3; Q = 1e-9
  z, q, alive = gaussian_line_bunch(sigma, Q, 20001)

  @testset "csr_kick_coef" begin
    # Electrons: ds * r_e / (e β² γ); Bmad (compat) ds * r_e / (e γ)
    gamma = 100.0
    beta2 = 1 - 1 / gamma^2
    @test csr_kick_coef(electron, gamma, 0.01) ≈ 0.01 * R_ELECTRON / (E_CHARGE * beta2 * gamma)
    @test csr_kick_coef(electron, gamma, 0.01; bmad_compat = true) ≈ 0.01 * R_ELECTRON / (E_CHARGE * gamma)
    @test csr_kick_coef(Species("positron"), gamma, 0.01) == csr_kick_coef(electron, gamma, 0.01)
    # Same field, same γ: Δpz ∝ 1/m. Bmad's has an extra 1/rel_mass (V2).
    proton = Species("proton")
    rel_mass = massof(proton) / massof(electron)
    @test csr_kick_coef(proton, gamma, 0.01) ≈ csr_kick_coef(electron, gamma, 0.01) / rel_mass rtol = 1e-14
    @test csr_kick_coef(proton, gamma, 0.01; bmad_compat = true) ≈
          csr_kick_coef(proton, gamma, 0.01) * beta2 / rel_mass rtol = 1e-14
    @test_throws ErrorException csr_kick_coef(Species("neutron"), 100.0, 0.01)
  end

  @testset "csr_n_step" begin
    p = CSRParams(ds_step = 0.03)
    @test csr_n_step(0.1, p) == (3, 0.1 / 3)
    @test csr_n_step(0.01, p) == (1, 0.01)
    @test_throws ErrorException csr_n_step(0.1, CSRParams())
  end

  @testset "Steady state in a long bend" begin
    # ρ = 10 m, 80 bends of 0.1 m. Overtaking length for 6σ is (24ρ²·6σ)^(1/3) ≈ 2.4 m < 8 m, so the
    # kick at the end is in the steady-state regime. γ = 1e4 makes 1/γ² terms negligible.
    rho = 10.0; L_ele = 0.1; n_ele = 80; gamma = 1e4
    params = CSRParams(n_bin = 200, particle_bin_span = 2, ds_step = L_ele)
    geom = CSRGeometry(circle_lattice(rho, L_ele, n_ele)...)
    pz = zeros(length(z))
    res = csr_kick!(pz, z, q, alive, geom, n_ele, L_ele / 2, gamma, electron, params)

    N = Q / E_CHARGE
    dl(x) = dgaussian(x, sigma)
    expected(x) = steady_state_dpz_ds(x, dl, rho, N, gamma; u_max = (12sigma)^(2/3))
    core = findall(x -> abs(x) <= 2sigma, res.z_center)
    W = expected.(res.z_center[core])
    scale = maximum(abs, W)
    err = maximum(abs.(res.kick_per_meter[core] .- W)) / scale
    @test err < 2e-2

    # Kick applied to particles = interpolated bin kick × step length
    ds = L_ele * geom.ele[n_ele].L_chord / L_ele
    i_mid = (length(z) + 1) ÷ 2
    @test pz[i_mid] ≈ expected(0.0) * ds rtol = 2e-2

    # Charge per meter integrates to the bunch charge
    @test sum(res.charge_per_meter) * (res.z_center[2] - res.z_center[1]) ≈ Q rtol = 1e-10

    # Qualitative: tail and core lose energy, head gains
    @test res.kick_per_meter[core[1]] < 0
    @test maximum(res.kick_per_meter) > 0

    # bmad_compat: same geometry (c3 = 0 on axis) but Bmad's first-bin approximation
    pz2 = zeros(length(z))
    params_c = CSRParams(n_bin = 200, particle_bin_span = 2, ds_step = L_ele, bmad_compat = true)
    geom_c = CSRGeometry(circle_lattice(rho, L_ele, n_ele)...; bmad_compat = true)
    res2 = csr_kick!(pz2, z, q, alive, geom_c, n_ele, L_ele / 2, gamma, electron, params_c)
    # Differs by the source fields at zbrent's last evaluation vs. the root (V7; tolerance 1e-8 m) and by the
    # small-angle path excess (Ls²/(80ρ²) relative, ~7e-4 at the longest lags here)
    @test res2.I_csr_per_meter ≈ res.I_csr_per_meter rtol = 1e-3
    err_c = maximum(abs.(res2.kick_per_meter[core] .- W)) / scale
    @test err_c > err

    # Every bin by quadrature (default) vs first bin only (planning/upstream-bin.md, options B and A)
    res_a = csr_kick!(zeros(length(z)), z, q, alive, geom, n_ele, L_ele / 2, gamma, electron,
                      CSRParams(n_bin = 200, particle_bin_span = 2, ds_step = L_ele, all_bin_quadrature = false))
    err_a = maximum(abs.(res_a.kick_per_meter[core] .- W)) / scale
    @test err < err_a
    @test res_a.I_csr_per_meter == res.I_csr_per_meter

    # Half kick at element ends
    pz3 = zeros(length(z))
    csr_kick!(pz3, z, q, alive, geom, n_ele, L_ele / 2, gamma, electron, params; kick_factor = 0.5)
    @test pz3 ≈ pz / 2 rtol = 1e-12

    # Dead particles are not kicked
    alive2 = copy(alive); alive2[i_mid] = false
    pz4 = zeros(length(z))
    csr_kick!(pz4, z, q, alive2, geom, n_ele, L_ele / 2, gamma, electron, params)
    @test pz4[i_mid] == 0
  end

  @testset "Straight drift: no CSR" begin
    gamma = 100.0
    params = CSRParams(n_bin = 100, particle_bin_span = 2, ds_step = 0.5)
    geom = CSRGeometry(fill(0.5, 20), zeros(20), zeros(21, 6))
    pz = zeros(length(z))
    res = csr_kick!(pz, z, q, alive, geom, 20, 0.25, gamma, electron, params)
    # Compare with the bend kick scale for the same bunch and step
    N = Q / E_CHARGE
    bend_scale = abs(steady_state_dpz_ds(0.0, x -> dgaussian(x, sigma), 10.0, N, gamma; u_max = (12sigma)^(2/3)))
    @test maximum(abs, res.kick_per_meter) < 1e-6 * bend_scale
  end

  @testset "Bin integrals: every bin by quadrature" begin
    # Entrance transient of the Bmad regression-test bend (42 MeV, B = 0.17 T, L = 0.1 m), kick at s = 0.05: the
    # integrand is steep over the first bins (planning/upstream-bin.md). Reference: every bin by quadrature, 1600 bins.
    gam = 42e6 / massof(electron)
    rho2 = sqrt(42e6^2 - massof(electron)^2) / C_LIGHT / 0.17
    sig2 = 0.000899377
    z2, q2, a2 = gaussian_line_bunch(sig2, 77e-12, 40001)
    g2 = CSRGeometry([0.1], [1 / rho2], zeros(2, 6))
    kick(nb, ab) = csr_kick!(zeros(40001), z2, q2, a2, g2, 1, 0.05, gam, electron,
                             CSRParams(n_bin = nb, ds_step = 0.01, all_bin_quadrature = ab))
    ref = kick(1600, true)
    zt = range(-2sig2, 2sig2; length = 41)
    interp(r, x) = (i = searchsortedlast(r.z_center, x); f = (x - r.z_center[i]) / (r.z_center[2] - r.z_center[1]);
                    (1 - f) * r.kick_per_meter[i] + f * r.kick_per_meter[i+1])
    Wr = interp.(Ref(ref), zt)
    werr(r) = maximum(abs.(interp.(Ref(r), zt) .- Wr)) / maximum(abs, Wr)
    errs = [werr(kick(nb, true)) for nb in (40, 100, 400)]
    @test issorted(errs; rev = true)
    @test errs[3] < 5e-3                          # measured 2.6e-3
    @test werr(kick(400, false)) > 3e-2           # first bin only: trapezoid error, measured 5.0e-2

    # all_bin_quadrature = false: trapezoid for bins ≥ 2
    kp = CSR.CSRKickPoint(g2, 1, 0.05, gam)
    I_csr, I_int, _ = CSR.csr_wake_table(g2, kp, 1e-4, 40, CSR.CPU(); all_bin_quadrature = false)
    @test I_int[2:end] == (I_csr[2:end] .+ I_csr[1:end-1]) .* 1e-4 ./ 2
    # bmad_compat ignores all_bin_quadrature (Bmad: trapezoid)
    g2c = CSRGeometry([0.1], [1 / rho2], zeros(2, 6); bmad_compat = true)
    kc(ab) = csr_kick!(zeros(40001), z2, q2, a2, g2c, 1, 0.05, gam, electron,
                       CSRParams(n_bin = 40, ds_step = 0.01, all_bin_quadrature = ab, bmad_compat = true)).kick_per_meter
    @test kc(true) ≈ kc(false) rtol = 1e-12
  end

  @testset "Too few particles" begin
    params = CSRParams(n_bin = 100, ds_step = 0.1)
    geom = CSRGeometry([0.1], [0.1], zeros(2, 6))
    zz, qq, aa = gaussian_line_bunch(1e-3, 1e-9, 50)
    @test_throws ErrorException csr_kick!(zeros(50), zz, qq, aa, geom, 1, 0.05, 100.0, electron, params)
  end

  @testset "bmad_compat must match" begin
    params = CSRParams(n_bin = 40, ds_step = 0.1, bmad_compat = true)
    geom = CSRGeometry([0.1], [0.1], zeros(2, 6))
    @test_throws ErrorException csr_kick!(zeros(length(z)), z, q, alive, geom, 1, 0.05, 100.0, electron, params)
  end
end
