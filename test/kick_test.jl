@testset "CSR kick" begin
  electron = Species("electron")
  sigma = 1e-3; Q = 1e-9
  z, q, alive = gaussian_line_bunch(sigma, Q, 20001)

  @testset "csr_kick_coef" begin
    # Electrons: ds * r_e / (e γ)
    @test csr_kick_coef(electron, 100.0, 0.01) ≈ 0.01 * R_ELECTRON / (E_CHARGE * 100)
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
    @test res2.I_csr_per_meter ≈ res.I_csr_per_meter rtol = 1e-12
    err_c = maximum(abs.(res2.kick_per_meter[core] .- W)) / scale
    @test err_c > err

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
