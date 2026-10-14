using CSR: CSRKickPoint, CSRKick1, s_source_calc, ddz_calc_csr, I_csr_calc, zbrent

@testset "Source point and kick integral" begin
  @testset "zbrent" begin
    x, ok = zbrent(x -> x^3 - 2, 0.0, 2.0, 1e-12, 1e-14)
    @test ok
    @test x ≈ cbrt(2) rtol = 1e-12
    @test !zbrent(x -> x^2 + 1, 0.0, 1.0, 1e-12, 1e-14)[2]
  end

  # Circle made of many short bends, centroid on the reference orbit; kick point mid-way through the
  # last element. For the source an angle φ behind: L = 2ρ sin(φ/2), Ls = ρφ, θ_sl = θ_lk = φ/2.
  rho = 10.0; L_ele = 0.1; n_ele = 80; gamma = 1e4
  geom = CSRGeometry(circle_lattice(rho, L_ele, n_ele)...)
  s_kick = L_ele / 2
  kp = CSRKickPoint(geom, n_ele, s_kick, gamma)

  @testset "Source point on a circle" begin
    for dz in (1e-6, 1e-4, 1e-3, 5e-3)
      k = CSRKick1(; ix_ele_source = n_ele, dz_particles = dz)
      k, _ = s_source_calc(k, geom, kp)
      @test 1 <= k.ix_ele_source <= n_ele
      # Retardation condition at the root (tolerance set by zbrent's abs_tol = 1e-8 on s)
      @test abs(k.L / (2gamma^2) + k.dL - dz) < 1e-10 + 1e-6 * dz
      # Path-length excess vs the circle (spline approximation of each 0.01 rad arc)
      phi = 2asin(k.L / (2rho))
      @test k.dL ≈ rho * phi - k.L rtol = 2e-3
      @test k.theta_sl ≈ phi / 2 rtol = 2e-3
      @test k.theta_lk ≈ phi / 2 rtol = 2e-3
      # Kick integral vs the circle closed form, and vs the γ → ∞ steady-state kernel
      k = I_csr_calc(k, 2, dz, geom, kp, 0.0)
      th = phi / 2
      I_circle = -2 * (k.dL / dz + gamma^2 * th^2 / (1 + gamma^2 * th^2)) / k.L
      @test k.I_csr ≈ I_circle rtol = 5e-3
      if gamma * phi > 100
        @test k.I_csr ≈ -2 / cbrt(3) * rho^(-2/3) * dz^(-1/3) rtol = 1e-2
      end
    end
  end

  # Default mode: path excess exact (path_excess). bmad_compat: small-angle form, too large by Ls⁵/(1920ρ⁴), i.e. a
  # relative error Ls²/(80ρ²) of the excess Ls³/(24ρ²).
  @testset "Path excess on a circle: exact (default), small-angle (compat)" begin
    geom_c = CSRGeometry(circle_lattice(rho, L_ele, n_ele)...; bmad_compat = true)
    kp_c = CSRKickPoint(geom_c, n_ele, s_kick, gamma)
    for dz in (1e-4, 1e-3, 5e-3), (g, p, bc) in ((geom, kp, false), (geom_c, kp_c, true))
      k = CSRKick1(; ix_ele_source = n_ele, dz_particles = dz)
      k, _ = s_source_calc(k, g, p)
      phi = 2asin(k.L / (2rho))
      rel = (k.dL - (rho * phi - k.L)) / (rho * phi - k.L)
      if bc
        @test rel ≈ (rho * phi)^2 / (80rho^2) rtol = 0.1
      else
        @test abs(rel) < 1e-7
      end
    end
  end

  @testset "First-bin integral" begin
    ss_int(dz) = -3^(2/3) * rho^(-2/3) * dz^(2/3)   # ∫₀ᶻ of the γ → ∞ steady-state kernel
    function first_bin(g, dz)
      kpg = CSRKickPoint(g, n_ele, s_kick, gamma)
      k = CSRKick1(; ix_ele_source = n_ele, dz_particles = dz)
      k, _ = s_source_calc(k, g, kpg)
      k = I_csr_calc(k, 1, dz, g, kpg, 0.0)
      return k
    end
    # Source inside the kick element: Bmad's closed form
    k = first_bin(geom, 2e-8)
    @test k.ix_ele_source == n_ele
    @test k.I_int_csr ≈ ss_int(2e-8) rtol = 3e-2   # the 1/γ² log term is still ~2% at this z
    # Source upstream of the kick element: closed form + quadrature
    for dz in (1e-6, 6e-5, 1e-3)
      k = first_bin(geom, dz)
      @test k.ix_ele_source < n_ele
      @test k.I_int_csr ≈ ss_int(dz) rtol = 3e-3
    end
    # Bmad's approximation (bmad_compat) tends to 2/3 of the integral in steady state
    geom_c = CSRGeometry(circle_lattice(rho, L_ele, n_ele)...; bmad_compat = true)
    @test first_bin(geom_c, 1e-3).I_int_csr / ss_int(1e-3) ≈ 2 / 3 rtol = 1e-2
  end

  @testset "Source upstream of the lattice" begin
    # Short bend: large lags put the source on the infinite upstream drift (ix_ele_source = 0)
    g2 = CSRGeometry([0.2], [0.1], zeros(2, 6))
    kp2 = CSRKickPoint(g2, 1, 0.1, 50.0)
    k = CSRKick1(; ix_ele_source = 1, dz_particles = 1e-3)
    k, s = s_source_calc(k, g2, kp2)
    @test k.ix_ele_source == 0
    @test s < 0
    # Exact retardation Ls - βL = dz on the straight upstream line
    beta = sqrt(1 - 1 / 50.0^2)
    @test (k.L + k.dL) - beta * k.L ≈ 1e-3 rtol = 1e-8
  end
  @testset "Source fields match the returned root (V7)" begin
    # At dz = zz the root is the kick element's entrance with ddz(0) == 0 exactly; zbrent returns s = 0 without
    # evaluating there. k must still describe the source at s = 0, not zbrent's last evaluation.
    c = zeros(3, 6); c[2, 2] = -1.7e-18   # round-off level centroid angle, as from tracking
    geom = CSRGeometry([0.1, 0.1], [0.1, 0.1], c)
    kp = CSR.CSRKickPoint(geom, 2, 0.1, 1e4)
    spl = geom.ele[2].spline
    L = kp.s_chord_kick
    zz = L / (2kp.gamma2) + CSR.path_excess(0.0, L, spl, spl[2] + spl[3] * L + spl[4] * L^2)
    k = CSR.CSRKick1(; ix_ele_source = 2, dz_particles = zz)
    k, s = CSR.s_source_calc(k, geom, kp)
    @test k.ix_ele_source == 2 && s == 0
    @test k.L ≈ L rtol = 1e-6
    # Kick insensitive to the round-off centroid angle
    z, q, alive = gaussian_line_bunch(1e-3, 1e-9, 201)
    p = CSRParams(n_bin = 40, ds_step = 0.05)
    r0 = csr_kick!(zeros(201), z, q, alive, CSRGeometry([0.1, 0.1], [0.1, 0.1], zeros(3, 6)), 2, 0.1, 1e4, Species("electron"), p)
    r1 = csr_kick!(zeros(201), z, q, alive, geom, 2, 0.1, 1e4, Species("electron"), p)
    @test r1.kick_per_meter ≈ r0.kick_per_meter rtol = 1e-8
  end
end
