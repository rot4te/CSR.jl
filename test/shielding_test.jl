# Phase 5: shielding by parallel plates (image charges).

# Longitudinal field of a charge moving uniformly with Lorentz factor
# gamma, at present-time separation dz (kicked minus source) and
# transverse distance y, without q/(4πε₀).
uniform_field(dz, y, gamma) = gamma * dz / (gamma^2 * dz^2 + y^2)^1.5

# Image kick for lags dz (increasing, warm-started as in
# csr_image_table)
function image_kicks(geom, kp, dzs)
  k = CSR.CSRKick1(; ix_ele_source = kp.ix_ele_kick)
  out = Float64[]; ixs = Int[]
  for dz in dzs
    k = CSR.with_fields(k; dz_particles = dz)
    k, _ = CSR.s_source_calc(k, geom, kp)
    push!(out, CSR.image_charge_kick(k, geom, kp))
    push!(ixs, k.ix_ele_source)
  end
  return out, ixs
end

eloss(r) = sum(r.kick_per_meter .* r.charge_per_meter) *
    (r.z_center[2] - r.z_center[1])

@testset "Shielding (image charges)" begin
  electron = Species("electron")

  @testset "Parameters" begin
    @test_throws ErrorException CSRParams(n_shield_images = -1)
    @test_throws ErrorException CSRParams(n_shield_images = 2)
    @test_throws ErrorException CSRParams(n_shield_images = 2,
        beam_chamber_height = -0.01)
    p = CSRParams(n_shield_images = 2, beam_chamber_height = 0.01)
    @test p.n_shield_images == 2 && p.beam_chamber_height == 0.01
    # Without images the chamber height is ignored
    z, q, alive = gaussian_line_bunch(1e-3, 1e-9, 401)
    geom = CSRGeometry(circle_lattice(10.0, 0.1, 10)...)
    ds = 0.1 / max(1, round(Int, Beamlines.scalarize_all(0.1) / 0.1,
                            RoundNearestTiesAway))
    r1 = csr_kick!(zeros(401), z, q, alive, geom, 10, ds, 0.05, 1e3,
                   electron, CSRParams(n_bin = 40))
    r2 = csr_kick!(zeros(401), z, q, alive, geom, 10, ds, 0.05, 1e3,
                   electron,
                   CSRParams(n_bin = 40, beam_chamber_height = 0.01))
    @test r1.kick_per_meter == r2.kick_per_meter
  end

  # In a drift the image kick is the field of a uniformly moving
  # charge (velocity term only), for sources behind and ahead of the
  # kick point, in the kick element, downstream elements, and on the
  # drifts outside the lattice.
  @testset "Drift: uniform-motion field, $name" for (name, L, ix,
      s_kick) in (
      ("one element", [10.0], 1, 5.0),
      ("split into elements", fill(0.01, 20), 10, 0.005),
      ("kick near the lattice end", fill(0.01, 3), 3, 0.009),
      ("kick at the lattice end", [0.05, 0.05], 2, 0.05))
    gamma = 100.0; y = 0.01
    peak = 0.385 / y^2   # maximum of uniform_field over dz
    dzs = range(-0.04, 0.04; length = 161)
    geom = CSRGeometry(L, zeros(length(L)), zeros(length(L) + 1, 6))
    kp = CSR.CSRKickPoint(geom, ix, s_kick, gamma;
                                 y_source = y)
    img, ixs = image_kicks(geom, kp, dzs)
    # Tolerance: Bmad's retardation z = L/(2γ²) + dL inside elements
    # (error ~ L/(8γ⁴); 3e-5 of the peak at dz = 0 in the
    # one-element case)
    @test maximum(abs.(img .- uniform_field.(dzs, y,
        gamma))) < 1e-4 * peak
    # some sources are downstream of the kick element
    length(L) > 1 && @test maximum(ixs) > ix
    # bmad_compat: no downstream search, and the source direction is
    # reversed for sources ahead (V8)
    geom_c = CSRGeometry(L, zeros(length(L)), zeros(length(L) + 1, 6);
                         bmad_compat = true)
    img_c, ixs_c = image_kicks(geom_c, CSR.CSRKickPoint(geom_c,
        ix, s_kick, gamma; y_source = y), dzs)
    @test maximum(ixs_c) == ix
    # At dz = -0.03 the source is ahead of the kick point: Bmad's
    # field is several times off (wrong direction of the source, or
    # the source left at the end of the kick element)
    i = findfirst(≈(-0.03), dzs)
    ana = uniform_field(dzs[i], y, gamma)
    @test img[i] ≈ ana rtol = 1e-6
    @test abs(img_c[i] - ana) > 5abs(ana)
  end

  @testset "V8: source direction for a source ahead in the kick element" begin
    for g in (0.0, 0.5), bc in (false, true)
      geom = CSRGeometry([1.0], [g], zeros(2, 6); bmad_compat = bc)
      kp = CSR.CSRKickPoint(geom, 1, 0.3, 100.0;
                                   y_source = 0.01)
      k = CSR.CSRKick1(; ix_ele_source = 1,
                              dz_particles = -0.05)
      k, s = CSR.s_source_calc(k, geom, kp)
      @test s > kp.s_chord_kick   # ahead, in the kick element
      e = geom.ele[1]
      true_theta = e.theta_chord + CSR.spline_d1(e.spline, s)
      d = CSR.modulo2(k.floor_s.theta - true_theta, pi)
      @test abs(bc ? abs(d) - pi : d) < 1e-3
    end
  end

  @testset "Downstream infinite drift: exact retardation" begin
    # Kick in the last element (a bend): sources ahead beyond the
    # lattice end
    for g in (0.0, 2.0)
      geom = CSRGeometry([0.1, 0.05], [0.0, g], zeros(3, 6))
      gamma = 50.0; y = 0.02
      kp = CSR.CSRKickPoint(geom, 2, 0.03, gamma; y_source = y)
      k = CSR.CSRKick1(; ix_ele_source = 2,
                              dz_particles = -0.08)
      k, t = CSR.s_source_calc(k, geom, kp)
      @test k.ix_ele_source == 3
      @test t > 0
      beta = sqrt(1 - 1 / gamma^2)
      # Ls - βL = dz, Ls < 0
      @test (k.L + k.dL) - beta * k.L ≈ -0.08 rtol = 1e-10
      @test k.floor_s.theta == geom.ele[end].floor1.theta
      @test CSR.source_s(geom, k.ix_ele_source, t) ≈ 0.15 + t
      # A search warm-started past the lattice end walks back when the
      # source is inside the lattice
      k = CSR.with_fields(k; dz_particles = -0.001)
      k, _ = CSR.s_source_calc(k, geom, kp)
      k2 = CSR.CSRKick1(; ix_ele_source = 2,
                               dz_particles = -0.001)
      k2, _ = CSR.s_source_calc(k2, geom, kp)
      @test k.ix_ele_source == k2.ix_ele_source <= 2
      @test k.floor_s == k2.floor_s
    end
  end

  @testset "Upstream infinite drift: warm start walks back" begin
    for g in (0.0, 2.0)
      geom = CSRGeometry([0.05, 0.1], [g, 0.0], zeros(3, 6))
      kp = CSR.CSRKickPoint(geom, 2, 0.07, 50.0;
                                   y_source = 0.02)
      k = CSR.CSRKick1(; ix_ele_source = 2,
                              dz_particles = 0.08)
      k, _ = CSR.s_source_calc(k, geom, kp)
      @test k.ix_ele_source == 0
      # A search warm-started upstream of the lattice walks back when
      # the source is inside the lattice sources in element 1, and
      # ahead of the kick point
      for dz in (-0.002, -0.03)
        k = CSR.with_fields(k; ix_ele_source = 0,
                                   dz_particles = dz)
        k, _ = CSR.s_source_calc(k, geom, kp)
        k2 = CSR.CSRKick1(; ix_ele_source = 2,
                                 dz_particles = dz)
        k2, _ = CSR.s_source_calc(k2, geom, kp)
        @test k.ix_ele_source == k2.ix_ele_source >= 1
        @test k.floor_s == k2.floor_s
      end
    end
  end

  # Hat-weighted bin integral of the uniform-motion field: second
  # difference of G with G'' = uniform_field (G(u) =
  # -asinh(γu/y)/γ²), divided by dz².
  hat_uniform_field(l, dz, y, gamma) =
    -(asinh(gamma * (l + 1) * dz / y) - 2asinh(gamma * l * dz / y) +
      asinh(gamma * (l - 1) * dz / y)) /
    (gamma * dz)^2

  @testset "Drift: hat-integrated image table, γ = $gamma, y = $y, dz = $dz" for (gamma, y, dz) in (
      (100.0, 0.01, 2e-4), (1e4, 3e-3, 1e-4))
    n_bin = 40
    geom = CSRGeometry([10.0], [0.0], zeros(2, 6))
    kp = CSR.CSRKickPoint(geom, 1, 5.0, gamma; y_source = y)
    hat, _ = CSR.csr_image_table(geom, kp, dz, n_bin,
                                        CSR.CPU();
                                        bin_integrate = true)
    ana = [hat_uniform_field(l, dz, y, gamma) for l in -n_bin:n_bin]
    @test maxrel(hat, ana) < 1e-4
    # Point sampling misses the structure when dz ≫ y/γ
    point, _ = CSR.csr_image_table(geom, kp, dz, n_bin,
        CSR.CPU())
    gamma * dz > 10y && @test maxrel(point, ana) > 0.5
  end

  # Kernels vs. serial reference, image table independent of the lag
  # blocking
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, 5001)
  cases = (
    ("off-axis drift-bend-drift, kick in bend", 2, 0.17, 0.01),
    ("kick at element entrance", 3, 0.0, 0.01),
    ("kick near the lattice end, sources past it", 3, 0.195, 0.002),
  )
  @testset "csr_kick! vs reference: $name, bmad_compat = $bc" for (
      name, ix, s_kick, h) in cases, bc in (false, true)
    p = CSRParams(n_bin = 120, n_shield_images = 3,
                  beam_chamber_height = h, bmad_compat = bc)
    geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
    ds = L3[ix] / max(1, round(Int,
                               Beamlines.scalarize_all(L3[ix]) / 0.1,
                               RoundNearestTiesAway))
    pz1 = zeros(length(z)); pz2 = zeros(length(z))
    r1 = csr_kick!(pz1, z, q, alive, geom, ix, ds, s_kick, 200.0,
                   electron, p; kick_factor = 0.5)
    r2 = CSR.csr_kick_reference!(pz2, z, q, alive, geom, ix,
                                        ds, s_kick, 200.0, electron,
                                        p; kick_factor = 0.5)
    @test maxrel(r1.kick_per_meter, r2.kick_per_meter) < 1e-12
    @test maxrel(pz1, pz2) < 1e-12
    # Images change the kick
    r0 = csr_kick!(zeros(length(z)), z, q, alive, geom, ix, ds,
                   s_kick, 200.0, electron,
                   CSRParams(n_bin = 120, bmad_compat = bc);
                   kick_factor = 0.5)
    @test maxrel(r1.kick_per_meter, r0.kick_per_meter) > 1e-3
  end

  @testset "Image table independent of the lag blocking" begin
    for bc in (false, true)
      geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
      kp = CSR.CSRKickPoint(geom, 3, 0.195, 200.0;
                                   y_source = 0.002,
                                   kick_factor = -2.0)
      tables = [CSR.csr_image_table(geom, kp, 1e-4, 120,
          CSR.CPU(); n_block) for n_block in (1, 7, 241)]
      @test tables[2] == tables[1]
      @test tables[3] == tables[1]
      # sources past the lattice end are reached
      bc || @test maximum(tables[1][2]) > sum(L3)
      hats = [CSR.csr_image_table(geom, kp, 1e-4, 120,
          CSR.CPU(); n_block, bin_integrate = true)
              for n_block in (1, 7, 241)]
      @test hats[2] == hats[1]
      @test hats[3] == hats[1]
      @test hats[1][2] == tables[1][2]   # s_source at the lags
    end
  end

  # Steady-state bend (ρ = 10 m, σ = 1 mm, γ = 1e4). Converged in
  # n_shield_images and n_bin (checked up to 64 layers and 7680 bins;
  # Bmad's point-sampled image sum needs bins much finer than the
  # image kernel's structure).
  @testset "Steady-state bend: free-space limit and shielding" begin
    N = 40001
    zb, qb, aliveb = gaussian_line_bunch(1e-3, 1e-9, N)
    geom = CSRGeometry(circle_lattice(10.0, 0.1, 80)...)
    ds = 0.1 / max(1, round(Int, Beamlines.scalarize_all(0.1) / 0.1,
                            RoundNearestTiesAway))
    kick(p) = csr_kick!(zeros(N), zb, qb, aliveb, geom, 80, ds, 0.05,
                        1e4, electron, p)
    r0 = kick(CSRParams(n_bin = 480))
    @test eloss(r0) < 0
    ratio = Dict{Float64,Float64}()
    for h in (0.01, 0.03, 0.1, 1.0, 10.0)
      r = kick(CSRParams(n_bin = 480, n_shield_images = 32,
                         beam_chamber_height = h))
      ratio[h] = eloss(r) / eloss(r0)
      h >= 1 && @test maxrel(r.kick_per_meter,
                             # free-space limit
                             r0.kick_per_meter) < 1e-3
    end
    @test 0 < ratio[0.01] < 0.01    # strong shielding
    @test 0 < ratio[0.03] < 0.06
    @test 0.9 < ratio[0.1] < 0.95   # mild shielding
    @test ratio[1.0] ≈ 1 rtol = 1e-3
    # h = 3 mm: the image kernel's structure is much finer than a bin.
    # The hat-weighted table gives the same result for coarse and fine
    # bins (fixed layer count, plain sum); point sampling
    # (bmad_compat) does not.
    geom_c = CSRGeometry(circle_lattice(10.0, 0.1, 80)...;
                         bmad_compat = true)
    kick3(p) = csr_kick!(zeros(N), zb, qb, aliveb,
                         p.bmad_compat ? geom_c : geom, 80, ds, 0.05,
                         1e4, electron, p)
    ratio3(n_bin; bc = false) =
      eloss(kick3(CSRParams(n_bin = n_bin, n_shield_images = 16,
                            beam_chamber_height = 0.003,
                            image_sum_order = 0, bmad_compat = bc))) /
      eloss(kick3(CSRParams(n_bin = n_bin, bmad_compat = bc)))
    @test ratio3(60) ≈ ratio3(480) rtol = 0.03
    @test !isapprox(ratio3(60; bc = true), ratio3(480; bc = true);
                    rtol = 0.5)
  end

  @testset "Averaged image-layer sum (image_sum_order)" begin
    @test CSRParams().image_sum_order == 8
    @test_throws ErrorException CSRParams(image_sum_order = -1)
    @test_throws ErrorException CSRParams(image_sum_order = 61)
    w(M, k; bc = false) = [CSR.image_layer_weight(n,
        CSRParams(n_shield_images = M, beam_chamber_height = 0.01,
                  image_sum_order = k,
                  bmad_compat = bc)) for n in 1:M]
    @test all(==(1), w(20, 0))
    @test all(==(1), w(20, 8; bc = true))
    # Weights reproduce the binomial average of the partial sums
    # S_{M-k} … S_M (S_0 = free space only)
    a = [(-1)^n * (1 + 0.3sin(n)) / n for n in 1:20]
    S = 1 .+ cumsum(a)
    @test 1 + sum(w(20, 8) .* a) ≈ sum(binomial(8,
        j) * S[12+j] for j in 0:8) / 2^8 rtol = 1e-14
    # k capped at M; weights non-increasing in [0, 1], summing to M -
    # k/2
    S0 = [1.0; S]
    @test 1 + sum(w(3, 8) .* a[1:3]) ≈ sum(binomial(3,
        j) * S0[1+j] for j in 0:3) / 2^3 rtol = 1e-14
    for (M, k) in ((20, 8), (3, 8), (64, 60))
      ws = w(M, k)
      @test issorted(ws; rev = true) && all(0 .<= ws .<= 1)
      @test sum(ws) ≈ M - min(k, M) / 2 rtol = 1e-14
    end

    # Steady-state bend on a 40 m arc (sources of high layers stay on
    # the arc), h = 3 mm: the averaged sum is converged at 32 layers,
    # the plain sum is not (PLAN.md, D24).
    N = 20001
    zb, qb, aliveb = gaussian_line_bunch(1e-3, 1e-9, N)
    geom = CSRGeometry(circle_lattice(10.0, 0.1, 400)...)
    ds = 0.1 / max(1, round(Int, Beamlines.scalarize_all(0.1) / 0.1,
                            RoundNearestTiesAway))
    kick(M, k) = csr_kick!(zeros(N), zb, qb, aliveb, geom, 400, ds,
                           0.05, 1e4, electron,
                           CSRParams(n_bin = 120, n_shield_images = M,
                               beam_chamber_height = 0.003,
                               image_sum_order = k)).kick_per_meter
    k32, k64 = kick(32, 8), kick(64, 8)
    @test maxrel(k32, k64) < 1e-4
    @test maxrel(kick(32, 0), k64) > 1
  end
end
