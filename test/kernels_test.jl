# Phase 4 gate: KernelAbstractions kernels (CPU backend) == serial
# phase 1 reference (src/reference.jl). The deposit uses atomic adds,
# so bin charges (and everything downstream of them) may differ in the
# last bits when Julia runs with several threads. Geometry-only
# quantities (grid, wake table) must match exactly.

# Gaussian bunch on a jittered grid (deterministic), so that particles
# do not sit on bin-aligned positions.
function kernel_test_bunch(sigma, Q, n)
  z = collect(range(-6sigma, 6sigma;
                    length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n)
  alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end

maxrel(a, b) = maximum(abs, a .- b) / maximum(abs, b)

@testset "Kernels vs. serial reference" begin
  electron = Species("electron")
  sigma = 1e-3; Q = 1e-9
  z, q, alive = kernel_test_bunch(sigma, Q, 5001)
  params = CSRParams(n_bin = 120, particle_bin_span = 2)

  @testset "Binning" begin
    b = csr_bin_particles(z, q, alive, params)
    r = CSR.csr_bin_particles_reference(z, q, alive, params)
    @test b.dz == r.dz
    @test b.z0_edge == r.z0_edge
    @test b.z_center == r.z_center
    @test b.charge ≈ r.charge rtol = 1e-13
    @test b.n_particle ≈ r.n_particle rtol = 1e-13
    @test maxrel(b.edge_dcharge_density_dz,
                 r.edge_dcharge_density_dz) < 1e-12
    @test_throws ErrorException csr_bin_particles(z, q, fill(false,
        length(z)), params)
  end

  @testset "Convolution and kick application" begin
    r = CSR.csr_bin_particles_reference(z, q, alive, params)
    I_int = [sin(0.1i) / i for i in 1:params.n_bin]
    k1 = zeros(params.n_bin)
    k2 = zeros(params.n_bin)
    CSR.csr_convolve!(k1, I_int, r.edge_dcharge_density_dz,
                             1.7)
    CSR.csr_convolve_reference!(k2, I_int,
                                       r.edge_dcharge_density_dz, 1.7)
    @test k1 == k2
    pz1 = fill(1e-4, length(z))
    pz2 = copy(pz1)
    CSR.csr_apply_kicks!(pz1, z, alive, r, k1)
    CSR.csr_apply_kicks_reference!(pz2, z, alive, r, k2)
    @test pz1 == pz2
    @test all(pz1[.!alive] .== 1e-4)
    # A particle outside the grid is an internal error
    zbad = copy(z); zbad[1] = 1.0
    @test_throws ErrorException CSR.csr_apply_kicks!(copy(pz1),
        zbad, alive, r, k1)
  end

  # Kicks: off-axis centroid (c3 ≠ 0), drift-bend-drift with sources
  # upstream of the lattice, kicks at an element entrance (first-bin
  # walk back), and bmad_compat.
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  rho = 10.0; L_ele = 0.1; n_ele = 80
  cases = (
    ("off-axis drift-bend-drift, kick in bend", (L3, g3, c3), 2, 0.17,
     200.0),
    ("kick at element entrance", (L3, g3, c3), 3, 0.0, 200.0),
    ("source upstream of the lattice", ([0.05, 0.05], [2.0, 2.0],
                                        zeros(3, 6)), 1, 0.03, 50.0),
  )
  @testset "csr_kick!: $name, bmad_compat = $bc" for (name, (L, g,
      cen), ix, s_kick, gamma) in cases, bc in (false, true)
    p = CSRParams(n_bin = 120, particle_bin_span = 2,
                  bmad_compat = bc)
    geom = CSRGeometry(L, g, cen; bmad_compat = bc)
    ds = L[ix] / max(1, round(Int,
                              Beamlines.scalarize_all(L[ix]) / 0.1,
                              RoundNearestTiesAway))
    pz1 = zeros(length(z))
    pz2 = zeros(length(z))
    r1 = csr_kick!(pz1, z, q, alive, geom, ix, ds, s_kick, gamma,
                   electron, p; kick_factor = 0.5)
    r2 = CSR.csr_kick_reference!(pz2, z, q, alive, geom, ix,
                                        ds, s_kick, gamma, electron,
                                        p; kick_factor = 0.5)
    @test r1.z_center == r2.z_center
    @test r1.I_csr_per_meter == r2.I_csr_per_meter
    @test r1.s_source == r2.s_source
    @test r1.charge_per_meter ≈ r2.charge_per_meter rtol = 1e-13
    @test maxrel(r1.kick_per_meter, r2.kick_per_meter) < 1e-12
    @test maxrel(pz1, pz2) < 1e-12
    @test maximum(abs, pz1) > 0
  end

  @testset "Wake table independent of the lag blocking" begin
    # One block = the reference's warm-started walk; one lag per block
    # = every search starts in the kick element.
    for (L, g, cen, ix, s_kick) in ((circle_lattice(rho, L_ele,
        n_ele)..., n_ele, 0.05), (L3, g3, c3, 3, 0.0))
      geom = CSRGeometry(L, g, cen)
      kp = CSR.CSRKickPoint(geom, ix, s_kick, 200.0)
      dz = 12sigma / 100
      tables = [CSR.csr_wake_table(geom, kp, dz, 120,
          CSR.CPU(); n_block) for n_block in (1, 7, 120)]
      @test tables[2] == tables[1]
      @test tables[3] == tables[1]
      tables = [CSR.csr_wake_table(geom, kp, dz, 120,
          CSR.CPU(); n_block,
          all_bin_quadrature = true) for n_block in (1, 7, 120)]
      @test tables[2] == tables[1]
      @test tables[3] == tables[1]
    end
  end

  @testset "Sources upstream of the lattice are reached" begin
    geom = CSRGeometry([0.05, 0.05], [2.0, 2.0], zeros(3, 6))
    r = csr_kick!(zeros(length(z)), z, q, alive, geom, 1,
                  0.05 / max(1, round(Int,
                                      Beamlines.scalarize_all(0.05) /
                                      0.1, RoundNearestTiesAway)),
                  0.03, 50.0, electron, params)
    @test any(<(0), r.s_source)
  end

  @testset "Fortran nint" begin
    for x in (0.5, -0.5, 1.5, 2.4999999999999996, -2.5,
              0.49999999999999994, 3.0, -0.0)
      @test CSR.nint(big(x)) == CSR.nint(x) ==
          round(Int, x, RoundNearestTiesAway)
    end
  end

  @testset "Generic element types" begin
    geom = CSRGeometry(L3, g3, c3)
    ds = L3[2] / max(1, round(Int,
                              Beamlines.scalarize_all(L3[2]) / 0.1,
                              RoundNearestTiesAway))
    pzr = zeros(length(z))
    ref = CSR.csr_kick_reference!(pzr, z, q, alive, geom, 2,
        ds, 0.17, 200.0, electron, params)

    # Float32 particles (atomic deposit on the promoted charge type)
    z32, q32 = Float32.(z), Float32.(q)
    pz32 = zeros(Float32, length(z))
    r32 = csr_kick!(pz32, z32, q32, alive, geom, 2, ds, 0.17, 200.0,
                    electron, params)
    @test eltype(pz32) == Float32
    @test maxrel(r32.kick_per_meter, ref.kick_per_meter) < 1e-3

    # BigFloat particles: serial deposit (no atomic add for this type)
    zb, qb = big.(z), big.(q)
    pzb = zeros(BigFloat, length(z))
    rb = csr_kick!(pzb, zb, qb, alive, geom, 2, ds, 0.17, 200.0,
                   electron, params)
    @test eltype(rb.charge_per_meter) == BigFloat
    @test maxrel(rb.charge_per_meter, ref.charge_per_meter) < 1e-12
    @test maxrel(Float64.(pzb), pzr) < 1e-10
  end

  @testset "Work struct and geometry types are isbits (GPU-ready)" begin
    D = ForwardDiff.Dual{Nothing,Float64,3}
    for T in (Float64, Float32, D)
      @test isbitstype(CSR.CSRKick1{T})
      @test isbitstype(CSRKickPoint{T})
      @test isbitstype(CSRElementInfo{T})
    end
    k = CSR.CSRKick1(; ix_ele_source = 3, dz_particles = 1e-3)
    k2 = CSR.with_fields(k; L = 2, ix_ele_source = 1)
    @test k2.L === 2.0 && k2.ix_ele_source == 1 && k2.dz_particles ==
        1e-3 && k.L == 0
  end
end
