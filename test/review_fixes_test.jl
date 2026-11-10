# Stage 04 review fixes (2026-10-18), core parts: sub-stage B (W5),
# sub-stage D (S10 reference range) and the final review fixes (input
# lengths and non-finite z). S2f removed the derivative parts and W4
# (test/MOVED.md). The BeamTracking-dependent parts (W3, S2, S7, S10
# SaganCavity, D.10, and the SaganCavity and gamma-ramp checks of the
# final review fixes) are not here: per spec 4.7.4 they go to CSR.jl's
# CSR-effect tests (stage S7). test/MOVED.md lists them. Uses
# kernel_test_bunch (kernels_test.jl), included before this file.

@testset "Review fixes 2026-10-18" begin
  L3 = [0.2, 0.3, 0.2]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]

  @testset "W5 gamma check" begin
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    electron = Species("electron")
    for g in (1.0, NaN)
      @test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1, g)
      @test_throws "gamma must be > 1" csr_kick_coef(electron, g,
          0.01)
      @test_throws "gamma must be > 1" csr_kick_coef(electron, g,
          0.01; bmad_compat = true)
    end
    # Dual gamma: the check uses the scalarized value
    @test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1,
        ForwardDiff.Dual(1.0, 1.0))
    @test CSRKickPoint(geom, 3, 0.1, 1.0001) isa CSRKickPoint
  end

  @testset "S10 reference range test" begin
    z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
    params = CSRParams(n_bin = 40)
    bins = csr_bin_particles(z, q, alive, params)
    kick = zeros(40)
    for zf in (bins.z_center[end] + 10bins.dz,
               bins.z_center[1] - 10bins.dz)
      @test_throws(ErrorException,
          CSR.csr_apply_kicks_reference!(
              [0.0], [zf], [true], bins, kick))
    end
  end

  @testset "Final review fixes" begin
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
    params = CSRParams(n_bin = 40)
    e = Species("electron")
    ds = L3[3] / max(1, round(Int, L3[3] / 0.05,
                              RoundNearestTiesAway))
    @test_throws "equal lengths" csr_bin_particles(z, q[1:end-1],
        alive, params)
    @test_throws "equal lengths" csr_kick!(zeros(length(z) - 1), z, q,
        alive, geom, 3, ds, 0.1, 200.0, e, params)
    z2 = copy(z)
    z2[findfirst(alive)] = NaN
    @test_throws "non-finite" csr_bin_particles(z2, q, alive, params)
    z2[findfirst(alive)] = Inf
    @test_throws "non-finite" csr_bin_particles(z2, q, alive, params)
    z3 = copy(z)
    a3 = copy(alive)
    z3[1] = NaN
    a3[1] = false
    @test csr_bin_particles(z3, q, a3, params) isa CSR.CSRBins
  end

  # Keep last: Descriptor(...) changes GTPSA.desc_current.
  @testset "TPS cases" begin
    d = Descriptor(1, 1)
    x = @vars(d)
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    @test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1,
        1.0 + x[1])
    @test CSRKickPoint(geom, 3, 0.1, 200.0 + x[1]) isa CSRKickPoint
  end
end
