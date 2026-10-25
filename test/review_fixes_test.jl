# Stage 04 review fixes (2026-10-18), sub-stage B (W3, W4, W5): shifted_to_slice number types, device number types
# (backends without Float64), gamma > 1 checks. Uses TC_EXT, TC_E, tc_pq (track_csr_test.jl) and kernel_test_bunch
# (kernels_test.jl), included before this file.

import KernelAbstractions

struct NoFloat64Backend <: KernelAbstractions.Backend end
KernelAbstractions.supports_float64(::NoFloat64Backend) = false

@testset "Review fixes 2026-10-18" begin
  L3 = [0.2, 0.3, 0.2]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]

  @testset "W3 shifted_to_slice" begin
    f(x, y, s, t, pp = nothing) = (s, t)
    ele = (; species_ref = TC_E, p_over_q_ref = ForwardDiff.Dual(tc_pq(200), 1.0))
    sp = TC_EXT.shifted_to_slice(f, ele, 0.25)
    @test sp.s0 isa ForwardDiff.Dual
    @test typeof(sp.s0) == typeof(sp.t0)
    @test ForwardDiff.value(sp.s0) == 0.25
    @test ForwardDiff.value(sp(0.0, 0.0, 0.1, 0.0)[1]) ≈ 0.35

    ele64 = (; species_ref = TC_E, p_over_q_ref = tc_pq(200))
    sp64 = TC_EXT.shifted_to_slice(f, ele64, 0.25)
    @test sp64.s0 === 0.25
    @test sp64.t0 isa Float64
  end

  @testset "W4 device_number_type" begin
    @test CSR.device_number_type(NoFloat64Backend(), Float64) === Float32
    @test CSR.device_number_type(NoFloat64Backend(), Float32) === Float32
    @test CSR.device_number_type(NoFloat64Backend(), ForwardDiff.Dual{Nothing,Float64,1}) ===
          ForwardDiff.Dual{Nothing,Float64,1}
    @test CSR.device_number_type(KernelAbstractions.CPU(), Float64) === Float64

    # CPU: csr_kick_pz keeps Float64 output for a Float32 pz
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
    params = CSRParams(n_bin = 40, ds_step = 0.05)
    pz = zeros(Float32, length(z))
    pz_new = csr_kick_pz(pz, z, q, alive, geom, 3, 0.1, 200.0, Species("electron"), params)
    @test eltype(pz_new) == Float64
  end

  @testset "W5 gamma check" begin
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    electron = Species("electron")
    for g in (1.0, 0.5, NaN)
      @test_throws ErrorException CSRKickPoint(geom, 3, 0.1, g)
      @test_throws ErrorException csr_kick_coef(electron, g, 0.01)
      @test_throws ErrorException csr_kick_coef(electron, g, 0.01; bmad_compat = true)
    end
    # TPS gamma with primal value 1: raises an error (no abort)
    d = Descriptor(1, 1)
    x = @vars(d)
    @test_throws ErrorException CSRKickPoint(geom, 3, 0.1, 1.0 + x[1])
    @test CSRKickPoint(geom, 3, 0.1, 1.0001) isa CSRKickPoint
  end
end
