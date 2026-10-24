# Stage 04 sub-stage A (W1, W2, W6): return types (`@inferred`) of the core source-point, kick and image-charge
# functions for Float64, Float32 scalars (the geometry is always at least Float64), a Float32 spline in
# `path_excess`, Dual, nested Dual, static- and dynamic-descriptor TPS. Detects A.4 regressions; the W1/W2
# local-variable unions leave the return types unchanged and are not detected here.

@testset "Type stability (core)" begin
  L3 = [0.2, 0.3, 0.2]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
  tag_in = ForwardDiff.Tag(sin, Float64)
  D1 = ForwardDiff.Dual{typeof(tag_in),Float64,1}
  tag_out = ForwardDiff.Tag(cos, D1)
  d_tps = Descriptor(1, 2)
  x_tps = @vars(d_tps)
  d_dyn = Descriptor(1, 2)
  x_dyn = @vars(d_dyn, dynamic = true)

  # (name, g2, s_kick, gamma, lag): g2 is the curvature of element 2
  cases = (
    ("Float64", 1.5, 0.1, 200.0, 1e-4),
    ("Float32", 1.5, 0.1f0, 200.0f0, 1f-4),
    ("Dual", ForwardDiff.Dual{typeof(tag_in)}(1.5, 1.0), 0.1, 200.0, 1e-4),
    ("nested Dual", ForwardDiff.Dual{typeof(tag_out)}(ForwardDiff.Dual{typeof(tag_in)}(1.5, 1.0),
                                                      ForwardDiff.Dual{typeof(tag_in)}(1.0, 0.0)), 0.1, 200.0, 1e-4),
    ("TPS", 1.5 + x_tps[1], 0.1, 200.0, 1e-4),
    ("TPS dynamic", 1.5 + x_dyn[1], 0.1, 200.0, 1e-4),
  )

  for (name, g2, s_kick, gamma, lag) in cases
    @testset "$name" begin
      geom = CSRGeometry(L3, [0.0, g2, 0.0], c3)
      T = CSR.number_type(geom)
      kp = CSRKickPoint(geom, 3, s_kick, gamma)
      search = CSR.primal_search(geom, kp, T)
      k0 = CSR.CSRKick1(geom, kp; dz_particles = lag)

      @test (@inferred CSR.path_excess(zero(s_kick), s_kick, geom.ele[2].spline, 0.01)) isa promote_type(T, typeof(s_kick))
      @test (@inferred CSR.path_excess(zero(s_kick), s_kick, geom.ele[2].spline, 0.01; bmad_compat = true)) isa promote_type(T, typeof(s_kick))

      @test (@inferred CSR.s_source_calc(k0, geom, kp, search)) isa Tuple
      k1, s1 = CSR.s_source_calc(k0, geom, kp, search)
      @test k1 isa CSR.CSRKick1
      @test (@inferred CSR.ddz_calc_csr(k1, s1, geom, kp)) isa Any
      @test (@inferred CSR.I_csr_calc(k1, 1, lag, geom, kp, 0.0, search)) isa Any
      @test (@inferred CSR.I_csr_calc(k1, 2, lag, geom, kp, 0.0, search)) isa Any
      @test (@inferred CSR.with_fields(k0; dz_particles = 2 * lag)) isa Any
      @test (@inferred CSR.with_fields(k0; s_chord_source = 0.0)) isa Any

      kp_i = CSRKickPoint(geom, 3, s_kick, gamma; y_source = 0.01)
      ki, _ = CSR.s_source_calc(CSR.CSRKick1(geom, kp_i; dz_particles = lag), geom, kp_i,
                                CSR.primal_search(geom, kp_i, T))
      @test 1 <= ki.ix_ele_source <= 3
      @test (@inferred CSR.image_charge_kick(ki, geom, kp_i)) isa Any

      kp_u = CSRKickPoint(geom, 1, 0.05, gamma; y_source = 0.01)
      ku, _ = CSR.s_source_calc(CSR.CSRKick1(geom, kp_u; dz_particles = 1e-3), geom, kp_u,
                                CSR.primal_search(geom, kp_u, T))
      @test ku.ix_ele_source == 0
      @test (@inferred CSR.image_charge_kick(ku, geom, kp_u)) isa Any
    end
  end

  @testset "Float32 spline" begin
    c32 = SVector{4,Float32}(1f-3, 2f-3, -1f-3, 5f-4)
    @test (@inferred CSR.path_excess(0f0, 0.1f0, c32, 0.01f0)) isa Float64
    @test (@inferred CSR.path_excess(0f0, 0.1f0, c32, 0.01f0; bmad_compat = true)) isa AbstractFloat
  end
end
