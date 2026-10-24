# Stage 04 sub-stage A (W1, W2, W6): type stability of the core source-point, kick and image-charge functions for
# Float64, Float32, Dual, nested Dual and TPS inputs (`@inferred`, concrete return types).

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

  # (name, g2, s_kick, gamma, lag): g2 is the curvature of element 2
  cases = (
    ("Float64", 1.5, 0.1, 200.0, 1e-4),
    ("Float32", 1.5, 0.1f0, 200.0f0, 1f-4),
    ("Dual", ForwardDiff.Dual{typeof(tag_in)}(1.5, 1.0), 0.1, 200.0, 1e-4),
    ("nested Dual", ForwardDiff.Dual{typeof(tag_out)}(ForwardDiff.Dual{typeof(tag_in)}(1.5, 1.0),
                                                      ForwardDiff.Dual{typeof(tag_in)}(1.0, 0.0)), 0.1, 200.0, 1e-4),
    ("TPS", 1.5 + x_tps[1], 0.1, 200.0, 1e-4),
  )

  for (name, g2, s_kick, gamma, lag) in cases
    @testset "$name" begin
      geom = CSRGeometry(L3, [0.0, g2, 0.0], c3)
      T = CSR.number_type(geom)
      kp = CSRKickPoint(geom, 3, s_kick, gamma)
      search = CSR.primal_search(geom, kp, T)
      k0 = CSR.CSRKick1(geom, kp; dz_particles = lag)

      r = @inferred CSR.path_excess(zero(s_kick), s_kick, geom.ele[2].spline, 0.01)
      @test isconcretetype(typeof(r))
      r = @inferred CSR.path_excess(zero(s_kick), s_kick, geom.ele[2].spline, 0.01; bmad_compat = true)
      @test isconcretetype(typeof(r))

      k1, s1 = @inferred CSR.s_source_calc(k0, geom, kp, search)
      @test k1 isa CSR.CSRKick1
      @inferred CSR.ddz_calc_csr(k1, s1, geom, kp)
      @inferred CSR.I_csr_calc(k1, 1, lag, geom, kp, 0.0, search)
      @inferred CSR.I_csr_calc(k1, 2, lag, geom, kp, 0.0, search)
      @inferred CSR.with_fields(k0; dz_particles = 2 * lag)
      @inferred CSR.with_fields(k0; s_chord_source = 0.0)

      kp_i = CSRKickPoint(geom, 3, s_kick, gamma; y_source = 0.01)
      ki, _ = CSR.s_source_calc(CSR.CSRKick1(geom, kp_i; dz_particles = lag), geom, kp_i,
                                CSR.primal_search(geom, kp_i, T))
      @test 1 <= ki.ix_ele_source <= 3
      @inferred CSR.image_charge_kick(ki, geom, kp_i)

      kp_u = CSRKickPoint(geom, 1, 0.05, gamma; y_source = 0.01)
      ku, _ = CSR.s_source_calc(CSR.CSRKick1(geom, kp_u; dz_particles = 1e-3), geom, kp_u,
                                CSR.primal_search(geom, kp_u, T))
      @test ku.ix_ele_source == 0
      @inferred CSR.image_charge_kick(ku, geom, kp_u)
    end
  end
end
