# Type inference of the core hot-path functions (no Union or Any results) in Float64, Float32, ForwardDiff Dual, nested
# Dual and GTPSA TPS. `@inferred` sees result types only: a Union local (a Float64 literal or accumulator mixed with the
# number type) that does not reach the result is not caught.
# Runs alone after `include("utils.jl")` (and `include("kernels_test.jl")`; no helper from it is needed) with CSR,
# Test, StaticArrays, ForwardDiff and GTPSA loaded.

@testset "Type inference of the core functions" begin
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]

  # (geometry arguments, gamma) for each number type; the geometry holds a knob in g[2] and gamma carries a knob
  Dual1(v, a, b) = ForwardDiff.Dual{Nothing}(v, a, b)
  nested(v) = ForwardDiff.Dual{Nothing}(Dual1(v, 1.0, 0.0), Dual1(1.0, 0.0, 0.0), Dual1(0.0, 1.0, 0.0))
  d = Descriptor(2, 2)
  x = @vars(d)
  f32 = FloorPosition(SVector(0.0f0, 0.0f0, 0.0f0), 0.0f0)
  cases = (
    ("Float64", (L3, g3, c3, FloorPosition()), 200.0),
    ("Float32", (Float32.(L3), Float32.(g3), Float32.(c3), f32), 200.0f0),
    ("Dual", (L3, [0.0, Dual1(1.5, 1.0, 0.0), 0.0], c3, FloorPosition()),
     Dual1(200.0, 0.0, 1.0)),
    ("nested Dual", (L3, [0.0, nested(1.5), 0.0], c3, FloorPosition()), nested(200.0)),
    ("TPS", (L3, [0.0, 1.5 + x[1], 0.0], c3, FloorPosition()), 200.0 + x[2]),
  )

  @testset "$name, bmad_compat = $bc" for (name, (L, g, c, start), gamma) in cases, bc in (false, true)
    geom = CSRGeometry(L, g, c; ref_start = start, bmad_compat = bc)
    ele = geom.ele[2]

    @testset "path_excess" begin
      @test (@inferred CSR.path_excess(0.0, ele.L_chord, ele.spline; bmad_compat = bc)) isa Number
      @test (@inferred CSR.path_excess(0.1, ele.L_chord, ele.spline, 1e-3; bmad_compat = bc)) isa Number
      @test (@inferred CSR.path_excess(ele.L_chord, 0.1, ele.spline, ele.theta_chord; bmad_compat = bc)) isa Number
    end

    # Kick in element 2 at s = 0.17; y_source > 0 exercises the image-charge branches
    for y in (0.0, 0.01)
      kp = @inferred CSRKickPoint(geom, 2, 0.17, gamma; y_source = y, kick_factor = 0.5)
      @test kp isa CSRKickPoint
      search = CSR.primal_search(geom, kp, CSR.number_type(kp))
      # TPS tables always search in primal values (the plain search needs `Dual{RootTag}` slopes)
      searches = search === nothing ? (nothing,) : CSR.number_type(kp) <: Real ? (nothing, search) : (search,)

      @testset "ddz_calc_csr, y_source = $y" begin
        k = CSR.CSRKick1(geom, kp; ix_ele_source = 1, dz_particles = 1e-3)
        @test (@inferred CSR.ddz_calc_csr(k, 0.1, geom, kp)) isa Tuple
        @test (@inferred CSR.ddz_calc_csr(CSR.with_fields(k; ix_ele_source = 2), 0.1, geom, kp)) isa Tuple
        @test (@inferred CSR.ddz_calc_csr(CSR.with_fields(k; ix_ele_source = 3), 0.1, geom, kp)) isa Tuple
        @test (@inferred CSR.with_fields(k; dz_particles = 2e-3, ix_ele_source = 2)) isa CSR.CSRKick1
      end

      # A source inside the lattice, one on the infinite drift upstream (ix = 0), one downstream (ix = n + 1, y > 0)
      for (dz, where_) in ((3e-4, :inside), (0.5, :upstream), (-2.0, :downstream))
        dz < 0 && y == 0 && continue
        @testset "source $where_, y_source = $y" begin
          k = CSR.CSRKick1(geom, kp; dz_particles = dz)
          for s in searches
            k1, s1 = @inferred CSR.s_source_calc(k, geom, kp, s)
            @test k1 isa CSR.CSRKick1
            if where_ == :inside
              @test 1 <= k1.ix_ele_source <= length(geom.ele)
            elseif where_ == :upstream
              @test k1.ix_ele_source == 0
            else
              bc || @test k1.ix_ele_source == length(geom.ele) + 1
            end
            @test (@inferred CSR.image_charge_kick(k1, geom, kp)) isa Number
            @test (@inferred CSR.I_csr_calc(k1, 1, 1e-4, geom, kp, 0.0, s)) isa CSR.CSRKick1
            @test (@inferred CSR.I_csr_calc(k1, 2, 1e-4, geom, kp, 0.0, s)) isa CSR.CSRKick1
          end
        end
      end
    end

    # Kick point in the first and last element (sources upstream/downstream of the kick element)
    for ix in (1, 3), y in (0.0, 0.01)
      kp = @inferred CSRKickPoint(geom, ix, 0.1, gamma; y_source = y)
      search = CSR.primal_search(geom, kp, CSR.number_type(kp))
      k = CSR.CSRKick1(geom, kp; dz_particles = 0.05)
      k1, _ = @inferred CSR.s_source_calc(k, geom, kp, search)
      @test (@inferred CSR.image_charge_kick(k1, geom, kp)) isa Number
      y == 0 && @test (@inferred CSR.I_csr_calc(k1, 1, 1e-4, geom, kp, 0.0, search)) isa CSR.CSRKick1
    end
  end
end
