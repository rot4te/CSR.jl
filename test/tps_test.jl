# Phase 7: GTPSA TPS. Stage 1: test particles in a frozen wake
# (csr_apply_test_kicks!, csr_kick! test_particles), checked against
# ForwardDiff Jacobians and Hessians of the same function of Float64
# coordinates.

@testset "TPS" begin
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  geom = CSRGeometry(L3, g3, c3)
  ds = L3[2] / max(1, round(Int, Beamlines.scalarize_all(L3[2]) / 0.1,
                            RoundNearestTiesAway))
  d = Descriptor(6, 3)
  x = @vars(d)

  @testset "primal, root_order" begin
    t = 0.3 + 2x[1] + x[5]^2
    @test Beamlines.scalarize_all(t) == 0.3
    @test CSR.root_order(t) == 3
    @test_throws MethodError CSR.root_order(1.0 + 2.0im)
  end

  @testset "Frozen-wake test particles (images = 0)" begin
    n_img = 0
    p = CSRParams(n_bin = 40, n_shield_images = n_img,
                  beam_chamber_height = n_img > 0 ? 0.01 : 0.0)
    st = CSR.csr_binned_kick(z, q, alive, geom, 2, ds, 0.17,
                                    200.0, electron, p)
    # Test particles across the bunch, coordinates nonlinear in the
    # variables; more than one default CPU workgroup (1024), so that
    # with threads the TPS arithmetic runs concurrently
    m = 3000
    z0 = range(-5e-3, 5e-3; length = m)
    zfun(v,
         j) = z0[j] + v[5] * (1 + 0.1j / m) + 0.3 * v[1]^2 - 0.2 *
             v[5] * v[6] + 0.5 * v[1] * v[5]^2
    pzfun(v, j) = v[6] + 1e-4j / m
    f(v) = CSR.csr_apply_test_kicks!([pzfun(v,
        j) for j in 1:m], [zfun(v, j) for j in 1:m], fill(true, m),
        st.bins,
                                     st.kick; pz_scale = 1.3)
    pzt = CSR.csr_apply_test_kicks!([pzfun(x, j) for j in 1:m],
        [zfun(x, j) for j in 1:m], fill(true, m), st.bins,
                                    st.kick; pz_scale = 1.3)
    v0 = zeros(6)
    y0 = f(v0)
    @test scalar.(pzt) == y0
    # Per column: column 6 (∂pz/∂pz = 1) would dominate a norm
    # over the whole matrix
    Jt, Jf = GTPSA.jacobian(pzt), ForwardDiff.jacobian(f, v0)
    for c in 1:6
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-14
    end
    for j in (1, m ÷ 2, m)
      @test GTPSA.hessian(pzt[j]) ≈ ForwardDiff.hessian(v -> f(v)[
          j], v0) rtol = 1e-13
      # Third order (truncation order of d): every
      # ∂³pz/∂a∂b∂c from the TPS vs nested ForwardDiff
      T3f = reshape(ForwardDiff.jacobian(u -> vec(ForwardDiff.hessian(
          v -> f(v)[j], u)), v0), 6, 6, 6)
      T3t = [scalar(GTPSA.deriv(pzt[j], [count(==(i), (a, b,
          c)) for i in 1:6])) for a in 1:6, b in 1:6, c in 1:6]
      @test any(!iszero, T3f)
      @test T3t ≈ T3f rtol = 1e-13
    end
    # Float64 test particles at bunch particles' positions get the
    # bunch particles' kicks
    pzb = zeros(n)
    idx = findall(alive)[1:37:end]
    pzt64 = zeros(length(idx))
    csr_kick!(pzb, z, q, alive, geom, 2, ds, 0.17, 200.0, electron, p;
              test_particles = (pzt64, z[idx], fill(true,
        length(idx))))
    @test pzt64 == pzb[idx]
  end

  @testset "Test particles outside the bunch" begin
    p = CSRParams(n_bin = 40)
    st = CSR.csr_binned_kick(z, q, alive, geom, 2, ds, 0.17,
                                    200.0, electron, p)
    @test_throws ErrorException CSR.csr_apply_test_kicks!([x[
        6]], [0.05 + x[5]], [true], st.bins, st.kick)
    # Accepted range [z_center[1] - dz/100, z_center[end]), as in the
    # error message
    zc, dz = st.bins.z_center, st.bins.dz
    @test CSR.csr_apply_test_kicks!([0.0, 0.0],
        [zc[1] - dz / 200, zc[end] - dz / 4], [true, true], st.bins,
                                    st.kick) isa Vector
    for zt in (zc[1] - dz / 50, zc[end] + dz / 4)
      err = try CSR.csr_apply_test_kicks!([0.0], [zt], [true],
          st.bins, st.kick); nothing catch e; e end
      @test err isa ErrorException && occursin("$(zc[end]))", err.msg)
    end
    # Dead test particles are skipped wherever they are
    pzt = [x[6]]
    CSR.csr_apply_test_kicks!(pzt, [0.05 + x[5]], [false],
                                     st.bins, st.kick)
    # every coefficient (TPS == compares the scalar parts only)
    @test isequal(pzt[1], x[6])
    # Nothing is kicked when any live test particle is outside
    # (finding 4)
    k0 = CSR.csr_apply_test_kicks!([0.0], [0.0], [true],
        st.bins, st.kick)[1]
    @test k0 != 0
    pzt = [0.0, 0.0]
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pzt,
        [0.0, 0.05], [true, true], st.bins, st.kick)
    @test pzt == [0.0, 0.0]
    pzb, pzt = zeros(n), [0.0, 0.0]
    @test_throws ErrorException csr_kick!(pzb, z, q, alive, geom, 2,
        ds, 0.17, 200.0, electron, p;
                                          test_particles = (pzt, [0.0,
        0.05], [true, true]))
    @test all(iszero, pzb) && pzt == [0.0, 0.0]
    # TPS z_t at the range edges: accepted where Float64 is, same
    # kicked value (review finding 6)
    for zt in (zc[1] - dz / 200, zc[end] - dz / 4)
      k64 = CSR.csr_apply_test_kicks!([0.0], [zt], [true],
          st.bins, st.kick)[1]
      kt = CSR.csr_apply_test_kicks!([0.0 + x[6]], [zt + x[5]],
          [true], st.bins, st.kick)[1]
      @test isapprox(scalar(kt), k64; rtol = 1e-13,
                     atol = 1e-13 * maximum(abs, st.kick))
    end
    for zt in (zc[1] - dz / 50, zc[end] + dz / 4)
      pzt = [0.0 + x[6]]
      @test_throws ErrorException CSR.csr_apply_test_kicks!(
          pzt, [zt + x[5]], [true], st.bins, st.kick)
      @test isequal(pzt[1], 0.0 + x[6])
    end
    # Non-finite or far-off z: the range error, not an InexactError
    # (review finding 1)
    for zt in (NaN, 1e30)
      pzt = [0.0]
      err = try CSR.csr_apply_test_kicks!(pzt, [zt], [true],
          st.bins, st.kick); nothing catch e; e end
      @test err isa ErrorException &&
          occursin("outside the bunch's bin range", err.msg)
      @test pzt == [0.0]
      errt = try CSR.csr_apply_test_kicks!([0.0 + x[6]],
          [zt + x[5]], [true], st.bins,
          st.kick); nothing catch e; e end
      @test errt isa ErrorException &&
          occursin("outside the bunch's bin range", errt.msg)
    end
    # Empty TPS test particles (review finding 3)
    T6 = typeof(x[6])
    @test isempty(CSR.csr_apply_test_kicks!(T6[], T6[], Bool[],
        st.bins, st.kick))
    # Different lengths (review finding 5)
    err = try CSR.csr_apply_test_kicks!([0.0, 0.0], [0.0],
        [true, true], st.bins, st.kick); nothing catch e; e end
    @test err isa ErrorException && occursin("must be equal", err.msg)
    pzb = zeros(n)
    err = try csr_kick!(pzb, z, q, alive, geom, 2, ds, 0.17, 200.0,
                        electron, p;
                        test_particles = ([0.0], [0.0, 0.0], [true,
        true])); nothing catch e; e end
    @test err isa ErrorException &&
        occursin("csr_kick!: test particle arrays", err.msg)
    @test all(iszero, pzb)
  end
end
