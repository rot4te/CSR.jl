# Phase 7: GTPSA TPS. Stage 1: test particles in a frozen wake (csr_apply_test_kicks!, csr_kick! test_particles), checked
# against ForwardDiff Jacobians and Hessians of the same function of Float64 coordinates.

@testset "TPS" begin
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  geom = CSRGeometry(L3, g3, c3)
  d = Descriptor(6, 3)
  x = @vars(d)

  @testset "primal, root_order" begin
    t = 0.3 + 2x[1] + x[5]^2
    @test CSR.primal(t) == 0.3
    @test CSR.root_order(t) == 3
  end

  @testset "Frozen-wake test particles (images = $n_img)" for n_img in (0, 2)
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0)
    st = CSR.csr_binned_kick(z, q, alive, geom, 2, 0.17, 200.0, electron, p)
    # Test particles across the bunch, coordinates nonlinear in the variables; more than one default CPU workgroup
    # (1024), so that with threads the TPS arithmetic runs concurrently
    m = 3000
    z0 = range(-5e-3, 5e-3; length = m)
    zfun(v, j) = z0[j] + v[5] * (1 + 0.1j / m) + 0.3 * v[1]^2 - 0.2 * v[5] * v[6]
    pzfun(v, j) = v[6] + 1e-4j / m
    f(v) = CSR.csr_apply_test_kicks!([pzfun(v, j) for j in 1:m], [zfun(v, j) for j in 1:m], fill(true, m), st.bins,
                                     st.kick; pz_scale = 1.3)
    pzt = CSR.csr_apply_test_kicks!([pzfun(x, j) for j in 1:m], [zfun(x, j) for j in 1:m], fill(true, m), st.bins,
                                    st.kick; pz_scale = 1.3)
    v0 = zeros(6)
    y0 = f(v0)
    @test scalar.(pzt) == y0
    @test GTPSA.jacobian(pzt) ≈ ForwardDiff.jacobian(f, v0) rtol = 1e-14
    for j in (1, m ÷ 2, m)
      @test GTPSA.hessian(pzt[j]) ≈ ForwardDiff.hessian(v -> f(v)[j], v0) rtol = 1e-13
    end
    # Float64 test particles at bunch particles' positions get the bunch particles' kicks
    pzb = zeros(n)
    idx = findall(alive)[1:37:end]
    pzt64 = zeros(length(idx))
    csr_kick!(pzb, z, q, alive, geom, 2, 0.17, 200.0, electron, p; test_particles = (pzt64, z[idx], fill(true, length(idx))))
    @test pzt64 == pzb[idx]
  end

  @testset "Test particles outside the bunch" begin
    p = CSRParams(n_bin = 40, ds_step = 0.1)
    st = CSR.csr_binned_kick(z, q, alive, geom, 2, 0.17, 200.0, electron, p)
    @test_throws ErrorException CSR.csr_apply_test_kicks!([x[6]], [0.05 + x[5]], [true], st.bins, st.kick)
    # Dead test particles are skipped wherever they are
    pzt = [x[6]]
    CSR.csr_apply_test_kicks!(pzt, [0.05 + x[5]], [false], st.bins, st.kick)
    @test pzt[1] == x[6]
  end
end
