# Phase 7 step 5: track_csr! with a test bunch (TPS, Dual or Float64) tracked alongside the bunch in the frozen wake.
# TPS order 1/2 vs ForwardDiff Jacobian/Hessian of the same tracking with a Float64 test particle.
# Uses tc_pq, tc_line_bunch (track_csr_test.jl) and lc_line_bunch, lc_lattice, LC_E, LC_E0 (lcavity_test.jl).
# Compile time dominates: every Dual tag type compiles BeamTracking's kernels anew (~1 min for nested Duals). The
# callable structs below give both lattices the same tags, so their common elements compile once. (Finite differences
# are no reference for the second order: the kick is piecewise linear in z between bin centres.)

# Final coordinates of one test particle starting at v_ref + v, tracked by `track(test_bunch)`
struct TPSTestTrack
  track::Any
  v_ref::Vector{Float64}
  species::Species
  pq::Float64
end
function (t::TPSTestTrack)(v)
  tb = Bunch(reshape(t.v_ref .+ v, 1, 6); species = t.species, p_over_q_ref = t.pq)
  t.track(tb)
  return tb.coords.v[1, :]
end

# Its Jacobian, differentiated again for the second derivatives
struct TPSTestJacobian
  f::TPSTestTrack
end
(j::TPSTestJacobian)(u) = ForwardDiff.jacobian(j.f, u)

@testset "track_csr! with a test bunch" begin
  e = Species("electron")
  pq = tc_pq(200)
  line = [Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5, Kn0 = 1.5), Drift(L = 0.2),
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = e)
  params = CSRParams(n_bin = 40, ds_step = 0.1)
  csr_on = ele -> ele !== bl.line[3]   # also the path for elements tracked whole
  bunch() = tc_line_bunch(1e-3, 1e-9, 400, pq; x = 1e-4)
  d = Descriptor(6, 2)
  x = @vars(d)

  # TPS order 2 in x vs ForwardDiff: values, Jacobian per column, all second derivatives (one nested-Dual pass,
  # H[j, a, b] = ∂²v_j/∂a∂b). Returns the TPS coordinates.
  function check_tps(f::TPSTestTrack)
    vt = f(x)
    v0 = zeros(6)
    @test maximum(abs, scalar.(vt) .- f(v0)) <= 1e-15
    Jt, Jf = GTPSA.jacobian(vt), ForwardDiff.jacobian(f, v0)
    # Separate bunch runs: with threads the atomic deposit order changes the frozen wake in the last bits (lcavity
    # Branch, 4 threads, 8 runs: Jacobian columns 1.9e-12, second derivatives 6.3e-15 of max |H|;
    # planning/diagnostics/track_tps_thread_spread.jl)
    for c in 1:6
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-10
    end
    H = reshape(ForwardDiff.jacobian(TPSTestJacobian(f), v0), 6, 6, 6)
    @test maximum(abs, H) > 1
    for j in 1:6
      @test GTPSA.hessian(vt[j]) ≈ H[j, :, :] rtol = 1e-12 atol = 1e-12 * maximum(abs, H)
    end
    return vt
  end

  @testset "TPS test bunch through a Beamline" begin
    v_ref = [2e-4, -1e-4, 0.0, 0.0, 3e-4, 1e-4]
    vt = check_tps(TPSTestTrack(tb -> track_csr!(bunch(), bl; params, csr_on, test_bunch = tb), v_ref, e, pq))
    # CSR changes the map (else the comparison above would not test the kicks)
    tb0 = Bunch(reshape(v_ref .+ x, 1, 6); species = e, p_over_q_ref = pq)
    track_csr!(bunch(), bl; params, csr_on = ele -> false, test_bunch = tb0)
    @test maximum(abs, GTPSA.jacobian(tb0.coords.v[1, :]) .- GTPSA.jacobian(vt)) > 0.1
  end

  @testset "TPS test bunch through a Branch with an lcavity" begin
    br = lc_lattice(4, 1e8)
    p_lc = CSRParams(n_bin = 80, ds_step = 0.05)
    check_tps(TPSTestTrack(tb -> track_csr!(lc_line_bunch(), br; params = p_lc, test_bunch = tb),
                           [1e-4, -2e-5, 0.0, 0.0, 1e-4, 2e-4], LC_E, BeamTracking.E_to_R(LC_E, LC_E0)))
  end

  @testset "Float64 test particles follow bunch particles" begin
    b = bunch()
    idx = 1:37:400
    tb = Bunch(b.coords.v[idx, :]; species = e, p_over_q_ref = pq)
    track_csr!(b, bl; params, csr_on, test_bunch = tb)
    # Not bitwise: BeamTracking's explicit SIMD lanes and its scalar remainder loop differ in the last bit
    @test maximum(abs, tb.coords.v .- b.coords.v[idx, :]) <= 1e-15
    # The test bunch does not change the bunch (atomic deposit order: tolerance)
    b2 = bunch()
    track_csr!(b2, bl; params, csr_on)
    @test maximum(abs, b2.coords.v .- b.coords.v) <= 1e-12 * maximum(abs, b.coords.v)
    # Through the lcavity (pz_scale != 1 for the test particles too)
    b = lc_line_bunch()
    idx = 1:397:4001
    tb = Bunch(b.coords.v[idx, :]; species = LC_E, p_over_q_ref = b.p_over_q_ref)
    track_csr!(b, lc_lattice(4, 1e8); params = CSRParams(n_bin = 80, ds_step = 0.05), test_bunch = tb)
    @test maximum(abs, tb.coords.v .- b.coords.v[idx, :]) <= 1e-15
    @test tb.p_over_q_ref == b.p_over_q_ref && tb.t_ref == b.t_ref
  end

  @testset "Errors" begin
    @test_throws ErrorException track_csr!(bunch(), bl; params, test_bunch = zeros(1, 6))
    @test_throws ErrorException track_csr!(bunch(), bl; params,
                                           test_bunch = Bunch(zeros(1, 6); species = Species("positron"), p_over_q_ref = pq))
    @test_throws ErrorException track_csr!(bunch(), bl; params,
                                           test_bunch = Bunch(zeros(1, 6); species = e, p_over_q_ref = 1.01pq))
    # Outside the bunch's bin range
    tb = Bunch(reshape([0.0, 0.0, 0.0, 0.0, 0.05, 0.0] .+ x, 1, 6); species = e, p_over_q_ref = pq)
    @test_throws ErrorException track_csr!(bunch(), bl; params, test_bunch = tb)
  end
end
