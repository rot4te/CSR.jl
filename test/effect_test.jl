# CSREffect / CSRReplay: CSR as an effect of BeamTracking's track!

const EF_EXT = Base.get_extension(CSR, :CSRBeamTrackingExt)
const EF_E = Species("electron")

# p_over_q_ref for electrons with Lorentz factor gamma
ef_pq(gamma) = Beamlines.E_to_R(EF_E, gamma * massof(EF_E))

# Line bunch: n particles on a z grid over ±6σ, Gaussian weights
# for a total charge Q [C]
function ef_line_bunch(sigma, Q, n, pq; x = 0.0)
  z = collect(range(-6sigma, 6sigma; length = n))
  w = gaussian.(z, sigma)
  w .*= (Q / E_CHARGE) / sum(w)
  v = zeros(n, 6)
  v[:, 1] .= x
  v[:, 5] .= z
  return Bunch(v, nothing, w; species = EF_E, p_over_q_ref = pq)
end

ef_beamline(line, pq) =
  Beamline(line; p_over_q_ref = pq, species_ref = EF_E)

@testset "CSREffect" begin
  @test EF_EXT isa Module
  @test_throws ErrorException CSREffect(CSRParams();
                                        centroid_start = zeros(5))

  @testset "Floor conventions match BeamTracking" begin
    # With no fields a particle moves on a straight floor line; its
    # floor position at each element exit, from the reference survey
    # and its coordinates, must stay on that line
    pq = ef_pq(1e3)
    line = [Drift(L = 0.3),
            SBend(L = 0.5, g_ref = 0.4),
            Drift(L = 0.2),
            Patch(L = 0.6, dx = 1e-3, dz = 0.55, dy_rot = 0.05),
            SBend(L = 0.4, g_ref = 0.3, tilt_ref = pi),
            Patch(L = 0.3, dx = -2e-3, dz = 0.3, dy_rot = -0.08),
            Drift(L = 0.5)]
    bl = ef_beamline(line, pq)
    b = Bunch(zeros(1, 6); species = EF_E, p_over_q_ref = pq)
    f = FloorPosition()
    for (i, ele) in enumerate(bl.line)
      track!(b, ele; notify = false)
      pp = ele.PatchParams
      g = isnothing(pp) ?
          EF_EXT.csr_ref_curvature(ele.BendParams, i) : 0.0
      f = EF_EXT.csr_ref_exit(f, float(ele.L), g, pp)
      cf = CSR.centroid_floor(f, b.coords.v[1, 1], b.coords.v[1, 2],
                              b.coords.v[1, 6])
      @test abs(cf.r[1]) < 1e-14
      @test abs(cf.theta) < 1e-14
    end
    @test f.theta ≈ -0.4 * 0.5 + 0.05 + 0.3 * 0.4 - 0.08
    @test_throws ErrorException EF_EXT.csr_ref_curvature(
      SBend(L = 0.4, g_ref = 0.3, tilt_ref = pi / 2).BendParams, 1)
  end

  @testset "Kicked slices" begin
    pq = ef_pq(100)
    mk() = [Drift(L = 0.2, n_slices = 2),
            SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0, n_slices = 3),
            Marker(),
            Quadrupole(L = 0.2, Kn1 = 2.0),
            SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0),
            Drift(L = 0.4, n_slices = 2)]
    params = CSRParams(n_bin = 40)
    calls = Tuple{Int,Int,Float64}[]
    diagnostics = (ix, k, s0, r) -> push!(calls, (ix, k, s0))
    b = ef_line_bunch(1e-3, 1e-9, 201, pq)
    track!(b, ef_beamline(mk(), pq); notify = false,
           effects = (CSREffect(params; diagnostics),))
    # Nothing before the first bend or in zero-length elements; one
    # kick per slice entrance from the first bend on
    @test calls == [(2, 1, 0.0), (2, 2, 0.3/3), (2, 3, 2*(0.3/3)),
                    (4, 1, 0.0), (5, 1, 0.0),
                    (6, 1, 0.0), (6, 2, 0.4/2)]

    # on restricts the kicked elements
    empty!(calls)
    b = ef_line_bunch(1e-3, 1e-9, 201, pq)
    on = ele -> ele.kind == "SBend"
    track!(b, ef_beamline(mk(), pq); notify = false,
           effects = (CSREffect(params; on, diagnostics),))
    @test [c[1:2] for c in calls] == [(2, 1), (2, 2), (2, 3), (5, 1)]

    # No kick anywhere: identical to track!
    b1 = ef_line_bunch(1e-3, 1e-9, 201, pq; x = 1e-4)
    b2 = ef_line_bunch(1e-3, 1e-9, 201, pq; x = 1e-4)
    track!(b1, ef_beamline(mk(), pq); notify = false)
    track!(b2, ef_beamline(mk(), pq); notify = false,
           effects = (CSREffect(params; on = ele -> false),))
    @test b1.coords.v == b2.coords.v
    @test b1.t_ref == b2.t_ref
    # ... and the kicks change the result
    @test b.coords.v != b1.coords.v
  end

  @testset "One kick at the entrance equals csr_kick!" begin
    pq = ef_pq(100)
    mk() = [SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0)]
    params = CSRParams(n_bin = 40)
    b = ef_line_bunch(1e-3, 1e-9, 201, pq)
    track!(b, ef_beamline(mk(), pq); notify = false,
           effects = (CSREffect(params),))
    h = ef_line_bunch(1e-3, 1e-9, 201, pq)
    bl = ef_beamline(mk(), pq)
    st = BeamTracking.effect_setup(CSREffect(params), h, bl)
    @test st.first_bend == 1
    v = h.coords.v
    csr_kick!(view(v, :, 6), view(v, :, 5), st.q, trues(201),
              st.geom, 1, 0.3, 0.0, 100.0, EF_E, params)
    track!(h, bl; notify = false)
    @test b.coords.v == h.coords.v

    # Linear in charge: twice the weights, twice the kick
    kicks = Float64[]
    diagnostics = (ix, k, s0, r) ->
      push!(kicks, maximum(abs, r.kick_per_meter))
    for scale in (1, 2)
      b = ef_line_bunch(1e-3, scale * 1e-9, 201, pq)
      track!(b, ef_beamline(mk(), pq); notify = false,
             effects = (CSREffect(params; diagnostics),))
    end
    @test kicks[2] ≈ 2 * kicks[1] rtol = 1e-12
  end

  @testset "Steady state in a long bend" begin
    # As the csr_kick! test (kick_test.jl): ρ = 10 m, 80 bends of
    # 0.1 m in 2 slices, γ = 1e4, kick in the last bend
    rho = 10.0; L_ele = 0.1; n_ele = 80; gamma = 1e4
    sigma = 1e-3; Q = 1e-9
    pq = ef_pq(gamma)
    bl = ef_beamline([SBend(L = L_ele, g_ref = 1 / rho, Kn0 = 1 / rho,
                            n_slices = 2) for _ in 1:n_ele], pq)
    b = ef_line_bunch(sigma, Q, 1201, pq)
    params = CSRParams(n_bin = 200, particle_bin_span = 2)
    results = Dict{Tuple{Int,Int},CSRStepResult}()
    diagnostics = (ix, k, s0, r) -> (results[(ix, k)] = r)
    track!(b, bl; notify = false,
           effects = (CSREffect(params; diagnostics),))
    @test length(results) == n_ele * 2

    res = results[(n_ele, 2)]
    N = Q / E_CHARGE
    expected(x) = steady_state_dpz_ds(x, x -> dgaussian(x, sigma),
                                      rho, N, gamma;
                                      u_max = (12sigma)^(2/3))
    core = findall(x -> abs(x) <= 2sigma, res.z_center)
    W = expected.(res.z_center[core])
    @test maximum(abs.(res.kick_per_meter[core] .- W)) /
          maximum(abs, W) < 2e-2
    @test sum(res.charge_per_meter) *
          (res.z_center[2] - res.z_center[1]) ≈ Q rtol = 1e-10
    # Energy loss: the bunch-averaged Δpz is negative (CSR radiates)
    @test sum(b.coords.v[:, 6] .* b.coords.weight) < 0
  end

  @testset "Record and replay" begin
    pq = ef_pq(100)
    mk(n = 3) = [SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0, n_slices = n),
                 Drift(L = 0.4, n_slices = 2)]
    params = CSRParams(n_bin = 40)
    rec = CSRRecord()
    b = ef_line_bunch(1e-3, 1e-9, 201, pq)
    track!(b, ef_beamline(mk(), pq); notify = false,
           effects = (CSREffect(params; record = rec),))
    @test [(p.ix, p.k) for p in rec.points] ==
          [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

    # Test particles at the bunch's positions get the bunch's kicks
    t = ef_line_bunch(1e-3, 1e-9, 201, pq)
    track!(t, ef_beamline(mk(), pq); notify = false,
           effects = (CSRReplay(rec),))
    @test t.coords.v == b.coords.v
    # They do not contribute: a single test particle, same kicks
    t1 = Bunch(zeros(1, 6); species = EF_E, p_over_q_ref = pq)
    track!(t1, ef_beamline(mk(), pq); notify = false,
           effects = (CSRReplay(rec),))
    @test t1.coords.v[1, :] ≈ b.coords.v[101, :] atol = 1e-15
    @test length(rec.points) == 5

    # Dual test particles: values as Float64, partials untouched by
    # the kick itself
    D = ForwardDiff.Dual{Nothing,Float64,1}
    vd = D.(zeros(1, 6))
    vd[1, 6] = ForwardDiff.Dual(0.0, 1.0)
    td = Bunch(vd; species = EF_E, p_over_q_ref = pq)
    track!(td, ef_beamline(mk(), pq); notify = false,
           effects = (CSRReplay(rec),))
    @test ForwardDiff.value.(td.coords.v[1, :]) ≈ t1.coords.v[1, :] atol = 1e-15

    # Other slicing than recorded
    t = ef_line_bunch(1e-3, 1e-9, 201, pq)
    @test_throws ErrorException track!(t, ef_beamline(mk(2), pq);
      notify = false, effects = (CSRReplay(rec),))

    # The record is emptied at the next track! of its effect
    b = ef_line_bunch(1e-3, 1e-9, 201, pq)
    track!(b, ef_beamline(mk(1), pq); notify = false,
           effects = (CSREffect(params; record = rec),))
    @test [(p.ix, p.k) for p in rec.points] == [(1, 1), (2, 1), (2, 2)]
  end

  @testset "Errors" begin
    pq = ef_pq(100)
    params = CSRParams(n_bin = 40)
    b = ef_line_bunch(1e-3, 1e-9, 201, pq)
    bl = ef_beamline([SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0,
                            tilt_ref = pi / 2)], pq)
    @test_throws ErrorException track!(b, bl; notify = false,
      effects = (CSREffect(params),))
    # Fewer live particles than bins
    b = ef_line_bunch(1e-3, 1e-9, 21, pq)
    bl = ef_beamline([SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0)], pq)
    @test_throws ErrorException track!(b, bl; notify = false,
      effects = (CSREffect(params),))
  end
end
