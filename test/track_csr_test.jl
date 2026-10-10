# track_csr!: Beamline tracking with 1D CSR (phase 2).

const TC_EXT = Base.get_extension(CSR, :CSRBeamTrackingBeamlinesExt)
const TC_E = Species("electron")

# p_over_q_ref for electrons with Lorentz factor gamma
tc_pq(gamma) = BeamTracking.E_to_R(TC_E, gamma * massof(TC_E))

# Line bunch: n particles on a z grid over ±6σ, Gaussian weights for a total charge Q [C].
function tc_line_bunch(sigma, Q, n, pq; x = 0.0)
  z = collect(range(-6sigma, 6sigma; length = n))
  w = gaussian.(z, sigma)
  w .*= (Q / E_CHARGE) / sum(w)
  v = zeros(n, 6)
  v[:, 1] .= x
  v[:, 5] .= z
  return Bunch(v, nothing, w; species = TC_E, p_over_q_ref = pq)
end

@testset "track_csr!" begin
  @testset "Floor conventions match BeamTracking (field-free particle stays on a straight line)" begin
    # With no fields a particle moves on a straight floor line; its floor position at each element exit,
    # built from the reference survey and its coordinates, must stay on that line.
    pq = tc_pq(1e3)
    line = [Drift(L = 0.3),
            SBend(L = 0.5, g_ref = 0.4),
            Drift(L = 0.2),
            LineElement(L = 0.6, dx = 1e-3, dz = 0.55, dy_rot = 0.05),
            SBend(L = 0.4, g_ref = 0.3, tilt_ref = pi),
            LineElement(L = 0.3, dx = -2e-3, dz = 0.3, dy_rot = -0.08),
            Drift(L = 0.5)]
    bl = Beamline(line; p_over_q_ref = pq, species_ref = TC_E)
    b = Bunch(zeros(1, 6); species = TC_E, p_over_q_ref = pq)
    f = FloorPosition()
    for (i, ele) in enumerate(bl.line)
      track!(b, ele)
      L = float(ele.L)
      pp = ele.PatchParams
      g = isnothing(pp) ? TC_EXT.csr_ref_curvature(ele.BendParams, i) : 0.0
      f = TC_EXT.csr_ref_exit(f, L, g, pp)
      cf = CSR.centroid_floor(f, b.coords.v[1, 1], b.coords.v[1, 2], b.coords.v[1, 6])
      @test abs(cf.r[1]) < 1e-14
      @test abs(cf.theta) < 1e-14
    end
    @test f.theta ≈ -0.4 * 0.5 + 0.05 + 0.3 * 0.4 - 0.08
    @test_throws ErrorException TC_EXT.csr_ref_curvature(Beamlines.BendParams(0.4, pi / 2, 0.0, 0.0, 0.0, 0.0), 1)
  end

  @testset "CSR off: identical to track!" begin
    pq = tc_pq(200)
    line = [Drift(L = 0.3), SBend(L = 0.5, g_ref = 0.4, Kn0 = 0.4, e1 = 0.1, e2 = 0.1), Quadrupole(L = 0.2, Kn1 = 2.0),
            Marker(), Drift(L = 0.4)]
    b1 = tc_line_bunch(1e-3, 1e-9, 101, pq; x = 1e-4)
    b2 = tc_line_bunch(1e-3, 1e-9, 101, pq; x = 1e-4)
    track!(b1, Beamline(line; p_over_q_ref = pq, species_ref = TC_E))
    track_csr!(b2, Beamline(line; p_over_q_ref = pq, species_ref = TC_E);
               params = CSRParams(n_bin = 40, ds_step = 0.05), csr_on = ele -> false)
    @test b1.coords.v == b2.coords.v
    @test b1.t_ref == b2.t_ref
  end

  @testset "Steady state in a long bend" begin
    # As the csr_kick! test (kick_test.jl): ρ = 10 m, 80 bends of 0.1 m, γ = 1e4, kick in the last bend.
    rho = 10.0; L_ele = 0.1; n_ele = 80; gamma = 1e4; sigma = 1e-3; Q = 1e-9
    pq = tc_pq(gamma)
    bl = Beamline([SBend(L = L_ele, g_ref = 1 / rho, Kn0 = 1 / rho) for _ in 1:n_ele]; p_over_q_ref = pq, species_ref = TC_E)
    b = tc_line_bunch(sigma, Q, 1201, pq)
    params = CSRParams(n_bin = 200, particle_bin_span = 2, ds_step = L_ele / 2)
    results = Dict{Tuple{Int,Int},CSRStepResult}()
    track_csr!(b, bl; params, diagnostics = (i, i_step, s, r) -> (results[(i, i_step)] = r))
    @test length(results) == n_ele * 3

    res = results[(n_ele, 1)]
    N = Q / E_CHARGE
    expected(x) = steady_state_dpz_ds(x, x -> dgaussian(x, sigma), rho, N, gamma; u_max = (12sigma)^(2/3))
    core = findall(x -> abs(x) <= 2sigma, res.z_center)
    W = expected.(res.z_center[core])
    @test maximum(abs.(res.kick_per_meter[core] .- W)) / maximum(abs, W) < 2e-2
    @test sum(res.charge_per_meter) * (res.z_center[2] - res.z_center[1]) ≈ Q rtol = 1e-10

    # Energy loss: the bunch-averaged Δpz is negative (CSR radiates).
    @test sum(b.coords.v[:, 6] .* b.coords.weight) < 0
  end

  @testset "Per-element selection, zero-length elements, weights" begin
    pq = tc_pq(100)
    line = [Drift(L = 0.2), SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0), Marker(), SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0)]
    params = CSRParams(n_bin = 40, ds_step = 0.1)
    calls = Tuple{Int,Int}[]
    bl = Beamline(line; p_over_q_ref = pq, species_ref = TC_E)
    b = tc_line_bunch(1e-3, 1e-9, 201, pq)
    track_csr!(b, bl; params, csr_on = ele -> ele.kind == "SBend", diagnostics = (i, j, s, r) -> push!(calls, (i, j)))
    @test calls == [(2, 0), (2, 1), (2, 2), (2, 3), (4, 0), (4, 1), (4, 2), (4, 3)]

    # Linear in charge: doubling the weights doubles the first kick.
    kicks = Float64[]
    for scale in (1, 2)
      b = tc_line_bunch(1e-3, scale * 1e-9, 201, pq)
      track_csr!(b, Beamline(line[2:2]; p_over_q_ref = pq, species_ref = TC_E); params,
                 diagnostics = (i, j, s, r) -> j == 1 && push!(kicks, maximum(abs, r.kick_per_meter)))
    end
    @test kicks[2] ≈ 2kicks[1] rtol = 1e-10

    # weight = nothing: one electron per macroparticle.
    b = Bunch(hcat(zeros(201, 4), collect(range(-6e-3, 6e-3; length = 201)), zeros(201)); species = TC_E, p_over_q_ref = pq)
    qtot = Ref(0.0)
    track_csr!(b, Beamline(line[2:2]; p_over_q_ref = pq, species_ref = TC_E); params,
               diagnostics = (i, j, s, r) -> j == 0 && (qtot[] = sum(r.charge_per_meter) * (r.z_center[2] - r.z_center[1])))
    @test qtot[] ≈ 201 * E_CHARGE rtol = 1e-10
  end

  @testset "Patch and misaligned elements inside a CSR line" begin
    pq = tc_pq(100)
    line = [SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0), LineElement(L = 0.2, dx = 1e-4, dz = 0.2, dy_rot = 1e-3),
            SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0, x_offset = 1e-4), Drift(L = 0.3)]
    b = tc_line_bunch(1e-3, 1e-9, 201, pq)
    track_csr!(b, Beamline(line; p_over_q_ref = pq, species_ref = TC_E); params = CSRParams(n_bin = 40, ds_step = 0.1))
    @test all(==(BeamTracking.STATE_ALIVE), b.coords.state)
    @test all(isfinite, b.coords.v)
  end

  @testset "Sliced centroid" begin
    pq = tc_pq(100)
    params = CSRParams(n_bin = 40, ds_step = 0.02)
    c0 = [2e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]
    function run(line, sliced_centroid; weight_scale = 1.0)
      b = tc_line_bunch(1e-3, 1e-9, 201, pq)
      b.coords.v[101, :] = c0   # a particle on the centroid start (z = 0 is the bunch centre)
      b.coords.weight .*= weight_scale
      res = Dict{Tuple{Int,Int},CSRStepResult}()
      exits = Dict{Int,Vector{Float64}}()
      n_step(i) = csr_n_step(float(line[i].L), params)[1]
      track_csr!(b, Beamline(line; p_over_q_ref = pq, species_ref = TC_E); params, centroid_start = c0, sliced_centroid,
                 diagnostics = (i, j, s, r) -> (res[(i, j)] = r; j == n_step(i) && (exits[i] = b.coords.v[101, :])))
      return res, exits, b
    end

    # Pure bend and drift: BeamTracking's integration is exact, so slicing the centroid changes nothing
    pure = [Drift(L = 0.1), SBend(L = 0.1, g_ref = 1.0, Kn0 = 1.0)]
    r1, _, b1 = run(pure, true)
    r0, _, b0 = run(pure, false)
    # (equal up to round-off: the exact maps are not bit-identical when split)
    @test maximum(maximum(abs, r1[k].I_csr_per_meter .- r0[k].I_csr_per_meter) for k in keys(r0)) <=
          1e-12 * maximum(maximum(abs, r0[k].I_csr_per_meter) for k in keys(r0))
    @test b1.coords.v ≈ b0.coords.v rtol = 1e-12

    # Combined-function bend: the one-step integration of a whole element differs from the sliced one
    cf = [Drift(L = 0.1), SBend(L = 0.1, g_ref = 1.0, Kn0 = 1.0, Kn1 = -20.0)]
    r1, _, _ = run(cf, true)
    r0, _, _ = run(cf, false)
    @test maximum(maxrel(r1[k].I_csr_per_meter, r0[k].I_csr_per_meter) for k in keys(r0) if any(!iszero, r0[k].I_csr_per_meter)) > 1e-8

    # The sliced centroid is the orbit of a bunch particle: rebuild the geometry from the exits of a zero-charge
    # particle started on the centroid and recompute I_csr
    res, exits, _ = run(cf, true; weight_scale = 1e-30)
    L = [0.1, 0.1]; g = [0.0, 1.0]
    geom = CSRGeometry(L, g, permutedims(hcat(c0, exits[1], exits[2])))
    gamma = BeamTracking.R_to_E(TC_E, pq) / massof(TC_E)
    # Compared relative to the largest I_csr over all steps: in the drift I_csr is at round-off level
    dmax = 0.0; Imax = 0.0
    for ((i, j), r) in res
      n, ds = csr_n_step(L[i], params)
      kf = (j == 0 || j == n) ? 0.5 : 1.0
      kp = CSR.CSRKickPoint(geom, i, j * ds, gamma; kick_factor = kf)
      I, _, _ = CSR.csr_wake_table(geom, kp, r.z_center[2] - r.z_center[1], params.n_bin, CSR.CPU())
      e = geom.ele[i]
      dmax = max(dmax, maximum(abs, I ./ (kf * ds * e.L_chord / e.L) .- r.I_csr_per_meter))
      Imax = max(Imax, maximum(abs, r.I_csr_per_meter))
    end
    @test dmax / Imax < 1e-10
  end

  @testset "Shielding: Bmad csr_and_space_charge_test parameters" begin
    # ben01: sbend, l = 0.1, b_field = 0.17, b1_gradient = -10; e_tot = 42 MeV; 77 pC, σ_z = 0.899377 mm;
    # n_bin = 40, particle_bin_span = 2, ds_track_step = 0.01; n_shield_images = 4, beam_chamber_height = 0.01.
    # No Bmad numbers to compare with (its output.correct is for its own random bunch).
    gamma = 42e6 / massof(TC_E)
    pq = tc_pq(gamma)
    brho = float(pq)
    line = [SBend(L = 0.1, g_ref = 0.17 / brho, Kn0 = 0.17 / brho, Kn1 = -10 / brho)]
    eloss = Float64[]
    for n_shield_images in (0, 4)
      b = tc_line_bunch(0.899377e-3, 77e-12, 1001, pq; x = 2e-4)
      params = CSRParams(n_bin = 40, particle_bin_span = 2, ds_step = 0.01, n_shield_images = n_shield_images, beam_chamber_height = 0.01)
      track_csr!(b, Beamline(line; p_over_q_ref = pq, species_ref = TC_E); params)
      @test all(isfinite, b.coords.v)
      push!(eloss, sum(b.coords.v[:, 6] .* b.coords.weight))
    end
    @test eloss[1] < 0
    @test eloss[2] != eloss[1]
  end

  @testset "Errors" begin
    pq = tc_pq(100)
    bl = Beamline([SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0)]; p_over_q_ref = pq, species_ref = TC_E)
    @test_throws ErrorException track_csr!(tc_line_bunch(1e-3, 1e-9, 20, pq), bl; params = CSRParams(n_bin = 40, ds_step = 0.1))
    @test_throws ErrorException track_csr!(tc_line_bunch(1e-3, 1e-9, 201, pq), bl; params = CSRParams(n_bin = 40))
    blv = Beamline([SBend(L = 0.3, g_ref = 1.0, Kn0 = 1.0, tilt_ref = pi / 2)]; p_over_q_ref = pq, species_ref = TC_E)
    @test_throws ErrorException track_csr!(tc_line_bunch(1e-3, 1e-9, 201, pq), blv; params = CSRParams(n_bin = 40, ds_step = 0.1))
  end

  @testset "csr_ref_energy" begin
    @test CSR.csr_ref_energy(1.0, 3.0, 0.0, 2.0) == 1.0
    @test CSR.csr_ref_energy(1.0, 3.0, 2.0, 2.0) == 3.0
    @test CSR.csr_ref_energy(1.0, 3.0, 0.5, 2.0) ≈ 1.5
  end
end
