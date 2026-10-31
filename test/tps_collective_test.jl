# Phase 7 Stage 2 (D37): collective TPS. The bunch's z and q, the
# geometry (element L, g, centroid), s_kick and gamma are TPS in
# shared knobs; deposit and kick tables in TPS. Checked against
# ForwardDiff Jacobians and nested-Dual second (and third) derivatives
# of the same function of Float64 knobs. Stage 2b: test particles in
# 6D + knobs with the collective TPS kick; range, element-type and
# descriptor checks.

@testset "Collective TPS" begin
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  pz0 = zeros(n)
  ds0 = L3[2] / max(1, round(Int,
                             Beamlines.scalarize_all(L3[2]) / 0.1,
                             RoundNearestTiesAway))

  @testset "Helpers" begin
    d = Descriptor(2, 2)
    x = @vars(d)
    @test CSR.thread_safe_number(Float64,
                                        ForwardDiff.Dual{Nothing,
        Float64,2})
    @test !CSR.thread_safe_number(Float64, typeof(x[1]))
    @test CSR.cpu_workgroupsize(100, Float64) === nothing
    @test CSR.cpu_workgroupsize(100, Float64,
                                       typeof(x[1])) == 100
    m = CSR.modulo2(4.0 + 2x[1], pi / 2)
    @test scalar(m) ≈ CSR.modulo2(4.0,
        pi / 2) && m[[1 => 1]] == 2
    for s in (8.0, -8.0)
      r = CSR.cube_root(s + x[1])
      @test scalar(r) ≈ cbrt(s) && r[[1 => 1]] ≈ 1 / (3 *
          cbrt(s)^2) && r[[1 => 2]] ≈ -1 / (9 * cbrt(s)^5)
    end
    @test_throws ErrorException CSR.cube_root(0.0 + x[1])
  end

  # Knobs v: element 2's L and g, centroid x at the kick element's
  # exit, s_kick, gamma, and a bunch knob scaling the charge and the
  # length
  v0 = [0.3, 1.5, 1.1e-3, 0.17, 200.0, 0.0]
  function F(v, p)
    c = similar(c3, eltype(v))
    c .= c3
    c[3, 1] = v[3]
    geom = CSRGeometry([L3[1], v[1], L3[3]], [g3[1], v[2], g3[3]], c;
                       bmad_compat = p.bmad_compat)
    ds = v[1] / max(1, round(Int, Beamlines.scalarize_all(v[1]) / 0.1,
                             RoundNearestTiesAway))
    return csr_kick_pz(pz0, z .* (1 + 0.2v[6]), q .* (1 + v[6]),
                       alive, geom, 2, ds, v[4], v[5], electron, p)
  end

  @testset "Order 2 (images = $n_img, bmad_compat = $bc)" for (
      n_img, bc) in ((0, false), (2, false), (2, true))
    p = CSRParams(n_bin = 40, n_shield_images = n_img,
                  beam_chamber_height = n_img > 0 ? 3e-3 : 0.0,
                  bmad_compat = bc)
    d = Descriptor(6, 2)
    x = @vars(d)
    t = F(v0 .+ x, p)
    y0 = F(v0, p)
    @test maximum(abs, scalar.(t) .- y0) <= 1e-13 * maximum(abs, y0)
    Jt, Jf = GTPSA.jacobian(t), ForwardDiff.jacobian(v -> F(v, p), v0)
    for c in 1:6
      @test any(!iszero, Jf[:, c])
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-11
    end
    Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(
        v -> F(v, p), u), v0), n, 6, 6)
    Ht = permutedims(cat((GTPSA.hessian(ti) for ti in t)...;
                         dims = 3), (3, 1, 2))
    Hmax = maximum(abs, Hf)
    for a in 1:6, b in a:6
      ref = maximum(abs, Hf[:, a, b])
      err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
      @test err <= 1e-10 * ref + 1e-14 * Hmax
    end
  end

  @testset "Order 3 (images = 0)" begin
    n_img = 0
    # g and gamma: three Newton steps per source root (root_order of
    # the TPS)
    p = CSRParams(n_bin = 40, n_shield_images = n_img,
                  beam_chamber_height = n_img > 0 ? 3e-3 : 0.0)
    F2(u) = F([v0[1], u[1], v0[3], v0[4], u[2], v0[6]], p)
    d = Descriptor(2, 3)
    x = @vars(d)
    t = F2([v0[2], v0[5]] .+ x)
    u0 = [v0[2], v0[5]]
    T3f = reshape(ForwardDiff.jacobian(w -> ForwardDiff.jacobian(
        u -> ForwardDiff.jacobian(F2, u), w), u0), n, 2, 2, 2)
    T3t = [scalar(GTPSA.deriv(t[j], [count(==(i), (a, b,
        c)) for i in 1:2])) for j in 1:n, a in 1:2, b in 1:2,
           c in 1:2]
    @test any(!iszero, T3f)
    Tmax = maximum(abs, T3f)
    for a in 1:2, b in a:2, c in b:2
      ref = maximum(abs, T3f[:, a, b, c])
      err = maximum(abs, T3t[:, a, b, c] .- T3f[:, a, b, c])
      @test err <= 1e-9 * ref + 1e-13 * Tmax
    end
  end

  @testset "csr_kick! with TPS pz" begin
    p = CSRParams(n_bin = 40, n_shield_images = 2,
                  beam_chamber_height = 3e-3)
    d = Descriptor(2, 2)
    x = @vars(d)
    qx = q .* (1 + x[2])
    geom = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)
    ref = csr_kick_pz(pz0, z, qx, alive, geom, 2, ds0, 0.17, 200.0,
                      electron, p)
    pz_t = [zero(x[1]) for _ in 1:n]
    csr_kick!(pz_t, z, qx, alive, geom, 2, ds0, 0.17, 200.0, electron,
              p)
    @test scalar.(pz_t) == scalar.(ref)
    @test GTPSA.jacobian(pz_t) == GTPSA.jacobian(ref)
    @test all(GTPSA.hessian(pz_t[i]) ==
              GTPSA.hessian(ref[i]) for i in 1:n)
    @test any(!iszero, GTPSA.jacobian(ref))
  end

  @testset "csr_kick! pz element type" begin
    p = CSRParams(n_bin = 40)
    geom0 = CSRGeometry(L3, g3, c3)
    d = Descriptor(2, 2)
    x = @vars(d)
    errs = (() -> csr_kick!(zeros(n), z, q, alive, geom0, 2, ds0,
                            0.17 + x[1], 200.0, electron, p),
            () -> csr_kick!(zeros(n), z, q .* (1 + x[2]), alive,
                            geom0, 2, ds0, 0.17, 200.0, electron, p),
            () -> csr_kick!(zeros(n), z, q, alive, geom0, 2, ds0,
                            0.17, ForwardDiff.Dual{Nothing}(200.0,
        1.0), electron, p))
    for f in errs
      @test_throws ErrorException f()
      @test occursin("cannot hold the kick", try f(); "" catch err;
                     sprint(showerror, err) end)
    end
    @test csr_kick!(zeros(Float32, n), z, q, alive, geom0, 2, ds0,
                    0.17, 200.0, electron, p) isa CSRStepResult
    @test csr_kick!(zeros(n), z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                    electron, p) isa CSRStepResult
  end

  # Stage 2b (D37): test particles in 6D + knobs with a collective TPS
  # kick (frozen in phase space, collective in knobs)
  m = 50
  z0t = range(-2.5e-3, 2.5e-3; length = m)
  w0 = [zeros(6); 1.5; 200.0; 0.0]
  function G(w, p)
    geom = CSRGeometry(L3, [g3[1], w[7], g3[3]], c3;
                       bmad_compat = p.bmad_compat)
    zt = [z0t[j] + w[5] * (1 + 0.1j / m) + 0.3 * w[1]^2 - 0.2 * w[5] *
          w[6] for j in 1:m]
    pzt = [w[6] + 1e-4j / m for j in 1:m]
    pzb = [zero(w[1]) for _ in 1:n]
    csr_kick!(pzb, z .* (1 + 0.2w[9]), q .* (1 + w[9]), alive, geom,
              2, ds0, 0.17, w[8], electron, p;
              test_particles = (pzt, zt, fill(true, m)))
    return pzt
  end
  pzt0 = [1e-4j / m for j in 1:m]

  @testset "Test particles in 6D + knobs (images = 0, bmad_compat = false)" begin
    n_img = 0
    bc = false
    p = CSRParams(n_bin = 40, n_shield_images = n_img,
                  beam_chamber_height = n_img > 0 ? 3e-3 : 0.0,
                  bmad_compat = bc)
    d = Descriptor(9, 2)
    x = @vars(d)
    t = G(w0 .+ x, p)
    y0 = G(w0, p)
    @test maximum(abs, y0 .- pzt0) > 0
    @test maximum(abs, scalar.(t) .- y0) <= 1e-13 * maximum(abs,
        y0 .- pzt0)
    Jt, Jf = GTPSA.jacobian(t), ForwardDiff.jacobian(w -> G(w, p), w0)
    for c in 1:4
      @test all(iszero, Jt[:, c]) && all(iszero, Jf[:, c])
    end
    for c in 5:9
      @test any(!iszero, Jf[:, c])
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-11
    end
    Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(
        w -> G(w, p), u), w0), m, 9, 9)
    Ht = permutedims(cat((GTPSA.hessian(ti) for ti in t)...;
                         dims = 3), (3, 1, 2))
    Hmax = maximum(abs, Hf)
    for (a, b) in ((1, 1), (5, 6), (5, 7), (5, 8), (5, 9), (7, 7), (7,
        8), (7, 9), (8, 8), (8, 9), (9, 9))
      # phase space × knob and knob × knob blocks are exercised
      @test any(!iszero, Hf[:, a, b])
    end
    for a in 1:9, b in a:9
      ref = maximum(abs, Hf[:, a, b])
      err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
      @test err <= 1e-10 * ref + 1e-14 * Hmax
    end
    # Test particles at bunch particles' positions (same TPS z) get
    # the bunch particles' kicks, every coefficient
    w = w0 .+ x
    geom = CSRGeometry(L3, [g3[1], w[7], g3[3]], c3; bmad_compat = bc)
    zb, qb = z .* (1 + 0.2w[9]), q .* (1 + w[9])
    idx = findall(alive)[1:37:end]
    pzb = [zero(x[1]) for _ in 1:n]
    pzt = [zero(x[1]) for _ in idx]
    csr_kick!(pzb, zb, qb, alive, geom, 2, ds0, 0.17, w[8], electron,
              p; pz_scale = 1.3,
              test_particles = (pzt, zb[idx], fill(true,
        length(idx))))
    @test all(i -> isequal(pzt[i], pzb[idx[i]]), eachindex(idx))
    @test any(!iszero, GTPSA.jacobian(pzt))
    pzb1 = [zero(x[1]) for _ in 1:n]
    pzt1 = [zero(x[1]) for _ in idx]
    csr_kick!(pzb1, zb, qb, alive, geom, 2, ds0, 0.17, w[8], electron,
              p; test_particles = (pzt1, zb[idx], fill(true,
        length(idx))))
    @test scalar.(pzt) ≈ 1.3 .* scalar.(pzt1) rtol = 1e-14
    @test GTPSA.jacobian(pzt) ≈ 1.3 .*
        GTPSA.jacobian(pzt1) rtol = 1e-14
  end

  @testset "Test particles in 6D, knobs as GTPSA parameters" begin
    p = CSRParams(n_bin = 40, n_shield_images = 2,
                  beam_chamber_height = 3e-3)
    d = Descriptor(6, 2, 3, 2)
    x, k = @vars(d), @params(d)
    t = G(vcat(w0[1:6] .+ x, w0[7:9] .+ k), p)
    Jt = GTPSA.jacobian(t; include_params = true)
    Jf = ForwardDiff.jacobian(w -> G(w, p), w0)
    @test size(Jt) == (m, 9)
    for c in 5:9
      @test Jt[:, c] ≈ Jf[:, c] rtol = 1e-11
    end
    Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(
        w -> G(w, p), u), w0), m, 9, 9)
    Ht = permutedims(cat((GTPSA.hessian(ti; include_params = true)
                          for ti in t)...; dims = 3), (3, 1, 2))
    Hmax = maximum(abs, Hf)
    for a in 1:9, b in a:9
      ref = maximum(abs, Hf[:, a, b])
      err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
      @test err <= 1e-10 * ref + 1e-14 * Hmax
    end
  end

  @testset "Test particles: pz_t element type" begin
    p = CSRParams(n_bin = 40)
    geom0 = CSRGeometry(L3, g3, c3)
    d = Descriptor(2, 2)
    x = @vars(d)
    geom1 = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)
    errmsg(f) = try f(); "" catch err; sprint(showerror, err) end
    zt3 = [0.0, 1e-3, -1e-3]
    # TPS kick, Float64 pz_t: error, nothing kicked
    pzb, pzt = [zero(x[1]) for _ in 1:n], zeros(3)
    f1 = () -> csr_kick!(pzb, z, q, alive, geom1, 2, ds0, 0.17, 200.0,
                         electron, p; test_particles = (pzt, zt3,
        fill(true, 3)))
    @test_throws ErrorException f1()
    msg = errmsg(f1)
    @test occursin("cannot hold the kick", msg) && occursin("pz_t",
        msg)
    @test occursin("csr_kick!: pz_t", msg)
    @test all(t -> isequal(t, zero(x[1])), pzb) && pzt == zeros(3)
    # Dual kick (gamma), Float64 pz_t
    γd = ForwardDiff.Dual{Nothing}(200.0, 1.0)
    pzbd = fill(zero(γd), n)
    pzt3 = zeros(3)
    f2 = () -> csr_kick!(pzbd, z, q, alive, geom0, 2, ds0, 0.17, γd,
                         electron, p; test_particles = (pzt3, zt3,
        fill(true, 3)))
    @test_throws ErrorException f2()
    @test occursin("cannot hold the kick", errmsg(f2))
    @test all(iszero, pzbd) && pzt3 == zeros(3)
    # Float64 kick, TPS z_t, Float64 pz_t
    st = CSR.csr_binned_kick(z, q, alive, geom0, 2, ds0, 0.17,
                                    200.0, electron, p)
    f3 = () -> CSR.csr_apply_test_kicks!(zeros(1),
        [0.0 + x[2]], [true], st.bins, st.kick)
    @test_throws ErrorException f3()
    @test occursin("cannot hold the kick", errmsg(f3))
    # Float32 pz_t with a Float64 kick still runs
    pz32 = CSR.csr_apply_test_kicks!(zeros(Float32, 3), zt3,
        fill(true, 3), st.bins, st.kick)
    @test pz32 isa Vector{Float32}
    @test pz32 == Float32.(CSR.csr_apply_test_kicks!(zeros(3),
        zt3, fill(true, 3), st.bins, st.kick))
    @test any(!iszero, pz32)
  end

  @testset "Zero-length element in a TPS geometry" begin
    k = @vars(Descriptor(2, 2))
    L = [0.3, 0.0, 0.3]
    g64 = [1.0, 0.0, 1.0]
    gk = [1.0 + k[1], zero(k[1]), 1.0 + k[2]]
    c64 = zeros(4, 6)
    # rows 2 and 3 equal: element 2's chord is exactly zero
    c64[:, 1] .= [1e-4, 2e-4, 2e-4, 3e-4]
    ck = map(x -> x + zero(k[1]), c64)
    gt = CSRGeometry(L, gk, ck)
    g0 = CSRGeometry(L, g64, c64)
    for i in 1:3
      if i == 2
        @test scalar(gt.ele[i].theta_chord) == g0.ele[i].theta_chord
        @test scalar(gt.ele[i].L_chord) == g0.ele[i].L_chord
      else
        # GTPSA's C libm and Julia's may differ in the last bit
        @test scalar(gt.ele[i].theta_chord) ≈ g0.ele[
            i].theta_chord rtol = 4eps()
        @test scalar(gt.ele[i].L_chord) ≈ g0.ele[
            i].L_chord rtol = 4eps()
      end
    end
    @test g0.ele[2].L_chord == 0
    @test all(iszero, GTPSA.jacobian([gt.ele[2].L_chord,
                                      gt.ele[2].theta_chord]))
    gd = CSRGeometry(L, [1.0 + ForwardDiff.Dual(0.0, 1.0), 0.0, 1.0],
                     c64)
    @test all(isfinite, ForwardDiff.partials(gd.ele[2].theta_chord))
    @test ForwardDiff.value(gd.ele[2].L_chord) == 0
  end

  # Last: leaves GTPSA.desc_current at a descriptor no later code
  # uses.
  @testset "Descriptor check" begin
    errmsg(f) = try f(); "" catch err; sprint(showerror, err) end
    p = CSRParams(n_bin = 40)
    geom0 = CSRGeometry(L3, g3, c3)
    d1 = Descriptor(2, 2)
    x = @vars(d1, dynamic = true)
    xs = @vars(d1)
    geom1 = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)
    qx = q .* (1 + x[2])
    pzf, zf = [x[1] + 0.0], [0.0 + x[2]]
    csr_kick!(copy(pz0), z, q, alive, geom0, 2, ds0, 0.17, 200.0,
              electron, p; test_particles = (pzf, zf, [true]))
    d2 = Descriptor(3, 2)
    cases = [
      ("q", () -> csr_kick_pz(pz0, z, qx, alive, geom0, 2, ds0, 0.17,
                              200.0, electron, p)),
      ("geom", () -> csr_kick_pz(pz0, z, q, alive, geom1, 2, ds0,
                                 0.17, 200.0, electron, p)),
      ("s_kick", () -> csr_kick!(copy(pz0), z, q, alive, geom0, 2,
                                 ds0, 0.17 + x[1], 200.0, electron,
                                 p)),
      ("q", () -> csr_bin_particles(z, qx, alive, p)),
      ("g", () -> CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)),
      ("g", () -> planar_survey(L3, [g3[1], 1.5 + x[1], g3[3]])),
      ("gamma", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + x[1])),
    ]
    for (name, f) in cases
      @test_throws ErrorException f()
      msg = errmsg(f)
      @test occursin("GTPSA's current Descriptor",
                     msg) && occursin("`$name`", msg)
    end
    pzf2, zf2 = [x[1] + 0.0], [0.0 + x[2]]
    csr_kick!(copy(pz0), z, q, alive, geom0, 2, ds0, 0.17, 200.0,
              electron, p; test_particles = (pzf2, zf2, [true]))
    @test scalar(pzf2[1]) == scalar(pzf[1])
    @test GTPSA.jacobian(pzf2) == GTPSA.jacobian(pzf)
    y = @vars(d2, dynamic = true)
    @test length(csr_kick_pz(pz0, z, q .* (1 + y[1]), alive, geom0, 2,
                             ds0, 0.17, 200.0, electron, p)) == n
    @test !CSR.descriptor_mismatch(1.0)
    @test !CSR.descriptor_mismatch(zeros(3))
    @test !CSR.descriptor_mismatch(geom0)
    @test CSR.descriptor_mismatch(x[1])
    @test CSR.descriptor_mismatch(geom1)
    @test !CSR.descriptor_mismatch(y[1])
    @test !CSR.descriptor_mismatch(xs[1])
    # Stage 2b: with a TPS kick, test particles are checked (pz_t
    # first, then z_t); nothing is kicked on error
    qy = q .* (1 + y[1])
    pzb = [zero(y[1]) for _ in 1:n]
    pzt1 = [x[1] + 0.0]
    f1 = () -> csr_kick!(pzb, z, qy, alive, geom0, 2, ds0, 0.17,
                         200.0, electron, p; test_particles = (pzt1,
        [0.0 + x[2]], [true]))
    @test_throws ErrorException f1()
    msg = errmsg(f1)
    @test occursin("GTPSA's current Descriptor",
                   msg) && occursin("`pz_t`", msg)
    @test occursin("csr_kick!:", msg)
    @test all(t -> isequal(t, zero(y[1])), pzb) && isequal(pzt1[1],
        x[1] + 0.0)
    f2 = () -> csr_kick!(pzb, z, qy, alive, geom0, 2, ds0, 0.17,
                         200.0, electron, p;
                         test_particles = ([y[2] + 0.0], [0.0 + x[2]],
        [true]))
    @test occursin("`z_t`", errmsg(f2))
    pzt2 = [y[2] + 0.0]
    csr_kick!(pzb, z, qy, alive, geom0, 2, ds0, 0.17, 200.0, electron,
              p; test_particles = (pzt2, [0.0 + y[3]], [true]))
    @test pzt2[1][[1 => 1]] != 0
    @test pzt2[1][[3 => 1]] != 0   # z_t dependence (∂Δpz/∂y3)
  end
end
