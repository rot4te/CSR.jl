# Phase 7 Stage 2 (D37): collective TPS. The bunch's z and q, the geometry (element L, g, centroid), s_kick and gamma
# are TPS in shared knobs; deposit and kick tables in TPS. Checked against ForwardDiff Jacobians and nested-Dual
# second (and third) derivatives of the same function of Float64 knobs.

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

  @testset "Helpers" begin
    d = Descriptor(2, 2)
    x = @vars(d)
    @test CSR.thread_safe_number(Float64, ForwardDiff.Dual{Nothing,Float64,2})
    @test !CSR.thread_safe_number(Float64, typeof(x[1]))
    @test CSR.cpu_workgroupsize(100, Float64) === nothing
    @test CSR.cpu_workgroupsize(100, Float64, typeof(x[1])) == 100
    m = CSR.modulo2(4.0 + 2x[1], pi / 2)
    @test scalar(m) ≈ CSR.modulo2(4.0, pi / 2) && m[[1 => 1]] == 2
    for s in (8.0, -8.0)
      r = CSR.cube_root(s + x[1])
      @test scalar(r) ≈ cbrt(s) && r[[1 => 1]] ≈ 1 / (3 * cbrt(s)^2) && r[[1 => 2]] ≈ -1 / (9 * cbrt(s)^5)
    end
    @test_throws ErrorException CSR.cube_root(0.0 + x[1])
  end

  # Knobs v: element 2's L and g, centroid x at the kick element's exit, s_kick, gamma, and a bunch knob scaling the
  # charge and the length
  v0 = [0.3, 1.5, 1.1e-3, 0.17, 200.0, 0.0]
  function F(v, p)
    c = similar(c3, eltype(v))
    c .= c3
    c[3, 1] = v[3]
    geom = CSRGeometry([L3[1], v[1], L3[3]], [g3[1], v[2], g3[3]], c; bmad_compat = p.bmad_compat)
    return csr_kick_pz(pz0, z .* (1 + 0.2v[6]), q .* (1 + v[6]), alive, geom, 2, v[4], v[5], electron, p)
  end

  @testset "Order 2 (images = $n_img, bmad_compat = $bc)" for n_img in (0, 2), bc in (false, true)
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0,
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
    Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(v -> F(v, p), u), v0), n, 6, 6)
    Ht = permutedims(cat((GTPSA.hessian(ti) for ti in t)...; dims = 3), (3, 1, 2))
    Hmax = maximum(abs, Hf)
    for a in 1:6, b in a:6
      ref = maximum(abs, Hf[:, a, b])
      err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
      @test err <= 1e-10 * ref + 1e-14 * Hmax
    end
  end

  @testset "Order 3 (images = $n_img)" for n_img in (0, 2)
    # g and gamma: three Newton steps per source root (root_order of the TPS)
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0)
    F2(u) = F([v0[1], u[1], v0[3], v0[4], u[2], v0[6]], p)
    d = Descriptor(2, 3)
    x = @vars(d)
    t = F2([v0[2], v0[5]] .+ x)
    u0 = [v0[2], v0[5]]
    T3f = reshape(ForwardDiff.jacobian(w -> ForwardDiff.jacobian(u -> ForwardDiff.jacobian(F2, u), w), u0), n, 2, 2, 2)
    T3t = [scalar(GTPSA.deriv(t[j], [count(==(i), (a, b, c)) for i in 1:2])) for j in 1:n, a in 1:2, b in 1:2, c in 1:2]
    @test any(!iszero, T3f)
    Tmax = maximum(abs, T3f)
    for a in 1:2, b in a:2, c in b:2
      ref = maximum(abs, T3f[:, a, b, c])
      err = maximum(abs, T3t[:, a, b, c] .- T3f[:, a, b, c])
      @test err <= 1e-9 * ref + 1e-13 * Tmax
    end
  end

  @testset "csr_kick! with TPS pz" begin
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 3e-3)
    d = Descriptor(2, 2)
    x = @vars(d)
    qx = q .* (1 + x[2])
    geom = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)
    ref = csr_kick_pz(pz0, z, qx, alive, geom, 2, 0.17, 200.0, electron, p)
    pz_t = [zero(x[1]) for _ in 1:n]
    csr_kick!(pz_t, z, qx, alive, geom, 2, 0.17, 200.0, electron, p)
    @test scalar.(pz_t) == scalar.(ref)
    @test GTPSA.jacobian(pz_t) == GTPSA.jacobian(ref)
    @test all(GTPSA.hessian(pz_t[i]) == GTPSA.hessian(ref[i]) for i in 1:n)
    @test any(!iszero, GTPSA.jacobian(ref))
  end

  @testset "csr_kick! pz element type" begin
    p = CSRParams(n_bin = 40, ds_step = 0.1)
    geom0 = CSRGeometry(L3, g3, c3)
    d = Descriptor(2, 2)
    x = @vars(d)
    errs = (() -> csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, electron, p),
            () -> csr_kick!(zeros(n), z, q .* (1 + x[2]), alive, geom0, 2, 0.17, 200.0, electron, p),
            () -> csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17, ForwardDiff.Dual{Nothing}(200.0, 1.0), electron, p))
    for f in errs
      @test_throws ErrorException f()
      @test occursin("cannot hold the kick", try f(); "" catch err; sprint(showerror, err) end)
    end
    @test csr_kick!(zeros(Float32, n), z, q, alive, geom0, 2, 0.17, 200.0, electron, p) isa CSRStepResult
    @test csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17, 200.0, electron, p) isa CSRStepResult
  end

  # Last: leaves GTPSA.desc_current at a descriptor no later code uses.
  @testset "Descriptor check" begin
    errmsg(f) = try f(); "" catch err; sprint(showerror, err) end
    p = CSRParams(n_bin = 40, ds_step = 0.1)
    geom0 = CSRGeometry(L3, g3, c3)
    d1 = Descriptor(2, 2)
    x = @vars(d1, dynamic = true)
    xs = @vars(d1)
    geom1 = CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)
    qx = q .* (1 + x[2])
    pzf, zf = [x[1] + 0.0], [0.0 + x[2]]
    csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17, 200.0, electron, p; test_particles = (pzf, zf, [true]))
    d2 = Descriptor(3, 2)
    cases = [
      ("q", () -> csr_kick_pz(pz0, z, qx, alive, geom0, 2, 0.17, 200.0, electron, p)),
      ("geom", () -> csr_kick_pz(pz0, z, q, alive, geom1, 2, 0.17, 200.0, electron, p)),
      ("s_kick", () -> csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, electron, p)),
      ("q", () -> csr_bin_particles(z, qx, alive, p)),
      ("g", () -> CSRGeometry(L3, [g3[1], 1.5 + x[1], g3[3]], c3)),
      ("g", () -> planar_survey(L3, [g3[1], 1.5 + x[1], g3[3]])),
      ("gamma", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + x[1])),
    ]
    for (name, f) in cases
      @test_throws ErrorException f()
      msg = errmsg(f)
      @test occursin("GTPSA's current Descriptor", msg) && occursin("`$name`", msg)
    end
    pzf2, zf2 = [x[1] + 0.0], [0.0 + x[2]]
    csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17, 200.0, electron, p; test_particles = (pzf2, zf2, [true]))
    @test scalar(pzf2[1]) == scalar(pzf[1])
    @test GTPSA.jacobian(pzf2) == GTPSA.jacobian(pzf)
    y = @vars(d2, dynamic = true)
    @test length(csr_kick_pz(pz0, z, q .* (1 + y[1]), alive, geom0, 2, 0.17, 200.0, electron, p)) == n
    @test !CSR.descriptor_mismatch(1.0)
    @test !CSR.descriptor_mismatch(zeros(3))
    @test !CSR.descriptor_mismatch(geom0)
    @test CSR.descriptor_mismatch(x[1])
    @test CSR.descriptor_mismatch(geom1)
    @test !CSR.descriptor_mismatch(y[1])
    @test !CSR.descriptor_mismatch(xs[1])
  end
end
