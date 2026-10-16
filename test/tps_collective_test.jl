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
    @test maximum(abs, Ht .- Hf) <= 1e-11 * maximum(abs, Hf)
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
    @test maximum(abs, T3t .- T3f) <= 1e-10 * maximum(abs, T3f)
  end
end
