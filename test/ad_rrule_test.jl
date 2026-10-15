# Phase 6 step 3b: rrule of csr_kick_pz with respect to the geometry, s_kick and gamma (ForwardDiff through the kick
# tables), and rrules of the CSRGeometry constructors. Checked against ForwardDiff directly (exact up to round-off)
# and with ChainRulesTestUtils (finite differences; tolerances set by the source-point tolerance, zbrent abs_tol 1e-8).

using ChainRulesTestUtils
using ChainRulesTestUtils: FiniteDifferences

# Finite differences perturb the numbers of a CSRGeometry, not its bmad_compat flag
function FiniteDifferences.to_vec(g::CSRGeometry)
  n, bc = length(g.ele), g.bmad_compat
  return CSR.geometry_to_vector(g), v -> CSR.geometry_from_vector(v, n, bc)
end
ChainRulesTestUtils.test_approx(a::CSRGeometry, b::CSRGeometry, msg = ""; kw...) =
  ChainRulesTestUtils.test_approx(CSR.geometry_to_vector(a), CSR.geometry_to_vector(b), msg; kw...)

@testset "rrule w.r.t. geometry, s_kick, gamma" begin
  CE = Base.get_extension(CSR, :CSRChainRulesCoreExt)
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]

  @testset "geometry_to_vector round trip" begin
    geom = CSRGeometry(L3, g3, c3)
    v = CSR.geometry_to_vector(geom)
    @test length(v) == 4 + 26 * 3
    g2 = CSR.geometry_from_vector(v, 3, false)
    @test CSR.geometry_to_vector(g2) == v
    @test g2.start == geom.start && g2.ele == geom.ele
    @test CE.geometry_vector(CE.geometry_tangent(geom, v), 3) == v
  end

  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  pz = 1e-4 .* sin.(1:n)
  kw = (kick_factor = 0.5, pz_scale = 1.3)
  cases = (
    ("free space", 0, false, true),
    ("images", 2, false, true),
    ("images, bmad_compat", 2, true, false),
  )
  @testset "vs ForwardDiff: $name" for (name, n_img, bc, abq) in cases
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                  bmad_compat = bc, all_bin_quadrature = abq)
    geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
    f(gm, s, γ) = csr_kick_pz(pz, z, q, alive, gm, 2, s, γ, electron, p; kw...)
    ybar = cos.(1:n)
    y, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, 2, 0.17, 200.0, electron, p; kw...)
    @test y == f(geom, 0.17, 200.0)
    t = back(ybar)
    @test t[6] isa Tangent{typeof(geom)}
    v0 = CSR.geometry_to_vector(geom)
    ref = ForwardDiff.gradient(v -> sum(ybar .* f(CSR.geometry_from_vector(v, 3, bc), 0.17, 200.0)), v0)
    @test maxrel(CE.geometry_vector(t[6], 3), ref) < 1e-12
    @test t[8] ≈ ForwardDiff.derivative(s -> sum(ybar .* f(geom, s, 200.0)), 0.17) rtol = 1e-12
    @test t[9] ≈ ForwardDiff.derivative(γ -> sum(ybar .* f(geom, 0.17, γ)), 200.0) rtol = 1e-12

    # Composed with the CSRGeometry rrule: cotangents of L, g, centroid. The L cotangent is a small difference of
    # large terms (an upstream length change moves the downstream geometry rigidly).
    _, gback = ChainRulesCore.rrule(CSRGeometry, L3, g3, c3; bmad_compat = bc)
    tg = gback(t[6])
    F(Lv, gv, cm) = sum(ybar .* f(CSRGeometry(Lv, gv, cm; bmad_compat = bc), 0.17, 200.0))
    @test maxrel(tg[2], ForwardDiff.gradient(Lv -> F(Lv, g3, c3), L3)) < 1e-6
    @test maxrel(tg[3], ForwardDiff.gradient(gv -> F(L3, gv, c3), g3)) < 1e-9
    @test maxrel(vec(tg[4]), ForwardDiff.gradient(cv -> F(L3, g3, reshape(cv, 4, 6)), vec(c3))) < 1e-9

    # Same through the CSRGeometry method with reference exit floor positions
    ref_exit = planar_survey(L3, g3)
    _, gback4 = ChainRulesCore.rrule(CSRGeometry, L3, g3, ref_exit, c3; bmad_compat = bc)
    tg4 = gback4(t[6])
    @test maxrel(tg4[5], tg[4]) < 1e-9
    G4(fl) = sum(ybar .* f(CSRGeometry(L3, g3, fl, c3; bmad_compat = bc), 0.17, 200.0))
    x0 = reduce(vcat, [[f.r; f.theta] for f in ref_exit])
    tofl(x) = [FloorPosition(SVector(x[4i-3], x[4i-2], x[4i-1]), x[4i]) for i in 1:3]
    rx = ForwardDiff.gradient(x -> G4(tofl(x)), x0)
    @test maxrel(reduce(vcat, [[ft.r; ft.theta] for ft in tg4[4]]), rx) < 1e-9
  end

  # Per-evaluation local derivatives (csr_kick_param_adjoint) == ForwardDiff through csr_bins_kick with every geometry
  # number seeded, in cases that exercise each source branch and table type
  @testset "Parameter adjoint == ForwardDiff: $name" for (name, Lv, gv, ix, sk, n_img, bc, abq) in (
      ("mid-lattice kick, long lattice", fill(0.1, 20), fill(0.1, 20), 15, 0.05, 0, false, true),
      ("trapezoid bins", fill(0.1, 20), fill(0.1, 20), 15, 0.05, 0, false, false),
      ("kick at element entrance", fill(0.1, 20), fill(0.1, 20), 15, 0.0, 0, false, true),
      ("sources on the upstream drift", [0.05, 0.1], [0.0, 1.0], 2, 0.03, 0, false, true),
      ("images, sources past the lattice end", [0.1, 0.1, 0.05], [0.0, 1.0, 2.0], 3, 0.03, 2, false, true),
      ("images, mid-lattice", fill(0.1, 8), fill(0.5, 8), 5, 0.05, 2, false, true),
      ("free space, bmad_compat", fill(0.1, 20), fill(0.1, 20), 15, 0.05, 0, true, false),
      ("images, bmad_compat", [0.1, 0.1, 0.05], [0.0, 1.0, 2.0], 3, 0.03, 2, true, false))
    n_ele = length(Lv)
    c = zeros(n_ele + 1, 6)
    c[:, 1] .= 1e-4 .* sin.(1:n_ele+1)
    c[:, 2] .= 1e-3 .* cos.(1:n_ele+1)
    geom = CSRGeometry(Lv, gv, c; bmad_compat = bc)
    p = CSRParams(n_bin = 40, ds_step = 0.05, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                  bmad_compat = bc, all_bin_quadrature = abq)
    zz, qq, aa = kernel_test_bunch(3e-3, 1e-9, 400)
    st = CSR.csr_binned_kick(zz, qq, aa, geom, ix, sk, 100.0, electron, p; kick_factor = 0.5)
    kbar = cos.(1:40)
    g, s, γ = CSR.csr_kick_param_adjoint(kbar, st.bins, geom, ix, sk, 100.0, electron, p; kick_factor = 0.5)
    n_g = 4 + 26n_ele
    phi(x) = sum(kbar .* CSR.csr_bins_kick(st.bins, CSR.CPU(), CSR.geometry_from_vector(view(x, 1:n_g), n_ele, bc), ix,
                                           x[n_g+1], x[n_g+2], electron, p; kick_factor = 0.5).kick)
    ref = ForwardDiff.gradient(phi, [CSR.geometry_to_vector(geom); sk; 100.0])
    # Relative to the largest component: the s_kick derivative can be much smaller than the geometry's (cancellation)
    scale = maximum(abs, ref)
    @test maximum(abs, g .- ref[1:n_g]) <= 1e-10 * scale
    @test abs(s - ref[n_g+1]) <= 1e-10 * scale
    @test abs(γ - ref[n_g+2]) <= 1e-10 * scale
  end

  @testset "Zero cotangents" begin
    p = CSRParams(n_bin = 40, ds_step = 0.1)
    geom = CSRGeometry(L3, g3, c3)
    _, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, 2, 0.17, 200.0, electron, p)
    @test all(x -> x isa AbstractZero, back(ZeroTangent()))
    _, gback = ChainRulesCore.rrule(CSRGeometry, L3, g3, c3)
    @test all(x -> x isa AbstractZero, gback(ZeroTangent()))
  end

  # ChainRulesTestUtils (finite differences). Small bunch to keep the number of FD directions down. z is held fixed
  # (wrapper, rrule via `rrule_f`): the bin grid moves with the extreme particles; ad_test.jl checks z by hand.
  # Steps capped at 1e-6: the kick curves strongly in some geometry fields (FD at h = 1e-4 off by ~10% in start.r[3]
  # and L_chord of element 1, at h ≤ 1e-6 converged to the AD value). At such steps the source-root noise (zbrent
  # abs_tol 1e-8) limits FD to ~1e-3 relative, so rtol = 1e-2 here: these tests check the tangent structure and the
  # wiring; the accuracy checks are rrule == ForwardDiff above (1e-12) and ForwardDiff vs FD in ad_geometry_test.jl.
  # γ enters as γ/Γ: the step cap is absolute, and a step of 1e-6 in γ = 200 is lost in the source-root noise (FD
  # error of ∂kick/∂γ 4e-3 with images in compat, so ȳ⋅∂kick/∂γ failed rtol 1e-2 for ~23% of random ȳ; with Γ = 1e5,
  # 1.6e-6 and ~1e-4 of ȳ).
  fdm = FiniteDifferences.central_fdm(5, 1; max_range = 1e-6)
  Γ = 1e5
  @testset "test_rrule: $name" for (name, n_img, bc, abq) in (("free space", 0, false, true),
                                                              ("images, bmad_compat", 2, true, false))
    zs, qs, as = kernel_test_bunch(1e-3, 1e-9, 60)
    p = CSRParams(n_bin = 24, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                  bmad_compat = bc, all_bin_quadrature = abq)
    geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
    h(pz, q, gm, s, w) = csr_kick_pz(pz, zs, q, as, gm, 2, s, Γ * w, electron, p; kw...)
    function h_rrule(::RuleConfig, _, pz, q, gm, s, w)
      y, back = ChainRulesCore.rrule(csr_kick_pz, pz, zs, q, as, gm, 2, s, Γ * w, electron, p; kw...)
      h_pullback(ȳ) = (t = back(ȳ); (NoTangent(), t[2], t[4], t[6], t[8], Γ * t[9]))
      return y, h_pullback
    end
    test_rrule(h ⊢ NoTangent(), 1e-4 .* sin.(1:60), qs, geom, 0.17, 200.0 / Γ;
               rrule_f = h_rrule, fdm, check_inferred = false, rtol = 1e-2, atol = 1e-10)
    test_rrule(CSRGeometry, L3, g3, c3; fkwargs = (; bmad_compat = bc), fdm, check_inferred = false, rtol = 1e-7)
  end
end
