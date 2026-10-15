# Phase 6 step 3: forward-mode derivatives of one CSR kick with respect to the geometry (element L, g, centroid),
# s_kick and gamma (ForwardDiff Duals through the kick tables; source roots by implicit differentiation).
# Checked against central finite differences. The kick tables have finite-difference noise from the source-point
# tolerance (zbrent abs_tol 1e-8) and, with images, from the adaptive hat quadrature and the kernel's structure
# near lag 0 (width ~y/γ): each comparison takes the best of several step sizes.

@testset "Derivatives w.r.t. geometry, s_kick, gamma" begin
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  pz0 = zero(z)
  unit(j, m) = (1:m) .== j
  # Best agreement of central differences (steps h * scale) with the derivative d
  function fd_best(F, x0, d; scale = max(abs(x0), 1e-2), steps = (1e-5, 1e-6, 1e-7, 1e-8))
    return minimum(maxrel((F(x0 + h * scale) .- F(x0 - h * scale)) ./ (2h * scale), d) for h in steps)
  end

  @testset "advance_floor at g = 0" begin
    gd = ForwardDiff.Dual(0.0, 1.0)
    f = CSR.advance_floor(FloorPosition(), 0.3, gd)
    @test ForwardDiff.value.(f.r) == CSR.advance_floor(FloorPosition(), 0.3, 0.0).r
    # x = (cos(gL) - 1)/g → dx/dg = -L²/2, z = sin(gL)/g → dz/dg = 0, θ = -gL
    @test ForwardDiff.partials(f.r[1], 1) ≈ -0.3^2 / 2
    @test ForwardDiff.partials(f.r[3], 1) == 0
    @test ForwardDiff.partials(f.theta, 1) == -0.3
  end

  @testset "modulo2 at a fold boundary" begin
    # ForwardDiff's mod gives NaN partials at exactly 0 (an on-axis straight element, a source on an element boundary)
    @test CSR.modulo2(ForwardDiff.Dual(0.0, 1.0), pi / 2) == ForwardDiff.Dual(0.0, 1.0)
    @test CSR.modulo2(ForwardDiff.Dual(4.0, 2.0), pi / 2) == ForwardDiff.Dual(CSR.modulo2(4.0, pi / 2), 2.0)
    J = ForwardDiff.jacobian(Lv -> CSR.geometry_to_vector(CSRGeometry(Lv, [0.0, 0.5], zeros(3, 6))), [0.2, 0.3])
    @test !any(isnan, J)
  end

  @testset "Source root: residual stationary (bmad_compat = $bc)" for bc in (false, true)
    G = CSRGeometry(L3, ForwardDiff.Dual.(g3, 1.0), c3; bmad_compat = bc)
    kp = CSRKickPoint(G, 2, 0.17, ForwardDiff.Dual(200.0, 1.0))
    for dz in (2e-5, 3e-4, 1e-3)
      k = CSR.CSRKick1(G, kp; ix_ele_source = 2, dz_particles = dz)
      k, s = CSR.s_source_calc(k, G, kp)
      @test ForwardDiff.partials(s, 1) != 0
      # ddz at the Dual root: zero derivative (implicit function theorem); at the primal root: not
      k2 = CSR.CSRKick1(G, kp; ix_ele_source = k.ix_ele_source, dz_particles = dz)
      _, r_dual = CSR.ddz_calc_csr(k2, s, G, kp)
      _, r_fixed = CSR.ddz_calc_csr(k2, ForwardDiff.value(s), G, kp)
      @test abs(ForwardDiff.partials(r_dual, 1)) < 1e-10 * abs(ForwardDiff.partials(r_fixed, 1))
    end
  end

  # Dual tables search their sources in primal values (PrimalSearch) and evaluate them once in Duals: the same work
  # struct as searching in Duals, for sources in the lattice, ahead of the kick point and on both infinite drifts
  @testset "Primal search == search in Duals (bmad_compat = $bc)" for bc in (false, true)
    G = CSRGeometry(L3, ForwardDiff.Dual.(g3, 1.0, 0.5), c3; bmad_compat = bc)
    for (ix, y) in ((1, 0.0), (2, 0.0), (2, 0.01), (3, 0.01))
      kp = CSRKickPoint(G, ix, 0.17, ForwardDiff.Dual(200.0, 1.0, 0.0); y_source = y)
      search = CSR.primal_search(G, kp, CSR.number_type(kp))
      @test search isa CSR.PrimalSearch
      for dz in (-0.5, -1e-3, -2e-5, 2e-5, 3e-4, 1e-3, 0.5)
        k = CSR.CSRKick1(G, kp; dz_particles = dz)
        @test CSR.s_source_calc(k, G, kp, search) == CSR.s_source_calc(k, G, kp)
      end
    end
    @test CSR.primal_search(CSR.primal(G), CSR.primal(CSRKickPoint(G, 2, 0.17, 200.0)), Float64) === nothing
  end

  @testset "Hat-integrated image table" begin
    dz = csr_bin_particles(z, q, alive, CSRParams(n_bin = 40)).dz
    function tab(gv, cv)
      G = CSRGeometry(L3, gv, reshape(cv, 4, 6))
      kp = CSRKickPoint(G, 2, 0.17, 200.0; y_source = 0.01, kick_factor = -1.0)
      return first(CSR.csr_image_table(G, kp, dz, 40, CSR.CPU(); bin_integrate = true, rtol = 1e-10))
    end
    c = vec(c3)
    Jg = ForwardDiff.jacobian(gv -> tab(gv, c), g3)
    Jc = ForwardDiff.jacobian(cv -> tab(g3, cv), c)
    steps = (1e-6, 1e-7, 1e-8, 1e-9)
    for j in 1:2
      @test fd_best(t -> tab(g3 .+ t .* unit(j, 3), c), 0.0, Jg[:, j]; scale = 1, steps) < 2e-5
    end
    for j in (1, 2, 3, 5, 6, 7)
      @test fd_best(t -> tab(g3, c .+ t .* unit(j, 24)), 0.0, Jc[:, j]; scale = 1, steps) < 2e-5
    end
  end

  cases = (
    ("free space", 0, false, true, 2, 0.17, 1e-5),
    ("free space, bmad_compat", 0, true, false, 2, 0.17, 1e-5),
    # Default-mode images: the shielded kick's derivatives are 7-25x smaller than the free-space part's, which
    # amplifies the hat quadrature's finite-difference noise (table level: below, ~1e-5)
    ("images", 2, false, true, 2, 0.17, 1e-3),
    ("images, bmad_compat", 2, true, false, 2, 0.17, 1e-4),
    ("kick at element entrance, images", 2, false, false, 3, 0.0, 1e-4),
  )
  @testset "$name" for (name, n_img, bc, abq, ix, sk, tol) in cases
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                  bmad_compat = bc, all_bin_quadrature = abq)
    c = vec(c3)
    f(Lv, gv, cv, s, γ) = csr_kick_pz(pz0, z, q, alive, CSRGeometry(Lv, gv, reshape(cv, 4, 6); bmad_compat = bc),
                                      ix, s, γ, electron, p; kick_factor = 0.5)
    y = f(L3, g3, c, sk, 200.0)

    # Values unchanged by the Duals
    Jc = ForwardDiff.jacobian(cv -> f(L3, g3, cv, sk, 200.0), c)
    yd = f(L3, g3, ForwardDiff.Dual.(c, 1.0), sk, 200.0)
    @test ForwardDiff.value.(yd) == y

    dγ = ForwardDiff.derivative(γ -> f(L3, g3, c, sk, γ), 200.0)
    @test maximum(abs, dγ) > 0
    @test fd_best(γ -> f(L3, g3, c, sk, γ), 200.0, dγ) < tol
    # s_kick away from the element entrance (a branch point: the first-bin integral switches element there)
    ds = ForwardDiff.derivative(s -> f(L3, g3, c, s, 200.0), sk + 0.01)
    @test fd_best(s -> f(L3, g3, c, s, 200.0), sk + 0.01, ds) < tol

    # Centroid x and px at the first three rows (column-major index in vec(centroid))
    for j in (1, 2, 3, 5, 6, 7)
      @test fd_best(t -> f(L3, g3, c .+ t .* unit(j, 24), sk, 200.0), 0.0, Jc[:, j]; scale = 1e-2) < tol
    end
    # pz enters only through the angle asin(px / (1 + 1e-14 + pz)): exact relation between the columns
    for r in 1:3
      @test maxrel(Jc[:, 20+r], Jc[:, 4+r] .* (-c3[r, 2] / (1 + 1e-14 + c3[r, 6]))) < 1e-8
    end
    # Element 4 exit (row 4) does not affect a kick in free space without downstream sources
    n_img == 0 && @test all(iszero, Jc[:, 4])

    Jg = ForwardDiff.jacobian(gv -> f(L3, gv, c, sk, 200.0), g3)
    for j in 1:2
      @test fd_best(t -> f(L3, g3 .+ t .* unit(j, 3), c, sk, 200.0), 0.0, Jg[:, j]; scale = 1e-2) < tol
    end
    JL = ForwardDiff.jacobian(Lv -> f(Lv, g3, c, sk, 200.0), L3)
    @test fd_best(t -> f(L3 .+ t .* unit(2, 3), g3, c, sk, 200.0), 0.0, JL[:, 2]; scale = 1e-2) < tol
  end
end
