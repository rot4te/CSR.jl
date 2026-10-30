using CSR: FloorPosition, planar_survey, chord_spline,
    spline_val, spline_d1, dspline_len, CSRGeometry,
    s_ref_to_s_chord, modulo2, CSRKickPoint, advance_floor, x_axis,
    z_axis

@testset "Geometry" begin
  @testset "planar_survey" begin
    # A bend with g > 0 curves towards -x (Bmad convention); theta
    # decreases by g*L
    f = planar_survey([1.0], [0.1])[1]
    @test f.r[1] < 0
    @test f.theta ≈ -0.1
    @test f.r[3] ≈ sin(0.1) / 0.1
    @test f.r[1] ≈ (cos(0.1) - 1) / 0.1

    # Four quarter circles close the ring
    rho = 3.0
    fs = planar_survey(fill(rho * pi / 2, 4), fill(1 / rho, 4))
    @test isapprox(fs[end].r, zeros(3); atol = 1e-12)
    @test fs[end].theta ≈ -2pi

    # Drift then bend from a rotated start
    start = FloorPosition([1.0, 0.0, 2.0], 0.3)
    fs = planar_survey([2.0, 0.0], [0.0, 0.0]; start)
    @test fs[1].r ≈ start.r + 2 * [sin(0.3), 0, cos(0.3)]
    @test fs[2].r ≈ fs[1].r
  end

  @testset "modulo2" begin
    @test modulo2(0.1, pi / 2) ≈ 0.1
    @test modulo2(pi - 0.1, pi / 2) ≈ -0.1
    @test modulo2(-pi / 2, pi / 2) ≈ -pi / 2
  end

  @testset "chord_spline" begin
    c = chord_spline(2.0, 0.01, -0.03)
    @test abs(spline_val(c, 0.0)) < 1e-15
    @test abs(spline_val(c, 2.0)) < 1e-15
    @test spline_d1(c, 0.0) ≈ 0.01
    @test spline_d1(c, 2.0) ≈ -0.03
  end

  @testset "dspline_len vs quadrature" begin
    for (c, s0, s1, dth) in ((SVector(0.0, 0.01, 0.02, 0.03), 0.3,
                              1.4, 1e-3),
                             (SVector(0.0, -0.02, 0.005, -0.01), 0.0,
                              2.0, 0.0),
                             (SVector(0.0, 0.0, 0.0, 0.01), 0.5, 1.5,
                              -2e-3))
      quad = simpson(s -> (c[2] + 2c[3] * s + 3c[4] * s^2 - dth)^2 /
                     2, s0, s1, 2000)
      @test dspline_len(s0, s1, c, dth) ≈ quad rtol = 1e-12
      # Bmad form differs only in the s⁴ term: 9c₃² instead of
      # 9c₃²/5
      P4 = 5s0^4 + 10s0^3 * (s1 - s0) + 10s0^2 * (s1 - s0)^2 + 5s0 *
          (s1 - s0)^3 + (s1 - s0)^4
      @test dspline_len(s0, s1, c, dth;
                        bmad_compat = true) - dspline_len(s0, s1, c,
                            dth) ≈ (s1 - s0) / 2 * P4 * 9c[4]^2 *
                                4 / 5
    end
    # Identical when c3 = 0
    c = SVector(0.0, 0.01, -0.02, 0.0)
    @test dspline_len(0.0, 1.0, c, 0.0;
                      bmad_compat = true) == dspline_len(0.0, 1.0, c,
                          0.0)
  end

  @testset "path_excess vs quadrature" begin
    # Exact integrand √(1 + x′²) - cos dθ - x′ sin dθ, small
    # slopes (as in an element) and large ones. The quadrature
    # reference loses ~1e-11 to cancellation in √(1 + x′²) - 1
    # for small slopes.
    for (c, s0, s1, rtol) in ((SVector(0.0, 0.01, 0.02, 0.03), 0.0,
                               0.1, 1e-10),
                              (SVector(0.0, 0.05, -0.3, 0.2), 0.1,
                               0.4, 1e-7)), dth in (0.0, 0.01, 0.3,
                                   -0.8)
      xp(s) = CSR.spline_d1(c, s)
      quad = simpson(s -> sqrt(1 + xp(s)^2) - cos(dth) - xp(s) *
                     sin(dth), s0, s1, 2000)
      @test CSR.path_excess(s0, s1, c,
                                   dth) ≈ quad rtol = rtol
      @test CSR.path_excess(s1, s0, c,
                                   dth) ≈ -CSR.path_excess(
                                       s0, s1, c, dth) rtol = 1e-14
      @test CSR.path_excess(s0, s1, c, dth;
                                   bmad_compat = true) ==
                                       dspline_len(s0, s1, c, dth;
                                           bmad_compat = true)
    end
    # Small angles: agrees with the small-angle form
    c = SVector(0.0, 1e-4, 2e-4, -1e-4)
    @test CSR.path_excess(0.0, 0.1, c,
                                 1e-4) ≈ dspline_len(0.0, 0.1, c,
                                     1e-4) rtol = 1e-6
  end

  @testset "CSRGeometry, on-axis bend" begin
    rho = 5.0; L = 0.5; phi = L / rho
    geom = CSRGeometry([1.0, L], [0.0, 1 / rho], zeros(3, 6))
    e = geom.ele[2]
    @test e.floor0.r ≈ e.ref_floor0.r
    @test e.floor1.r ≈ e.ref_floor1.r
    @test e.L_chord ≈ 2rho * sin(phi / 2)
    @test e.theta_chord ≈ -phi / 2
    # entrance slope relative to the chord
    @test e.spline[2] ≈ phi / 2
    @test abs(e.spline[4]) < 1e-14     # symmetric: c3 = 0
    @test e.dL_s ≈ rho * phi^3 / 24 rtol = 1e-3
    @test e.s_start == 1.0
    @test geom.ele[1].dL_s == 0
    @test s_ref_to_s_chord(0.0, e) == 0
    @test s_ref_to_s_chord(0.3, geom.ele[1]) ≈ 0.3
  end

  @testset "s_ref_to_s_chord in a bend (V3)" begin
    rho = 5.0; L = 0.5; phi = L / rho
    geom = CSRGeometry([1.0, L], [0.0, 1 / rho], zeros(3, 6))
    gc = CSRGeometry([1.0, L], [0.0, 1 / rho], zeros(3, 6);
                     bmad_compat = true)
    e = geom.ele[2]
    # On axis: the arc point at s projected on the chord, to the
    # accuracy of the cubic chord spline (~ρφ⁴/384 = 1.3e-6 m
    # here); Bmad's ρ atan(s/ρ) is off by up to 1.4e-3 m
    chord(s) = rho * (sin(s / rho - phi / 2) + sin(phi / 2))
    for s in (0.01, 0.1, 0.25, 0.4, 0.49)
      @test s_ref_to_s_chord(s, e) ≈ chord(s) atol = 1e-7
      @test s_ref_to_s_chord(s, e;
                             bmad_compat = true) ≈ rho * atan(s /
                                 rho) rtol = 1e-14
      @test CSRKickPoint(geom, 2, s,
                         100.0).floor_k.r ≈ advance_floor(
                             e.ref_floor0, s, 1 / rho).r atol = 3e-6
      @test CSRKickPoint(gc, 2, s,
                         100.0).s_chord_kick ≈ rho * atan(s /
                             rho) rtol = 1e-14
    end
    @test s_ref_to_s_chord(L, e) == e.L_chord
    @test s_ref_to_s_chord(0.0, e) == 0
    # Derivatives along s (first and second order, nested Duals) vs
    # central differences
    f(s) = s_ref_to_s_chord(s, e)
    df(s) = ForwardDiff.derivative(f, s)
    h = 1e-5
    for s in (0.0, 0.3)
      @test df(s) ≈ (f(s + h) - f(s - h)) / 2h rtol = 1e-9
      @test ForwardDiff.derivative(df,
                                   s) ≈ (df(s + h) - df(s - h)) /
                                       2h rtol = 1e-6
    end
    @test df(0.3) ≈ cos(0.3 / rho - phi / 2) atol = 1e-6

    # Off-axis centroid: the kick point is in the reference normal
    # plane, at the centroid offset (≈ x)
    centroid = zeros(3, 6)
    centroid[2, 1] = 2e-3; centroid[2, 2] = 1e-3
    centroid[3, 1] = -1e-3; centroid[3, 2] = -2e-3
    geom = CSRGeometry([1.0, L], [0.0, 1 / rho], centroid)
    e = geom.ele[2]
    for s in (0.1, 0.25, 0.4)
      ref = advance_floor(e.ref_floor0, s, 1 / rho)
      dr = CSRKickPoint(geom, 2, s, 100.0).floor_k.r - ref.r
      @test abs(dr' * z_axis(ref.theta)) < 1e-15
      @test -1e-3 < dr' * x_axis(ref.theta) < 2e-3
    end
    @test s_ref_to_s_chord(L, e) == e.L_chord
    # Derivative w.r.t. g: ForwardDiff vs central differences
    sc(g) = s_ref_to_s_chord(0.3, CSRGeometry([1.0, L], [0.0, g],
        centroid).ele[2])
    h = 1e-6
    @test ForwardDiff.derivative(sc,
                                 1 / rho) ≈ (sc(1 / rho + h) -
                                     sc(1 / rho - h)) / 2h rtol = 1e-6
  end

  @testset "CSRGeometry, off-axis centroid" begin
    # Centroid offset changes across a drift: chord is tilted, spline
    # has c3 ≠ 0 when end angles differ
    centroid = zeros(3, 6)
    centroid[2, 1] = 1e-3; centroid[2, 2] = 2e-3
    centroid[3, 1] = 1e-3; centroid[3, 2] = -1e-3
    geom = CSRGeometry([1.0, 1.0], [0.0, 0.0], centroid)
    e = geom.ele[2]
    @test e.floor0.r[1] ≈ 1e-3
    @test e.floor0.theta ≈ asin(2e-3 / (1 + 1e-14))
    @test abs(e.spline[4]) > 0
    geom_c = CSRGeometry([1.0, 1.0], [0.0, 0.0], centroid;
                         bmad_compat = true)
    @test geom_c.ele[2].dL_s > e.dL_s
  end
end
