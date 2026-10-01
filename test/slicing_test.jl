# Phase 2 gate: with CSR off, tracking an element as n slices must equal one unsliced track!.

const SLICE_PQ = -60.0  # p_over_q_ref [T·m], electrons

function slice_test_coords()
  return [ 1e-3 -2e-4  5e-4  3e-4  1e-3  2e-3;
          -2e-3  1e-4 -1e-3 -2e-4 -5e-4 -1e-3;
           0.0   0.0   0.0   0.0   0.0   0.0;
           4e-3  1e-3 -3e-3  5e-4  2e-3 -4e-3]
end

slice_test_bunch(t_ref = 0.0) = Bunch(slice_test_coords(); species = Species("electron"), p_over_q_ref = SLICE_PQ, t_ref)

slice_test_line(line) = Beamline(line; p_over_q_ref = SLICE_PQ, species_ref = Species("electron"))

# Track element `i` of `line` whole, or as `n` slices.
function track_unsliced(line::Vector, i = 1; t_ref = 0.0)
  b = slice_test_bunch(t_ref)
  track!(b, slice_test_line(line).line[i])
  return b
end
track_unsliced(ele::LineElement) = track_unsliced([ele])

function track_sliced(line::Vector, n, i = 1; t_ref = 0.0)
  e = slice_test_line(line).line[i]
  L = float(e.L)
  b = slice_test_bunch(t_ref)
  for k in 1:n
    track!(b, CSR.slice_element(e, (k - 1) * L / n, L / n, k == 1, k == n); _p_over_q_ref = SLICE_PQ)
  end
  return b
end
track_sliced(ele::LineElement, n) = track_sliced([ele], n)

slice_maxdiff(b1, b2) = maximum(abs, b1.coords.v .- b2.coords.v)

function test_gate(b1, b2; tol = 1e-15)
  @test slice_maxdiff(b1, b2) <= tol
  @test b1.t_ref ≈ b2.t_ref rtol = 1e-15
  @test b1.coords.state == b2.coords.state
end

# s- and t-dependent normalized four-potential: As = -k(s)/2 (x² - y²), ϕ = e0 x cos(ω t).
function slice_test_potential(x, y, s, t, p = nothing)
  k0, k1, e0, w = 1.0, 2.0, 3e5, 2e9  # ϕ/c ~ 1e-3
  k = k0 * (1 + k1 * s)
  c, sn = cos(w * t), sin(w * t)
  return ((e0 * x * c, 0.0, 0.0, -k / 2 * (x^2 - y^2)),
          (e0 * c, 0.0, 0.0, -e0 * x * w * sn,
           0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0,
           -k * x, k * y, -k0 * k1 / 2 * (x^2 - y^2), 0.0))
end

@testset "Element slicing (CSR off)" begin
  sym8 = Symplectic(order = 4, n_steps = 8)
  bend_kw = (L = 0.5, g_ref = 0.4, Kn0 = 0.4, e1 = 0.1, e2 = -0.07)
  ext = Base.get_extension(CSR, :CSRBeamTrackingBeamlinesExt)

  @testset "Gate: sliced == unsliced, $name" for (name, ele) in (
      ("drift", Drift(L = 1.0)),
      ("sbend", SBend(; bend_kw...)),
      ("sbend Symplectic", SBend(; bend_kw..., tracking_method = sym8)),
      ("sbend Exact", SBend(; bend_kw..., tracking_method = Exact())),
      ("quadrupole Symplectic", Quadrupole(L = 0.3, Kn1 = 1.2, tracking_method = sym8)),
      ("quadrupole integrated", Quadrupole(L = 0.3, Kn1L = 0.36, tracking_method = sym8)),
      ("combined-function bend", SBend(; bend_kw..., Kn1 = -0.8, tracking_method = sym8)),
      ("combined-function integrated", SBend(; bend_kw..., Kn0L = 0.2, Kn1L = -0.4, tracking_method = sym8)),
    )
    test_gate(track_unsliced(ele), track_sliced(ele, 4))
  end

  @testset "One slice reproduces the element" begin
    ele = SBend(; bend_kw..., Kn1 = -0.8)
    @test slice_maxdiff(track_unsliced(ele), track_sliced(ele, 1)) == 0
  end

  @testset "SciBmadStandard: slicing adds integrator steps, converging to the reference" begin
    # Unsliced uses one order-4 step; 4 slices use four. Not a slicing error.
    mk(tm) = SBend(; bend_kw..., Kn1 = -0.8, tracking_method = tm)
    ref = track_unsliced(mk(Symplectic(order = 6, n_steps = 64)))
    err1 = slice_maxdiff(track_unsliced(mk(SciBmadStandard())), ref)
    err4 = slice_maxdiff(track_sliced(mk(SciBmadStandard()), 4), ref)
    @test err4 < err1 / 50
  end

  @testset "Edges stay on inner slices only if fringe_at is not reset (control)" begin
    # Without NoEnd on inner slices, the e1/e2 edge kicks would be applied 4 times.
    ele = SBend(; bend_kw..., tracking_method = Symplectic(order = 4, n_steps = 2))
    e = slice_test_line([ele]).line[1]
    b = slice_test_bunch()
    for k in 1:4
      s = CSR.slice_element(e, (k - 1) * 0.125, 0.125, true, true)  # every slice treated as both ends
      track!(b, s; _p_over_q_ref = SLICE_PQ)
    end
    @test slice_maxdiff(b, track_unsliced(ele)) > 1e-6
  end

  @testset "fringe_at and n_steps of slices" begin
    bl = slice_test_line([Quadrupole(L = 0.4, Kn1 = 1.0, tracking_method = Symplectic(n_steps = 8)),
                          Quadrupole(L = 0.4, Kn1 = 1.0, tracking_method = Symplectic(n_steps = 8, fringe_at = Fringe.ExitEnd)),
                          Quadrupole(L = 0.4, Kn1 = 1.0)])
    e, ex, es = bl.line
    @test CSR.slice_element(e, 0.0, 0.1, true, false).tracking_method.fringe_at == Fringe.EntranceEnd
    @test CSR.slice_element(e, 0.1, 0.1, false, false).tracking_method.fringe_at == Fringe.NoEnd
    @test CSR.slice_element(e, 0.3, 0.1, false, true).tracking_method.fringe_at == Fringe.ExitEnd
    @test CSR.slice_element(e, 0.0, 0.4, true, true).tracking_method.fringe_at == Fringe.BothEnds
    @test CSR.slice_element(ex, 0.0, 0.1, true, false).tracking_method.fringe_at == Fringe.NoEnd
    @test CSR.slice_element(ex, 0.3, 0.1, false, true).tracking_method.fringe_at == Fringe.ExitEnd
    @test CSR.slice_element(e, 0.0, 0.1, true, false).tracking_method.n_steps == 2
    @test CSR.slice_element(e, 0.0, 0.01, true, false).tracking_method.n_steps == 1
    tm = CSR.slice_element(es, 0.1, 0.1, false, false).tracking_method
    @test tm isa Symplectic && tm.order == 4 && tm.fringe_at == Fringe.NoEnd
    @test CSR.slice_element(Drift(L = 1.0), 0.0, 0.5, true, false).tracking_method.order == 2
    @test CSR.slice_element(es, 0.0, 0.1, true, false).L == 0.1
    @test CSR.slice_element(es, 0.0, 0.1, true, false).Kn1 == 1.0
  end

  @testset "Apertures only at element ends" begin
    ap = (x1_limit = -1.5e-3, x2_limit = 1.5e-3, y1_limit = -1.0, y2_limit = 1.0,
          aperture_shape = ApertureShape.Rectangular)
    for at in (ApertureAt.Entrance, ApertureAt.Exit, ApertureAt.BothEnds)
      ele = Quadrupole(; L = 0.3, Kn1 = 3.0, tracking_method = sym8, ap..., aperture_at = at)
      b1 = track_unsliced(ele)
      b2 = track_sliced(ele, 4)
      @test b1.coords.state == b2.coords.state
      @test count(!=(BeamTracking.STATE_ALIVE), b1.coords.state) > 0
    end
    ele = Quadrupole(; L = 0.3, Kn1 = 3.0, ap..., aperture_at = ApertureAt.BothEnds)
    @test isnothing(CSR.slice_element(ele, 0.1, 0.1, false, false).ApertureParams)
    @test CSR.slice_element(ele, 0.0, 0.1, true, false).aperture_at == ApertureAt.Entrance
    @test CSR.slice_element(ele, 0.2, 0.1, false, true).aperture_at == ApertureAt.Exit
    @test ele.aperture_at == ApertureAt.BothEnds  # original untouched
  end

  @testset "Electric dipole" begin
    # 1 MV/m horizontal field, integrated and non-integrated forms.
    for ele in (LineElement(L = 0.4, En0 = 1e6, tracking_method = sym8),
                LineElement(L = 0.4, En0L = 4e5, tracking_method = sym8),
                LineElement(L = 0.4, En0 = 1e6))
      b1 = track_unsliced(ele)
      test_gate(b1, track_sliced(ele, 4))
      # Field acts: differs from a drift of the same length.
      @test slice_maxdiff(b1, track_unsliced(Drift(L = 0.4))) > 1e-6
    end
    ele = LineElement(L = 0.4, En0L = 4e5)
    @test CSR.slice_element(ele, 0.0, 0.1, true, false).En0L ≈ 1e5
    @test CSR.slice_element(ele, 0.0, 0.1, true, false).tracking_method.fringe_at == Fringe.EntranceEnd
    @test CSR.slice_element(ele, 0.1, 0.1, false, false).tracking_method.fringe_at == Fringe.NoEnd
  end

  @testset "Alignment: slices compose to the misaligned element" begin
    al = (x_offset = 1e-3, y_offset = -2e-3, z_offset = 3e-3, x_rot = 2e-3, y_rot = -1.5e-3, tilt = 0.1)
    @testset "$name" for (name, ele) in (
        ("drift", Drift(; L = 1.0, al...)),
        ("quadrupole", Quadrupole(; L = 0.3, Kn1 = 1.2, tracking_method = sym8, al...)),
        ("quadrupole x_offset", Quadrupole(; L = 0.3, Kn1 = 1.2, tracking_method = sym8, x_offset = 1e-3)),
        ("sbend with edges", SBend(; bend_kw..., tracking_method = sym8, al...)),
        ("tilted combined-function bend", SBend(; L = 0.5, g_ref = 0.4, Kn0 = 0.4, Kn1 = -0.8, tilt_ref = 0.3,
                                                 tracking_method = sym8, al...)),
        ("sbend roll", SBend(; L = 0.5, g_ref = 0.4, Kn0 = 0.4, tracking_method = sym8, tilt = 0.2)),
        ("sbend SciBmadStandard", SBend(; bend_kw..., al...)),
      )
      test_gate(track_unsliced(ele), track_sliced(ele, 4))
    end
    # Control: the unchanged alignment on every slice is wrong.
    ele = SBend(; bend_kw..., tracking_method = sym8, al...)
    e = slice_test_line([ele]).line[1]
    b = slice_test_bunch()
    for k in 1:4
      s = CSR.slice_element(e, (k - 1) * 0.125, 0.125, k == 1, k == 4)
      s.AlignmentParams = deepcopy(e.AlignmentParams)
      track!(b, s; _p_over_q_ref = SLICE_PQ)
    end
    @test slice_maxdiff(b, track_unsliced(ele)) > 1e-5
    # Straight: rotation kept, offsets follow the slice center.
    q = Quadrupole(; L = 0.4, Kn1 = 1.0, x_rot = 1e-3)
    s = CSR.slice_element(q, 0.0, 0.1, true, false)
    @test s.x_rot == 1e-3
    @test s.x_offset ≈ -0.15 * BeamTracking.quat_rotate((0, 0, 1), BeamTracking.rot_quaternion(1e-3, 0, 0))[1] atol = 1e-18
  end

  @testset "Patch: transform on the first slice, drifts after" begin
    @testset "$name" for (name, ele) in (
        ("offsets and rotations", LineElement(L = 0.6, dx = 1e-3, dy = -2e-3, dz = 0.55, dx_rot = 2e-3, dy_rot = 0.05,
                                              dz_rot = 0.1, dt = 1e-12)),
        ("offsets only", LineElement(L = 0.6, dx = 1e-3, dy = -2e-3, dz = 0.55)),
        ("yaw only", LineElement(L = 0.6, dz = 0.6, dy_rot = 0.05)),
      )
      b1 = track_unsliced(ele)
      test_gate(b1, track_sliced(ele, 4); tol = 1e-16)
      @test slice_maxdiff(b1, slice_test_bunch()) > 1e-3
    end
    ele = LineElement(L = 0.6, dx = 1e-3, dz = 0.55, dy_rot = 0.05)
    s2 = CSR.slice_element(ele, 0.15, 0.15, false, false)
    @test (s2.dx, s2.dy, s2.dz, s2.dx_rot, s2.dy_rot, s2.dz_rot, s2.dt) == (0, 0, 0.15, 0, 0, 0, 0)
    s1 = CSR.slice_element(ele, 0.0, 0.15, true, false)
    @test s1.dy_rot == 0.05
    @test s1.dz ≈ 0.55 - 0.45 * cos(0.05)
  end

  @testset "Four-potential: field seen at element s and t" begin
    ele = LineElement(L = 0.5, four_potential = slice_test_potential, four_potential_normalized = true,
                      tracking_method = sym8)
    b1 = track_unsliced(ele)
    test_gate(b1, track_sliced(ele, 4))
    @test slice_maxdiff(b1, slice_test_bunch()) > 1e-4
    # Control: without the t shift the slices see the wrong RF phase.
    e = slice_test_line([ele]).line[1]
    b = slice_test_bunch()
    for k in 1:4
      s = CSR.slice_element(e, (k - 1) * 0.125, 0.125, k == 1, k == 4)
      f = s.four_potential
      s.four_potential = ext.ShiftedPotential(f.f, f.s0, 0.0)
      track!(b, s; _p_over_q_ref = SLICE_PQ)
    end
    @test slice_maxdiff(b, b1) > 1e-6
    @test_throws Exception CSR.slice_element(ele, 0.0, 0.125, true, false)  # detached: no reference energy
  end

  @testset "RF cavity" begin
    t0 = 1.3e-9
    rf_kw = (L = 0.4, voltage = 5e6, phi0 = 0.3)
    line = [RFCavity(; rf_kw..., rf_frequency = 1.3e9, tracking_method = sym8)]
    b1 = track_unsliced(line; t_ref = t0)
    test_gate(b1, track_sliced(line, 4; t_ref = t0))
    @test abs(b1.coords.v[1, 6] - slice_test_coords()[1, 6]) > 1e-4
    # Harmonic number: converted with the original beamline's circumference.
    line = [Drift(L = 9.6), RFCavity(; rf_kw..., harmon = 33, tracking_method = sym8)]
    test_gate(track_unsliced(line, 2; t_ref = t0), track_sliced(line, 4, 2; t_ref = t0))
    # SciBmadStandard (one order-2 step unsliced): agrees to integrator accuracy.
    line = [RFCavity(; rf_kw..., rf_frequency = 1.3e9)]
    @test slice_maxdiff(track_unsliced(line; t_ref = t0), track_sliced(line, 4; t_ref = t0)) < 1e-11
    e = slice_test_line(line).line[1]
    s = CSR.slice_element(e, 0.1, 0.1, false, false)
    @test s.voltage ≈ 1.25e6
    @test s.rf_frequency == 1.3e9
    @test s.tracking_method.order == 2
    e = slice_test_line([RFCavity(; rf_kw..., rf_frequency = 1.3e9, tracking_method = SaganCavity())]).line[1]
    @test_throws ErrorException CSR.slice_element(e, 0.0, 0.1, true, false)
  end

  @testset "Map elements: whole only, no CSR" begin
    shear(v, q, p) = ((v[1] + p * v[2], v[2], v[3], v[4], v[5], v[6]), q)
    ele = LineElement(L = 0.5, transport_map = shear, transport_map_params = 0.5)
    b1 = track_unsliced(ele)
    b2 = track_sliced(ele, 1)
    @test slice_maxdiff(b1, b2) == 0
    @test b1.t_ref == b2.t_ref
    @test b1.coords.v[1, 1] ≈ 1e-3 + 0.5 * -2e-4
    @test_throws ErrorException CSR.slice_element(ele, 0.0, 0.25, true, false)
    @test !CSR.csr_sliceable(ele)
    @test !CSR.csr_sliceable(Marker())
    @test !CSR.csr_sliceable(Quadrupole(L = 0.0, Kn1L = 0.1))
    @test CSR.csr_sliceable(Drift(L = 1.0))
    @test CSR.csr_sliceable(SBend(; bend_kw...))
  end

  @testset "Zero-length elements cannot be sliced" begin
    @test_throws ErrorException CSR.slice_element(Marker(), 0.0, 0.0, true, true)
  end
end
