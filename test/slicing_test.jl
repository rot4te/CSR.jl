# Phase 2 gate: with CSR off, tracking an element as n slices must equal one unsliced track!.

const SLICE_PQ = -60.0  # p_over_q_ref [T·m], electrons

function slice_test_coords()
  return [ 1e-3 -2e-4  5e-4  3e-4  1e-3  2e-3;
          -2e-3  1e-4 -1e-3 -2e-4 -5e-4 -1e-3;
           0.0   0.0   0.0   0.0   0.0   0.0;
           4e-3  1e-3 -3e-3  5e-4  2e-3 -4e-3]
end

slice_test_bunch() = Bunch(slice_test_coords(); species = Species("electron"), p_over_q_ref = SLICE_PQ)

function track_unsliced(ele)
  bl = Beamline([ele]; p_over_q_ref = SLICE_PQ, species_ref = Species("electron"))
  b = slice_test_bunch()
  track!(b, bl)
  return b
end

function track_sliced(ele, n)
  bl = Beamline([ele]; p_over_q_ref = SLICE_PQ, species_ref = Species("electron"))
  e = bl.line[1]
  L = float(e.L)
  b = slice_test_bunch()
  for k in 1:n
    track!(b, CSR.slice_element(e, L / n, k == 1, k == n); _p_over_q_ref = SLICE_PQ)
  end
  return b
end

slice_maxdiff(b1, b2) = maximum(abs, b1.coords.v .- b2.coords.v)

@testset "Element slicing (CSR off)" begin
  sym8 = Symplectic(order = 4, n_steps = 8)
  bend_kw = (L = 0.5, g_ref = 0.4, Kn0 = 0.4, e1 = 0.1, e2 = -0.07)

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
    b1 = track_unsliced(ele)
    b2 = track_sliced(ele, 4)
    @test slice_maxdiff(b1, b2) <= 1e-15
    @test b1.t_ref ≈ b2.t_ref rtol = 1e-15
    @test b1.coords.state == b2.coords.state
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
    bl = Beamline([ele]; p_over_q_ref = SLICE_PQ, species_ref = Species("electron"))
    e = bl.line[1]
    b = slice_test_bunch()
    for k in 1:4
      s = CSR.slice_element(e, 0.125, true, true)  # every slice treated as both ends
      track!(b, s; _p_over_q_ref = SLICE_PQ)
    end
    @test slice_maxdiff(b, track_unsliced(ele)) > 1e-6
  end

  @testset "fringe_at and n_steps of slices" begin
    bl = Beamline([Quadrupole(L = 0.4, Kn1 = 1.0, tracking_method = Symplectic(n_steps = 8)),
                   Quadrupole(L = 0.4, Kn1 = 1.0, tracking_method = Symplectic(n_steps = 8, fringe_at = Fringe.ExitEnd)),
                   Quadrupole(L = 0.4, Kn1 = 1.0)];
                  p_over_q_ref = SLICE_PQ, species_ref = Species("electron"))
    e, ex, es = bl.line
    @test CSR.slice_element(e, 0.1, true, false).tracking_method.fringe_at == Fringe.EntranceEnd
    @test CSR.slice_element(e, 0.1, false, false).tracking_method.fringe_at == Fringe.NoEnd
    @test CSR.slice_element(e, 0.1, false, true).tracking_method.fringe_at == Fringe.ExitEnd
    @test CSR.slice_element(e, 0.4, true, true).tracking_method.fringe_at == Fringe.BothEnds
    @test CSR.slice_element(ex, 0.1, true, false).tracking_method.fringe_at == Fringe.NoEnd
    @test CSR.slice_element(ex, 0.1, false, true).tracking_method.fringe_at == Fringe.ExitEnd
    @test CSR.slice_element(e, 0.1, true, false).tracking_method.n_steps == 2
    @test CSR.slice_element(e, 0.01, true, false).tracking_method.n_steps == 1
    tm = CSR.slice_element(es, 0.1, false, false).tracking_method
    @test tm isa Symplectic && tm.order == 4 && tm.fringe_at == Fringe.NoEnd
    @test CSR.slice_element(Drift(L = 1.0), 0.5, true, false).tracking_method.order == 2
    @test CSR.slice_element(es, 0.1, true, false).L == 0.1
    @test CSR.slice_element(es, 0.1, true, false).Kn1 == 1.0
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
    @test isnothing(CSR.slice_element(ele, 0.1, false, false).ApertureParams)
    @test CSR.slice_element(ele, 0.1, true, false).aperture_at == ApertureAt.Entrance
    @test CSR.slice_element(ele, 0.1, false, true).aperture_at == ApertureAt.Exit
    @test ele.aperture_at == ApertureAt.BothEnds  # original untouched
  end

  @testset "Electric dipole" begin
    # 1 MV/m horizontal field, integrated and non-integrated forms.
    for (name, ele) in (("En0", LineElement(L = 0.4, En0 = 1e6, tracking_method = sym8)),
                        ("En0L", LineElement(L = 0.4, En0L = 4e5, tracking_method = sym8)),
                        ("SciBmadStandard", LineElement(L = 0.4, En0 = 1e6)))
      b1 = track_unsliced(ele)
      b2 = track_sliced(ele, 4)
      d = slice_maxdiff(b1, b2)
      @test d <= 1e-15
      @test b1.t_ref ≈ b2.t_ref rtol = 1e-15
      # Field acts: differs from a drift of the same length.
      @test slice_maxdiff(b1, track_unsliced(Drift(L = 0.4))) > 1e-6
    end
    ele = LineElement(L = 0.4, En0L = 4e5)
    @test CSR.slice_element(ele, 0.1, true, false).En0L ≈ 1e5
    @test CSR.slice_element(ele, 0.1, true, false).tracking_method.fringe_at == Fringe.EntranceEnd
    @test CSR.slice_element(ele, 0.1, false, false).tracking_method.fringe_at == Fringe.NoEnd
  end

  @testset "Map elements: whole only, no CSR" begin
    shear(v, q, p) = ((v[1] + p * v[2], v[2], v[3], v[4], v[5], v[6]), q)
    ele = LineElement(L = 0.5, transport_map = shear, transport_map_params = 0.5)
    b1 = track_unsliced(ele)
    b2 = track_sliced(ele, 1)
    @test slice_maxdiff(b1, b2) == 0
    @test b1.t_ref == b2.t_ref
    @test b1.coords.v[1, 1] ≈ 1e-3 + 0.5 * -2e-4
    @test_throws ErrorException CSR.slice_element(ele, 0.25, true, false)
    @test !CSR.csr_sliceable(ele)
    @test !CSR.csr_sliceable(Marker())
    @test !CSR.csr_sliceable(Quadrupole(L = 0.0, Kn1L = 0.1))
    @test CSR.csr_sliceable(Drift(L = 1.0))
    @test CSR.csr_sliceable(SBend(; bend_kw...))
  end

  @testset "Unsupported elements" begin
    @test_throws ErrorException CSR.slice_element(Quadrupole(L = 0.3, Kn1 = 1.0, x_offset = 1e-3), 0.1, true, false)
    @test_throws ErrorException CSR.slice_element(RFCavity(L = 0.3, voltage = 1e6), 0.1, true, false)
    @test_throws ErrorException CSR.slice_element(Marker(), 0.0, true, true)
  end
end
