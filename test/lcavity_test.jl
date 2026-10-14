# lcavity mode: SaganCavity slices built from BeamTracking's Sagan kernels, reference energy ramp (dE_ref),
# track_csr! through a Branch.

const LC_EXT = Base.get_extension(CSR, :CSRBeamTrackingBeamlinesExt)
const LC_E = Species("electron")
const LC_E0 = 1e8   # Reference energy at the lattice start [eV]

lc_cavity(; kw...) = RFCavity(; L = 2.0, voltage = 0.2LC_E0, rf_frequency = 1e9, dE_ref = 0.1LC_E0, phi0 = 0.1, kw...)

# The cavity as placed in a Branch (dE_ref puts it in its own Beamline).
lc_in_branch(cav) = Branch([Marker(E_ref = LC_E0, species_ref = LC_E), cav]).beamlines[2].line[1]

function lc_bunch(ele)
  v = [ 1e-3 -2e-4  5e-4  3e-4  1e-3  2e-3;
       -2e-3  1e-4 -1e-3 -2e-4 -5e-4 -1e-3;
        0.0   0.0   0.0   0.0   0.0   0.0]
  return Bunch(v; species = LC_E, p_over_q_ref = BeamTracking.E_to_R(LC_E, ele.E_ref - ele.dE_ref))
end

# Track `ele` as n slices with boundaries as in track_csr! (s0 + L_slice == next s0 exactly).
function lc_track_sliced(ele, n)
  b = lc_bunch(ele)
  ds = float(ele.L) / n
  local s
  for k in 1:n
    s0 = (k - 1) * ds
    s = CSR.slice_element(ele, s0, k * ds - s0, k == 1, k == n)
    LC_EXT.track_slice!(b, s)
  end
  return b, s
end

lc_track_whole(ele) = (b = lc_bunch(ele); track!(b, ele); b)

# Line bunch for CSR tracking
function lc_line_bunch()
  n = 4001
  z = collect(range(-6e-4, 6e-4; length = n))
  w = gaussian.(z, 1e-4)
  w .*= (1e-9 / E_CHARGE) / sum(w)
  v = zeros(n, 6)
  v[:, 5] .= z
  return Bunch(v, nothing, w; species = LC_E, p_over_q_ref = BeamTracking.E_to_R(LC_E, LC_E0))
end

# Bends (CSR source), an lcavity, a drift.
function lc_lattice(n_cells, dE)
  bends = [SBend(L = 0.1, g_ref = 0.5, Kn0 = 0.5) for _ in 1:10]
  cav = RFCavity(L = 1.0, voltage = 2e8, rf_frequency = 1.3e9, dE_ref = dE, phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = n_cells),
                 traveling_wave = true)
  return Branch([Marker(E_ref = LC_E0, species_ref = LC_E), bends..., cav, Drift(L = 0.5)])
end

is_sagan(ele) = ele.tracking_method isa SaganCavity

@testset "lcavity (SaganCavity) mode" begin
  # Exact up to round-off: only drifts are split; the half ponderomotive and multipole kicks of each drift segment
  # are point events at its ends (D22). With a solenoid the exact solenoid map's own round-off shows (BeamTracking's
  # Solenoid split once in two: 6e-17).
  @testset "Slices reproduce the whole cavity: $name" for (name, cav, tol) in (
      ("traveling wave", lc_cavity(tracking_method = SaganCavity(n_cells = 3), traveling_wave = true), 1e-17),
      ("single kick (n_cells = 0)", lc_cavity(tracking_method = SaganCavity(n_cells = 0), traveling_wave = true), 1e-17),
      ("L_active = 0", lc_cavity(tracking_method = SaganCavity(n_cells = 3, L_active = 0.0)), 1e-17),
      ("cells on slice boundaries", lc_cavity(tracking_method = SaganCavity(n_cells = 8), traveling_wave = true), 1e-17),
      ("automatic cells", lc_cavity(tracking_method = SaganCavity(), traveling_wave = true), 1e-17),
      ("aperture and misalignment", lc_cavity(tracking_method = SaganCavity(n_cells = 3), traveling_wave = true,
                                              x_offset = 1e-4, tilt = 0.1, x1_limit = -1.5e-3, x2_limit = 1.5e-3,
                                              y1_limit = -1.0, y2_limit = 1.0), 1e-17),
      ("standing wave", lc_cavity(tracking_method = SaganCavity(n_cells = 2), traveling_wave = false), 1e-17),
      ("standing wave, n_cells = 0", lc_cavity(tracking_method = SaganCavity(n_cells = 0, L_active = 1.5), Kn1L = 0.001,
                                               traveling_wave = false), 1e-17),
      ("multipoles, radiation damping", lc_cavity(tracking_method = SaganCavity(n_cells = 3, L_active = 1.0,
                                                  radiation_damping_on = true), Kn1L = 0.001, Kn2L = 0.01,
                                                  traveling_wave = false), 1e-17),
      ("L_active < L, multipoles", lc_cavity(tracking_method = SaganCavity(n_cells = 3, L_active = 1.0), Ksol = 0.01,
                                             Kn1L = 0.001, traveling_wave = true), 1e-15),
      ("standing wave, multipoles", lc_cavity(tracking_method = SaganCavity(n_cells = 3, L_active = 1.0), Ksol = 0.01,
                                              Kn1L = 0.001, traveling_wave = false), 1e-15),
    )
    ele = lc_in_branch(cav)
    b1 = lc_track_whole(ele)
    b0, s1 = lc_track_sliced(ele, 1)
    @test b0.coords.v == b1.coords.v
    @test b0.coords.state == b1.coords.state
    for n in (3, 4, 7, 20)
      b, s = lc_track_sliced(ele, n)
      @test maximum(abs, b.coords.v .- b1.coords.v) <= tol
      @test b.coords.state == b1.coords.state
      @test b.t_ref ≈ b1.t_ref rtol = 1e-14
      # Reference momentum after the last slice: all cells applied
      @test LC_EXT.sagan_exit_P0c(s) ≈ sqrt(ele.E_ref^2 - massof(LC_E)^2) rtol = 1e-12
    end
    @test maximum(abs, b1.coords.v .- lc_bunch(ele).coords.v) > 1e-4
  end

  @testset "Reference momentum steps at the cell kicks" begin
    ele = lc_in_branch(lc_cavity(tracking_method = SaganCavity(n_cells = 4, L_active = 1.0)))
    pc(E) = sqrt(E^2 - massof(LC_E)^2)
    E0 = ele.E_ref - ele.dE_ref
    s = CSR.slice_element(ele, 0.0, 0.5, true, false)          # Before the active region (starts at 0.5)
    @test LC_EXT.sagan_exit_P0c(s) ≈ pc(E0) rtol = 1e-14
    s = CSR.slice_element(ele, 0.5, 0.25, false, false)        # Half-kick at 0.5, full kick at 0.75 not included
    @test LC_EXT.sagan_exit_P0c(s) ≈ pc(E0 + ele.dE_ref / 8) rtol = 1e-12
    s = CSR.slice_element(ele, 0.75, 1.25, false, true)
    @test LC_EXT.sagan_exit_P0c(s) ≈ pc(ele.E_ref) rtol = 1e-12
  end

  @testset "Errors" begin
    # Not in a Beamline: no reference energy or dE_ref
    @test_throws ErrorException CSR.slice_element(lc_cavity(tracking_method = SaganCavity()), 0.0, 0.5, true, false)
  end

  @testset "track_csr! through a Branch" begin
    p = CSRParams(n_bin = 80, ds_step = 0.05)
    for dE in (0.0, 1e8)
      br = lc_lattice(4, dE)
      # CSR off: identical to element-by-element track!
      b1 = lc_line_bunch()
      for bl in br.beamlines, e in bl.line
        track!(b1, e)
      end
      b2 = lc_line_bunch()
      track_csr!(b2, br; params = p, csr_on = e -> false)
      @test b2.coords.v == b1.coords.v

      # CSR inside the cavity does not depend on how finely the reference energy steps (Δpz rescaled from the
      # linearly ramped reference momentum to the stepped one). Without the rescale: 4% (dE = 1e8).
      dpz = map((2, 8)) do n_cells
        bon = lc_line_bunch()
        track_csr!(bon, lc_lattice(n_cells, dE); params = p)
        boff = lc_line_bunch()
        track_csr!(boff, lc_lattice(n_cells, dE); params = p, csr_on = !is_sagan)
        bon.coords.v[:, 6] .- boff.coords.v[:, 6]
      end
      @test maximum(abs, dpz[2]) > 1e-4
      @test maximum(abs, dpz[1] .- dpz[2]) / maximum(abs, dpz[2]) < 5e-3
    end

    # The cavity gets n_step + 1 = 21 kicks (s = 0, 0.05, ..., 1.0)
    br = lc_lattice(4, 1e8)
    ix_cav = findfirst(is_sagan, reduce(vcat, [bl.line for bl in br.beamlines]))
    s_kicks = Float64[]
    track_csr!(lc_line_bunch(), br; params = p, diagnostics = (i, k, s, r) -> i == ix_cav && push!(s_kicks, s))
    @test s_kicks ≈ 0:0.05:1.0

    # A Branch of plain elements gives the same result as the Beamline
    line() = [Marker(E_ref = LC_E0, species_ref = LC_E), (SBend(L = 0.1, g_ref = 0.5, Kn0 = 0.5) for _ in 1:5)...]
    b1 = lc_line_bunch()
    track_csr!(b1, Beamline(line()); params = p)
    b2 = lc_line_bunch()
    track_csr!(b2, Branch(line()); params = p)
    @test maximum(abs, b2.coords.v .- b1.coords.v) <= 1e-12 * maximum(abs, b1.coords.v)   # atomic deposit order
  end

  @testset "csr_kick! pz_scale" begin
    z, q, alive = gaussian_line_bunch(1e-3, 1e-9, 2001)
    params = CSRParams(n_bin = 60, ds_step = 0.1)
    geom = CSRGeometry(circle_lattice(10.0, 0.1, 20)...)
    pz1 = zeros(length(z)); pz2 = zeros(length(z))
    r1 = csr_kick!(pz1, z, q, alive, geom, 20, 0.05, 1e3, LC_E, params)
    r2 = csr_kick!(pz2, z, q, alive, geom, 20, 0.05, 1e3, LC_E, params; pz_scale = 1.25)
    @test pz2 ≈ 1.25 .* pz1 rtol = 1e-14
    @test r2.kick_per_meter == r1.kick_per_meter   # diagnostics are not scaled
  end
end
