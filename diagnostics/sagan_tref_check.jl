# Upstream refresh (2026-10-15): BeamTracking 0.8.7 contains 8c302c6 ("Fix bugs in SaganCavity universal!":
# t_exit = t_enter + dt_ref). Does CSR.jl's SaganSlice tracking (which reproduced the old t_ref reset) still match
# BeamTracking's t_ref? And is BeamTracking item 3 (p_over_q_ref not updated after a SaganCavity) still real?
# Run: julia --project=<test env> sagan_tref_check.jl
using CSR, BeamTracking, Beamlines
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

const E = Species("electron")
const E0 = 1e8
pq0 = BeamTracking.E_to_R(E, E0)
cav(dE) = RFCavity(L = 1.0, voltage = 2e8, rf_frequency = 1.3e9, dE_ref = dE, phi0 = 0.0,
                   zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
lattice(dE) = Branch([Marker(E_ref = E0, species_ref = E), [SBend(L = 0.1, g_ref = 0.5, Kn0 = 0.5) for _ in 1:3]...,
                      cav(dE), Drift(L = 0.5)])
# 400 particles of weight 1 (charge 400 e: CSR kicks negligible, so the coordinates differ only slightly)
bunch() = (v = zeros(400, 6); v[:, 5] = range(-1e-4, 1e-4; length = 400); Bunch(v; species = E, p_over_q_ref = pq0))

for dE in (0.0, 1e8)
  br = lattice(dE)
  b1 = bunch()
  t_before = NaN
  for bl in br.beamlines, ele in bl.line
    ele.tracking_method isa SaganCavity && (t_before = b1.t_ref)
    track!(b1, ele)
  end
  b2 = bunch()
  track_csr!(b2, br; params = CSRParams(n_bin = 40, ds_step = 0.05), csr_on = e -> e.tracking_method isa SaganCavity)
  println("dE_ref = $dE: t_ref before cavity (track!) ", t_before)
  println("  t_ref at end: track! ", b1.t_ref, "  track_csr! (CSR off except slicing the cavity) ", b2.t_ref,
          "  diff ", b2.t_ref - b1.t_ref)
  println("  coords equal: ", b1.coords.v == b2.coords.v, "  max diff ", maximum(abs, b1.coords.v .- b2.coords.v))
  println("  p_over_q_ref: start ", pq0, "  after track! ", b1.p_over_q_ref, "  after track_csr! ", b2.p_over_q_ref,
          "  cavity element exit ", br.beamlines[end].line[end].p_over_q_ref)
end
