# Stage 2c (2026-10-17): does BeamTracking 0.8.7 shift pz twice after a SaganCavity with constant p_over_q_ref?
# On-crest cavity with voltage == dE_ref: the reference particle (all coordinates 0) gains dE_ref, so its pz must be
# ~0 after the cavity and stay so through a following drift. Prints pz and bunch.p_over_q_ref after each element.
# Run: julia --project=<test env> stage2c_sagan_pq_check.jl
using BeamTracking, Beamlines
const E = Species("electron")
for n_cells in (4, 0)
  cav = RFCavity(L = 1.0, voltage = 1e8, rf_frequency = 1.3e9, dE_ref = 1e8, phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = n_cells), traveling_wave = true)
  br = Branch([Marker(E_ref = 1e8, species_ref = E), cav, Drift(L = 0.5), Drift(L = 0.5)])
  b = Bunch(zeros(1, 6); species = E, p_over_q_ref = BeamTracking.E_to_R(E, 1e8))
  println("n_cells = ", n_cells)
  for (ib, bl) in enumerate(br.beamlines), ele in bl.line
    track!(b, ele; context = bl.context)
    println("  beamline ", ib, " ", rpad(ele.kind, 9), " pz = ", b.coords.v[1, 6], "   bunch E_ref = ",
            BeamTracking.R_to_E(E, b.p_over_q_ref), "   element E_ref = ", BeamTracking.R_to_E(E, ele.p_over_q_ref))
  end
end
