# Stage 2c (2026-10-17): dE_ref knob at primal 0, plain BeamTracking (no CSR.jl): TPS derivative vs central FD of the
# final coordinates, for the cavity alone and for the cavity followed by a drift (Branch). Also dE_ref nonzero.
# Run: julia --project=<test env> stage2c_dE0_bt.jl
using BeamTracking, Beamlines, GTPSA
const E = Species("electron")
k = @vars(Descriptor(1, 1))
v0 = [1e-3 -2e-4 5e-4 3e-4 1e-3 2e-3; -2e-3 1e-4 -1e-3 -2e-4 -5e-4 -1e-3]
function f(p, dE0, tail)
  c(x) = x + zero(p)
  cav = RFCavity(L = 1.0, voltage = 2e8, rf_frequency = 1.3e9, dE_ref = dE0 + 1e7 * p, phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  br = Branch(tail ? [Marker(E_ref = 1e8, species_ref = E), cav, Drift(L = 0.5)] : [Marker(E_ref = 1e8, species_ref = E), cav])
  b = Bunch(c.(v0); species = E, p_over_q_ref = c(BeamTracking.E_to_R(E, 1e8)), t_ref = c(0.0))
  for bl in br.beamlines, ele in bl.line
    track!(b, ele; context = bl.context)
  end
  return vcat(vec(b.coords.v), [b.p_over_q_ref])
end
for dE0 in (0.0, 1e8), tail in (false, true)
  Jt = GTPSA.jacobian(f(k[1], dE0, tail))[:, 1]
  h = 1e-5
  fd = (f(h, dE0, tail) .- f(-h, dE0, tail)) ./ (2h)
  n = length(fd) - 1
  println("dE_ref = ", dE0, " + 1e7 p, drift after: ", tail, "   coords |TPS|/|FD| = ",
          round(sqrt(sum(abs2, Jt[1:n])) / sqrt(sum(abs2, fd[1:n])); sigdigits = 6),
          "   p_over_q_ref TPS/FD = ", round(Jt[end] / fd[end]; sigdigits = 6))
end

# Per coordinate, per particle: TPS vs FD after the cavity and after the drift (dE_ref = 0 + 1e7 p)
function g(p, stop_after_cavity)
  c(x) = x + zero(p)
  cav = RFCavity(L = 1.0, voltage = 2e8, rf_frequency = 1.3e9, dE_ref = 1e7 * p, phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  br = Branch([Marker(E_ref = 1e8, species_ref = E), cav, Drift(L = 0.5)])
  b = Bunch(c.(v0); species = E, p_over_q_ref = c(BeamTracking.E_to_R(E, 1e8)), t_ref = c(0.0))
  for (ib, bl) in enumerate(br.beamlines), ele in bl.line
    stop_after_cavity && ele isa LineElement && ele.kind == "Drift" && break
    track!(b, ele; context = bl.context)
  end
  return b
end
for stop in (true, false)
  bt = g(k[1], stop); h = 1e-5
  bp, bm = g(h, stop), g(-h, stop)
  Jt = GTPSA.jacobian(vec(bt.coords.v))[:, 1]
  fd = (vec(bp.coords.v) .- vec(bm.coords.v)) ./ (2h)
  println(stop ? "after cavity:" : "after drift:", "  t_ref TPS=", GTPSA.jacobian([bt.t_ref])[1], " FD=", (bp.t_ref - bm.t_ref) / 2h)
  for j in 1:6
    println("  coord ", j, ": TPS ", round.(Jt[2j-1:2j]; sigdigits = 6), "  FD ", round.(fd[2j-1:2j]; sigdigits = 6))
  end
end
println("element kinds: ", [(bl_i, ele.kind) for (bl_i, bl) in enumerate(Branch([Marker(E_ref = 1e8, species_ref = E), RFCavity(L = 1.0, voltage = 2e8, rf_frequency = 1.3e9, dE_ref = 1e7 * k[1], phi0 = 0.0, tracking_method = SaganCavity(n_cells = 4)), Drift(L = 0.5)]).beamlines) for ele in bl.line])
