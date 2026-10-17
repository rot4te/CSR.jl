using CSR, GTPSA, ForwardDiff, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore
using AtomicAndPhysicalConstants
using AtomicAndPhysicalConstants
const Species_ = try; AtomicAndPhysicalConstants.Species; catch; nothing; end
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom0 = CSRGeometry([0.2, 0.3, 0.2], [0.0, 1.5, 0.0], c3)
z = collect(range(-6e-3, 6e-3; length = 400)) .+ 0.37 * (12e-3 / 400) .* sin.(1:400)
w = exp.(-z.^2 ./ 2e-6) .* (1 .+ 0.2 .* cos.(3 .* (1:400))); q = 1e-9 .* w ./ sum(w)
alive = fill(true, 400)
params = CSRParams(n_bin = 40, ds_step = 0.1)
sp = Species("electron")
function tryit(name, f)
  print(rpad(name, 50))
  try
    r = f()
    println("OK  ", typeof(r))
  catch e
    println("FAIL ", first(split(sprint(showerror, e), '\n')))
    if get(ENV, "BT", "") == name
      Base.showerror(stdout, e, catch_backtrace()); println()
    end
  end
end
for (label, mk) in (("static", () -> @vars(Descriptor(2,2))), ("dynamic", () -> @vars(Descriptor(2,2), dynamic = true)))
  println("=== ", label)
  x = mk()
  tryit("KP gamma TPS", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + x[1]))
  tryit("KP s_kick TPS", () -> CSRKickPoint(geom0, 2, 0.17 + x[1], 200.0))
  tryit("KP kick_factor TPS", () -> CSRKickPoint(geom0, 2, 0.17, 200.0; kick_factor = 1.0 + x[1]))
  tryit("KP y_source TPS", () -> CSRKickPoint(geom0, 2, 0.17, 200.0; y_source = 1e-3 + x[1]))
  tryit("kick gamma TPS", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17, 200.0 + x[1], sp, params))
  tryit("kick s_kick TPS", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, sp, params))
end
println("=== ForwardDiff")
D = ForwardDiff.Dual{:t}
tryit("KP gamma Dual", () -> CSRKickPoint(geom0, 2, 0.17, ForwardDiff.Dual{:t}(200.0, 1.0)))
tryit("KP s_kick Dual", () -> CSRKickPoint(geom0, 2, ForwardDiff.Dual{:t}(0.17, 1.0), 200.0))
tryit("kick gamma Dual", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17, ForwardDiff.Dual{:t}(200.0, 1.0), sp, params))
tryit("kick s_kick Dual", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, ForwardDiff.Dual{:t}(0.17, 1.0), 200.0, sp, params))
