# Phase 7 step 5: time track_csr! with Float64 / TPS / Dual / nested Dual test bunches (track_tps_test.jl lattice).
# Run: julia --project=<test env> track_tps_timing.jl
using CSR, BeamTracking, Beamlines, GTPSA, ForwardDiff
using AtomicAndPhysicalConstants: massof
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

e = Species("electron")
pq = BeamTracking.E_to_R(e, 200 * massof(e))
line = [Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5, Kn0 = 1.5), Quadrupole(L = 0.2, Kn1 = 3.0),
        SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
bl = Beamline(line; p_over_q_ref = pq, species_ref = e)
params = CSRParams(n_bin = 40, ds_step = 0.1)
function bunch()
  zz = collect(range(-6e-3, 6e-3; length = 400))
  w = gaussian.(zz, 1e-3); w .*= (1e-9 / E_CHARGE) / sum(w)
  v = zeros(400, 6); v[:, 1] .= 1e-4; v[:, 5] .= zz
  return Bunch(v, nothing, w; species = e, p_over_q_ref = pq)
end
v_ref = [2e-4, -1e-4, 0.0, 0.0, 3e-4, 1e-4]
f(v) = (tb = Bunch(reshape(v_ref .+ v, 1, 6); species = e, p_over_q_ref = pq);
        track_csr!(bunch(), bl; params, test_bunch = tb); tb.coords.v[1, :])
x = @vars(Descriptor(6, 2))
v0 = zeros(6)
for pass in 1:2
  println("pass $pass")
  t = @elapsed track_csr!(bunch(), bl; params); println("  no test bunch     ", t)
  t = @elapsed f(v0); println("  Float64           ", t)
  t = @elapsed f(x); println("  TPS order 2       ", t)
  t = @elapsed ForwardDiff.jacobian(f, v0); println("  ForwardDiff J     ", t)
  t = @elapsed ForwardDiff.hessian(v -> f(v)[1], v0); println("  ForwardDiff H     ", t)
end
