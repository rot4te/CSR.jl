# Phase 7 step 5: run-to-run spread of the test particle's Jacobian and second derivatives on several threads
# (atomic deposit order changes the frozen wake), for the lcavity Branch and the Beamline of track_tps_test.jl.
# Sets the tolerances there. Run: julia --project=<test env> -t 4 track_tps_thread_spread.jl [n_runs]
using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, ForwardDiff, GTPSA
const TD = joinpath(homedir(), ".julia/dev/CSR.jl/test")
include(joinpath(TD, "utils.jl"))
maxrel(a, b) = maximum(abs, a .- b) / maximum(abs, b)
norm_rel(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))   # as isapprox on vectors
include(joinpath(TD, "track_csr_test.jl"))
include(joinpath(TD, "lcavity_test.jl"))
include(joinpath(TD, "track_tps_test.jl"))   # defines TPSTestTrack, TPSTestJacobian

n_runs = parse(Int, get(ARGS, 1, "8"))
println("threads: ", Threads.nthreads())
e = Species("electron"); pq = tc_pq(200)
bl = Beamline([Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5, Kn0 = 1.5), Drift(L = 0.2),
               SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]; p_over_q_ref = pq, species_ref = e)
cases = [
  ("Beamline", TPSTestTrack(tb -> track_csr!(tc_line_bunch(1e-3, 1e-9, 400, pq; x = 1e-4), bl;
                                             params = CSRParams(n_bin = 40, ds_step = 0.1),
                                             csr_on = ele -> ele !== bl.line[3], test_bunch = tb),
                            [2e-4, -1e-4, 0.0, 0.0, 3e-4, 1e-4], e, pq)),
  ("lcavity", TPSTestTrack(tb -> track_csr!(lc_line_bunch(), lc_lattice(4, 1e8);
                                            params = CSRParams(n_bin = 80, ds_step = 0.05), test_bunch = tb),
                           [1e-4, -2e-5, 0.0, 0.0, 1e-4, 2e-4], LC_E, BeamTracking.E_to_R(LC_E, LC_E0)))]
for (name, f) in cases
  Js = [ForwardDiff.jacobian(f, zeros(6)) for _ in 1:n_runs]
  Hs = [reshape(ForwardDiff.jacobian(TPSTestJacobian(f), zeros(6)), 6, 6, 6) for _ in 1:n_runs]
  # Per column / per output, relative to its norm, as the tests compare
  dJ = maximum(maximum(norm_rel(J[:, c], Js[1][:, c]) for c in 1:6 if any(!iszero, Js[1][:, c])) for J in Js)
  dH = maximum(maximum(maximum(abs, H[j, :, :] .- Hs[1][j, :, :]) / maximum(abs, Hs[1][j, :, :]) for j in 1:6) for H in Hs)
  dHabs = maximum(maximum(abs, H .- Hs[1]) / maximum(abs, Hs[1]) for H in Hs)
  println("$name ($n_runs runs): J column spread ", dJ, "; H per output spread ", dH, "; H vs max|H| ", dHabs)
end
