# Does BeamTracking crash tracking many TPS particles on several threads? Usage: julia -t 4 bt_tps_threads.jl [n]
using BeamTracking, Beamlines, GTPSA
n = parse(Int, get(ARGS, 1, "2000"))
d = Descriptor(6, 2); x = @vars(d)
v = [x[j] + 1e-4 * (i + j) for i in 1:n, j in 1:6]
mt = get(ARGS, 2, "false") == "true"
b = Bunch(v; species = Species("electron"))
bl = Beamline([SBend(L = 0.5, g = 0.2), Drift(L = 1.0), Quadrupole(L = 0.3, Kn1 = 1.0)]; p_over_q_ref = -1.0, species_ref = Species("electron"))
for _ in 1:20; track!(b, bl; use_cpu_multithreading = mt); end
println("ok, mt = $mt, n = $n, threads = $(Threads.nthreads())")
