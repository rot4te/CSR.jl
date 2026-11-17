# Phase 7 step 5: runs the track_csr! test_bunch example of docs/src/usage.md (not executed by the docs build: the
# docs environment does not load GTPSA), with the usage page's bunch and lattice as far as needed.
# Run: julia --project=<test env> track_tps_docs_example.jl
using CSR, BeamTracking, Beamlines, GTPSA
using AtomicAndPhysicalConstants: massof
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

e = Species("electron")
pq = BeamTracking.E_to_R(e, 1e4 * massof(e))
bl = Beamline([SBend(L = 0.1, g_ref = 0.1, Kn0 = 0.1) for _ in 1:20]; p_over_q_ref = pq, species_ref = e)
zz = collect(range(-6e-3, 6e-3; length = 2001))
w = gaussian.(zz, 1e-3); w .*= (1e-9 / E_CHARGE) / sum(w)
v = zeros(2001, 6); v[:, 5] .= zz
bunch = Bunch(v, nothing, w; species = e, p_over_q_ref = pq)
params = CSRParams(n_bin = 200, ds_step = 0.05)
x = @vars(Descriptor(6, 2))

# Example as in usage.md
tb = Bunch(reshape([0.0, 0.0, 0.0, 0.0, 0.2e-3, 0.0] .+ x, 1, 6); species = e, p_over_q_ref = bunch.p_over_q_ref)
track_csr!(bunch, bl; params, test_bunch = tb)
display(GTPSA.jacobian(tb.coords.v[1, :]))
