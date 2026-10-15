# Phase 7 step 5: where the nested-Dual compile time goes (BeamTracking track! alone vs track_csr! with a test bunch),
# and whether a second Hessian closure recompiles. Fresh session per run.
# Run: julia --project=<test env> track_tps_compile.jl
include(joinpath(@__DIR__, "track_tps_timing.jl"))   # defines bl, f, v0 (and runs the timing passes)
using ForwardDiff: Dual, Tag
println("compile study (after the timing script's warm-up of the same tag types):")
T2 = Dual{Tag{Nothing,Float64},Dual{Tag{Nothing,Float64},Float64,6},6}
vb = zeros(T2, 1, 6)
t = @elapsed track!(Bunch(vb; species = e, p_over_q_ref = pq), bl); println("  BeamTracking track!, new nested Dual type  ", t)
T3 = Dual{Tag{Int,Float64},Dual{Tag{Int,Float64},Float64,6},6}
t = @elapsed track_csr!(bunch(), bl; params, test_bunch = Bunch(zeros(T3, 1, 6) .+ v_ref'; species = e, p_over_q_ref = pq))
println("  track_csr! test bunch, another nested Dual type  ", t)
t = @elapsed ForwardDiff.hessian(v -> f(v)[5], v0); println("  ForwardDiff H, second closure  ", t)
