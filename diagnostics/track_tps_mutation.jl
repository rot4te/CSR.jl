# Phase 7 step 5: can track_tps_test.jl fail? Overrides csr_apply_test_kicks! with a mutation and runs the test file
# (expect failures). Mutations: "none" (test kicks skipped), "scale" (pz_scale ignored for test particles).
# Run: julia --project=<test env> ~/.julia/dev/CSR.jl/planning/diagnostics/track_tps_mutation.jl <mutation>
using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, ForwardDiff, GTPSA
const TD = joinpath(homedir(), ".julia/dev/CSR.jl/test")
include(joinpath(TD, "utils.jl"))
maxrel(a, b) = maximum(abs, a .- b) / maximum(abs, b)
include(joinpath(TD, "track_csr_test.jl"))
include(joinpath(TD, "lcavity_test.jl"))

mutation = get(ARGS, 1, "none")
const ORIG = CSR.csr_apply_test_kicks!
if mutation == "none"
  @eval CSR csr_apply_test_kicks!(pz_t::AbstractVector, z_t::AbstractVector, alive_t::AbstractVector{Bool},
                                  bins::CSRBins, kick::AbstractVector; pz_scale = 1) = pz_t
elseif mutation == "scale"
  @eval CSR function csr_apply_test_kicks!(pz_t::AbstractVector, z_t::AbstractVector, alive_t::AbstractVector{Bool},
                                           bins::CSRBins, kick::AbstractVector; pz_scale = 1)
    n = apply_kicks_outside!(pz_t, z_t, alive_t, Array(bins.z_center), Array(kick), primal(bins.dz), 1)
    n == 0 || error("outside")
    return pz_t
  end
end
println("mutation: ", mutation)
try
  include(joinpath(TD, "track_tps_test.jl"))
catch err
  println("track_tps_test.jl: ", sprint(showerror, err))
end
