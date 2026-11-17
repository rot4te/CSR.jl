using CSR
using Test
using StaticArrays
using Beamlines, ForwardDiff, GTPSA
using BeamTracking

include("utils.jl")

@testset "CSR.jl" begin
  include("geometry_test.jl")
  include("binning_test.jl")
  include("wake_test.jl")
  include("kick_test.jl")
  include("kernels_test.jl")
  include("shielding_test.jl")
  include("tps_test.jl")
  include("tps_collective_test.jl")
  include("type_stability_test.jl")
  include("review_fixes_test.jl")
  include("effect_test.jl")
end
