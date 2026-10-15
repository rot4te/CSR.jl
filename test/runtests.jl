using CSR
using Test
using StaticArrays
using AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, ForwardDiff, GTPSA

include("utils.jl")

@testset "CSR.jl" begin
  @testset "Package extensions load" begin
    for ext in (:CSRAbstractFFTsExt, :CSRBeamTrackingBeamlinesExt, :CSRChainRulesCoreExt, :CSRGTPSAExt)
      @test !isnothing(Base.get_extension(CSR, ext))
    end
  end

  include("geometry_test.jl")
  include("binning_test.jl")
  include("wake_test.jl")
  include("kick_test.jl")
  include("kernels_test.jl")
  include("shielding_test.jl")
  include("ad_test.jl")
  include("ad_geometry_test.jl")
  include("ad_rrule_test.jl")
  include("tps_test.jl")
  include("slicing_test.jl")
  include("track_csr_test.jl")
  include("lcavity_test.jl")
end
