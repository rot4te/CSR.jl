using CSR
using Test
using StaticArrays
using AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, GTPSA

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
  include("slicing_test.jl")
end
