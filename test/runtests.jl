using ModularCSR
using Test
using StaticArrays
using AbstractFFTs, Beamlines, ChainRulesCore, ForwardDiff, GTPSA

include("utils.jl")

@testset "ModularCSR" begin
  @testset "Package extensions load" begin
    for ext in (:ModularCSRAbstractFFTsExt,
                :ModularCSRChainRulesCoreExt, :ModularCSRGTPSAExt)
      @test !isnothing(Base.get_extension(ModularCSR, ext))
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
  include("tps_collective_test.jl")
  include("type_stability_test.jl")
  include("review_fixes_test.jl")
end
