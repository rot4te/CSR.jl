using CSR
using Test
using AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, GTPSA

@testset "CSR.jl" begin
  @testset "Package extensions load" begin
    for ext in (:CSRAbstractFFTsExt, :CSRBeamTrackingBeamlinesExt, :CSRChainRulesCoreExt, :CSRGTPSAExt)
      @test !isnothing(Base.get_extension(CSR, ext))
    end
  end
end
