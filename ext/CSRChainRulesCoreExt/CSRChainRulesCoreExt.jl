module CSRChainRulesCoreExt

using CSR, ChainRulesCore

# Pullback of csr_kick_pz with respect to pz, z and q by explicit adjoint kernels (src/adjoint.jl). The bin grid and
# the kick tables are constants; derivatives with respect to the geometry, s_kick and gamma are not implemented.
function ChainRulesCore.rrule(::typeof(csr_kick_pz), pz::AbstractVector, z::AbstractVector, q::AbstractVector,
                              alive::AbstractVector{Bool}, geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma,
                              species, params::CSRParams; kick_factor = 1.0, pz_scale = 1.0)
  st = CSR.csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor)
  pz_new = similar(pz, promote_type(eltype(pz), eltype(z), eltype(q)))
  copyto!(pz_new, pz)
  CSR.csr_apply_kicks!(pz_new, z, alive, st.bins, st.kick; pz_scale)

  function csr_kick_pz_pullback(ybar)
    ybar = unthunk(ybar)
    if ybar isa AbstractZero
      pzbar, zbar, qbar = ybar, ybar, ybar
      geombar, s_kickbar, gammabar = ybar, ybar, ybar
    else
      pzbar, zbar, qbar = CSR.csr_kick_adjoint(ybar, z, q, alive, st, params; pz_scale)
      geombar = @not_implemented("csr_kick_pz: derivative with respect to the geometry")
      s_kickbar = @not_implemented("csr_kick_pz: derivative with respect to s_kick")
      gammabar = @not_implemented("csr_kick_pz: derivative with respect to gamma")
    end
    return (NoTangent(), pzbar, zbar, qbar, NoTangent(), geombar, NoTangent(), s_kickbar, gammabar, NoTangent(),
            NoTangent())
  end
  return pz_new, csr_kick_pz_pullback
end

end
