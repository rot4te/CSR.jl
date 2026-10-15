module CSRGTPSAExt

using CSR, GTPSA

# Branch decisions on TPS values use the scalar part, as BeamTracking does (GTPSA's comparisons also use only the
# scalar part); the TPS is the expansion of the code path the reference point takes.
CSR.primal(t::TPS{Float64}) = scalar(t)

# Newton steps from the primal source root: one order per step, up to the TPS's truncation order (field `mo`).
CSR.root_order(t::TPS) = Int(t.mo)

end
