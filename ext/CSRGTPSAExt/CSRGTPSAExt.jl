module CSRGTPSAExt

using CSR, GTPSA

# Branch decisions on TPS values use the scalar part, as BeamTracking does (GTPSA's comparisons also use only the
# scalar part); the TPS is the expansion of the code path the reference point takes.
CSR.primal(t::TPS{Float64}) = scalar(t)

# Newton steps from the primal source root: one order per step, up to the TPS's truncation order (field `mo`).
CSR.root_order(t::TPS) = Int(t.mo)

# GTPSA has no `mod`: fold the scalar part only, as for `Dual`s (the derivative of the fold is 1).
function CSR.modulo2(x::TPS, amp)
  s = scalar(x)
  return x + (CSR.modulo2(s, amp) - s)
end

# GTPSA has no `cbrt`: the power series of x^(1/3) about the scalar part, which must not be 0.
function CSR.cube_root(x::TPS)
  s = scalar(x)
  s == 0 && error("cube_root: TPS with zero scalar part")
  return s > 0 ? x^(1 / 3) : -(-x)^(1 / 3)
end

# CSR.jl creates TPS constants from types, which GTPSA puts in the current Descriptor; a TPS in another one is a mismatch.
CSR.descriptor_mismatch(t::TPS{Float64,GTPSA.Dynamic}) = GTPSA.getdesc(t) != GTPSA.desc_current

end
