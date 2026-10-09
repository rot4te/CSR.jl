module CSRForwardDiffExt

using CSR, ForwardDiff

# The bin grid is built from primal values (held constant under differentiation).
CSR.primal(d::ForwardDiff.Dual) = CSR.primal(ForwardDiff.value(d))

end
