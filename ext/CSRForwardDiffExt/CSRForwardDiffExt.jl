module CSRForwardDiffExt

using CSR, ForwardDiff
using ForwardDiff: Dual, Partials, value, partials

# The bin grid is built from primal values (held constant under differentiation).
CSR.primal(d::Dual) = CSR.primal(value(d))

# Tag of the derivative ∂f/∂s in root_tangent: ordered inside any other tag (including `Nothing`, untagged Duals),
# so that it nests within the parameters' Duals.
struct RootTag end
ForwardDiff.:≺(::Type, ::Type{RootTag}) = true
ForwardDiff.:≺(::Type{RootTag}, ::Type) = false
ForwardDiff.:≺(::Type{RootTag}, ::Type{RootTag}) = false

# Implicit differentiation of a root found from primal values: ds/dp = -(∂f/∂p)/(∂f/∂s), ∂f/∂s by a nested Dual at
# s0. First order in Tg only.
function CSR.root_tangent(f, s0::Real, f0::Dual{Tg,V,N}) where {Tg,V<:AbstractFloat,N}
  sv = Dual{Tg}(V(s0), zero(Partials{N,V}))
  fs = f(Dual{RootTag}(sv, Partials((one(sv),))))
  dfds = value(partials(fs, 1))
  return Dual{Tg}(V(s0), -partials(f0) / dfds)
end

function CSR.root_tangent(f, s0::Real, f0::Dual{Tg,V}) where {Tg,V}
  error("root_tangent: nested Duals (value type $V) are not supported")
end

end
