module CSRChainRulesCoreExt

using CSR, ChainRulesCore, StaticArrays, ForwardDiff
using CSR: CSRGeometry, CSRElementInfo, FloorPosition, N_ELE_PARAMS, geometry_to_vector, geometry_from_vector

# Pullback of csr_kick_pz: pz, z and q by explicit adjoint kernels (src/adjoint.jl, src/kernels/adjoint.jl), the
# bin grid held constant; geometry, s_kick and gamma by ForwardDiff through the kick tables (csr_kick_param_adjoint).
# Keyword arguments (kick_factor, pz_scale) get no cotangents.
function ChainRulesCore.rrule(::typeof(csr_kick_pz), pz::AbstractVector, z::AbstractVector, q::AbstractVector,
                              alive::AbstractVector{Bool}, geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma,
                              species, params::CSRParams; kick_factor = 1.0, pz_scale = 1.0)
  st = CSR.csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor)
  backend = CSR.get_backend(pz)
  pz_new = similar(pz, CSR.device_eltype(backend, promote_type(eltype(pz), eltype(z), eltype(q), CSR.number_type(geom),
                                                               typeof(s_kick), typeof(gamma), typeof(kick_factor),
                                                               typeof(pz_scale))))
  copyto!(pz_new, pz)
  pz_scale_d = CSR.device_scalar(backend, pz_scale)
  CSR.csr_apply_kicks!(pz_new, z, alive, st.bins, st.kick; pz_scale = pz_scale_d)

  function csr_kick_pz_pullback(ybar)
    ybar = unthunk(ybar)
    if ybar isa AbstractZero
      pzbar, zbar, qbar = ybar, ybar, ybar
      geombar, s_kickbar, gammabar = ybar, ybar, ybar
    else
      pzbar, zbar, qbar, kbar = CSR.csr_kick_adjoint(ybar, z, q, alive, st, params; pz_scale = pz_scale_d)
      gv, s_kickbar, gammabar = CSR.csr_kick_param_adjoint(kbar, st.bins, geom, ix_ele_kick, s_kick, gamma, species,
                                                           params; kick_factor)
      geombar = geometry_tangent(geom, gv)
    end
    return (NoTangent(), pzbar, zbar, qbar, NoTangent(), geombar, NoTangent(), s_kickbar, gammabar, NoTangent(),
            NoTangent())
  end
  return pz_new, csr_kick_pz_pullback
end

# CSRGeometry from lengths, curvatures, centroid (and reference exit floor positions): cotangents by the transposed
# ForwardDiff Jacobian of geometry_to_vector. The keywords (ref_start, bmad_compat) get no cotangents.
function ChainRulesCore.rrule(::Type{CSRGeometry}, L::AbstractVector, g::AbstractVector, centroid::AbstractMatrix;
                              ref_start::FloorPosition = FloorPosition(), bmad_compat::Bool = false)
  geom = CSRGeometry(L, g, centroid; ref_start, bmad_compat)
  n = length(L)
  function CSRGeometry_pullback(gbar)
    gbar = unthunk(gbar)
    gbar isa AbstractZero && return (NoTangent(), gbar, gbar, gbar)
    gv = geometry_vector(gbar, n)
    x0 = [float.(L); float.(g); vec(float.(centroid))]
    unpack(x) = (x[1:n], x[n+1:2n], reshape(x[2n+1:end], n + 1, 6))
    grad = ForwardDiff.gradient(x -> sum(gv .* geometry_to_vector(CSRGeometry(unpack(x)...; ref_start, bmad_compat))), x0)
    Lbar, gbar_, cbar = unpack(grad)
    return (NoTangent(), Lbar, gbar_, cbar)
  end
  return geom, CSRGeometry_pullback
end

function ChainRulesCore.rrule(::Type{CSRGeometry}, L::AbstractVector, g::AbstractVector,
                              ref_exit::AbstractVector{<:FloorPosition}, centroid::AbstractMatrix;
                              ref_start::FloorPosition = FloorPosition(), bmad_compat::Bool = false)
  geom = CSRGeometry(L, g, ref_exit, centroid; ref_start, bmad_compat)
  n = length(L)
  function CSRGeometry_pullback(gbar)
    gbar = unthunk(gbar)
    gbar isa AbstractZero && return (NoTangent(), gbar, gbar, gbar, gbar)
    gv = geometry_vector(gbar, n)
    x0 = [float.(L); float.(g); reduce(vcat, [[f.r; f.theta] for f in ref_exit]); vec(float.(centroid))]
    function unpack(x)
      r = x[2n+1:6n]
      floors = [FloorPosition(SVector(r[4i-3], r[4i-2], r[4i-1]), r[4i]) for i in 1:n]
      return x[1:n], x[n+1:2n], floors, reshape(x[6n+1:end], n + 1, 6)
    end
    grad = ForwardDiff.gradient(x -> sum(gv .* geometry_to_vector(CSRGeometry(unpack(x)...; ref_start, bmad_compat))), x0)
    Lbar, gbar_, _, cbar = unpack(grad)
    r = grad[2n+1:6n]
    fbar = [Tangent{eltype(ref_exit)}(r = SVector(r[4i-3], r[4i-2], r[4i-1]), theta = r[4i]) for i in 1:n]
    return (NoTangent(), Lbar, gbar_, fbar, cbar)
  end
  return geom, CSRGeometry_pullback
end

# Tangent{CSRGeometry} from a vector in the order of geometry_to_vector, and back. bmad_compat (a Bool) is left out
# of the Tangent (absent properties are zero).
function geometry_tangent(geom::CSRGeometry, v::AbstractVector)
  FT = typeof(geom.start)
  ftan(i) = Tangent{FT}(r = SVector(v[i], v[i+1], v[i+2]), theta = v[i+3])
  ET = eltype(geom.ele)
  ele = map(1:length(geom.ele)) do j
    i = 5 + N_ELE_PARAMS * (j - 1)
    Tangent{ET}(L = v[i], g = v[i+1], s_start = v[i+2], ref_floor0 = ftan(i + 3), ref_floor1 = ftan(i + 7),
                floor0 = ftan(i + 11), floor1 = ftan(i + 15), theta_chord = v[i+19], L_chord = v[i+20],
                spline = SVector(v[i+21], v[i+22], v[i+23], v[i+24]), dL_s = v[i+25])
  end
  return Tangent{typeof(geom)}(start = ftan(1), ele = ele)
end

function geometry_vector(t, n_ele::Integer)
  v = zeros(4 + N_ELE_PARAMS * n_ele)
  setv!(i, x) = (x isa AbstractZero || (v[i] = x); nothing)
  function put_floor!(i, f)
    f = unthunk(f)
    f isa AbstractZero && return
    r = unthunk(f.r)
    r isa AbstractZero || (v[i:i+2] .= r)
    setv!(i + 3, unthunk(f.theta))
  end
  t = unthunk(t)
  put_floor!(1, t.start)
  ele = unthunk(t.ele)
  ele isa AbstractZero && return v
  for j in 1:n_ele
    e = unthunk(ele[j])
    e isa AbstractZero && continue
    i = 5 + N_ELE_PARAMS * (j - 1)
    setv!(i, unthunk(e.L)); setv!(i + 1, unthunk(e.g)); setv!(i + 2, unthunk(e.s_start))
    put_floor!(i + 3, e.ref_floor0); put_floor!(i + 7, e.ref_floor1)
    put_floor!(i + 11, e.floor0); put_floor!(i + 15, e.floor1)
    setv!(i + 19, unthunk(e.theta_chord)); setv!(i + 20, unthunk(e.L_chord))
    sp = unthunk(e.spline)
    sp isa AbstractZero || (v[i+21:i+24] .= sp)
    setv!(i + 25, unthunk(e.dL_s))
  end
  return v
end

end
