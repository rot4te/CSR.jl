# Reverse-mode pullback of one CSR kick (host side); kernels in kernels/adjoint.jl.

"""
    csr_kick_adjoint(ybar, z, q, alive, st, params; pz_scale = 1) -> (pzbar, zbar, qbar, kbar)

Pullback of `csr_kick_pz` with respect to `pz`, `z` and `q` for the output cotangent `ybar`, given the forward
intermediates `st = csr_binned_kick(...)`; `kbar` is the cotangent of the kick per bin (`st.kick`), input to
`csr_kick_param_adjoint`. The bin grid and kick tables are constants. Dead particles get zero `z`
and `q` cotangents. Runs as kernels on the backend of `z`; the kick-cotangent scatter uses atomic adds for
`Float32`/`Float64` and a serial CPU loop otherwise.
"""
function csr_kick_adjoint(ybar::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, st,
                          params::CSRParams; pz_scale = 1)
  backend = get_backend(z)
  bins = st.bins
  n_bin = length(st.kick)
  T = promote_type(eltype(ybar), eltype(st.kick), typeof(pz_scale))
  kbar = KernelAbstractions.zeros(backend, T, n_bin)
  zbar = KernelAbstractions.allocate(backend, promote_type(T, eltype(q)), length(z))
  if supports_atomic_add(kbar)
    csr_kick_adjoint_kernel!(backend)(kbar, zbar, ybar, z, alive, bins.z_center, st.kick, bins.dz, pz_scale;
                                      ndrange = length(z))
  else
    backend isa CPU || error("csr_kick_adjoint: element type $T needs the CPU backend (no atomic add)")
    for i in eachindex(z)
      if alive[i]
        kick_adjoint_particle!(PlainAdd(), kbar, zbar, i, ybar, z, bins.z_center, st.kick, bins.dz, pz_scale)
      else
        zbar[i] = zero(eltype(zbar))
      end
    end
  end

  gbar = KernelAbstractions.allocate(backend, promote_type(T, eltype(st.I_int), typeof(st.coef)), n_bin)
  csr_convolve_adjoint_kernel!(backend)(gbar, st.I_int, kbar, st.coef; ndrange = n_bin)
  Qbar = similar(gbar)
  csr_edge_gradient_adjoint_kernel!(backend)(Qbar, gbar, bins.dz; ndrange = n_bin)
  for image in st.images
    csr_image_convolve_adjoint_kernel!(backend)(Qbar, image, kbar, st.coef; ndrange = n_bin)
  end

  z_min = bins.z0_edge[1]
  qbar = KernelAbstractions.allocate(backend, eltype(Qbar), length(q))
  csr_deposit_adjoint_kernel!(backend)(qbar, zbar, Qbar, z, q, alive, z_min, bins.dz,
                                       params.particle_bin_span * bins.dz, params.particle_bin_span;
                                       ndrange = length(z))
  KernelAbstractions.synchronize(backend)
  return copy(ybar), zbar, qbar, kbar
end

# Geometry as a flat vector of its numbers, for derivatives with respect to every field of a CSRGeometry.

"Number of numbers per `CSRElementInfo` in `geometry_to_vector`: `L, g, s_start`, four floor positions, `theta_chord, L_chord`, spline, `dL_s`."
const N_ELE_PARAMS = 26

"""
    geometry_to_vector(geom) -> Vector

All numbers of `geom`: `start` (`r`, `theta`), then per element `L, g, s_start, ref_floor0, ref_floor1, floor0,
floor1` (each `r`, `theta`), `theta_chord, L_chord, spline[1:4], dL_s`. Length `4 + 26 n_ele`. Inverse:
`geometry_from_vector`.
"""
function geometry_to_vector(geom::CSRGeometry{T}) where {T}
  v = Vector{T}(undef, 4 + N_ELE_PARAMS * length(geom.ele))
  push_floor!(v, i, f) = (v[i:i+2] .= f.r; v[i+3] = f.theta; i + 4)
  i = push_floor!(v, 1, geom.start)
  for e in geom.ele
    v[i] = e.L; v[i+1] = e.g; v[i+2] = e.s_start
    i += 3
    for f in (e.ref_floor0, e.ref_floor1, e.floor0, e.floor1)
      i = push_floor!(v, i, f)
    end
    v[i] = e.theta_chord; v[i+1] = e.L_chord
    v[i+2:i+5] .= e.spline
    v[i+6] = e.dL_s
    i += 7
  end
  return v
end

"""
    geometry_from_vector(v, n_ele, bmad_compat) -> CSRGeometry

`CSRGeometry{eltype(v)}` with `n_ele` elements from the numbers `v` in the order of `geometry_to_vector`.
"""
function geometry_from_vector(v::AbstractVector{T}, n_ele::Integer, bmad_compat::Bool) where {T}
  length(v) == 4 + N_ELE_PARAMS * n_ele || error("geometry_from_vector: length(v) must be 4 + $N_ELE_PARAMS n_ele")
  floor_at(i) = FloorPosition{T}(SVector(v[i], v[i+1], v[i+2]), v[i+3])
  ele = Vector{CSRElementInfo{T}}(undef, n_ele)
  for j in 1:n_ele
    i = 5 + N_ELE_PARAMS * (j - 1)
    ele[j] = CSRElementInfo{T}(v[i], v[i+1], v[i+2], floor_at(i + 3), floor_at(i + 7), floor_at(i + 11), floor_at(i + 15),
                               v[i+19], v[i+20], SVector(v[i+21], v[i+22], v[i+23], v[i+24]), v[i+25])
  end
  return CSRGeometry{T}(floor_at(1), ele, bmad_compat)
end

"""
    geometry_active_indices(geom, ix_ele_kick, s_kick, gamma, params, dz; kick_factor = 1.0) -> Vector{Int}

Indices (in the order of `geometry_to_vector`) of the geometry numbers the kick tables can depend on: all fields of
the kick element; `floor0`, `theta_chord`, `L_chord`, `spline`, `dL_s` of the elements the source searches walk
through (found by searching at the extreme lags: free space `n_bin dz`, images `±(n_bin + 1) dz` per layer; the kick
element's predecessor for a kick at its entrance); with a source on the upstream infinite drift, `start` and
`L_chord`, `dL_s` of all upstream elements; with one past the lattice end, `L_chord`, `dL_s` of the downstream
elements and `floor1` of the last. Assumes the source element varies monotonically with the lag between the
extreme lags (true without patches that reverse the centroid).
"""
function geometry_active_indices(geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, params::CSRParams, dz;
                                 kick_factor = 1.0)
  n = length(geom.ele)
  lo = hi = ix_ele_kick
  up = down = false
  function visit(kp, u)
    k, _ = s_source_calc(CSRKick1(geom, kp; dz_particles = u), geom, kp)
    ix = k.ix_ele_source
    ix == 0 ? (up = true; lo = 1) : (lo = min(lo, ix))
    ix > n ? (down = true; hi = n) : (hi = max(hi, ix))
    return nothing
  end
  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)
  visit(kp, params.n_bin * dz)
  primal(kp.s_chord_kick) == 0 && ix_ele_kick > 1 && (lo = min(lo, ix_ele_kick - 1))
  for ns in 1:params.n_shield_images
    kp_ns = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source = ns * params.beam_chamber_height, kick_factor)
    visit(kp_ns, (params.n_bin + 1) * dz)
    visit(kp_ns, -(params.n_bin + 1) * dz)
  end

  base(j) = 4 + N_ELE_PARAMS * (j - 1)   # element j's numbers are base(j) .+ (1:26)
  idx = Int[]
  up && append!(idx, 1:4)
  for j in 1:n
    b = base(j)
    if j == ix_ele_kick
      append!(idx, b .+ (1:N_ELE_PARAMS))
    elseif lo <= j <= hi
      append!(idx, b .+ [12:15; 20:26])        # floor0, theta_chord, L_chord, spline, dL_s
    elseif (up && j < lo) || (down && j > hi)
      append!(idx, b .+ [21, 26])              # L_chord, dL_s
    end
    down && j == n && append!(idx, b .+ (16:19))   # floor1 of the last element
  end
  return sort!(unique!(idx))
end

"""
    csr_kick_param_adjoint(kbar, bins, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0,
                           sparse = true) -> (geombar, s_kickbar, gammabar)

Cotangents of the geometry (as a vector in the order of `geometry_to_vector`), `s_kick` and `gamma` for the
cotangent `kbar` of the kick per bin (from `csr_kick_adjoint`), with the bins held constant: the gradient of
`kbar ⋅ kick` by ForwardDiff through the kick tables (`csr_bins_kick` on the CPU; source roots by implicit
differentiation). With `sparse = true` only the geometry numbers of `geometry_active_indices` are seeded (the others
get zero); `sparse = false` seeds all `4 + 26 n_ele`. Cost: about one table evaluation per seeded number, in dual
arithmetic (ForwardDiff chunks).
"""
function csr_kick_param_adjoint(kbar::AbstractVector, bins::CSRBins, geom::CSRGeometry, ix_ele_kick::Integer, s_kick,
                                gamma, species::Species, params::CSRParams; kick_factor = 1.0, sparse::Bool = true)
  cpu(A) = A isa Array ? A : Array(A)
  bins_cpu = CSRBins(primal(bins.dz), cpu(bins.z0_edge), cpu(bins.z_center), primal.(cpu(bins.charge)),
                     cpu(bins.n_particle), primal.(cpu(bins.edge_dcharge_density_dz)))
  kb = primal.(cpu(kbar))
  n_ele = length(geom.ele)
  n_g = 4 + N_ELE_PARAMS * n_ele
  p0 = [primal.(geometry_to_vector(geom)); primal(s_kick); primal(gamma)]
  idx = sparse ? [geometry_active_indices(geom, ix_ele_kick, primal(s_kick), primal(gamma), params, bins_cpu.dz;
                                          kick_factor = primal(kick_factor)); n_g + 1; n_g + 2] : collect(1:n_g+2)
  function phi(x)
    p = Vector{eltype(x)}(p0)
    p[idx] = x
    g = geometry_from_vector(view(p, 1:n_g), n_ele, geom.bmad_compat)
    st = csr_bins_kick(bins_cpu, CPU(), g, ix_ele_kick, p[n_g+1], p[n_g+2], species, params; kick_factor)
    return sum(kb .* st.kick)
  end
  grad = zeros(n_g + 2)
  grad[idx] = ForwardDiff.gradient(phi, p0[idx])
  return grad[1:n_g], grad[n_g+1], grad[n_g+2]
end
