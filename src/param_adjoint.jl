# Pullback of one CSR kick with respect to the geometry, s_kick and gamma (host side, CPU): the kick tables are
# linear combinations of source evaluations at fixed lags; each evaluation is differentiated with local Duals
# (PLAN.md, D30).

# Geometry as a flat vector of its numbers, for derivatives with respect to every field of a CSRGeometry.

"Number of numbers per `CSRElementInfo` in `geometry_to_vector`: `L, g, s_start`, four floor positions, `theta_chord, L_chord`, spline, `dL_s`."
const N_ELE_PARAMS = 26

"""
    element_numbers(e) -> SVector{26}

The numbers of one `CSRElementInfo` in the order of `geometry_to_vector`: `L, g, s_start, ref_floor0, ref_floor1,
floor0, floor1` (each `r`, `theta`), `theta_chord, L_chord, spline[1:4], dL_s`. Inverse: `element_from_numbers`.
"""
function element_numbers(e::CSRElementInfo)
  f(x) = (x.r[1], x.r[2], x.r[3], x.theta)
  return SVector(e.L, e.g, e.s_start, f(e.ref_floor0)..., f(e.ref_floor1)..., f(e.floor0)..., f(e.floor1)...,
                 e.theta_chord, e.L_chord, e.spline..., e.dL_s)
end

"""
    element_from_numbers(v::SVector{26}) -> CSRElementInfo

`CSRElementInfo{eltype(v)}` from the numbers of `element_numbers`.
"""
function element_from_numbers(v::SVector{N_ELE_PARAMS,T}) where {T}
  fl(i) = FloorPosition{T}(SVector(v[i], v[i+1], v[i+2]), v[i+3])
  return CSRElementInfo{T}(v[1], v[2], v[3], fl(4), fl(8), fl(12), fl(16), v[20], v[21],
                           SVector(v[22], v[23], v[24], v[25]), v[26])
end

"""
    geometry_to_vector(geom) -> Vector

All numbers of `geom`: `start` (`r`, `theta`), then per element its `element_numbers`. Length `4 + 26 n_ele`.
Inverse: `geometry_from_vector`.
"""
function geometry_to_vector(geom::CSRGeometry{T}) where {T}
  v = Vector{T}(undef, 4 + N_ELE_PARAMS * length(geom.ele))
  v[1:3] .= geom.start.r
  v[4] = geom.start.theta
  for (j, e) in enumerate(geom.ele)
    v[geometry_base(j) .+ (1:N_ELE_PARAMS)] .= element_numbers(e)
  end
  return v
end

"""
    geometry_from_vector(v, n_ele, bmad_compat) -> CSRGeometry

`CSRGeometry{eltype(v)}` with `n_ele` elements from the numbers `v` in the order of `geometry_to_vector`.
"""
function geometry_from_vector(v::AbstractVector{T}, n_ele::Integer, bmad_compat::Bool) where {T}
  length(v) == 4 + N_ELE_PARAMS * n_ele || error("geometry_from_vector: length(v) must be 4 + $N_ELE_PARAMS n_ele")
  ele = [element_from_numbers(SVector{N_ELE_PARAMS,T}(view(v, geometry_base(j) .+ (1:N_ELE_PARAMS)))) for j in 1:n_ele]
  return CSRGeometry{T}(FloorPosition{T}(SVector(v[1], v[2], v[3]), v[4]), ele, bmad_compat)
end

"Offset in `geometry_to_vector` before element `j`'s numbers (they are at `geometry_base(j) .+ (1:26)`)."
geometry_base(j) = 4 + N_ELE_PARAMS * (j - 1)

"""
    widen_source_range(lo, hi, up, down, ix, n) -> (lo, hi, up, down)

Element range `lo:hi` and the upstream/downstream infinite-drift flags after a source search ended in element `ix`
(`0`: upstream of the lattice, `> n`: past its end), for `geometry_active_indices`.
"""
widen_source_range(lo, hi, up, down, ix, n) =
  (ix == 0 ? 1 : min(lo, ix), ix > n ? n : max(hi, ix), up || ix == 0, down || ix > n)

"""
    geometry_active_indices(geom, ix_ele_kick, s_kick, gamma, params, dz; kick_factor = 1.0,
                            lag = params.n_bin * dz, n_images = params.n_shield_images) -> Vector{Int}

Indices (in the order of `geometry_to_vector`) of the geometry numbers the kick tables can depend on, for free-space
lags up to `lag` and `n_images` image layers: all fields of the kick element; `floor0`, `theta_chord`, `L_chord`,
`spline`, `dL_s` of the elements the source searches walk through (found by searching at the extreme lags: free
space `lag`, images `±(n_bin + 1) dz` per layer; the kick element's predecessor for a kick at its entrance); with a
source on the upstream infinite drift, `start` and `L_chord`, `dL_s` of all upstream elements; with one past the
lattice end, `L_chord`, `dL_s` of the downstream elements and `floor1` of the last. Assumes the source element
varies monotonically with the lag between the extreme lags (true without patches that reverse the centroid).
`csr_kick_param_adjoint` uses it for the first bin (`lag = dz`, `n_images = 0`).
"""
function geometry_active_indices(geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, params::CSRParams, dz;
                                 kick_factor = 1.0, lag = params.n_bin * dz, n_images = params.n_shield_images)
  n = length(geom.ele)
  lo = hi = ix_ele_kick
  up = down = false
  src_ix(kp, u) = s_source_calc(CSRKick1(geom, kp; dz_particles = u), geom, kp)[1].ix_ele_source
  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)
  lo, hi, up, down = widen_source_range(lo, hi, up, down, src_ix(kp, lag), n)
  primal(kp.s_chord_kick) == 0 && ix_ele_kick > 1 && (lo = min(lo, ix_ele_kick - 1))
  for ns in 1:n_images
    kp_ns = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source = ns * params.beam_chamber_height, kick_factor)
    lo, hi, up, down = widen_source_range(lo, hi, up, down, src_ix(kp_ns, (params.n_bin + 1) * dz), n)
    lo, hi, up, down = widen_source_range(lo, hi, up, down, src_ix(kp_ns, -(params.n_bin + 1) * dz), n)
  end

  idx = Int[]
  up && append!(idx, 1:4)
  for j in 1:n
    b = geometry_base(j)
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

# Local derivatives of one source evaluation. Slots of the local Dual: s_kick, gamma, the kick element's numbers that
# the tables use, the source element's, the lattice start or the last element's floor1 (sources on the infinite
# drifts), and offsets of the middle-element path-length sums at the root (root_tangent) and at the evaluated source.

"Tag of the local `Dual`s of `csr_kick_param_adjoint`."
struct LocalTag end

"Number of local partials per source evaluation in `csr_kick_param_adjoint` (see the slot constants)."
const N_LOCAL = 33

"Local `Dual` of `csr_kick_param_adjoint`."
const LocalDual = Dual{LocalTag,Float64,N_LOCAL}

"Local slot of `s_kick`."
const SLOT_S_KICK = 1
"Local slot of `gamma`."
const SLOT_GAMMA = 2
"Local slots of the kick element's numbers (`KICK_OFFSETS`)."
const SLOTS_KICK = 3:18
"Local slots of the source element's numbers (`SOURCE_OFFSETS`)."
const SLOTS_SOURCE = 19:27
"Local slots of the lattice start (upstream drift) or the last element's `floor1` (downstream drift)."
const SLOTS_FLOOR = 28:31
"Local slot of the middle-element sum offset in the root's implicit derivative."
const SLOT_MID_ROOT = 32
"Local slot of the middle-element sum offset at the evaluated source."
const SLOT_MID = 33

"Numbers of the kick element the tables use (offsets in `element_numbers`): `L, g, ref_floor0, floor0, theta_chord, L_chord, spline`."
const KICK_OFFSETS = (1, 2, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25)
"Numbers of a source element the tables use: `floor0.r, theta_chord, L_chord, spline`."
const SOURCE_OFFSETS = (12, 13, 14, 20, 21, 22, 23, 24, 25)
"Offsets of `floor1` in `element_numbers`."
const FLOOR1_OFFSETS = (16, 17, 18, 19)

"Tag of the `Dual` of `middle_excess` on a `LocalDual` geometry."
struct MidTag end

"""
    middle_excess(acc, geom::CSRGeometry{LocalDual}, r, theta_L)

On the local-Dual geometry of `csr_kick_param_adjoint` the elements between source and kick element carry no
partials (only the kick and source elements and the lattice ends are seeded), so the sum depends on the parameters
only through `theta_L`: first order exactly `M(θ₀) + M′(θ₀)(theta_L - θ₀)`, with `M` and `M′` in `Float64` (one
`Dual` of width 1) instead of in `theta_L`'s number type. Exact for every first-order partial, also for a nested
`theta_L` (`root_tangent`, which uses only the primal of `∂ddz/∂s`); the mixed second-order partials are dropped.
"""
function middle_excess(acc, geom::CSRGeometry{LocalDual}, r, theta_L)
  isempty(r) && return acc
  theta0 = primal(theta_L)
  t = Dual{MidTag}(theta0, 1.0)
  M = zero(t)
  for i in r
    ce = geom.ele[i]
    M += path_excess(0.0, primal(ce.L_chord), primal.(ce.spline), modulo2(t - primal(ce.theta_chord), pi / 2);
                     bmad_compat = geom.bmad_compat)
  end
  return acc + (value(M) + partials(M, 1) * (theta_L - theta0))
end

"`x` as a `LocalDual` with zero partials."
nodual(x) = LocalDual(x, zero(Partials{N_LOCAL,Float64}))

"`x` as a `LocalDual` with partial 1 in `slot`."
seeded(x, slot) = LocalDual(x, Partials(ntuple(j -> Float64(j == slot), N_LOCAL)))

"""
    seed_numbers(v, offsets, slots, w = nodual.(v))

`w` with the numbers `v[offsets]` seeded in the local `slots`.
"""
function seed_numbers(v::SVector{N}, offsets, slots, w = nodual.(v)) where {N}
  return SVector(ntuple(Val(N)) do i
    j = findfirst(==(i), offsets)
    j === nothing ? w[i] : seeded(v[i], slots[j])
  end)
end

"""
    ParamAdjointWork(geom0, ix_ele_kick, n_grad)
    ParamAdjointWork(wk)

Per-task state of `csr_kick_param_adjoint`: the primal geometry, the local-Dual geometry (`geom_d`, and `geom_up`
with the start seeded; both share `ele`, where the source element is swapped in per evaluation from `ele_src`, or
the last element from `ele_end`; `ele_d0` holds the unseeded copies, kick element seeded) and the gradient
accumulator. `ParamAdjointWork(wk)`: a new task's state, sharing the seeded elements of `wk`.
"""
struct ParamAdjointWork
  geom0::CSRGeometry{Float64}
  ele_d0::Vector{CSRElementInfo{LocalDual}}
  ele_src::Vector{CSRElementInfo{LocalDual}}
  ele_end::CSRElementInfo{LocalDual}
  ele::Vector{CSRElementInfo{LocalDual}}
  geom_d::CSRGeometry{LocalDual}
  geom_up::CSRGeometry{LocalDual}
  ix_ele_kick::Int
  grad::Vector{Float64}
end

function ParamAdjointWork(geom0::CSRGeometry{Float64}, ix_ele_kick::Integer, n_grad::Integer)
  ele_d0 = map(enumerate(geom0.ele)) do (j, e)
    v = element_numbers(e)
    element_from_numbers(j == ix_ele_kick ? seed_numbers(v, KICK_OFFSETS, SLOTS_KICK) : nodual.(v))
  end
  ele_src = [element_from_numbers(seed_numbers(element_numbers(e), SOURCE_OFFSETS, SLOTS_SOURCE)) for e in geom0.ele]
  n = length(geom0.ele)
  ele_end = element_from_numbers(seed_numbers(element_numbers(geom0.ele[n]), FLOOR1_OFFSETS, SLOTS_FLOOR,
                                              element_numbers(ele_d0[n])))
  ele = copy(ele_d0)
  s = geom0.start
  start_d = FloorPosition{LocalDual}(nodual.(s.r), nodual(s.theta))
  start_up = FloorPosition{LocalDual}(SVector(ntuple(i -> seeded(s.r[i], SLOTS_FLOOR[i]), 3)), seeded(s.theta, SLOTS_FLOOR[4]))
  return ParamAdjointWork(geom0, ele_d0, ele_src, ele_end, ele, CSRGeometry{LocalDual}(start_d, ele, geom0.bmad_compat),
                          CSRGeometry{LocalDual}(start_up, ele, geom0.bmad_compat), ix_ele_kick, zeros(n_grad))
end

function ParamAdjointWork(wk::ParamAdjointWork)
  ele = copy(wk.ele_d0)
  bc = wk.geom0.bmad_compat
  return ParamAdjointWork(wk.geom0, wk.ele_d0, wk.ele_src, wk.ele_end, ele, CSRGeometry{LocalDual}(wk.geom_d.start, ele, bc),
                          CSRGeometry{LocalDual}(wk.geom_up.start, ele, bc), wk.ix_ele_kick, zeros(length(wk.grad)))
end

"""
    scatter_partials!(grad, base, offsets, slots, p, w)

`grad[base + offsets[i]] += w * p[slots[i]]`.
"""
function scatter_partials!(grad, base, offsets, slots, p, w)
  for (o, s) in zip(offsets, slots)
    grad[base+o] += w * p[s]
  end
  return grad
end

"""
    middle_adjoint!(grad, c, ix_source, ix_kick, theta_L, geom)

Cotangent `c` of the middle-element path-length sum of a source in element `ix_source` (`ddz_calc_csr`'s `dmid`)
passed to the elements strictly between source and kick element: `∂path_excess/∂(L_chord, spline, theta_chord)`
at `theta_L`; on the infinite drifts, the sums of `dL_s + L_chord` (upstream or downstream of the kick element).
"""
function middle_adjoint!(grad, c, ix_source, ix_kick, theta_L, geom::CSRGeometry{Float64})
  c == 0 && return grad
  n = length(geom.ele)
  if ix_source == 0 || ix_source > n
    for i in (ix_source == 0 ? (1:ix_kick-1) : (ix_kick+1:n))
      b = geometry_base(i)
      grad[b+21] += c
      grad[b+26] += c
    end
    return grad
  end
  bc = geom.bmad_compat
  for i in (ix_source < ix_kick ? (ix_source+1:ix_kick-1) : (ix_kick+1:ix_source-1))
    e = geom.ele[i]
    pe(x) = path_excess(0.0, x[1], SVector(x[2], x[3], x[4], x[5]), modulo2(theta_L - x[6], pi / 2); bmad_compat = bc)
    gr = ForwardDiff.gradient(pe, SVector(e.L_chord, e.spline..., e.theta_chord))
    b = geometry_base(i)
    grad[b+20] += c * gr[6]
    grad[b+21] += c * gr[1]
    for m in 1:4
      grad[b+21+m] += c * gr[1+m]
    end
  end
  return grad
end

"""
    node_adjoint!(wk, image, w, u, k0, kp0, kp_d) -> k0

Add `w` times the gradient of one source evaluation at lag `u` to `wk.grad`: `I_csr_value` (free space) or
`image_charge_kick` (`image = true`) with kick point `kp0` (primal) / `kp_d` (local Duals). The source is searched in
`Float64` (`s_source_walk`, warm-started from `k0`) and evaluated once in local Duals (`s_source_eval`; the root's
derivatives by implicit differentiation). Returns the primal work struct for the next warm start.
"""
function node_adjoint!(wk::ParamAdjointWork, image::Bool, w, u, k0, kp0, kp_d)
  geom0 = wk.geom0
  n = length(geom0.ele)
  ik = wk.ix_ele_kick
  k0, _, o = s_source_walk(with_fields(k0; dz_particles = u), geom0, kp0)
  w == 0 && return k0
  js = o.ix
  in_lattice = 1 <= js <= n && js != ik
  g = js == 0 ? wk.geom_up : wk.geom_d
  in_lattice && (wk.ele[js] = wk.ele_src[js])
  js > n && (wk.ele[n] = wk.ele_end)
  kd = CSRKick1{LocalDual}(; ix_ele_source = js, dz_particles = nodual(u))
  kd, _ = s_source_eval(o, kd, g, kp_d, (seeded(0.0, SLOT_MID_ROOT), seeded(0.0, SLOT_MID)))
  F = image ? image_charge_kick(kd, g, kp_d) : I_csr_value(kd, kp_d)
  in_lattice && (wk.ele[js] = wk.ele_d0[js])
  js > n && (wk.ele[n] = wk.ele_d0[n])

  p = partials(F)
  grad = wk.grad
  n_g = length(grad) - 2
  grad[n_g+1] += w * p[SLOT_S_KICK]
  grad[n_g+2] += w * p[SLOT_GAMMA]
  scatter_partials!(grad, geometry_base(ik), KICK_OFFSETS, SLOTS_KICK, p, w)
  if in_lattice
    scatter_partials!(grad, geometry_base(js), SOURCE_OFFSETS, SLOTS_SOURCE, p, w)
  elseif js == 0
    scatter_partials!(grad, 0, (1, 2, 3, 4), SLOTS_FLOOR, p, w)
  elseif js > n
    scatter_partials!(grad, geometry_base(n), FLOOR1_OFFSETS, SLOTS_FLOOR, p, w)
  end
  theta_L = value(kd.theta_L)
  # In compat the source is evaluated at zbrent's last point (V7), not at the root of the implicit derivative.
  theta_root = o.code == SRC_ROOT && geom0.bmad_compat ? first(ddz_calc_csr(k0, o.s_root, geom0, kp0)).theta_L : theta_L
  middle_adjoint!(grad, w * p[SLOT_MID], js, ik, theta_L, geom0)
  middle_adjoint!(grad, w * p[SLOT_MID_ROOT], js, ik, theta_root, geom0)
  return k0
end

"""
    nodes_adjoint!(grad, geom0, ix_ele_kick, nodes, image, s_kick, gamma; y_source = 0.0, kick_factor = 1.0)

`node_adjoint!` for the `(lag, weight)` pairs in `nodes` (sorted by lag, split into blocks on threads), added to
`grad`.
"""
function nodes_adjoint!(grad, geom0, ix_ele_kick, nodes, image, s_kick, gamma; y_source = 0.0, kick_factor = 1.0)
  isempty(nodes) && return grad
  sort!(nodes)
  kp0 = CSRKickPoint(geom0, ix_ele_kick, s_kick, gamma; y_source, kick_factor)
  proto = ParamAdjointWork(geom0, ix_ele_kick, length(grad))
  blocks = Iterators.partition(nodes, cld(length(nodes), min(length(nodes), 4 * Threads.nthreads())))
  tasks = map(blocks) do block
    Threads.@spawn begin
      wk = ParamAdjointWork(proto)
      kp_d = CSRKickPoint(wk.geom_d, ix_ele_kick, seeded(s_kick, SLOT_S_KICK), seeded(gamma, SLOT_GAMMA);
                          y_source, kick_factor)
      k0 = CSRKick1(; ix_ele_source = ix_ele_kick)
      for (u, w) in block
        k0 = node_adjoint!(wk, image, w, u, k0, kp0, kp_d)
      end
      wk.grad
    end
  end
  for t in tasks
    grad .+= fetch(t)
  end
  return grad
end

"""
    first_bin_adjoint!(grad, w, geom0, ix_ele_kick, s_kick, gamma, params, dz, kick_factor)

Add `w` times the gradient of the first bin's integral `I_int[1]` (`I_csr_calc` with `i_bin = 1`; its quadrature
nodes move with the parameters) by ForwardDiff with the numbers of `geometry_active_indices` for lag `dz` seeded
(sources searched in `geom0`, `PrimalSearch`).
"""
function first_bin_adjoint!(grad, w, geom0, ix_ele_kick, s_kick, gamma, params, dz, kick_factor)
  w == 0 && return grad
  n = length(geom0.ele)
  n_g = 4 + N_ELE_PARAMS * n
  idx = [geometry_active_indices(geom0, ix_ele_kick, s_kick, gamma, params, dz; kick_factor, lag = dz, n_images = 0);
         n_g + 1; n_g + 2]
  p0 = [geometry_to_vector(geom0); s_kick; gamma]
  search = PrimalSearch(geom0, CSRKickPoint(geom0, ix_ele_kick, s_kick, gamma; kick_factor))
  function I_int1(x)
    p = Vector{eltype(x)}(p0)
    p[idx] = x
    g = geometry_from_vector(view(p, 1:n_g), n, geom0.bmad_compat)
    kp = CSRKickPoint(g, ix_ele_kick, p[n_g+1], p[n_g+2]; kick_factor)
    k, _ = s_source_calc(CSRKick1(g, kp; dz_particles = dz), g, kp, search)
    return I_csr_calc(k, 1, dz, g, kp, 0.0, search).I_int_csr
  end
  grad[idx] .+= w .* ForwardDiff.gradient(I_int1, p0[idx])
  return grad
end

"""
    hat_nodes(geom0, kp0, dz, n_bin, Wimg) -> Vector{Pair{Float64,Float64}}

Lags and weights of the hat-weighted image table's kernel evaluations (`csr_image_table` with `bin_integrate`) in
`Σ_l Wimg[l] image[l]`: the accepted adaptive-Simpson intervals of each lag interval (Boole's rule on 5 nodes),
combined into the hat moments and the table as in `csr_image_hat_combine_kernel!`. Coincident nodes are merged.
"""
function hat_nodes(geom0::CSRGeometry, kp0::CSRKickPoint, dz, n_bin, Wimg)
  n = 2n_bin + 1
  image0, _ = csr_image_table(geom0, kp0, dz, n_bin, CPU())
  u_min = image_u_min(kp0)
  tol = image_hat_tol(geom0, kp0, image0, dz, 1e-6, u_min)
  W(j) = 1 <= j <= n ? Wimg[j] : 0.0
  acc = Dict{Float64,Float64}()
  k = CSRKick1(geom0, kp0)
  for m in -n_bin-1:n_bin
    jr = m + n_bin + 2            # rise[jr] enters image[jr], fall[jr] enters image[jr - 1]
    Wr, Wf = W(jr), W(jr - 1)
    (Wr == 0 && Wf == 0) && continue
    u0 = m * dz
    function leaf(a, b)
      mid = (a + b) / 2
      h = b - a
      for (u, wb) in ((a, 7h / 90), ((a + mid) / 2, 32h / 90), (mid, 12h / 90), ((mid + b) / 2, 32h / 90), (b, 7h / 90))
        t = (u - u0) / dz
        acc[u] = get(acc, u, 0.0) + wb * (Wr * t + Wf * (1 - t)) / dz
      end
    end
    k, _ = image_interval_moments(k, geom0, kp0, m, dz, tol, u_min, leaf)
  end
  return collect(acc)
end

"""
    csr_kick_param_adjoint(kbar, bins, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0)
        -> (geombar, s_kickbar, gammabar)

Cotangents of the geometry (as a vector in the order of `geometry_to_vector`), `s_kick` and `gamma` for the
cotangent `kbar` of the kick per bin (from `csr_kick_adjoint`), with the bins held constant: the gradient of
`kbar ⋅ kick`, on the CPU.

The kick is `coef` times the tables convolved with the bins, and each table entry is a fixed linear combination of
source evaluations at fixed lags (bin quadrature nodes, trapezoid, image lags, or the accepted intervals of the
adaptive hat quadrature). So the gradient is the coefficient's part plus, per evaluation, its weight (from the
convolutions' transposes) times the evaluation's gradient. Each evaluation searches its source in `Float64` and is
then evaluated once with local `Dual`s (`N_LOCAL` partials: `s_kick`, `gamma`, the kick and source elements' numbers,
the lattice start or end, and offsets of the path-length sums over the elements in between, whose cotangents
`middle_adjoint!` passes to those elements). The first bin's integral, whose quadrature nodes move with the
parameters, is differentiated by ForwardDiff over its `geometry_active_indices` (`first_bin_adjoint!`). Exact up to
round-off, like ForwardDiff through `csr_bins_kick`; cost a few primal kicks.
"""
function csr_kick_param_adjoint(kbar::AbstractVector, bins::CSRBins, geom::CSRGeometry, ix_ele_kick::Integer, s_kick,
                                gamma, species::Species, params::CSRParams; kick_factor = 1.0)
  cpu(A) = A isa Array ? A : Array(A)
  bins0 = CSRBins(primal(bins.dz), cpu(bins.z0_edge), cpu(bins.z_center), primal.(cpu(bins.charge)),
                  cpu(bins.n_particle), primal.(cpu(bins.edge_dcharge_density_dz)))
  kb = primal.(cpu(kbar))
  n_ele = length(geom.ele)
  geom0 = primal(geom)
  s_kick, gamma, kick_factor = primal(s_kick), primal(gamma), primal(kick_factor)
  ik = ix_ele_kick
  n_bin = params.n_bin
  dz = bins0.dz
  n_g = 4 + N_ELE_PARAMS * n_ele
  grad = zeros(n_g + 2)

  # Coefficient: the kick is linear in coef.
  st = csr_bins_kick(bins0, CPU(), geom0, ik, s_kick, gamma, species, params; kick_factor)
  coef = st.coef
  e = geom0.ele[ik]
  dcoef = ForwardDiff.gradient(x -> csr_kick_coef(species, x[1], csr_n_step(x[2], params)[2] * (x[3] / x[2]);
                                                  params.bmad_compat),
                               SVector(gamma, e.L, e.L_chord))
  c = sum(kb .* st.kick) / coef
  grad[n_g+2] += c * dcoef[1]
  grad[geometry_base(ik)+1] += c * dcoef[2]
  grad[geometry_base(ik)+21] += c * dcoef[3]

  # Cotangents of the tables (transposes of the convolutions)
  edge = bins0.edge_dcharge_density_dz
  Ibar = [coef * sum(kb[i] * edge[i+1-m] for i in m:n_bin) for m in 1:n_bin]
  Wimg = zeros(2n_bin + 1)
  for i in 1:n_bin, j in 1:n_bin
    Wimg[i-j+n_bin+1] += coef * kb[i] * bins0.charge[j]
  end

  # Free space: I_int[1] as a whole, the other bins by their evaluations
  first_bin_adjoint!(grad, Ibar[1], geom0, ik, s_kick, gamma, params, dz, kick_factor)
  nodes = Tuple{Float64,Float64}[]
  if params.all_bin_quadrature && !params.bmad_compat
    n_q = N_BIN_QUAD
    for i in 2:n_bin
      t0 = cbrt((i - 1) * dz)
      h = (cbrt(i * dz) - t0) / n_q
      for j in 0:n_q
        t = j == n_q ? cbrt(i * dz) : t0 + j * h
        w = (j == 0 || j == n_q) ? 1 : (isodd(j) ? 4 : 2)
        push!(nodes, (t^3, Ibar[i] * w * 3t^2 * h / 3))
      end
    end
  else
    for i in 1:n_bin
      push!(nodes, (i * dz, ((i >= 2 ? Ibar[i] : 0.0) + (i < n_bin ? Ibar[i+1] : 0.0)) * dz / 2))
    end
  end
  nodes_adjoint!(grad, geom0, ik, nodes, false, s_kick, gamma; kick_factor)

  # Image layers (as csr_bins_kick)
  for ns in 1:params.n_shield_images
    y_source = ns * params.beam_chamber_height
    kf = (-1)^ns * 2 * image_layer_weight(ns, params) * kick_factor
    if params.bmad_compat
      nodes = [((j - n_bin - 1) * dz, Wimg[j]) for j in 1:2n_bin+1]
    else
      kp0 = CSRKickPoint(geom0, ik, s_kick, gamma; y_source, kick_factor = kf)
      nodes = hat_nodes(geom0, kp0, dz, n_bin, Wimg)
    end
    nodes_adjoint!(grad, geom0, ik, nodes, true, s_kick, gamma; y_source, kick_factor = kf)
  end
  return grad[1:n_g], grad[n_g+1], grad[n_g+2]
end
