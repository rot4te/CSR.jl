# CSR through a Beamline as a BeamTracking effect: setup (reference
# survey, centroid orbit, geometry) and one kick per slice entrance.

"""
    csr_ref_curvature(bp, ix)

Reference curvature [1/m] in the horizontal plane of element `ix`
(`g_ref`, sign from `tilt_ref ∈ {0, π}`), scalarized. Bends out of
the horizontal plane are not supported by the planar geometry
(error).

# Arguments
- `bp`: the element's `BendParams` (evaluated), or `nothing`.
- `ix`: element index, for the error message.
"""
function csr_ref_curvature(bp, ix)
  isactive(bp) || return 0.0
  g = scalarize(float(bp.g_ref))
  g == 0 && return 0.0
  tilt = scalarize(bp.tilt_ref)
  abs(sin(tilt)) < 1e-12 ||
    error("CSREffect: element $ix bends out of the horizontal plane (tilt_ref = $(bp.tilt_ref))")
  return g * sign(cos(tilt))
end

"""
    csr_ref_exit(f, L, g, pp)

Reference floor position at the exit of an element entered at `f`.

# Arguments
- `f`: `FloorPosition` at the entrance.
- `L`: element length [m].
- `g`: reference curvature [1/m] (`csr_ref_curvature`).
- `pp`: the element's `PatchParams` (evaluated), or `nothing`.
"""
function csr_ref_exit(f::FloorPosition, L, g, pp)
  isactive(pp) || return CSR.advance_floor(f, L, g)
  rot = scalarize.((pp.dx_rot, pp.dy_rot, pp.dz_rot))
  ez = quat_rotate((0, 0, 1), rot_quaternion(rot...))
  dr = scalarize.((pp.dx, pp.dy, pp.dz))
  return CSR.advance_floor_patch(f, dr, ez)
end

"""
    macro_weights(bunch)

Real particles per macroparticle as a vector (`weight = nothing`
means 1).
"""
function macro_weights(bunch)
  wt = bunch.coords.weight
  N = size(bunch.coords.v, 1)
  isnothing(wt) && return ones(N)
  wt isa Number && return fill(float(wt), N)
  return collect(float.(wt))
end

"""
    bunch_centroid(bunch, w, alive)

Charge-weighted mean of the live particles' coordinates.

# Arguments
- `bunch`: the `Bunch`.
- `w`: weights (`macro_weights`).
- `alive`: live-particle mask.
"""
function bunch_centroid(bunch, w, alive)
  v = bunch.coords.v
  wa = w .* alive
  scalarize(sum(wa)) > 0 || error("CSREffect: no live particles")
  return vec(sum(v .* wa; dims = 1)) ./ sum(wa)
end

"""
    centroid_bunch(bunch, c0)

Single-particle bunch at `c0` for the centroid orbit, with the
species, reference and weight type of `bunch` (weight 1).
"""
function centroid_bunch(bunch, c0)
  wt = bunch.coords.weight
  w1 = isnothing(wt) ? nothing : wt isa Number ? one(wt) :
       fill!(similar(wt, 1), one(eltype(wt)))
  return Bunch(reshape(c0, 1, 6), nothing, w1;
               species = bunch.species,
               p_over_q_ref = bunch.p_over_q_ref,
               t_ref = bunch.t_ref)
end

"""
    BeamTracking.effect_setup(e::CSREffect, bunch, bl) -> state

Reference survey of `bl.line` (with patches), centroid orbit (one
particle tracked element by element without CSR, in the elements'
slices), `CSRGeometry` and macroparticle charges. Empties
`e.record`. Errors if the centroid particle is lost.

# Returns
- `(; geom, q, first_bend)`: the `CSRGeometry`, the charges [C] and
  the index of the first bending element (`length(bl.line) + 1` if
  none).
"""
function BeamTracking.effect_setup(e::CSREffect, bunch::Bunch, bl)
  line = bl.line
  context = bl.context
  n = length(line)
  isnothing(e.record) || empty!(e.record.points)

  # Reference geometry
  L = Vector{Float64}(undef, n)
  g = Vector{Float64}(undef, n)
  ref_exit = Vector{FloorPosition{Float64}}(undef, n)
  f = FloorPosition()
  for (i, ele) in enumerate(line)
    L[i] = scalarize(float(deval(ele.UniversalParams, context).L))
    pp = deval(ele.PatchParams, context)
    g[i] = isactive(pp) ? 0.0 :
           csr_ref_curvature(deval(ele.BendParams, context), i)
    f = csr_ref_exit(f, L[i], g[i], pp)
    ref_exit[i] = f
  end
  first_bend = something(findfirst(!iszero, g), n + 1)

  # Centroid orbit (no CSR)
  w = macro_weights(bunch)
  alive = bunch.coords.state .== STATE_ALIVE
  c0 = isnothing(e.centroid_start) ? bunch_centroid(bunch, w, alive) :
       collect(float.(e.centroid_start))
  Tc = promote_type(eltype(bunch.coords.v), eltype(c0))
  eltype(c0) == Tc || (c0 = convert(Vector{Tc}, c0))
  cb = centroid_bunch(bunch, c0)
  centroid = Matrix{Float64}(undef, n + 1, 6)
  centroid[1, :] = scalarize.(c0)
  for (i, ele) in enumerate(line)
    track!(cb, ele; context, notify = false)
    cb.coords.state[1] == STATE_ALIVE ||
      error("CSREffect: centroid orbit lost in element $i")
    centroid[i+1, :] = scalarize.(cb.coords.v[1, :])
  end
  geom = CSRGeometry(L, g, ref_exit, centroid;
                     bmad_compat = e.params.bmad_compat)
  q = scalarize.(w) .* abs(chargeof(bunch.species; C = true))
  return (; geom, q, first_bend)
end

"""
    csr_kicked(e, st, ele, sp) -> Bool

Whether the slice `sp` of `ele` gets a CSR kick: nonzero length, a
bend or after the first bend of the line, not tracked with
`SaganCavity`, and `e.on(ele)`.
"""
function csr_kicked(e::CSREffect, st, ele, sp)
  scalarize(sp.L_slice) > 0 || return false
  sp.ix >= st.first_bend || return false
  ele.tracking_method isa SaganCavity && return false
  return e.on(ele)::Bool
end

"""
    BeamTracking.effect_kick!(e::CSREffect, st, bunch, ele, sp)

One CSR kick (`csr_kick!`) at the entrance of slice `sp`, covering
`sp.L_slice`, with the Lorentz factor of the bunch's reference
there; nothing if the slice is not kicked (`csr_kicked`). Then
`e.diagnostics` and `e.record`, if set.
"""
function BeamTracking.effect_kick!(e::CSREffect, st, bunch::Bunch,
                                   ele, sp)
  csr_kicked(e, st, ele, sp) || return nothing
  species = bunch.species
  gamma = R_to_E(species, scalarize(bunch.p_over_q_ref)) /
          massof(species)
  v = bunch.coords.v
  alive = bunch.coords.state .== STATE_ALIVE
  rec = e.record
  binned = isnothing(rec) ? nothing :
           r -> push!(rec.points, (; ix = sp.ix, k = sp.k,
                                   n_slices = sp.n_slices,
                                   bins = r.bins, kick = r.kick))
  result = csr_kick!(view(v, :, PZI), view(v, :, ZI), st.q, alive,
                     st.geom, sp.ix, sp.L_slice, sp.s0, gamma,
                     species, e.params; binned)
  isnothing(e.diagnostics) ||
    e.diagnostics(sp.ix, sp.k, sp.s0, result)
  return nothing
end

"""
    BeamTracking.effect_setup(e::CSRReplay, bunch, bl) -> state

# Returns
- `Dict` from `(ix, k)` to the index of that kick in
  `e.record.points`.
"""
function BeamTracking.effect_setup(e::CSRReplay, bunch::Bunch, bl)
  return Dict{Tuple{Int,Int},Int}(
    (p.ix, p.k) => i for (i, p) in enumerate(e.record.points))
end

"""
    BeamTracking.effect_kick!(e::CSRReplay, st, bunch, ele, sp)

Apply the kick recorded for slice `sp` to `bunch`
(`csr_apply_test_kicks!`); nothing if none was recorded there.
"""
function BeamTracking.effect_kick!(e::CSRReplay, st, bunch::Bunch,
                                   ele, sp)
  i = get(st, (sp.ix, sp.k), 0)
  i == 0 && return nothing
  p = e.record.points[i]
  p.n_slices == sp.n_slices ||
    error("CSRReplay: element $(sp.ix) has $(sp.n_slices) slice(s), but $(p.n_slices) when the kicks were recorded")
  v = bunch.coords.v
  alive = bunch.coords.state .== STATE_ALIVE
  CSR.csr_apply_test_kicks!(view(v, :, PZI), view(v, :, ZI), alive,
                            p.bins, p.kick)
  return nothing
end
