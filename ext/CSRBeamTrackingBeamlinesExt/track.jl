# Bunch tracking with 1D CSR through a Beamline: port of Bmad's track1_bunch_csr step loop (route (a)).

"""
    csr_ref_curvature(bp, ix)

Reference curvature in the horizontal plane (`g_ref`, sign from `tilt_ref ∈ {0, π}`). Vertical bends are not
supported by the planar geometry (error). A TPS or `Dual` curvature with primal value 0 is returned as is (its knob
derivatives are kept).
"""
function csr_ref_curvature(bp, ix)
  isactive(bp) || return 0.0
  g = float(bp.g_ref)
  CSR.primal(g) == 0 && g isa AbstractFloat && return 0.0
  c, s = cos(bp.tilt_ref), sin(bp.tilt_ref)
  abs(CSR.primal(s)) < 1e-12 ||
    error("track_csr!: element $ix bends out of the horizontal plane (tilt_ref = $(bp.tilt_ref))")
  return g * sign(CSR.primal(c))
end

"""
    csr_ref_exit(f, L, g, pp)

Reference floor position at the exit of an element entered at `f`.
"""
function csr_ref_exit(f::FloorPosition, L, g, pp)
  isactive(pp) || return CSR.advance_floor(f, L, g)
  ez = quat_rotate((0, 0, 1), rot_quaternion(pp.dx_rot, pp.dy_rot, pp.dz_rot))
  return CSR.advance_floor_patch(f, (pp.dx, pp.dy, pp.dz), ez)
end

"""
    bunch_centroid(bunch, w, alive)

Charge-weighted mean of the live particles' coordinates.
"""
function bunch_centroid(bunch, w, alive)
  v = bunch.coords.v
  wa = w .* alive
  CSR.primal(sum(wa)) > 0 || error("track_csr!: no live particles")
  return vec(sum(v .* wa; dims = 1)) ./ sum(wa)
end

"""
    centroid_bunch(bunch, c0)

Single-particle bunch at c0 for the centroid orbit. Its weight is 1 (as `nothing`: only IBS reads it, through
log N) with the type of the bunch's weight, so BeamTracking reuses the kernels compiled for the bunch.
"""
function centroid_bunch(bunch, c0)
  wt = bunch.coords.weight
  w1 = isnothing(wt) ? nothing : wt isa Number ? one(wt) : fill!(similar(wt, 1), one(eltype(wt)))
  return Bunch(reshape(c0, 1, 6), nothing, w1; species = bunch.species, p_over_q_ref = bunch.p_over_q_ref,
               t_ref = bunch.t_ref)
end

"""
    macro_weights(bunch)

Real particles per macroparticle as a vector (`weight = nothing` means 1).
"""
function macro_weights(bunch)
  wt = bunch.coords.weight
  N = size(bunch.coords.v, 1)
  isnothing(wt) && return ones(N)
  wt isa Number && return fill(float(wt), N)
  return collect(float.(wt))
end

"""
    csr_ref_energies(ele, species, p_over_q_ref, context)

Reference energies `(E_start, E_end, ramp)` at the entrance and exit of an element, and whether the reference momentum
ramps inside it. Only SaganCavity tracking uses dE_ref (BeamTracking), so only there does the reference energy change
inside the element (lcavity): `ramp` is `true` for every SaganCavity, also with `dE_ref = 0` (then the ramp is flat;
D49), and `false` for other elements, where `E_start = E_end`.
"""
function csr_ref_energies(ele, species, p_over_q_ref, context)
  E_end = R_to_E(species, p_over_q_ref)
  deval(ele.UniversalParams, context).tracking_method isa SaganCavity || return E_end, E_end, false
  return E_end - float(deval(ele.BeamlineParams, context).dE_ref), E_end, true
end

"""
    csr_slice_P0c(s)

Reference momentum [eV] of the particles after slice `s` (and at s = 0), when it differs from p_over_q_ref.
"""
csr_slice_P0c(s::SaganSlice) = sagan_exit_P0c(s)
csr_slice_P0c(_) = nothing

"""
    CSR.track_csr!(bunch, bl::Beamline; params, csr_on = ele -> true, centroid_start = nothing,
                   sliced_centroid = true, diagnostics = nothing, test_bunch = nothing) -> bunch
    CSR.track_csr!(bunch, br::Branch; ...) -> bunch

Track `bunch` through `bl` (or the Beamlines of `br` in order) with 1D CSR kicks between slices of each element
(Bmad `csr_method = one_dim`).

- `params::CSRParams` (`ds_step > 0` required; error otherwise).
- `csr_on(ele)`: elements for which it returns `false` are tracked whole without CSR, like Bmad's per-element
  `csr_method = off`. Zero-length and map elements are always tracked whole without CSR.
- `centroid_start`: 6-vector where the centroid orbit starts (Bmad/Tao: the supplied `centroid` orbit).
  Default: charge-weighted mean of the live particles. The centroid is tracked through `bl` without CSR
  before the bunch, and its orbit defines the CSR geometry.
- `sliced_centroid = true`: track the centroid through CSR elements in the same slices as the bunch, so that it
  follows the orbit of a bunch particle. BeamTracking's integration depends on the slicing (the default Symplectic
  method takes one step per element or slice), so the centroid then depends on `params.ds_step` and `csr_on`.
  `false`: track it through whole elements (before 2026-10-06).
- `diagnostics(ix_ele, i_step, s_kick, result::CSRStepResult)` is called after each kick.
- `test_bunch`: a `Bunch` (e.g. GTPSA TPS or ForwardDiff `Dual` coordinates) tracked through the same slices that
  gets the bunch's CSR kicks without contributing to them or to the centroid (frozen wake, `csr_kick!`
  `test_particles`): its final coordinates are the transfer map with the bunch's wake held fixed. Same species and
  reference momentum as `bunch`; its live particles must stay within the bunch's bin range at every kick (error).

Knobs: the bunch may carry GTPSA TPS (or ForwardDiff `Dual`) coordinates, weights, `t_ref` and `p_over_q_ref`,
and element parameters may be TPS/`Dual` (e.g. `L`, `g_ref`, `Kn1`, patch offsets, RF `voltage`, `dE_ref`, the
reference energy). The final coordinates are then the collective map in the knobs: the deposit, geometry,
centroid orbit, γ and kick tables are computed in that number type. Discrete choices (number of CSR steps, slice
step counts, bin grid, SaganCavity cells and event windows) use the primal values and are held constant.

When `L`, `p_over_q_ref`, `dE_ref`, patch parameters or the RF parameters of a SaganCavity are TPS/`Dual`, the
reference time is too: the `Bunch` (and `test_bunch`) must be built with `t_ref`, `p_over_q_ref` and coordinates
of a type that holds them (e.g. `t_ref = zero(k)` for a TPS knob `k`); otherwise `track_csr!` raises an error before
tracking. Dynamic-descriptor TPS in the bunch, `centroid_start`, the test bunch or those element parameters (and
`g_ref`) must be in GTPSA's current Descriptor (error otherwise). Knobs in other parameters (e.g. `g_ref`, `Kn1`,
alignment) are not part of this check; they also need TPS/`Dual` bunch coordinates (otherwise BeamTracking or
`csr_kick!` raises an error).

Limits: a SaganCavity with automatic `n_cells` and a knob in its length or RF frequency fails in BeamTracking
(`round` of a TPS); derivatives of order ≥ 3 in a curvature knob of an element with `g = 0` are truncated.
BeamTracking 0.8.7 leaves `bunch.p_over_q_ref` at the entrance value after a SaganCavity (the coordinates are
already normalized to the exit reference), so the next element shifts `pz` a second time; CSR.jl's lcavity
slices follow BeamTracking. With a `dE_ref` knob of primal value 0 that second shift is skipped for the
derivatives (equal reference energies in value).

Macroparticle charge: `weight × |charge of species|` [C] (`weight = nothing` means 1).

Runs on the CPU: the bunch's arrays must be CPU arrays (GPU bunches are not supported). Each kick uses `csr_kick!`, which
also runs on GPU backends when called directly.

Reference energy: constant inside elements, except SaganCavity elements (lcavity; with `dE_ref = 0` the ramp is flat),
where γ for the CSR kick ramps linearly from `E_ref - dE_ref` to `E_ref` (`csr_ref_energy`, Bmad). The particles'
reference momentum there steps at each RF cell (BeamTracking), so the applied Δpz is rescaled from the linearly ramped
reference momentum to the stepped one.
"""
function CSR.track_csr!(bunch::Bunch, bl::Beamline; test_bunch = nothing, centroid_start = nothing, kwargs...)
  isempty(bl.line) && return bunch
  check_track_inputs(bunch, test_bunch, bl.line, bl.context, centroid_start)
  check_bunches!(bunch, test_bunch, bl)
  check_ref_types(bunch, test_bunch, ref_param_type(bl.line, bl.context))
  return track_csr_line!(bunch, bl.line, bl.context; test_bunch, centroid_start, kwargs...)
end

function CSR.track_csr!(bunch::Bunch, br::Branch; test_bunch = nothing, centroid_start = nothing, kwargs...)
  bls = filter(bl -> !isempty(bl.line), br.beamlines)
  isempty(bls) && return bunch
  line = reduce(vcat, [bl.line for bl in bls])
  check_track_inputs(bunch, test_bunch, line, br.context, centroid_start)
  check_bunches!(bunch, test_bunch, first(bls))
  check_ref_types(bunch, test_bunch, ref_param_type(line, br.context))
  return track_csr_line!(bunch, line, br.context; test_bunch, centroid_start, kwargs...)
end

"""
    check_bunches!(bunch, test_bunch, bl)

BeamTracking's bunch check (species and reference momentum, set from `bl` where missing) for the bunch and the test
bunch; the test bunch must then have the bunch's species and reference momentum.
"""
function check_bunches!(bunch, test_bunch, bl)
  bt_ext().check_bl_bunch!(bunch, bl)
  isnothing(test_bunch) && return
  test_bunch isa Bunch || error("track_csr!: test_bunch must be a Bunch or nothing")
  bt_ext().check_bl_bunch!(test_bunch, bl)
  test_bunch.species == bunch.species || error("track_csr!: test_bunch species differs from the bunch's")
  CSR.primal(test_bunch.p_over_q_ref) == CSR.primal(bunch.p_over_q_ref) ||
    error("track_csr!: test_bunch p_over_q_ref ($(test_bunch.p_over_q_ref)) differs from the bunch's " *
          "($(bunch.p_over_q_ref))")
  return
end

"""
    csr_read_params(line, context)

`(; L, g_ref, patch, dE_ref, sagan_rf, p_over_q_ref)`: the element parameters CSR.jl reads from `line`, as flat
`Vector{Any}`s. `L` of every element; `g_ref` of the active bends; the 7 patch fields of the active patches; `dE_ref`
and the RF `rate`, `voltage`, `phi0` (when active) of the SaganCavity elements; `p_over_q_ref` of the elements that
have a number.
"""
function csr_read_params(line, context)
  L = Any[]
  g_ref = Any[]
  patch = Any[]
  dE_ref = Any[]
  sagan_rf = Any[]
  p_over_q_ref = Any[]
  for ele in line
    push!(L, float(deval(ele.UniversalParams, context).L))
    bp = deval(ele.BendParams, context)
    isactive(bp) && push!(g_ref, bp.g_ref)
    pp = deval(ele.PatchParams, context)
    isactive(pp) && append!(patch, (pp.dt, pp.dx, pp.dy, pp.dz, pp.dx_rot, pp.dy_rot, pp.dz_rot))
    if deval(ele.UniversalParams, context).tracking_method isa SaganCavity
      push!(dE_ref, float(deval(ele.BeamlineParams, context).dE_ref))
      rp = deval(ele.RFParams, context)
      isactive(rp) && append!(sagan_rf, (rp.rate, rp.voltage, rp.phi0))
    end
    ele.p_over_q_ref isa Number && push!(p_over_q_ref, ele.p_over_q_ref)
  end
  return (; L, g_ref, patch, dE_ref, sagan_rf, p_over_q_ref)
end

"""
    check_track_inputs(bunch, test_bunch, line, context, centroid_start)

GTPSA Descriptor check (`CSR.check_descriptors`) of the bunch, the test bunch, `centroid_start` and the element
parameters that CSR.jl reads from `line` (see `csr_read_params`). Errors before anything is tracked.
"""
function check_track_inputs(bunch, test_bunch, line, context, centroid_start)
  r = csr_read_params(line, context)
  tb = test_bunch
  CSR.check_descriptors("track_csr!"; coords = bunch.coords.v, weight = bunch.coords.weight, t_ref = bunch.t_ref,
                        p_over_q_ref = bunch.p_over_q_ref, centroid_start,
                        test_coords = tb isa Bunch ? tb.coords.v : nothing,
                        test_t_ref = tb isa Bunch ? tb.t_ref : nothing,
                        test_p_over_q_ref = tb isa Bunch ? tb.p_over_q_ref : nothing,
                        L = r.L, g_ref = r.g_ref, patch = r.patch, dE_ref = r.dE_ref, sagan_rf = r.sagan_rf,
                        ele_p_over_q_ref = r.p_over_q_ref)
  return nothing
end

"""
    ref_param_type(line, context)

Number type of the element parameters that enter the reference time and momentum (`L`, patch, `dE_ref`, the
SaganCavity RF parameters, which set the `SaganSlice` number type, and `p_over_q_ref`), promoted from `Float64`.
"""
function ref_param_type(line, context)
  r = csr_read_params(line, context)
  return mapreduce(typeof, promote_type, vcat(r.L, r.patch, r.dE_ref, r.sagan_rf, r.p_over_q_ref); init = Float64)
end

"""
    check_ref_types(bunch, test_bunch, Tp)

Error unless the `t_ref`, `p_over_q_ref` and coordinates of `bunch` and `test_bunch` can hold the element parameter type
`Tp` (see `ref_param_type`). Nothing is checked for a hardware float `Tp`.
"""
function check_ref_types(bunch, test_bunch, Tp)
  Tp <: AbstractFloat && return nothing
  holds(F) = promote_type(F, Tp) == F
  hint = "Build the Bunch with t_ref, p_over_q_ref and coordinates of a type that holds $Tp, e.g. t_ref = zero(k) " *
         "for a knob k."
  why = "cannot hold the element parameters' type $Tp (L, p_over_q_ref, dE_ref, patch or SaganCavity RF parameters)"
  for (name, b) in (("bunch", bunch), ("test_bunch", test_bunch))
    isnothing(b) && continue
    F = typeof(b.t_ref)
    holds(F) || error("track_csr!: $name.t_ref has type $F, which $why. $hint")
    F = typeof(b.p_over_q_ref)
    holds(F) || error("track_csr!: $name.p_over_q_ref has type $F, which $why. $hint")
    F = eltype(b.coords.v)
    holds(F) || error("track_csr!: $name coordinates have element type $F, which $why. $hint")
  end
  return nothing
end

"""
    test_kick_particles(test_bunch)

`(pz, z, alive)` of the test bunch for `csr_kick!`'s `test_particles`, or `nothing`.
"""
test_kick_particles(::Nothing) = nothing
function test_kick_particles(tb::Bunch)
  v = tb.coords.v
  return (view(v, :, BeamTracking.PZI), view(v, :, BeamTracking.ZI), tb.coords.state .== BeamTracking.STATE_ALIVE)
end

"""
    track_csr_line!(bunch, line, context; params, csr_on, centroid_start, sliced_centroid, diagnostics, test_bunch)

Body of `track_csr!` for a vector of elements: reference survey (with patches), centroid orbit, `CSRGeometry`,
then the step loop over elements and slices with half kicks at the element ends (Bmad `track1_bunch_csr`).
"""
function track_csr_line!(bunch::Bunch, line, context; params::CSRParams, csr_on = ele -> true,
                         centroid_start = nothing, sliced_centroid = true, diagnostics = nothing, test_bunch = nothing)
  params.ds_step > 0 || error("track_csr!: params.ds_step must be > 0 (got $(params.ds_step))")
  isnothing(centroid_start) || length(centroid_start) == 6 ||
    error("track_csr!: centroid_start must have length 6 (got $(length(centroid_start)))")
  n = length(line)
  species = bunch.species
  q_abs = abs(chargeof(species; C = true))

  # Reference geometry
  Lv = Vector{Any}(undef, n)
  gv = Vector{Any}(undef, n)
  fv = Vector{Any}(undef, n)
  f = FloorPosition()
  for (i, ele) in enumerate(line)
    Lv[i] = float(deval(ele.UniversalParams, context).L)
    pp = deval(ele.PatchParams, context)
    gv[i] = isactive(pp) ? 0.0 : csr_ref_curvature(deval(ele.BendParams, context), i)
    f = csr_ref_exit(f, Lv[i], gv[i], pp)
    fv[i] = f
  end
  Tg = promote_type(mapreduce(typeof, promote_type, Lv; init = Float64), mapreduce(typeof, promote_type, gv; init = Float64))
  L = convert(Vector{Tg}, Lv)
  g = convert(Vector{Tg}, gv)
  Tf = mapreduce(CSR.number_type, promote_type, fv; init = Tg)
  ref_exit = FloorPosition{Tf}[convert(FloorPosition{Tf}, f) for f in fv]

  # Centroid orbit (no CSR)
  w = macro_weights(bunch)
  alive0 = bunch.coords.state .== BeamTracking.STATE_ALIVE
  c0 = isnothing(centroid_start) ? bunch_centroid(bunch, w, alive0) : collect(float.(centroid_start))
  Tc = promote_type(eltype(bunch.coords.v), eltype(c0))
  eltype(c0) == Tc || (c0 = convert(Vector{Tc}, c0))
  cb = centroid_bunch(bunch, c0)
  centroid = Matrix{promote_type(Float64, eltype(c0))}(undef, n + 1, 6)
  centroid[1, :] = c0
  for (i, ele) in enumerate(line)
    if sliced_centroid && csr_on(ele) && CSR.csr_sliceable(ele; context)
      p_over_q_ref = ele.p_over_q_ref
      p_over_q_ref isa Number || error("track_csr!: time-dependent reference energy is not supported (element $i)")
      track_centroid_sliced!(cb, ele, L[i], p_over_q_ref, params, context)
    else
      track!(cb, ele; context)
    end
    cb.coords.state[1] == BeamTracking.STATE_ALIVE || error("track_csr!: centroid orbit lost in element $i")
    centroid[i+1, :] = cb.coords.v[1, :]
  end
  geom = CSRGeometry(L, g, ref_exit, centroid; bmad_compat = params.bmad_compat)

  # Bunch tracking
  v = bunch.coords.v
  z = view(v, :, BeamTracking.ZI)
  pz = view(v, :, BeamTracking.PZI)
  q = w .* q_abs
  for (i, ele) in enumerate(line)
    if !(csr_on(ele) && CSR.csr_sliceable(ele; context))
      track!(bunch, ele; context)
      isnothing(test_bunch) || track!(test_bunch, ele; context)
      continue
    end
    p_over_q_ref = ele.p_over_q_ref
    p_over_q_ref isa Number || error("track_csr!: time-dependent reference energy is not supported (element $i)")
    E_start, E_end, ramp = csr_ref_energies(ele, species, p_over_q_ref, context)
    track_csr_element!(bunch, ele, i, L[i], E_start, E_end, ramp, p_over_q_ref, z, pz, q, geom, params, context,
                       diagnostics, test_bunch)
  end
  return bunch
end

"""
    csr_slice(ele, i_step, n_step, ds, context)

Slice i_step (1..n_step) of an element cut into n_step slices of length ds, as tracked between CSR kicks.
s0 + L_slice equals the next slice's s0 exactly (Sterbenz), so SaganCavity cell kicks on a boundary fall in
exactly one slice.
"""
function csr_slice(ele, i_step, n_step, ds, context)
  s0 = (i_step - 1) * ds
  return CSR.slice_element(ele, s0, i_step * ds - s0, i_step == 1, i_step == n_step; context)
end

"""
    track_centroid_sliced!(cb, ele, L, p_over_q_ref, params, context)

Centroid through an element in the bunch's slices (BeamTracking's integration depends on the slicing: the
default Symplectic method takes one step per element or slice). Function barrier as track_csr_element!.
"""
function track_centroid_sliced!(cb, ele, L, p_over_q_ref, params, context)
  n_step, ds = csr_n_step(L, params)
  for i_step in 1:n_step
    track_slice!(cb, csr_slice(ele, i_step, n_step, ds, context), p_over_q_ref)
  end
  return cb
end

"""
    track_csr_element!(bunch, ele, i, L, E_start, E_end, ramp, p_over_q_ref, z, pz, q, geom, params, context,
                       diagnostics, test_bunch = nothing)

CSR steps through element i. Function barrier: Beamlines parameters (p_over_q_ref, dE_ref) are not inferred.
The test bunch is tracked through the same slices and gets the bunch's kicks (`csr_kick!` `test_particles`).
"""
function track_csr_element!(bunch, ele, i, L, E_start, E_end, ramp, p_over_q_ref, z, pz, q, geom, params, context,
                            diagnostics, test_bunch = nothing)
  species = bunch.species
  mass = massof(species)
  n_step, ds = csr_n_step(L, params)
  TP = float(promote_type(typeof(E_start), typeof(E_end)))
  P0c = convert(TP, sqrt(E_start^2 - mass^2))::TP   # Particles' reference momentum (used in an lcavity only)
  for i_step in 0:n_step
    if i_step != 0
      s = csr_slice(ele, i_step, n_step, ds, context)
      track_slice!(bunch, s, p_over_q_ref)
      isnothing(test_bunch) || track_slice!(test_bunch, s, p_over_q_ref)
      ramp && (P0c = convert(TP, csr_slice_P0c(s))::TP)
    end
    s_kick = i_step * ds
    E_kick = CSR.csr_ref_energy(E_start, E_end, s_kick, L)
    gamma = E_kick / mass
    pz_scale = ramp ? sqrt(E_kick^2 - mass^2) / P0c : one(E_kick) / one(P0c)
    alive = bunch.coords.state .== BeamTracking.STATE_ALIVE
    kick_factor = (i_step == 0 || i_step == n_step) ? 0.5 : 1.0
    result = csr_kick!(pz, z, q, alive, geom, i, s_kick, gamma, species, params; kick_factor, pz_scale,
                       test_particles = test_kick_particles(test_bunch))
    isnothing(diagnostics) || diagnostics(i, i_step, s_kick, result)
  end
  return bunch
end
