# Bunch tracking with 1D CSR through a Beamline: port of Bmad's track1_bunch_csr step loop (route (a)).

"""
    csr_ref_curvature(bp, ix)

Reference curvature in the horizontal plane (`g_ref`, sign from `tilt_ref ∈ {0, π}`). Vertical bends are not
supported by the planar geometry (error).
"""
function csr_ref_curvature(bp, ix)
  isactive(bp) || return 0.0
  g = float(bp.g_ref)
  g == 0 && return 0.0
  c, s = cos(bp.tilt_ref), sin(bp.tilt_ref)
  abs(s) < 1e-12 || error("track_csr!: element $ix bends out of the horizontal plane (tilt_ref = $(bp.tilt_ref))")
  return g * sign(c)
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
  sum(wa) > 0 || error("track_csr!: no live particles")
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

Reference energies at the entrance and exit of an element. Only SaganCavity tracking uses dE_ref (BeamTracking),
so only there does the reference energy change inside the element (lcavity).
"""
function csr_ref_energies(ele, species, p_over_q_ref, context)
  E_end = R_to_E(species, p_over_q_ref)
  deval(ele.UniversalParams, context).tracking_method isa SaganCavity || return E_end, E_end
  return E_end - float(deval(ele.BeamlineParams, context).dE_ref), E_end
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

- `params::CSRParams` (`ds_step` must be set).
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

Macroparticle charge: `weight × |charge of species|` [C] (`weight = nothing` means 1).

Reference energy: constant inside elements, except SaganCavity elements with `dE_ref` (lcavity), where γ for
the CSR kick ramps linearly from `E_ref - dE_ref` to `E_ref` (`csr_ref_energy`, Bmad). The particles' reference
momentum there steps at each RF cell (BeamTracking), so the applied Δpz is rescaled from the linearly ramped
reference momentum to the stepped one.
"""
function CSR.track_csr!(bunch::Bunch, bl::Beamline; test_bunch = nothing, kwargs...)
  isempty(bl.line) && return bunch
  check_bunches!(bunch, test_bunch, bl)
  return track_csr_line!(bunch, bl.line, bl.context; test_bunch, kwargs...)
end

function CSR.track_csr!(bunch::Bunch, br::Branch; test_bunch = nothing, kwargs...)
  bls = filter(bl -> !isempty(bl.line), br.beamlines)
  isempty(bls) && return bunch
  check_bunches!(bunch, test_bunch, first(bls))
  return track_csr_line!(bunch, reduce(vcat, [bl.line for bl in bls]), br.context; test_bunch, kwargs...)
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
  test_bunch.p_over_q_ref == bunch.p_over_q_ref ||
    error("track_csr!: test_bunch p_over_q_ref ($(test_bunch.p_over_q_ref)) differs from the bunch's ($(bunch.p_over_q_ref))")
  return
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
  n = length(line)
  species = bunch.species
  q_abs = abs(chargeof(species; C = true))

  # Reference geometry
  L = zeros(n)
  g = zeros(n)
  ref_exit = Vector{FloorPosition{Float64}}(undef, n)
  f = FloorPosition()
  for (i, ele) in enumerate(line)
    L[i] = float(deval(ele.UniversalParams, context).L)
    pp = deval(ele.PatchParams, context)
    g[i] = isactive(pp) ? 0.0 : csr_ref_curvature(deval(ele.BendParams, context), i)
    f = csr_ref_exit(f, L[i], g[i], pp)
    ref_exit[i] = f
  end

  # Centroid orbit (no CSR)
  w = macro_weights(bunch)
  alive0 = bunch.coords.state .== BeamTracking.STATE_ALIVE
  c0 = isnothing(centroid_start) ? bunch_centroid(bunch, w, alive0) : collect(float.(centroid_start))
  cb = centroid_bunch(bunch, c0)
  centroid = zeros(n + 1, 6)
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
    E_start, E_end = csr_ref_energies(ele, species, p_over_q_ref, context)
    track_csr_element!(bunch, ele, i, L[i], E_start, E_end, p_over_q_ref, z, pz, q, geom, params, context, diagnostics,
                       test_bunch)
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
    track_csr_element!(bunch, ele, i, L, E_start, E_end, p_over_q_ref, z, pz, q, geom, params, context, diagnostics,
                       test_bunch = nothing)

CSR steps through element i. Function barrier: Beamlines parameters (p_over_q_ref, dE_ref) are not inferred.
The test bunch is tracked through the same slices and gets the bunch's kicks (`csr_kick!` `test_particles`).
"""
function track_csr_element!(bunch, ele, i, L, E_start, E_end, p_over_q_ref, z, pz, q, geom, params, context, diagnostics,
                            test_bunch = nothing)
  species = bunch.species
  mass = massof(species)
  n_step, ds = csr_n_step(L, params)
  ramp = E_start != E_end
  P0c = sqrt(E_start^2 - mass^2)   # Particles' reference momentum (used in an lcavity only)
  for i_step in 0:n_step
    if i_step != 0
      s = csr_slice(ele, i_step, n_step, ds, context)
      track_slice!(bunch, s, p_over_q_ref)
      isnothing(test_bunch) || track_slice!(test_bunch, s, p_over_q_ref)
      ramp && (P0c = oftype(P0c, csr_slice_P0c(s)))
    end
    s_kick = i_step * ds
    E_kick = CSR.csr_ref_energy(E_start, E_end, s_kick, L)
    gamma = E_kick / mass
    pz_scale = ramp ? sqrt(E_kick^2 - mass^2) / P0c : one(P0c)
    alive = bunch.coords.state .== BeamTracking.STATE_ALIVE
    kick_factor = (i_step == 0 || i_step == n_step) ? 0.5 : 1.0
    result = csr_kick!(pz, z, q, alive, geom, i, s_kick, gamma, species, params; kick_factor, pz_scale,
                       test_particles = test_kick_particles(test_bunch))
    isnothing(diagnostics) || diagnostics(i, i_step, s_kick, result)
  end
  return bunch
end
