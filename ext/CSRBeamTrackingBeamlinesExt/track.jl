# Bunch tracking with 1D CSR through a Beamline: port of Bmad's track1_bunch_csr step loop (route (a)).

# Reference curvature in the horizontal plane. Vertical bends are not supported by the planar geometry.
function csr_ref_curvature(bp, ix)
  isactive(bp) || return 0.0
  g = float(bp.g_ref)
  g == 0 && return 0.0
  c, s = cos(bp.tilt_ref), sin(bp.tilt_ref)
  abs(s) < 1e-12 || error("track_csr!: element $ix bends out of the horizontal plane (tilt_ref = $(bp.tilt_ref))")
  return g * sign(c)
end

# Reference floor position at the exit of an element entered at `f`.
function csr_ref_exit(f::FloorPosition, L, g, pp)
  isactive(pp) || return CSR.advance_floor(f, L, g)
  ez = quat_rotate((0, 0, 1), rot_quaternion(pp.dx_rot, pp.dy_rot, pp.dz_rot))
  return CSR.advance_floor_patch(f, (pp.dx, pp.dy, pp.dz), ez)
end

# Charge-weighted mean of the live particles' coordinates.
function bunch_centroid(bunch, w, alive)
  v = bunch.coords.v
  wa = w .* alive
  sum(wa) > 0 || error("track_csr!: no live particles")
  return vec(sum(v .* wa; dims = 1)) ./ sum(wa)
end

function macro_weights(bunch)
  wt = bunch.coords.weight
  N = size(bunch.coords.v, 1)
  isnothing(wt) && return ones(N)
  wt isa Number && return fill(float(wt), N)
  return collect(float.(wt))
end

"""
    CSR.track_csr!(bunch, bl::Beamline; params, csr_on = ele -> true, centroid_start = nothing,
                   diagnostics = nothing) -> bunch

Track `bunch` through `bl` with 1D CSR kicks between slices of each element (Bmad `csr_method = one_dim`).

- `params::CSRParams` (`ds_step` must be set).
- `csr_on(ele)`: elements for which it returns `false` are tracked whole without CSR, like Bmad's per-element
  `csr_method = off`. Zero-length and map elements are always tracked whole without CSR.
- `centroid_start`: 6-vector where the centroid orbit starts (Bmad/Tao: the supplied `centroid` orbit).
  Default: charge-weighted mean of the live particles. The centroid is tracked through `bl` without CSR
  before the bunch, and its orbit defines the CSR geometry.
- `diagnostics(ix_ele, i_step, s_kick, result::CSRStepResult)` is called after each kick.

Macroparticle charge: `weight × |charge of species|` [C] (`weight = nothing` means 1).
The reference energy is constant inside every BeamTracking element, so γ is constant within an element
(`csr_ref_energy` with equal end energies).
"""
function CSR.track_csr!(bunch::Bunch, bl::Beamline; params::CSRParams, csr_on = ele -> true,
                        centroid_start = nothing, diagnostics = nothing)
  line = bl.line
  n = length(line)
  n == 0 && return bunch
  Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt).check_bl_bunch!(bunch, bl)
  context = bl.context
  species = bunch.species
  q_abs = abs(chargeof(species; C = true))

  # Reference geometry
  L = zeros(n)
  g = zeros(n)
  ref_exit = Vector{FloorPosition}(undef, n)
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
  cb = Bunch(reshape(c0, 1, 6); species, p_over_q_ref = bunch.p_over_q_ref, t_ref = bunch.t_ref)
  centroid = zeros(n + 1, 6)
  centroid[1, :] = c0
  for (i, ele) in enumerate(line)
    track!(cb, ele; context)
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
      continue
    end
    p_over_q_ref = ele.p_over_q_ref
    p_over_q_ref isa Number || error("track_csr!: time-dependent reference energy is not supported (element $i)")
    E_ref = R_to_E(species, p_over_q_ref)
    n_step, ds = csr_n_step(L[i], params)
    for i_step in 0:n_step
      if i_step != 0
        s = CSR.slice_element(ele, (i_step - 1) * ds, ds, i_step == 1, i_step == n_step; context)
        track!(bunch, s; _p_over_q_ref = p_over_q_ref)
      end
      s_kick = i_step * ds
      gamma = CSR.csr_ref_energy(E_ref, E_ref, s_kick, L[i]; bmad_compat = params.bmad_compat) / massof(species)
      alive = bunch.coords.state .== BeamTracking.STATE_ALIVE
      kick_factor = (i_step == 0 || i_step == n_step) ? 0.5 : 1.0
      result = csr_kick!(pz, z, q, alive, geom, i, s_kick, gamma, species, params; kick_factor)
      isnothing(diagnostics) || diagnostics(i, i_step, s_kick, result)
    end
  end
  return bunch
end
