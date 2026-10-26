# One CSR kick (host side): bin, build the kick table, convolve, apply.
# Port of the per-step part of Bmad's track1_bunch_csr and csr_bin_kicks (CSR only, with image charges).

"""
    csr_convolve!(kick, I_int, edge_grad, coef)

`kick[i] = coef * Σ_{k=1}^{i} I_int[k] * edge_grad[i+1-k]`. One work item per bin, on the backend of `kick`.
"""
function csr_convolve!(kick::AbstractVector, I_int::AbstractVector, edge_grad::AbstractVector, coef)
  backend = get_backend(kick)
  n = length(kick)
  wg = cpu_workgroupsize(n, eltype(kick), eltype(I_int), eltype(edge_grad), typeof(coef))
  csr_convolve_kernel!(backend)(kick, I_int, edge_grad, coef; ndrange = n, workgroupsize = wg)
  KernelAbstractions.synchronize(backend)
  return kick
end

"""
    csr_image_convolve!(kick, image, charge, coef)

`kick[i] += coef * Σ_{j=1}^{n_bin} image(i - j) * charge[j]`, with the image kick for lag `l` stored at
`image[l + n_bin + 1]` (`csr_image_table`). Bmad's image-charge sum (all lags, bin charge rather than its
gradient). One work item per bin.
"""
function csr_image_convolve!(kick::AbstractVector, image::AbstractVector, charge::AbstractVector, coef)
  length(image) == 2length(kick) + 1 || error("csr_image_convolve!: image must have 2n_bin + 1 entries")
  backend = get_backend(kick)
  n = length(kick)
  wg = cpu_workgroupsize(n, eltype(kick), eltype(image), eltype(charge), typeof(coef))
  csr_image_convolve_kernel!(backend)(kick, image, charge, coef; ndrange = n, workgroupsize = wg)
  KernelAbstractions.synchronize(backend)
  return kick
end

"""
    csr_apply_kicks!(pz, z, alive, bins, kick; pz_scale = 1)

Add the kick, linearly interpolated between bin centers and multiplied by `pz_scale`, to `pz` of each live particle.
As in Bmad's CSR-only branch, `z` is not rescaled for the change in velocity.
One work item per particle, on the backend of `pz`.
"""
function csr_apply_kicks!(pz::AbstractVector, z::AbstractVector, alive::AbstractVector{Bool}, bins::CSRBins, kick::AbstractVector;
                          pz_scale = 1)
  length(pz) == length(z) == length(alive) ||
    error("csr_apply_kicks!: pz, z, alive must have equal lengths (got $(length(pz)), $(length(z)), $(length(alive)))")
  length(kick) == length(bins.z_center) || error("csr_apply_kicks!: kick and bins.z_center must have equal lengths")
  apply_kicks_outside!(pz, z, alive, bins.z_center, kick, bins.dz, pz_scale) == 0 ||
    error("csr_apply_kicks!: particle outside the bin range (internal error)")
  return pz
end

"""
    apply_kicks_outside!(pz, z, alive, z_center, kick, dz, pz_scale) -> n_outside

Launch `csr_apply_kicks_kernel!` on the backend of `pz`; returns the number of live particles outside the table.
"""
function apply_kicks_outside!(pz, z, alive, z_center, kick, dz, pz_scale)
  length(pz) == 0 && return 0
  backend = get_backend(pz)
  n_outside = KernelAbstractions.zeros(backend, Int32, 1)   # Int32: no 64-bit atomics on some GPUs
  n = length(pz)
  wg = cpu_workgroupsize(n, eltype(pz), eltype(z), eltype(kick), typeof(dz), typeof(pz_scale))
  csr_apply_kicks_kernel!(backend)(pz, n_outside, z, alive, z_center, kick, dz, pz_scale; ndrange = n, workgroupsize = wg)
  KernelAbstractions.synchronize(backend)
  return Int(Array(n_outside)[1])
end

"""
    count_outside(backend, z, alive, z_center, dz) -> n_outside

Number of live particles outside the kick table (`csr_count_outside_kernel!` on `backend`), without kicking.
"""
function count_outside(backend, z, alive, z_center, dz)
  length(z) == 0 && return 0
  n_outside = KernelAbstractions.zeros(backend, Int32, 1)   # Int32: no 64-bit atomics on some GPUs
  n = length(z)
  wg = cpu_workgroupsize(n, eltype(z), typeof(dz))
  csr_count_outside_kernel!(backend)(n_outside, z, alive, z_center, dz; ndrange = n, workgroupsize = wg)
  KernelAbstractions.synchronize(backend)
  return Int(Array(n_outside)[1])
end

"""
    check_test_particles(fname, pz_t, z_t, alive_t, Tkick, pz_scale)

Checks on test particles that need only their types and lengths, for a kick of number type `Tkick` (errors name
function `fname`): equal lengths of `pz_t`, `z_t`, `alive_t`; a `pz_t` that can hold the kick (an `AbstractFloat`
`pz_t` needs an `AbstractFloat` kick, `z_t` and `pz_scale`); and, for a non-`Real` `Tkick` (TPS), `check_descriptors`
on `pz_t` and `z_t`.
"""
function check_test_particles(fname, pz_t, z_t, alive_t, Tkick, pz_scale)
  length(pz_t) == length(z_t) == length(alive_t) ||
    error("$fname: test particle arrays pz_t, z_t, alive_t have lengths $(length(pz_t)), $(length(z_t)), " *
          "$(length(alive_t)); they must be equal.")
  Tt = promote_type(Tkick, eltype(z_t), typeof(pz_scale))
  eltype(pz_t) <: AbstractFloat && !(Tt <: AbstractFloat) &&
    error("$fname: pz_t has element type $(eltype(pz_t)), which cannot hold the kick in number type " *
          "$Tt; use a pz_t with element type $(promote_type(eltype(pz_t), Tt)).")
  Tkick <: Real || check_descriptors(fname; pz_t, z_t)
  return nothing
end

"""
    csr_apply_test_kicks!(pz_t, z_t, alive_t, bins, kick; pz_scale = 1)

Frozen-wake kick on test particles: `csr_apply_kicks!` with the bins and kick of a bunch the test particles are not
part of (they deposit no charge). For TPS (or `Dual`) test particles this gives the kick's expansion in their
coordinates about the primal `z`, with the bunch's wake held fixed; with a TPS kick (a collective TPS bunch,
geometry, `gamma`, ...) the expansion is also in the bunch's knobs. The test particles' arrays may live on another
backend than the bunch (e.g. TPS on the CPU); `bins` and `kick` are copied there.

Errors, raised before any test particle is kicked (`pz_t` is then unchanged):
- a live test particle outside the bunch's bin range, `z_center[1] - dz/100 ≤ z < z_center[end]` (the range
  `csr_apply_kicks_kernel!` interpolates over; counted first by `count_outside`);
- `pz_t`, `z_t`, `alive_t` of different lengths;
- `pz_t` with an `AbstractFloat` element type that cannot hold the kick (TPS or `Dual` kick, `z_t` or `pz_scale`);
- with a TPS kick, a dynamic-descriptor TPS in `pz_t` or `z_t` that is not in GTPSA's current Descriptor
  (`check_descriptors`). With a `Real` kick the test particles are not checked.

Other element-type combinations that cannot hold the kick (e.g. a `Dual` `pz_t` with a TPS kick, a TPS `pz_t` with a
`Dual` kick) are not checked and fail inside the kick kernel, possibly after part of `pz_t` has been kicked.
"""
function csr_apply_test_kicks!(pz_t::AbstractVector, z_t::AbstractVector, alive_t::AbstractVector{Bool}, bins::CSRBins,
                               kick::AbstractVector; pz_scale = 1)
  check_test_particles("csr_apply_test_kicks!", pz_t, z_t, alive_t, eltype(kick), pz_scale)
  cpu(A) = A isa Array ? A : Array(A)
  backend = get_backend(pz_t)
  zc, k = backend isa CPU ? (cpu(bins.z_center), cpu(kick)) : (bins.z_center, kick)
  dz = primal(bins.dz)
  n = count_outside(backend, z_t, alive_t, zc, dz)
  n == 0 || error("csr_apply_test_kicks!: $n test particle(s) outside the bunch's bin range " *
                  "[$(primal(first(zc)) - primal(bins.dz) / 100), $(primal(last(zc))))")
  apply_kicks_outside!(pz_t, z_t, alive_t, zc, k, dz, pz_scale) == 0 ||
    error("csr_apply_test_kicks!: test particle outside the bin range (internal error)")
  return pz_t
end

"""
    csr_kick!(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0,
              pz_scale = 1.0, test_particles = nothing) -> CSRStepResult

Apply one 1D CSR kick at reference position `s_kick` [m] from the entrance of element `ix_ele_kick`. Errors if the
array lengths differ or a live particle has a non-finite `z`.

- `pz`, `z`: particle coordinates (`pz` is modified). `pz` must hold the kick's number type: with `Dual` or TPS
  inputs, a `pz` of that element type, or use `csr_kick_pz`.
- `q`: macroparticle charges [C]. `alive`: live-particle mask.
- `gamma`: Lorentz factor of the reference particle at the kick point.
- `kick_factor`: `1/2` at the first and last step of an element, `1` otherwise.
- `pz_scale`: multiplies the Δpz applied to the particles (not the returned diagnostics). Used where the particles'
  reference momentum differs from the one the kick is computed for (inside a SaganCavity, whose reference momentum
  steps at each cell while γ ramps linearly).
- `test_particles`: `(pz_t, z_t, alive_t)` or `nothing`. Test particles get the same kick (`pz_t` modified) without
  contributing to it, e.g. a GTPSA TPS particle for the transfer map with the bunch's wake frozen
  (`csr_apply_test_kicks!`). They are kicked before the bunch, so an error from `csr_apply_test_kicks!` (a live test
  particle outside the bin range, arrays of different lengths, a `pz_t` that cannot hold the kick, a descriptor
  mismatch; the last three are checked before the kick tables are built) leaves `pz` and `pz_t` unchanged. Other
  element-type combinations that cannot hold the kick (e.g. a `Dual` `pz` or `pz_t` with a TPS kick, a TPS `pz` or
  `pz_t` with a `Dual` kick) are not checked and fail inside the kick kernel, possibly after part of `pz` or `pz_t`
  has been kicked.

The kick covers one step, `ds = L / n_step` (see `csr_n_step`) scaled by `L_chord / L`, as in Bmad.

With `params.n_shield_images > 0` the image-charge kicks of the shielding plates are added (Bmad's image part of
`csr_bin_kicks`), with the layer sum averaged per `params.image_sum_order`. `kick_per_meter` includes them;
`I_csr_per_meter` and `s_source` are for free space only.

Runs as KernelAbstractions kernels on the backend of `pz` (deposit, edge gradient, convolutions, kick). The wake and
image tables are always computed on the CPU in `Float64` and copied to a GPU backend in `csr_float_type(backend)`
(`Float32` where the backend has no `Float64`, e.g. Metal); scalars passed to the kernels are converted likewise.
`csr_kick_reference!` is the serial phase 1 version.

Dynamic-descriptor TPS arguments (`@vars(d, dynamic = true)`) must all be in GTPSA's current Descriptor (error
otherwise; test particles are checked only when the kick is TPS, before the kick tables are built); static-descriptor TPS (the
`@vars` default) carry their Descriptor in the type and are not checked.
"""
function csr_kick!(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                   geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                   params::CSRParams; kick_factor = 1.0, pz_scale = 1.0, test_particles = nothing)
  check_descriptors("csr_kick!"; pz, z, q, geom, s_kick, gamma, kick_factor, pz_scale)
  Tk = promote_type(eltype(z), eltype(q), number_type(geom), typeof(s_kick), typeof(gamma), typeof(kick_factor),
                    typeof(pz_scale))
  eltype(pz) <: AbstractFloat && !(Tk <: AbstractFloat) &&
    error("csr_kick!: pz has element type $(eltype(pz)), which cannot hold the kick in number type $Tk; " *
          "use csr_kick_pz, or a pz with element type $(promote_type(eltype(pz), Tk)).")
  test_particles === nothing || check_test_particles("csr_kick!", test_particles..., Tk, pz_scale)
  st = csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor)
  test_particles === nothing || csr_apply_test_kicks!(test_particles..., st.bins, st.kick; pz_scale)
  csr_apply_kicks!(pz, z, alive, st.bins, st.kick; pz_scale = device_scalar(get_backend(pz), pz_scale))
  return csr_step_result(st)
end

"""
    csr_kick_pz(pz, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0,
                pz_scale = 1.0) -> pz_new

Non-mutating `csr_kick!`: returns `pz` after the kick in a new vector (element type promoted with `z`, `q`, the
geometry's number type, `s_kick`, `gamma`, `kick_factor` and `pz_scale`), leaving `pz` unchanged.
On a backend without `Float64` (e.g. Metal) the returned vector has element type `Float32` (`CSR.csr_float_type`).
The `rrule`'s pullback needs a backend with `Float64` (the CPU).

Differentiable with ForwardDiff `Dual`s in any of these: the particles (`pz`, `z`, `q`), the geometry (built by
`CSRGeometry` from `Dual` lengths, curvatures or centroid), `s_kick`, `gamma`, `kick_factor`, `pz_scale`. The kick
tables then carry derivatives; each retarded source point gets its derivatives by Newton steps from its primal
root (`root_tangent`: implicit differentiation at first order; nested `Dual`s, e.g. Hessians, also work). Held
constant: the bin grid (it depends on the extreme live particles) and the discrete choices (number of steps
`csr_n_step`, the element a source lies in, the adaptive quadrature's subdivision). GTPSA TPS in the same arguments
give the kick's expansion (one Newton step per order; computed on the calling thread). With ChainRulesCore loaded an
`rrule` gives the pullback with respect to `pz`, `z`, `q`, `geom`, `s_kick` and `gamma` (not the keywords).

Dynamic-descriptor TPS arguments (`@vars(d, dynamic = true)`) must all be in GTPSA's current Descriptor (error
otherwise); static-descriptor TPS (the `@vars` default) carry their Descriptor in the type and are not checked.
"""
function csr_kick_pz(pz::AbstractVector, z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool},
                     geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma, species::Species,
                     params::CSRParams; kick_factor = 1.0, pz_scale = 1.0)
  check_descriptors("csr_kick_pz"; pz, z, q, geom, s_kick, gamma, kick_factor, pz_scale)
  pz_new = similar(pz, device_number_type(get_backend(pz), promote_type(eltype(pz), eltype(z), eltype(q),
                                          number_type(geom), typeof(s_kick), typeof(gamma), typeof(kick_factor),
                                          typeof(pz_scale))))
  copyto!(pz_new, pz)
  csr_kick!(pz_new, z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor, pz_scale)
  return pz_new
end

"""
    csr_binned_kick(z, q, alive, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0)

Bin the particles and compute the binned kick (all of csr_kick! except applying it). Returns the intermediate
quantities the adjoint needs: bins, kick, free-space tables (I_csr, I_int, s_source), image tables per layer,
the kick coefficient and the step length.
"""
function csr_binned_kick(z::AbstractVector, q::AbstractVector, alive::AbstractVector{Bool}, geom::CSRGeometry,
                         ix_ele_kick::Integer, s_kick, gamma, species::Species, params::CSRParams; kick_factor = 1.0)
  n_bin = params.n_bin
  params.bmad_compat == geom.bmad_compat || error("csr_kick!: params.bmad_compat and geom.bmad_compat differ")
  count(alive) >= n_bin || error("csr_kick!: number of live particles ($(count(alive))) is less than n_bin ($n_bin)")
  bins = csr_bin_particles(z, q, alive, params)
  return csr_bins_kick(bins, get_backend(z), geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor)
end

"""
    csr_bins_kick(bins, backend, geom, ix_ele_kick, s_kick, gamma, species, params; kick_factor = 1.0)

The part of `csr_binned_kick` after binning: kick tables, convolutions and kick per bin, for given `bins` (arrays on
`backend`). Returns the same named tuple.
"""
function csr_bins_kick(bins::CSRBins, backend, geom::CSRGeometry, ix_ele_kick::Integer, s_kick, gamma,
                       species::Species, params::CSRParams; kick_factor = 1.0)
  n_bin = params.n_bin
  tb = table_backend(backend)

  e = geom.ele[ix_ele_kick]
  _, ds_step = csr_n_step(e.L, params)
  actual_track_step = ds_step * (e.L_chord / e.L)

  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)
  dz_t = tb === backend ? bins.dz : Float64(bins.dz)   # tables off the particle backend: Float64 lags
  I_csr, I_int, s_source = to_backend.(Ref(backend),
    csr_wake_table(geom, kp, dz_t, n_bin, tb; all_bin_quadrature = params.all_bin_quadrature && !params.bmad_compat))

  coef = device_scalar(backend, csr_kick_coef(species, gamma, actual_track_step; params.bmad_compat))
  edge = bins.edge_dcharge_density_dz
  kick = KernelAbstractions.zeros(backend, promote_type(typeof(coef), eltype(I_int), eltype(edge)), n_bin)
  csr_convolve!(kick, I_int, edge, coef)
  # Shielding: image layer ns at height ns*h with sign (-1)^ns; the factor 2 counts the images above and below.
  # image_layer_weight averages the slowly converging layer sum (params.image_sum_order).
  images = map(1:params.n_shield_images) do ns
    kp_ns = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; y_source = ns * params.beam_chamber_height,
                         kick_factor = (-1)^ns * 2 * image_layer_weight(ns, params) * kick_factor)
    image, _ = csr_image_table(geom, kp_ns, dz_t, n_bin, tb; bin_integrate = !params.bmad_compat)
    image = to_backend(backend, image)
    csr_image_convolve!(kick, image, bins.charge, coef)
    image
  end
  return (; bins, kick, I_csr, I_int, s_source, images, coef,
          ds_kick = device_scalar(backend, kick_factor * actual_track_step))
end

"""
    csr_step_result(st) -> CSRStepResult

Per-meter diagnostics from the intermediates of `csr_binned_kick`.
"""
csr_step_result(st) = CSRStepResult(st.bins.z_center, st.bins.charge ./ st.bins.dz, st.kick ./ st.ds_kick,
                                    st.I_csr ./ st.ds_kick, st.s_source)
