# One CSR kick (host side): bin, build the kick table, convolve,
# apply. Port of the per-step part of Bmad's track1_bunch_csr and
# csr_bin_kicks (CSR only, with image charges).

"""
    csr_convolve!(kick, I_int, edge_grad, coef)

`kick[i] = coef * Σ_{k=1}^{i} I_int[k] * edge_grad[i+1-k]`. One work
item per bin, on the backend of `kick`.

# Arguments
- `kick`: kick per bin (modified: overwritten), length `n_bin`.
- `I_int`: bin integrals of the kick integral, from
  `csr_wake_table` (at least `length(kick)` entries).
- `edge_grad`: charge density gradient at the bin edges
  (`bins.edge_dcharge_density_dz` from `csr_bin_particles`).
- `coef`: kick coefficient, from `csr_kick_coef`.

# Returns
- `kick`.
"""
function csr_convolve!(kick::AbstractVector, I_int::AbstractVector,
                       edge_grad::AbstractVector, coef)
  backend = get_backend(kick)
  n = length(kick)
  csr_convolve_kernel!(backend)(kick, I_int, edge_grad, coef;
                                ndrange = n)
  KernelAbstractions.synchronize(backend)
  return kick
end

"""
    csr_image_convolve!(kick, image, charge, coef)

`kick[i] += coef * Σ_{j=1}^{n_bin} image(i - j) * charge[j]`, with
the image kick for lag `l` stored at `image[l + n_bin + 1]`
(`csr_image_table`). Bmad's image-charge sum (all lags, bin charge
rather than its gradient). One work item per bin. Errors unless
`length(image) == 2length(kick) + 1`.

# Arguments
- `kick`: kick per bin (modified: added to), length `n_bin`.
- `image`: image-charge table for lags `-n_bin:n_bin`, from
  `csr_image_table`.
- `charge`: bin charges [C] (`bins.charge` from
  `csr_bin_particles`).
- `coef`: kick coefficient, from `csr_kick_coef`.

# Returns
- `kick`.
"""
function csr_image_convolve!(kick::AbstractVector,
                             image::AbstractVector,
                             charge::AbstractVector, coef)
  length(image) == 2length(kick) + 1 ||
      error("csr_image_convolve!: image must have 2n_bin + 1 entries")
  backend = get_backend(kick)
  n = length(kick)
  csr_image_convolve_kernel!(backend)(kick, image, charge, coef;
                                      ndrange = n)
  KernelAbstractions.synchronize(backend)
  return kick
end

"""
    csr_apply_kicks!(pz, z, alive, bins, kick; pz_scale = 1)

Add the kick, linearly interpolated between bin centers and multiplied
by `pz_scale`, to `pz` of each live particle. The interpolation uses
the scalar value of `z` (`Beamlines.scalarize`), and only the value of
`pz` changes (a `Dual` or TPS `pz` keeps its partials or higher
coefficients). As in Bmad's CSR-only branch, `z` is not rescaled for
the change in velocity. One work item per particle, on the backend of
`pz`. Errors if `pz`, `z`, `alive` differ in length, `kick` and
`bins.z_center` differ in length, `bins` or `kick` do not hold
floating-point numbers (`check_float_bins`), or a live particle lies
outside the bin range (internal error).

# Arguments
- `pz`: particle `pz` coordinates (modified).
- `z`: particle `z` coordinates [m].
- `alive`: live-particle mask.
- `bins`: `CSRBins` of these particles, from `csr_bin_particles`
  (floating-point numbers).
- `kick`: Δpz per bin (`st.kick` from `csr_bins_kick`,
  floating-point numbers).
- `pz_scale`: multiplies the applied kick.

# Returns
- `pz`.
"""
function csr_apply_kicks!(pz::AbstractVector, z::AbstractVector,
                          alive::AbstractVector{Bool}, bins::CSRBins,
                          kick::AbstractVector;
                          pz_scale = 1)
  length(pz) == length(z) == length(alive) ||
    error("csr_apply_kicks!: pz, z, alive must have equal lengths (got $(length(pz)), $(length(z)), $(length(alive)))")
  length(kick) == length(bins.z_center) ||
      error("csr_apply_kicks!: kick and bins.z_center must have equal lengths")
  check_float_bins("csr_apply_kicks!", bins, kick)
  apply_kicks_outside!(pz, z, alive, bins.z_center, kick, bins.dz,
                       pz_scale) == 0 ||
    error("csr_apply_kicks!: particle outside the bin range (internal error)")
  return pz
end

"""
    apply_kicks_outside!(pz, z, alive, z_center, kick, dz,
                         pz_scale) -> n_outside

Launch `csr_apply_kicks_kernel!` on the backend of `pz`; returns the
number of live particles outside the table.

# Arguments
- `pz`: particle `pz` coordinates (modified for the particles
  inside the table).
- `z`: particle `z` coordinates [m].
- `alive`: live-particle mask.
- `z_center`: bin centers [m] (`bins.z_center`).
- `kick`: Δpz per bin, same length as `z_center`.
- `dz`: bin width [m] (`bins.dz`).
- `pz_scale`: multiplies the applied kick.

# Returns
- `n_outside::Int` (0 for an empty `pz`).
"""
function apply_kicks_outside!(pz, z, alive, z_center, kick, dz,
                              pz_scale)
  length(pz) == 0 && return 0
  backend = get_backend(pz)
  # Int32: no 64-bit atomics on some GPUs
  n_outside = KernelAbstractions.zeros(backend, Int32, 1)
  n = length(pz)
  # TPS pz on the CPU backend: one workgroup, which the
  # KernelAbstractions CPU backend runs on the calling thread.
  # GTPSA's C library takes scratch TPSs from a per-thread stack
  # indexed by the OpenMP thread number (0 on every Julia thread), so
  # concurrent TPS operations may corrupt memory.
  wg = (eltype(pz) <: Real || !(backend isa CPU)) ? nothing : n
  csr_apply_kicks_kernel!(backend)(pz, n_outside, z, alive, z_center,
                                   kick, dz, pz_scale; ndrange = n,
                                   workgroupsize = wg)
  KernelAbstractions.synchronize(backend)
  return Int(Array(n_outside)[1])
end

"""
    count_outside(backend, z, alive, z_center, dz) -> n_outside

Number of live particles outside the kick table
(`csr_count_outside_kernel!` on `backend`), without kicking.

# Arguments
- `backend`: KernelAbstractions backend of `z` and `alive`.
- `z`: particle `z` coordinates [m].
- `alive`: live-particle mask.
- `z_center`: bin centers [m] (`bins.z_center`).
- `dz`: bin width [m] (`bins.dz`).

# Returns
- `n_outside::Int` (0 for an empty `z`).
"""
function count_outside(backend, z, alive, z_center, dz)
  length(z) == 0 && return 0
  # Int32: no 64-bit atomics on some GPUs
  n_outside = KernelAbstractions.zeros(backend, Int32, 1)
  n = length(z)
  csr_count_outside_kernel!(backend)(n_outside, z, alive, z_center,
                                     dz; ndrange = n)
  KernelAbstractions.synchronize(backend)
  return Int(Array(n_outside)[1])
end

"""
    check_float_pz(fname, name, pz)

Error naming function `fname` and argument `name` if `pz` has an
`Integer` element type (the kick needs floating-point values).

# Arguments
- `fname`: name of the calling function, for the error message.
- `name`: name of the argument (`"pz"` or `"pz_t"`).
- `pz`: the array the kick is added to.
"""
function check_float_pz(fname, name, pz)
  eltype(pz) <: Integer &&
    error("$fname: $name has element type $(eltype(pz)); it must hold floating-point values (e.g. float.($name)).")
  return nothing
end

"""
    check_float_bins(fname, bins, kick = nothing)

Error naming function `fname` unless `bins` (and `kick`, if given)
hold floating-point numbers, as they do when they come from
`csr_bin_particles` and `csr_binned_kick`.

# Arguments
- `fname`: name of the calling function, for the error message.
- `bins`: `CSRBins`.
- `kick`: Δpz per bin, or `nothing`.
"""
function check_float_bins(fname, bins::CSRBins, kick = nothing)
  bins.dz isa AbstractFloat &&
      eltype(bins.z0_edge) <: AbstractFloat &&
      eltype(bins.z_center) <: AbstractFloat &&
      eltype(bins.charge) <: AbstractFloat &&
      eltype(bins.n_particle) <: AbstractFloat &&
      eltype(bins.edge_dcharge_density_dz) <: AbstractFloat ||
    error("$fname: bins must hold floating-point numbers (use csr_bin_particles or csr_binned_kick).")
  kick === nothing || eltype(kick) <: AbstractFloat ||
    error("$fname: kick must hold floating-point numbers (use csr_binned_kick).")
  return nothing
end

"""
    check_test_particles(fname, pz_t, z_t, alive_t)

Error naming function `fname` unless the test particle arrays `pz_t`,
`z_t` and `alive_t` have equal lengths, or if `pz_t` has an `Integer`
element type (`check_float_pz`).

# Arguments
- `fname`: name of the calling function, for the error message.
- `pz_t`, `z_t`, `alive_t`: test particle `pz`, `z` [m] and
  live-particle mask (`test_particles` of `csr_kick!`).
"""
function check_test_particles(fname, pz_t, z_t, alive_t)
  length(pz_t) == length(z_t) == length(alive_t) ||
    error("$fname: test particle arrays pz_t, z_t, alive_t have lengths $(length(pz_t)), $(length(z_t)), " *
          "$(length(alive_t)); they must be equal.")
  check_float_pz(fname, "pz_t", pz_t)
  return nothing
end

"""
    csr_apply_test_kicks!(pz_t, z_t, alive_t, bins, kick;
                          pz_scale = 1)

Frozen-wake kick on test particles: `csr_apply_kicks!` with the bins
and kick of a bunch the test particles are not part of (they deposit
no charge). `z_t` and `pz_scale` are scalarized
(`Beamlines.scalarize`; `Dual` or TPS allowed), so each test particle
gets the kick at its scalar `z`, and the kick changes only the value
of `pz_t` (a `Dual` or TPS `pz_t` keeps its partials or higher
coefficients). The test particles' arrays may live on another backend
than the bunch; `bins` and `kick` are copied to the CPU only when the
test particles are on the CPU.

Errors, raised before any test particle is kicked (`pz_t` is then
unchanged):
- a live test particle outside the bunch's bin range, `z_center[1] -
  dz/100 ≤ z < z_center[end]` (the range `csr_apply_kicks_kernel!`
  interpolates over; counted first by `count_outside`);
- `pz_t`, `z_t`, `alive_t` of different lengths;
- `pz_t` with an `Integer` element type;
- `kick` and `bins.z_center` of different lengths;
- `bins` or `kick` not holding floating-point numbers
  (`check_float_bins`).

# Arguments
- `pz_t`: test particle `pz` coordinates (modified); a
  floating-point, `Dual` or TPS element type.
- `z_t`: test particle `z` coordinates [m].
- `alive_t`: test particle live mask.
- `bins`: the bunch's `CSRBins` (`st.bins` from `csr_binned_kick`,
  floating-point numbers).
- `kick`: the bunch's Δpz per bin (`st.kick` from
  `csr_binned_kick`, floating-point numbers).
- `pz_scale`: multiplies the applied kick.

# Returns
- `pz_t`.
"""
function csr_apply_test_kicks!(pz_t::AbstractVector,
                               z_t::AbstractVector,
                               alive_t::AbstractVector{Bool},
                               bins::CSRBins,
                               kick::AbstractVector; pz_scale = 1)
  z_t = scalar_array(z_t)
  pz_scale = Beamlines.scalarize(pz_scale)
  check_test_particles("csr_apply_test_kicks!", pz_t, z_t, alive_t)
  check_float_bins("csr_apply_test_kicks!", bins, kick)
  length(kick) == length(bins.z_center) ||
    error("csr_apply_test_kicks!: kick and bins.z_center must have equal lengths")
  cpu(A) = A isa Array ? A : Array(A)
  backend = get_backend(pz_t)
  zc, k = backend isa CPU ? (cpu(bins.z_center),
                             cpu(kick)) : (bins.z_center, kick)
  dz = bins.dz
  n = count_outside(backend, z_t, alive_t, zc, dz)
  n == 0 || error("csr_apply_test_kicks!: $n test particle(s) outside the bunch's bin range " *
                  "[$(first(zc) - bins.dz / 100), $(last(zc)))")
  apply_kicks_outside!(pz_t, z_t, alive_t, zc, k, dz,
                       device_scalar(backend, pz_scale)) == 0 ||
    error("csr_apply_test_kicks!: test particle outside the bin range (internal error)")
  return pz_t
end

"""
    csr_kick!(pz, z, q, alive, geom, ix_ele_kick, ds, s_kick, gamma,
              species, params; kick_factor = 1.0,
              pz_scale = 1.0,
              test_particles = nothing) -> CSRStepResult

Apply one 1D CSR kick at reference position `s_kick` [m] from the
entrance of element `ix_ele_kick`. Errors if `pz` (or `pz_t`) has an
`Integer` element type (before any kick), the array lengths differ
or a live particle has a non-finite `z`, or unless `ds > 0`; also if
fewer than `params.n_bin` particles are live, `params.bmad_compat`
differs from `geom.bmad_compat`, or `gamma ≤ 1`.

The kick covers one step of length `ds` scaled by `L_chord / L`, as in
Bmad.

With `params.n_shield_images > 0` the image-charge kicks of the
shielding plates are added (Bmad's image part of `csr_bin_kicks`),
with the layer sum averaged per `params.image_sum_order`.
`kick_per_meter` includes them; `I_csr_per_meter` and `s_source` are
for free space only.

Runs as KernelAbstractions kernels on the backend of `pz` (deposit,
edge gradient, convolutions, kick). The wake and image tables are
always computed on the CPU (`table_backend`; with `Float64` lags for
a GPU backend) and copied to a GPU backend in
`csr_float_type(backend)` (`Float32` where the backend has no
`Float64`, e.g. Metal); scalars passed to the kernels are converted
likewise. `csr_kick_reference!` is the serial phase 1 version.

The number inputs `z`, `q`, `geom`, `ds`, `s_kick`, `gamma`,
`kick_factor`, `pz_scale` and the test particles' `z_t` may be
ForwardDiff `Dual` or GTPSA TPS: they are scalarized
(`Beamlines.scalarize`) and the kick is computed in real numbers. The
kick changes only the value of `pz` and `pz_t` (a `Dual` or TPS `pz`
keeps its partials or higher coefficients): no derivatives are
tracked through the kick.

# Arguments
- `pz`, `z`: particle coordinates (`pz` is modified, with a
  floating-point, `Dual` or TPS element type; `z` [m]).
- `q`: macroparticle charges [C].
- `alive`: live-particle mask.
- `geom`: centroid geometry, from `CSRGeometry`.
- `ix_ele_kick`: index into `geom.ele` of the kick's element.
- `ds`: step length [m] the kick covers (`> 0`).
- `s_kick`: reference position [m] of the kick, from the entrance
  of element `ix_ele_kick`.
- `gamma`: Lorentz factor of the reference particle at the kick point.
- `species`: particle species (`Species`, AtomicAndPhysicalConstants;
  mass and charge enter `csr_kick_coef`).
- `params`: `CSRParams`.
- `kick_factor`: `1/2` at the first and last step of an element, `1`
  otherwise.
- `pz_scale`: multiplies the Δpz applied to the particles (not the
  returned diagnostics). Used where the particles' reference momentum
  differs from the one the kick is computed for (inside a SaganCavity,
  whose reference momentum steps at each cell while γ ramps
  linearly).
- `test_particles`: `(pz_t, z_t, alive_t)` or `nothing`. Test
  particles get the same kick (`pz_t` modified) without contributing
  to it, e.g. particles for a transfer map with the bunch's wake
  frozen (`csr_apply_test_kicks!`). They are kicked before the bunch,
  so an error from `csr_apply_test_kicks!` (a live test particle
  outside the bin range, or arrays of different lengths, checked
  before the kick tables are built) leaves `pz` and `pz_t` unchanged.

# Returns
- `CSRStepResult` with the per-bin diagnostics of this kick
  (`csr_step_result`).
"""
function csr_kick!(pz::AbstractVector, z::AbstractVector,
                   q::AbstractVector, alive::AbstractVector{Bool},
                   geom::CSRGeometry, ix_ele_kick::Integer, ds,
                   s_kick, gamma, species::Species,
                   params::CSRParams; kick_factor = 1.0,
                   pz_scale = 1.0, test_particles = nothing)
  check_float_pz("csr_kick!", "pz", pz)
  z = scalar_array(z)
  q = scalar_array(q)
  geom = Beamlines.scalarize(geom)
  ds = Beamlines.scalarize(ds)
  s_kick = Beamlines.scalarize(s_kick)
  gamma = Beamlines.scalarize(gamma)
  kick_factor = Beamlines.scalarize(kick_factor)
  pz_scale = Beamlines.scalarize(pz_scale)
  ds > 0 || error("csr_kick!: ds must be > 0 (got $ds)")
  test_particles === nothing || check_test_particles("csr_kick!",
      test_particles...)
  st = csr_binned_kick(z, q, alive, geom, ix_ele_kick, ds, s_kick,
                       gamma, species, params; kick_factor)
  test_particles === nothing ||
      csr_apply_test_kicks!(test_particles..., st.bins, st.kick;
                            pz_scale)
  csr_apply_kicks!(pz, z, alive, st.bins, st.kick;
                   pz_scale = device_scalar(get_backend(pz),
      pz_scale))
  return csr_step_result(st)
end

"""
    csr_kick_pz(pz, z, q, alive, geom, ix_ele_kick, ds, s_kick, gamma,
                species, params; kick_factor = 1.0,
                pz_scale = 1.0) -> pz_new

Non-mutating `csr_kick!`: returns `pz` after the kick in a new vector
`similar(pz)` (element type `eltype(pz)`), leaving `pz` unchanged.
Elements of a non-isbits type are copied with `copy` (a new object
for a TPS), so kicking the result never changes `pz`. A `Float32`
`pz` gives a `Float32` result also with `Float64` inputs. Number
inputs as in `csr_kick!` (scalarized; the kick changes only the value
of `pz`). Errors as `csr_kick!` (an `Integer` element type of `pz`
before any kick).

# Arguments
- `pz`: particle `pz` coordinates (not modified), with a
  floating-point, `Dual` or TPS element type.
- `z`, `q`, `alive`, `geom`, `ix_ele_kick`, `ds`, `s_kick`, `gamma`,
  `species`, `params`, `kick_factor`, `pz_scale`: as in `csr_kick!`.

# Returns
- `pz_new`: kicked copy of `pz`.
"""
function csr_kick_pz(pz::AbstractVector, z::AbstractVector,
                     q::AbstractVector, alive::AbstractVector{Bool},
                     geom::CSRGeometry, ix_ele_kick::Integer, ds,
                     s_kick, gamma, species::Species,
                     params::CSRParams; kick_factor = 1.0,
                     pz_scale = 1.0)
  check_float_pz("csr_kick_pz", "pz", pz)
  z = scalar_array(z)
  q = scalar_array(q)
  geom = Beamlines.scalarize(geom)
  ds = Beamlines.scalarize(ds)
  s_kick = Beamlines.scalarize(s_kick)
  gamma = Beamlines.scalarize(gamma)
  kick_factor = Beamlines.scalarize(kick_factor)
  pz_scale = Beamlines.scalarize(pz_scale)
  ds > 0 || error("csr_kick_pz: ds must be > 0 (got $ds)")
  pz_new = similar(pz)
  # Mutable elements (e.g. TPS) are copied, so that unkicked elements
  # of pz_new are not the objects of pz
  if isbitstype(eltype(pz))
    copyto!(pz_new, pz)
  else
    map!(copy, pz_new, pz)
  end
  csr_kick!(pz_new, z, q, alive, geom, ix_ele_kick, ds, s_kick, gamma,
            species, params; kick_factor, pz_scale)
  return pz_new
end

"""
    csr_binned_kick(z, q, alive, geom, ix_ele_kick, ds, s_kick, gamma,
                    species, params; kick_factor = 1.0)

Bin the particles and compute the binned kick (all of csr_kick! except
applying it). Returns the bins, the kick and the intermediate
quantities: free-space tables (I_csr, I_int, s_source), image tables
per layer, the kick coefficient and the step length. Number inputs
are scalarized as in `csr_kick!`. Errors as `csr_kick!` for these
arguments.

# Arguments
- `z`, `q`, `alive`, `geom`, `ix_ele_kick`, `ds`, `s_kick`, `gamma`,
  `species`, `params`, `kick_factor`: as in `csr_kick!`.

# Returns
- named tuple `(; bins, kick, I_csr, I_int, s_source, images, coef,
  ds_kick)`: `bins` from `csr_bin_particles`; `kick` the Δpz per
  bin; `I_csr`, `I_int`, `s_source` from `csr_wake_table`; `images`
  a vector of one `csr_image_table` `image` per layer; `coef` from
  `csr_kick_coef`; `ds_kick = kick_factor * ds * L_chord / L`.
  Arrays on the backend of `z`.
"""
function csr_binned_kick(z::AbstractVector, q::AbstractVector,
                         alive::AbstractVector{Bool},
                         geom::CSRGeometry,
                         ix_ele_kick::Integer, ds, s_kick, gamma,
                         species::Species, params::CSRParams;
                         kick_factor = 1.0)
  z = scalar_array(z)
  q = scalar_array(q)
  geom = Beamlines.scalarize(geom)
  ds = Beamlines.scalarize(ds)
  s_kick = Beamlines.scalarize(s_kick)
  gamma = Beamlines.scalarize(gamma)
  kick_factor = Beamlines.scalarize(kick_factor)
  ds > 0 || error("csr_binned_kick: ds must be > 0 (got $ds)")
  n_bin = params.n_bin
  params.bmad_compat == geom.bmad_compat ||
      error(
          "csr_kick!: params.bmad_compat and geom.bmad_compat differ")
  count(alive) >= n_bin || error("csr_kick!: number of live particles ($(count(alive))) is less than n_bin ($n_bin)")
  bins = csr_bin_particles(z, q, alive, params)
  return csr_bins_kick(bins, get_backend(z), geom, ix_ele_kick, ds,
                       s_kick, gamma, species, params; kick_factor)
end

"""
    csr_bins_kick(bins, backend, geom, ix_ele_kick, ds, s_kick, gamma,
                  species, params; kick_factor = 1.0)

The part of `csr_binned_kick` after binning: kick tables, convolutions
and kick per bin, for given `bins` (arrays on `backend`). Returns the
same named tuple. Number inputs are scalarized as in `csr_kick!`.
Errors if `bins` does not hold floating-point numbers
(`check_float_bins`).

# Arguments
- `bins`: `CSRBins`, from `csr_bin_particles` (floating-point
  numbers).
- `backend`: KernelAbstractions backend of `bins` (the particles'
  `get_backend(z)`).
- `geom`, `ix_ele_kick`, `ds`, `s_kick`, `gamma`, `species`,
  `params`, `kick_factor`: as in `csr_kick!`.
"""
function csr_bins_kick(bins::CSRBins, backend, geom::CSRGeometry,
                       ix_ele_kick::Integer, ds, s_kick, gamma,
                       species::Species, params::CSRParams;
                       kick_factor = 1.0)
  check_float_bins("csr_bins_kick", bins)
  geom = Beamlines.scalarize(geom)
  ds = Beamlines.scalarize(ds)
  s_kick = Beamlines.scalarize(s_kick)
  gamma = Beamlines.scalarize(gamma)
  kick_factor = Beamlines.scalarize(kick_factor)
  ds > 0 || error("csr_bins_kick: ds must be > 0 (got $ds)")
  n_bin = params.n_bin
  tb = table_backend(backend)

  e = geom.ele[ix_ele_kick]
  ds_step = ds
  actual_track_step = ds_step * (e.L_chord / e.L)

  kp = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; kick_factor)
  # tables off the particle backend: Float64 lags
  dz_t = tb === backend ? bins.dz : Float64(bins.dz)
  I_csr, I_int, s_source = to_backend.(Ref(backend),
    csr_wake_table(geom, kp, dz_t, n_bin, tb;
                   all_bin_quadrature = params.all_bin_quadrature &&
                   !params.bmad_compat))

  coef = device_scalar(backend, csr_kick_coef(species, gamma,
      actual_track_step; params.bmad_compat))
  edge = bins.edge_dcharge_density_dz
  kick = KernelAbstractions.zeros(backend, promote_type(typeof(coef),
      eltype(I_int), eltype(edge)), n_bin)
  csr_convolve!(kick, I_int, edge, coef)
  # Shielding: image layer ns at height ns*h with sign (-1)^ns; the
  # factor 2 counts the images above and below. image_layer_weight
  # averages the slowly converging layer sum (params.image_sum_order).
  # let: the closure captures the scalarized arguments (no Core.Box)
  images = let geom = geom, s_kick = s_kick, gamma = gamma,
               kick_factor = kick_factor
    map(1:params.n_shield_images) do ns
      kp_ns = CSRKickPoint(geom, ix_ele_kick, s_kick, gamma;
                           y_source = ns * params.beam_chamber_height,
                           kick_factor = (-1)^ns * 2 *
                           image_layer_weight(ns, params) *
                           kick_factor)
      image, _ = csr_image_table(geom, kp_ns, dz_t, n_bin, tb;
                                 bin_integrate = !params.bmad_compat)
      image = to_backend(backend, image)
      csr_image_convolve!(kick, image, bins.charge, coef)
      image
    end
  end
  return (; bins, kick, I_csr, I_int, s_source, images, coef,
          ds_kick = device_scalar(backend,
                                  kick_factor * actual_track_step))
end

"""
    csr_step_result(st) -> CSRStepResult

Per-meter diagnostics from the intermediates of `csr_binned_kick`.

# Arguments
- `st`: named tuple from `csr_binned_kick` or `csr_bins_kick`.

# Returns
- `CSRStepResult` (`kick` and `I_csr` divided by `st.ds_kick`,
  `charge` by `st.bins.dz`).
"""
csr_step_result(st) = CSRStepResult(st.bins.z_center,
                                    st.bins.charge ./ st.bins.dz,
                                    st.kick ./ st.ds_kick,
                                    st.I_csr ./ st.ds_kick,
                                    st.s_source)
