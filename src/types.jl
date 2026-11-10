# Types: parameters, floor positions, centroid geometry, bins,
# kick-point and per-lag work structs, results.

"""
    FloorPosition(r, theta)

Position `r = (X, Y, Z)` [m] and horizontal orientation angle `theta`
[rad] in floor coordinates. The number type `T` is promoted from the
arguments (as a float).

Produced by `FloorPosition()`, `planar_survey`, `advance_floor`,
`advance_floor_patch` and `centroid_floor`.

# Fields
- `r`: position `(X, Y, Z)` [m], an `SVector{3,T}` (any
  3-element vector in the constructor).
- `theta`: horizontal orientation angle [rad].
"""
struct FloorPosition{T<:Number}
  r::SVector{3,T}
  theta::T
end

function FloorPosition(r::AbstractVector, theta::Number)
  T = float(promote_type(eltype(r), typeof(theta)))
  return FloorPosition{T}(SVector{3,T}(r), T(theta))
end

Base.convert(::Type{FloorPosition{T}},
             f::FloorPosition) where {T} = FloorPosition{T}(f.r,
                 f.theta)

"""
    FloorPosition()

Origin, orientation angle 0.

# Returns
- `FloorPosition{Float64}` with `r = (0, 0, 0)`, `theta = 0`.
"""
FloorPosition() = FloorPosition(SVector(0.0, 0.0, 0.0), 0.0)

"""
    CSRElementInfo

Per-element centroid geometry (Bmad `csr_ele_info_struct`). The
centroid "chord" is the line from the centroid position at the element
entrance to the centroid position at the exit. `spline` gives the
centroid offset from the chord as a function of distance along the
chord. Number type `T` as `FloorPosition`. Built by `CSRGeometry`
(`geom.ele[i]`).

# Fields
- `L`: element (reference) length [m].
- `g`: reference curvature [1/m] (0 for straight elements).
- `s_start`: reference s [m] at the element entrance.
- `ref_floor0`, `ref_floor1`: reference `FloorPosition` at the
  entrance and exit.
- `floor0`, `floor1`: centroid `FloorPosition` at the entrance and
  exit.
- `theta_chord`: angle [rad] of the chord in the floor Z-X plane.
- `L_chord`: chord length [m]; negative if the centroid moves
  backwards in the element.
- `spline`: coefficients `c` of the centroid offset from the chord,
  `c[1] + c[2] u + c[3] u² + c[4] u³` at chord distance `u`
  (`spline_val`); zero for a chord shorter than `1e-8`.
- `dL_s`: centroid path length minus `L_chord` [m].
"""
struct CSRElementInfo{T<:Number}
  L::T                         # Element (reference) length
  # Reference curvature (0 for straight elements)
  g::T
  s_start::T                   # Reference s at the element entrance
  ref_floor0::FloorPosition{T} # Reference floor position at entrance
  ref_floor1::FloorPosition{T} # Reference floor position at exit
  floor0::FloorPosition{T}     # Centroid floor position at entrance
  floor1::FloorPosition{T}     # Centroid floor position at exit
  # Angle of the chord in the floor Z-X plane
  theta_chord::T
  # Chord length. Negative if the centroid moves backwards in the
  # element
  L_chord::T
  # Centroid offset from the chord vs. chord distance
  spline::SVector{4,T}
  dL_s::T                      # Centroid path length minus L_chord
end

function CSRElementInfo(L, g, s_start, ref_floor0::FloorPosition,
                        ref_floor1::FloorPosition,
                        floor0::FloorPosition,
                        floor1::FloorPosition, theta_chord, L_chord,
                        spline::StaticVector{4}, dL_s)
  T = float(promote_type(typeof(L), typeof(g), typeof(s_start),
                         number_type(ref_floor0),
                         number_type(ref_floor1),
                         number_type(floor0), number_type(floor1),
                         typeof(theta_chord), typeof(L_chord),
                         eltype(spline), typeof(dL_s)))
  return CSRElementInfo{T}(L, g, s_start, ref_floor0, ref_floor1,
                           floor0, floor1, theta_chord, L_chord,
                           spline, dL_s)
end

"""
    CSRGeometry

Centroid geometry for a beamline: `start` is the centroid floor
position at the start of the lattice (Bmad `eleinfo(0)%floor1`), and
`ele[i]` the information for element `i`. Number type `T` as
`FloorPosition`. Built by the `CSRGeometry(L, g, ...)` constructors.

# Fields
- `start`: centroid `FloorPosition` at the start of the lattice.
- `ele`: `Vector{CSRElementInfo{T}}`, one entry per element.
- `bmad_compat`: the constructor's `bmad_compat`; must equal
  `CSRParams` `bmad_compat` in `csr_kick!`.
"""
struct CSRGeometry{T<:Number}
  start::FloorPosition{T}
  ele::Vector{CSRElementInfo{T}}
  bmad_compat::Bool
end

"""
    CSRKickPoint

Kick-point information that is the same for all source lags (part of
Bmad's `csr_struct`). Number type `T` as `FloorPosition`. Built by
`CSRKickPoint(geom, ix_ele_kick, s_kick, gamma; ...)`.

# Fields
- `ix_ele_kick`: index into `geom.ele` of the element containing
  the kick point.
- `s_kick`: kick point, reference s [m] from the element entrance.
- `s_chord_kick`: kick point along the centroid chord [m]
  (`s_ref_to_s_chord`).
- `floor_k`: `FloorPosition` of the kick point.
- `gamma`, `gamma2`, `beta`: Lorentz factor `γ` of the reference
  particle, `γ²` and `β = √(1 - 1/γ²)`.
- `y_source`: height [m] of the source particle (image charges; 0
  for free space).
- `kick_factor`: multiplies every kick computed with this point
  (1/2 at element ends; image sign and weight).
"""
struct CSRKickPoint{T<:Number}
  ix_ele_kick::Int           # Element containing the kick point
  # Kick point, reference s from the element entrance
  s_kick::T
  s_chord_kick::T            # Kick point along the centroid chord
  floor_k::FloorPosition{T}  # Floor position of the kick point
  gamma::T
  gamma2::T
  beta::T
  # Height of the source particle (image charges; 0 for free space)
  y_source::T
  kick_factor::T             # Scales the kick (1/2 at element ends)
end

function CSRKickPoint(ix_ele_kick::Integer, s_kick, s_chord_kick,
                      floor_k::FloorPosition, gamma, gamma2, beta,
                      y_source, kick_factor)
  T = float(promote_type(typeof(s_kick), typeof(s_chord_kick),
                         number_type(floor_k), typeof(gamma),
                         typeof(gamma2),
                         typeof(beta), typeof(y_source),
                         typeof(kick_factor)))
  return CSRKickPoint{T}(ix_ele_kick, s_kick, s_chord_kick, floor_k,
                         gamma, gamma2, beta, y_source, kick_factor)
end

"""
    CSRKick1{T}

Kick quantities for one source-to-kick separation (Bmad
`csr_kick1_struct`), in number type `T`. Immutable (isbits for isbits
`T`): the source-point functions return an updated copy
(`with_fields`). Built by the `CSRKick1` keyword constructors.

# Fields
- `I_csr`: kick integral (Bmad `I_csr`).
- `I_int_csr`: kick integral integrated over the bin.
- `L_vec`: source-to-kick vector [m] in floor coordinates.
- `L`: distance [m] between the source and kick points.
- `dL`: `Ls - L` [m], `Ls` the centroid path length from source
  to kick (negative for a source ahead of the kick point).
- `dz_particles`: lag [m], kicked minus source particle position
  at constant time.
- `s_chord_source`: source point along the source element chord
  [m].
- `theta_L`: angle [rad] of `L_vec`.
- `theta_sl`: angle [rad] between the source velocity and `L_vec`.
- `theta_lk`: angle [rad] between `L_vec` and the kicked particle
  velocity.
- `ix_ele_source`: source element (0 = upstream of the lattice,
  `length(geom.ele) + 1` = downstream, `source_s`).
- `floor_s`: `FloorPosition` of the source point.
"""
struct CSRKick1{T<:Number}
  I_csr::T                    # Kick integral
  I_int_csr::T                # Integrated kick integral
  # Source-to-kick vector in floor coordinates
  L_vec::SVector{3,T}
  # Distance between source and kick points
  L::T
  dL::T                       # Ls - L
  # Kicked minus source particle position at constant time
  dz_particles::T
  # Source point along the source element chord
  s_chord_source::T
  theta_L::T                  # Angle of L_vec
  # Angle between source velocity and L_vec
  theta_sl::T
  # Angle between L_vec and kicked particle velocity
  theta_lk::T
  # Source element (0 = upstream of the lattice)
  ix_ele_source::Int
  floor_s::FloorPosition{T}   # Floor position of the source point
end

"""
    CSRKick1{T}(; ix_ele_source = 0, dz_particles = 0)
    CSRKick1(; ix_ele_source = 0, dz_particles = 0)
    CSRKick1(geom, kp; ix_ele_source = kp.ix_ele_kick,
             dz_particles = 0)

Zeroed work struct in number type `T` (default `Float64`;
`CSRKick1(geom, kp)`: promoted from the geometry and the kick point)
with the source search's start element `ix_ele_source` and the lag
`dz_particles`.

# Arguments
- `T`: number type of the fields.
- `geom`: `CSRGeometry`.
- `kp`: `CSRKickPoint` (from `CSRKickPoint(geom, ...)`).
- `ix_ele_source`: start element of the source search, an index
  into `geom.ele` (0 = upstream of the lattice).
- `dz_particles`: lag [m] (converted to `T`).

# Returns
- `CSRKick1{T}` with all other fields zero.
"""
function CSRKick1{T}(; ix_ele_source::Integer = 0,
                     dz_particles = zero(T)) where {T}
  z = zero(T)
  return CSRKick1{T}(z, z, SVector(z, z, z), z, z, dz_particles, z, z,
                     z, z, ix_ele_source,
                     FloorPosition(SVector(z, z, z), z))
end

CSRKick1(; kw...) = CSRKick1{Float64}(; kw...)

CSRKick1(geom, kp; ix_ele_source::Integer = kp.ix_ele_kick,
         dz_particles = 0) =
  CSRKick1{promote_type(number_type(geom), number_type(kp))}(;
      ix_ele_source, dz_particles)

"""
    with_fields(k::CSRKick1; fields...) -> CSRKick1

Copy of `k` with the given fields replaced (converted to `k`'s number
type).

# Arguments
- `k`: `CSRKick1` (from `CSRKick1(...)` or a source-point function).
- `fields...`: keyword arguments `name = value`, `name` a field of
  `CSRKick1`.

# Returns
- `CSRKick1{T}`, `T = number_type(k)`.
"""
@inline function with_fields(k::CSRKick1{T}; kw...) where {T}
  nt = NamedTuple{fieldnames(CSRKick1)}(ntuple(i -> getfield(k, i),
      Val(fieldcount(CSRKick1))))
  return CSRKick1{T}(merge(nt, values(kw))...)
end

"""
    number_type(x)

Number type `T` of a `FloorPosition{T}`, `CSRElementInfo{T}`,
`CSRGeometry{T}`, `CSRKickPoint{T}` or `CSRKick1{T}`.

# Arguments
- `x`: a `FloorPosition`, `CSRElementInfo`, `CSRGeometry`,
  `CSRKickPoint` or `CSRKick1`.

# Returns
- the type `T`.
"""
number_type(::FloorPosition{T}) where {T} = T
number_type(::CSRElementInfo{T}) where {T} = T
number_type(::CSRGeometry{T}) where {T} = T
number_type(::CSRKickPoint{T}) where {T} = T
number_type(::CSRKick1{T}) where {T} = T

"""
    CSRBins

Longitudinal bins. Bin `i` spans `[z0_edge[i], z0_edge[i] + dz)`.
`edge_dcharge_density_dz[i]` is the charge density gradient at the
left edge of bin `i`: `(charge[i] - charge[i-1]) / dz²` (zero for `i
= 1`). Arrays live on the backend of the particle coordinates.
Produced by `csr_bin_particles` (and `csr_bin_particles_reference`).

# Fields
- `dz`: bin width [m].
- `z0_edge`: left edge [m] of each bin.
- `z_center`: center [m] of each bin, `z0_edge .+ dz / 2`.
- `charge`: charge [C] deposited in each bin.
- `n_particle`: particle weight in each bin (sum of the triangle
  overlaps).
- `edge_dcharge_density_dz`: charge density gradient [C/m²] at the
  left edge of each bin.
"""
struct CSRBins{T,V<:AbstractVector,W<:AbstractVector}
  dz::T
  z0_edge::V
  z_center::V
  charge::W
  n_particle::V
  edge_dcharge_density_dz::W
end

"""
    CSRStepResult

Diagnostics for one kick, per bin (mirrors Bmad's
`diagnostic_output_file` columns): `z_center`, `charge_per_meter`
[C/m], `kick_per_meter` (Δpz per meter), `I_csr_per_meter`,
`s_source` (reference s of the source point). Arrays are on the
backend of the particle coordinates. Returned by `csr_kick!`
(`csr_step_result`) and `csr_kick_reference!`.

# Fields
- `z_center`: bin centers [m] (`CSRBins` `z_center`).
- `charge_per_meter`: bin charge / bin width [C/m].
- `kick_per_meter`: Δpz per bin divided by `kick_factor * ds *
  L_chord / L` of the kick element [1/m].
- `I_csr_per_meter`: free-space kick integral `I_csr` per lag,
  divided by the same length.
- `s_source`: reference s [m] of the free-space source point per
  lag.
"""
struct CSRStepResult{Z<:AbstractVector,C<:AbstractVector,
                     K<:AbstractVector,I<:AbstractVector,
                     S<:AbstractVector}
  z_center::Z
  charge_per_meter::C
  kick_per_meter::K
  I_csr_per_meter::I
  s_source::S
end

"""
    N_BIN_QUAD

Simpson intervals per bin in `u^(1/3)` for `all_bin_quadrature` (9
nodes per bin).
"""
const N_BIN_QUAD = 8

"""
    CSRParams(; n_bin = 40, particle_bin_span = 2,
              n_shield_images = 0, beam_chamber_height = 0.0,
              image_sum_order = 8, all_bin_quadrature = true,
              bmad_compat = false)

Parameters for the 1D CSR calculation. Names follow Bmad's
`space_charge_com` components. The constructor errors if a
constraint below is not met.

# Fields
- `n_bin`: Number of longitudinal bins. Must satisfy `n_bin - 2 -
  (particle_bin_span + 1) ≥ 1`.
- `particle_bin_span`: Width of each particle's triangular charge
  distribution, in units of the bin width. At least 1.
- `n_shield_images`: Number of image-charge layers for shielding by
  parallel plates (0 = free space; must be ≥ 0). Layer `n` is a pair
  of images at heights `±n * beam_chamber_height` with sign
  `(-1)^n`, as in Bmad.
- `beam_chamber_height`: Height of the beam chamber [m], i.e. the
  distance between the plates. Must be positive when `n_shield_images
  > 0`.
- `image_sum_order`: Order `k` of the averaging applied to the
  image-layer sum (0 = plain sum, at most 60). The layer terms
  alternate in sign and decay only as ~1/n, so the plain sum converges
  slowly. With `M = n_shield_images`, the kick is the binomially
  weighted average `2^-k Σ_j C(k, j) S_{M-k+j}` of the partial sums
  `S_m` (free space plus layers 1..m), i.e. layer `n` is weighted by
  `2^-k Σ_{j ≥ n-M+k} C(k, j)` (1 for `n ≤ M - k`); `k` is capped
  at `M`. Default 8: with `M ≥ 32` the steady-state kick agrees with
  a 1024-layer reference to ~1e-6 (docs, "Physics decisions", D24).
  Use `M` of 32–64 when the chamber height is comparable to or
  smaller than the bunch length. Ignored with `bmad_compat = true`
  (Bmad: plain sum).
- `all_bin_quadrature`: Integrate the kick integrand over every bin by
  quadrature (Simpson in `u^(1/3)`, 8 intervals (9 nodes) per bin),
  instead of the trapezoid `(I_csr[i] + I_csr[i-1]) dz / 2` for bins
  `i ≥ 2`. Default `true` (docs, "Physics decisions", V5). `false`
  integrates by quadrature over the first bin only. Ignored with
  `bmad_compat = true` (Bmad: trapezoid).
- `bmad_compat`: If `true`, reproduce Bmad expressions that CSR.jl
  replaces, for like-for-like comparison with Bmad output. Default
  `false`. 1. `dspline_len` s⁴ term: Bmad uses `9c₃²`, the
  integral gives `9c₃²/5` (docs, "Physics decisions", V1). 2.
  First-bin kick integral when the source lies outside the kick
  element: Bmad holds `I_csr` constant over the rest of the bin (2/3
  of the integral in steady state); CSR.jl integrates by quadrature
  (docs, "Physics decisions", V5). 3. Image charges, source ahead of
  the kick point: Bmad gives the source the reverse direction of
  motion, and does not look for sources beyond the end of the kick
  element (docs, "Physics decisions", V8). 4. Image charges: Bmad
  samples the image kernel at the bin centres and sums the layers
  plainly; CSR.jl integrates the kernel against the bin hat function
  and averages the layer sum (`image_sum_order`). 5. Path-length
  excess of the centroid over the source–kick line (retardation):
  Bmad's small-angle form `½∫(x′ - dθ)²` (`dspline_len`);
  CSR.jl the exact form (`path_excess`). They differ by
  `~Ls²/(80ρ²)` relative. 6. Kick coefficient (`csr_kick_coef`):
  Bmad's carries an extra `1/rel_mass` (`rel_mass = m/m_e`) and omits
  the `1/β²` of `Δpz = ΔE/(β²E)` (docs, "Physics decisions",
  V2). 7. Kick point in a bend (`s_ref_to_s_chord`, set by the
  geometry's `bmad_compat`): Bmad's formula places it upstream by `≈
  7L³/(24ρ²)` at the exit of an on-axis bend; CSR.jl takes the
  crossing of the centroid spline with the reference normal plane
  (V3). 8. Retardation condition (`ddz_calc_csr`): Bmad's source in
  the kick element uses the secant slope of the spline as the angle of
  the source–kick line and `2Δs_chord` for sources ahead, and
  `L/(2γ²)` for `L(1 - β)`. These make `ddz` jump at the kick
  element's ends and the lattice ends, giving two roots or none for
  some lags (the source then depends on where the search starts).
  CSR.jl uses the angle, `2 L_plane` and `L/(γ²(1 + β))`: `ddz` is
  continuous and each lag has one source (docs, "Physics decisions",
  seam). 9. First-bin closed form over the kick element: Bmad's
  `log(2γ² z/Ls)` vanishes on a straight line only with its
  `L/(2γ²)` retardation; CSR.jl uses `log(γ²(1 + β) z/Ls)` to
  match item 8 (docs, "Physics decisions", D42).
"""
Base.@kwdef struct CSRParams
  n_bin::Int = 40
  particle_bin_span::Int = 2
  n_shield_images::Int = 0
  beam_chamber_height::Float64 = 0.0
  image_sum_order::Int = 8
  all_bin_quadrature::Bool = true
  bmad_compat::Bool = false
  function CSRParams(n_bin, particle_bin_span, n_shield_images,
                     beam_chamber_height, image_sum_order,
                     all_bin_quadrature, bmad_compat)
    n_bin_eff = n_bin - 2 - (particle_bin_span + 1)
    n_bin_eff >= 1 || error("CSRParams: n_bin (= $n_bin) must be greater than 3 + particle_bin_span (= $particle_bin_span)")
    particle_bin_span >= 1 ||
        error("CSRParams: particle_bin_span must be at least 1")
    n_shield_images >= 0 ||
        error("CSRParams: n_shield_images must be non-negative")
    n_shield_images == 0 || beam_chamber_height > 0 ||
      error("CSRParams: beam_chamber_height must be positive when n_shield_images > 0")
    0 <= image_sum_order <= 60 ||
        error("CSRParams: image_sum_order must be in 0:60")
    return new(n_bin, particle_bin_span, n_shield_images,
               beam_chamber_height, image_sum_order,
               all_bin_quadrature, bmad_compat)
  end
end
