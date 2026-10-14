# Types: parameters, floor positions, centroid geometry, bins, kick-point and per-lag work structs, results.

"""
    FloorPosition(r, theta)

Position `r = (X, Y, Z)` [m] and horizontal orientation angle `theta` [rad] in floor coordinates. The number type
`T` is promoted from the arguments (`Float64`, or e.g. ForwardDiff `Dual`s for derivatives w.r.t. the geometry).
"""
struct FloorPosition{T<:Real}
  r::SVector{3,T}
  theta::T
end

function FloorPosition(r::AbstractVector, theta::Real)
  T = float(promote_type(eltype(r), typeof(theta)))
  return FloorPosition{T}(SVector{3,T}(r), T(theta))
end

Base.convert(::Type{FloorPosition{T}}, f::FloorPosition) where {T} = FloorPosition{T}(f.r, f.theta)

"""
    FloorPosition()

Origin, orientation angle 0.
"""
FloorPosition() = FloorPosition(SVector(0.0, 0.0, 0.0), 0.0)

"""
    CSRElementInfo

Per-element centroid geometry (Bmad `csr_ele_info_struct`). The centroid "chord" is the line from the
centroid position at the element entrance to the centroid position at the exit. `spline` gives the
centroid offset from the chord as a function of distance along the chord. Number type `T` as `FloorPosition`.
"""
struct CSRElementInfo{T<:Real}
  L::T                         # Element (reference) length
  g::T                         # Reference curvature (0 for straight elements)
  s_start::T                   # Reference s at the element entrance
  ref_floor0::FloorPosition{T} # Reference floor position at entrance
  ref_floor1::FloorPosition{T} # Reference floor position at exit
  floor0::FloorPosition{T}     # Centroid floor position at entrance
  floor1::FloorPosition{T}     # Centroid floor position at exit
  theta_chord::T               # Angle of the chord in the floor Z-X plane
  L_chord::T                   # Chord length. Negative if the centroid moves backwards in the element
  spline::SVector{4,T}         # Centroid offset from the chord vs. chord distance
  dL_s::T                      # Centroid path length minus L_chord
end

function CSRElementInfo(L, g, s_start, ref_floor0::FloorPosition, ref_floor1::FloorPosition, floor0::FloorPosition,
                        floor1::FloorPosition, theta_chord, L_chord, spline::StaticVector{4}, dL_s)
  T = float(promote_type(typeof(L), typeof(g), typeof(s_start), number_type(ref_floor0), number_type(ref_floor1),
                         number_type(floor0), number_type(floor1), typeof(theta_chord), typeof(L_chord),
                         eltype(spline), typeof(dL_s)))
  return CSRElementInfo{T}(L, g, s_start, ref_floor0, ref_floor1, floor0, floor1, theta_chord, L_chord, spline, dL_s)
end

"""
    CSRGeometry

Centroid geometry for a beamline: `start` is the centroid floor position at the start of the lattice
(Bmad `eleinfo(0)%floor1`), and `ele[i]` the information for element `i`. Number type `T` as `FloorPosition`.
"""
struct CSRGeometry{T<:Real}
  start::FloorPosition{T}
  ele::Vector{CSRElementInfo{T}}
  bmad_compat::Bool
end

"""
    CSRKickPoint

Kick-point information that is the same for all source lags (part of Bmad's `csr_struct`). Number type `T` as
`FloorPosition` (e.g. `Dual`s for derivatives w.r.t. `s_kick`, `gamma` or the geometry).
"""
struct CSRKickPoint{T<:Real}
  ix_ele_kick::Int           # Element containing the kick point
  s_kick::T                  # Kick point, reference s from the element entrance
  s_chord_kick::T            # Kick point along the centroid chord
  floor_k::FloorPosition{T}  # Floor position of the kick point
  gamma::T
  gamma2::T
  beta::T
  y_source::T                # Height of the source particle (image charges; 0 for free space)
  kick_factor::T             # Scales the kick (1/2 at element ends)
end

function CSRKickPoint(ix_ele_kick::Integer, s_kick, s_chord_kick, floor_k::FloorPosition, gamma, gamma2, beta,
                      y_source, kick_factor)
  T = float(promote_type(typeof(s_kick), typeof(s_chord_kick), number_type(floor_k), typeof(gamma), typeof(gamma2),
                         typeof(beta), typeof(y_source), typeof(kick_factor)))
  return CSRKickPoint{T}(ix_ele_kick, s_kick, s_chord_kick, floor_k, gamma, gamma2, beta, y_source, kick_factor)
end

"""
    CSRKick1{T}

Kick quantities for one source-to-kick separation (Bmad `csr_kick1_struct`), in number type `T`.
"""
mutable struct CSRKick1{T<:Real}
  I_csr::T                    # Kick integral
  I_int_csr::T                # Integrated kick integral
  L_vec::SVector{3,T}         # Source-to-kick vector in floor coordinates
  L::T                        # Distance between source and kick points
  dL::T                       # Ls - L
  dz_particles::T             # Kicked minus source particle position at constant time
  s_chord_source::T           # Source point along the source element chord
  theta_L::T                  # Angle of L_vec
  theta_sl::T                 # Angle between source velocity and L_vec
  theta_lk::T                 # Angle between L_vec and kicked particle velocity
  ix_ele_source::Int          # Source element (0 = upstream of the lattice)
  floor_s::FloorPosition{T}   # Floor position of the source point
end

"""
    CSRKick1{T}()
    CSRKick1()
    CSRKick1(geom, kp)

Zeroed work struct in number type `T` (default `Float64`; `CSRKick1(geom, kp)`: promoted from the geometry and the
kick point); set `ix_ele_source` (search start) and `dz_particles` (lag) before `s_source_calc!`.
"""
function CSRKick1{T}() where {T}
  z = zero(T)
  return CSRKick1{T}(z, z, SVector(z, z, z), z, z, z, z, z, z, z, 0, FloorPosition(SVector(z, z, z), z))
end

CSRKick1() = CSRKick1{Float64}()

CSRKick1(geom, kp) = CSRKick1{promote_type(number_type(geom), number_type(kp))}()

"""
    number_type(x)

Number type `T` of a `FloorPosition{T}`, `CSRElementInfo{T}`, `CSRGeometry{T}`, `CSRKickPoint{T}` or `CSRKick1{T}`.
"""
number_type(::FloorPosition{T}) where {T} = T
number_type(::CSRElementInfo{T}) where {T} = T
number_type(::CSRGeometry{T}) where {T} = T
number_type(::CSRKickPoint{T}) where {T} = T
number_type(::CSRKick1{T}) where {T} = T

"""
    CSRBins

Longitudinal bins. Bin `i` spans `[z0_edge[i], z0_edge[i] + dz)`. `edge_dcharge_density_dz[i]` is the
charge density gradient at the left edge of bin `i`: `(charge[i] - charge[i-1]) / dz²` (zero for `i = 1`).
Arrays live on the backend of the particle coordinates.
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

Diagnostics for one kick, per bin (mirrors Bmad's `diagnostic_output_file` columns):
`z_center`, `charge_per_meter` [C/m], `kick_per_meter` (Δpz per meter), `I_csr_per_meter`,
`s_source` (reference s of the source point). Arrays are on the backend of the particle coordinates.
"""
struct CSRStepResult{Z<:AbstractVector,C<:AbstractVector,K<:AbstractVector,I<:AbstractVector,S<:AbstractVector}
  z_center::Z
  charge_per_meter::C
  kick_per_meter::K
  I_csr_per_meter::I
  s_source::S
end

"""
    CSRParams(; n_bin = 40, particle_bin_span = 2, ds_step = 0.0, n_shield_images = 0, beam_chamber_height = 0.0,
              image_sum_order = 8, all_bin_quadrature = true, bmad_compat = false)

Parameters for the 1D CSR calculation. Names follow Bmad's `space_charge_com` components.

## Properties
- `n_bin`: Number of longitudinal bins. Must satisfy `n_bin - 2 - (particle_bin_span + 1) ≥ 1`.
- `particle_bin_span`: Width of each particle's triangular charge distribution, in units of the bin width.
- `ds_step`: Requested tracking step length [m]. The actual step is `L / max(1, round(L / ds_step))`.
- `n_shield_images`: Number of image-charge layers for shielding by parallel plates (0 = free space).
    Layer `n` is a pair of images at heights `±n * beam_chamber_height` with sign `(-1)^n`, as in Bmad.
- `beam_chamber_height`: Height of the beam chamber [m], i.e. the distance between the plates. Must be positive
    when `n_shield_images > 0`.
- `image_sum_order`: Order `k` of the averaging applied to the image-layer sum (0 = plain sum, at most 60). The layer
    terms alternate in sign and decay only as ~1/n, so the plain sum converges slowly. With `M = n_shield_images`, the
    kick is the binomially weighted average `2^-k Σ_j C(k, j) S_{M-k+j}` of the partial sums `S_m` (free space plus
    layers 1..m), i.e. layer `n` is weighted by `2^-k Σ_{j ≥ n-M+k} C(k, j)` (1 for `n ≤ M - k`); `k` is capped at `M`.
    Default 8: with `M ≥ 32` the steady-state kick agrees with a 1024-layer reference to ~1e-6
    (`planning/PLAN.md`, D24). Use `M` of 32–64 when the chamber height is comparable to or smaller than the
    bunch length. Ignored with `bmad_compat = true` (Bmad: plain sum).
- `all_bin_quadrature`: Integrate the kick integrand over every bin by quadrature (Simpson in `u^(1/3)`, 8 nodes per
    bin), instead of the trapezoid `(I_csr[i] + I_csr[i-1]) dz / 2` for bins `i ≥ 2`. Default `true`
    (`planning/upstream-bin.md`, option B). `false` gives the previous default (first bin only, option A). Ignored with
    `bmad_compat = true` (Bmad: trapezoid).
- `bmad_compat`: If `true`, reproduce Bmad expressions that CSR.jl replaces, for like-for-like comparison with
    Bmad output. Default `false`.
    1. `dspline_len` s⁴ term: Bmad uses `9c₃²`, the integral gives `9c₃²/5` (`planning/dspline_coeff.md`).
    2. First-bin kick integral when the source lies outside the kick element: Bmad holds `I_csr` constant
       over the rest of the bin (2/3 of the integral in steady state); CSR.jl integrates by quadrature
       (`planning/upstream-bin.md`).
    3. Image charges, source ahead of the kick point: Bmad gives the source the reverse direction of motion,
       and does not look for sources beyond the end of the kick element (`planning/PLAN.md`, V8).
    4. Image charges: Bmad samples the image kernel at the bin centres and sums the layers plainly; CSR.jl integrates
       the kernel against the bin hat function and averages the layer sum (`image_sum_order`).
    5. Path-length excess of the centroid over the source–kick line (retardation): Bmad's small-angle form
       `½∫(x′ - dθ)²` (`dspline_len`); CSR.jl the exact form (`path_excess`). They differ by `~Ls²/(80ρ²)` relative.
"""
Base.@kwdef struct CSRParams
  n_bin::Int = 40
  particle_bin_span::Int = 2
  ds_step::Float64 = 0.0
  n_shield_images::Int = 0
  beam_chamber_height::Float64 = 0.0
  image_sum_order::Int = 8
  all_bin_quadrature::Bool = true
  bmad_compat::Bool = false
  function CSRParams(n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height, image_sum_order,
                     all_bin_quadrature, bmad_compat)
    n_bin_eff = n_bin - 2 - (particle_bin_span + 1)
    n_bin_eff >= 1 || error("CSRParams: n_bin (= $n_bin) must be greater than 3 + particle_bin_span (= $particle_bin_span)")
    particle_bin_span >= 1 || error("CSRParams: particle_bin_span must be at least 1")
    ds_step >= 0 || error("CSRParams: ds_step must be non-negative")
    n_shield_images >= 0 || error("CSRParams: n_shield_images must be non-negative")
    n_shield_images == 0 || beam_chamber_height > 0 ||
      error("CSRParams: beam_chamber_height must be positive when n_shield_images > 0")
    0 <= image_sum_order <= 60 || error("CSRParams: image_sum_order must be in 0:60")
    return new(n_bin, particle_bin_span, ds_step, n_shield_images, beam_chamber_height, image_sum_order,
               all_bin_quadrature, bmad_compat)
  end
end
