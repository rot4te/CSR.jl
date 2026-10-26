# SaganCavity (lcavity) slices. A slice [s0, s1] runs the part of the whole cavity's kernel sequence that lies in
# the window: outer drifts, entrance/exit fringe, cell kicks with their reference-energy steps, cell drifts.
# Built from BeamTracking's internal Sagan kernels (ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl,
# src/kernels/sagan_cavity.jl), since SaganCavity has no fringe_at and cannot be split with track!.
# Point events (cell kicks, fringes, and the half multipole/ponderomotive kicks at drift-segment ends) at position p
# belong to the slice with s0 <= p < s1 (p <= s1 for the last slice). Only drifts are split, so slices reproduce the
# whole cavity to round-off (PLAN.md, D22).
# Consecutive slices must share their boundary exactly (s0 + L_slice of one == s0 of the next), or an event on a
# boundary can be dropped or applied twice.

"""
    bt_ext()

BeamTracking's Beamlines extension module (its internal helpers are used for SaganCavity slices).
"""
bt_ext() = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)

"""
    SaganSlice

Slice `[s0, s1]` of a SaganCavity element of length `L` (`csr_n_step` slicing), with the cavity constants
BeamTracking computes for the whole element. `ele` holds the slice length, alignment, aperture and multipoles.
`TE`: number type of the energy fields (`E0`, `dE_ref`, `P0c0`, `p_over_q_ref`); `T`: RF and length fields
(promoted from the cavity's parameters; TPS or `Dual` with knobs).
"""
Base.@kwdef struct SaganSlice{E,TM,M,K,TE<:Number,T<:Number}
  ele::E
  tm::TM                  # The cavity's SaganCavity
  species::Species
  mass::Float64
  q::Float64
  a::Float64              # Gyromagnetic anomaly
  traveling_wave::Bool
  E0::TE                  # Reference energy at the cavity entrance (E_ref - dE_ref)
  dE_ref::TE
  P0c0::TE                # Reference momentum at the cavity entrance [eV]
  p_over_q_ref::TE        # Reference p_over_q of the cavity's beamline (exit energy)
  rf_omega::T
  t_phi0::T
  q_gradient::T
  q_voltage::T
  n_cells::Float64
  L_active::T
  L::T
  m_order::M
  Bn::K                   # Multipole field strengths (BeamTracking: Kn * p_over_q at the entrance energy)
  Bs::K
  has_mult::Bool
  has_sol::Bool
  s0::T
  s1::T
  first::Bool
  last::Bool
end

"""
    sagan_kick_factor(i_step, n_cells)

2 for the end cell kicks (half strength), 1 otherwise, as BeamTracking's `sagan_cavity_thick!`.
"""
@inline sagan_kick_factor(i_step, n_cells) = (i_step == 0 || i_step == n_cells) ? 2 : 1

"""
    sagan_overlap(a, b, len, s0, s1)

Length of segment [a, b] (nominal length len) inside [s0, s1]; the nominal length if it is fully inside. Decided on
primal values (`Dual`s at exact boundary ties compare partials otherwise; D22 relies on exact ties).
"""
@inline sagan_overlap(a, b, len, s0, s1) = (CSR.primal(a) >= CSR.primal(s0) && CSR.primal(b) <= CSR.primal(s1)) ? len :
                                           CSR.primal_max(zero(len), CSR.primal_min(b, s1) - CSR.primal_max(a, s0))
"""
    sagan_in_window(p, s0, s1, last)

Whether a point event at `p` belongs to slice `[s0, s1]`: `s0 ≤ p < s1`, or `p ≤ s1` for the last slice. Decided on
primal values (`Dual`s at exact boundary ties compare partials otherwise; D22 relies on exact ties).
"""
@inline sagan_in_window(p, s0, s1, last) = CSR.primal(p) >= CSR.primal(s0) && (CSR.primal(p) < CSR.primal(s1) || last)

"""
    sagan_P0c_at(c, s, inclusive)

Reference momentum after the cell kicks at p < s (p <= s if inclusive), applied as in sagan_cavity_thick!.
"""
function sagan_P0c_at(c::SaganSlice, s, inclusive)
  P0c = c.P0c0
  before(p) = CSR.primal(p) < CSR.primal(s) || (inclusive && CSR.primal(p) <= CSR.primal(s))
  Lo = (c.L - c.L_active) / 2
  if c.n_cells == 0
    before(Lo + c.L_active / 2) && (P0c += BeamTracking.dpc_given_dE(P0c, c.dE_ref, c.mass))
  else
    for i_step in 0:c.n_cells
      before(Lo + i_step * c.L_active / c.n_cells) || continue
      P0c += BeamTracking.dpc_given_dE(P0c, c.dE_ref / (c.n_cells * sagan_kick_factor(i_step, c.n_cells)), c.mass)
    end
  end
  return P0c
end

"""
    sagan_dt_ref(c, s)

Reference time from the cavity entrance to s, with BeamTracking's per-segment reference velocities
(universal! for SaganCavity). sagan_dt_ref(c, c.L) is the whole-cavity value.
"""
function sagan_dt_ref(c::SaganSlice, s)
  sp = c.species
  E1 = c.E0 + c.dE_ref
  Lo = (c.L - c.L_active) / 2
  part(a, b) = sagan_overlap(a, b, b - a, 0.0, s)
  dt = part(0.0, Lo) / BeamTracking.E_to_v(sp, c.E0) + part(c.L - Lo, c.L) / BeamTracking.E_to_v(sp, E1)
  if c.n_cells == 0
    h = c.L_active / 2
    dt += part(Lo, Lo + h) / BeamTracking.E_to_v(sp, c.E0) + part(Lo + h, Lo + c.L_active) / BeamTracking.E_to_v(sp, E1)
  else
    lc = c.L_active / c.n_cells
    for i in 1:c.n_cells
      E_now = c.E0 + (i - 1/2) * c.dE_ref / c.n_cells
      dt += part(Lo + (i - 1) * lc, Lo + i * lc) / BeamTracking.E_to_v(sp, E_now)
    end
  end
  return dt
end

"""
    sagan_segment!(i, coords, val_rad_damping_on, val_rad_fluctuations_on, ::Val{traveling_wave}, ::Val{has_mult}, ::Val{has_sol}, m_order, Kn, Ks, q, q_gradient, a_spin, mass, P0c, a, b, len, s0, s1, last, inside)

Drift segment [a, b] (nominal length len) of the whole cavity, as BeamTracking's sagan_cavity_inside_drift!
(inside = true: half ponderomotive kicks, standing wave only) or sagan_cavity_outside_drift!: half kicks of strength
len/2 at a and b, drift or solenoid over len. The half kicks are point events at the segment ends (same window rule
as the cell kicks); only the drift is split, and drifts compose exactly, so slices reproduce the whole cavity.
"""
@inline function sagan_segment!(i, coords, val_rad_damping_on, val_rad_fluctuations_on, ::Val{traveling_wave},
                                ::Val{has_mult}, ::Val{has_sol}, m_order, Kn, Ks, q, q_gradient, a_spin, mass, P0c,
                                a, b, len, s0, s1, last, inside) where {traveling_wave, has_mult, has_sol}
  CSR.primal(len) > 0 || return nothing
  pond = inside && !traveling_wave
  beta0 = P0c / sqrt(P0c^2 + mass^2)
  gamma0 = P0c / (mass * beta0)
  if sagan_in_window(a, s0, s1, last)
    pond && BeamTracking.rf_pondermotive_kick!(i, coords, q_gradient, a_spin, mass, P0c, len / 2)
    has_mult && BeamTracking.multipole_kick_with_rad!(i, coords, val_rad_damping_on, val_rad_fluctuations_on,
                                                      q, m_order, Kn, Ks, a_spin, mass, P0c, beta0, len / 2)
  end
  d = sagan_overlap(a, b, len, s0, s1)
  if CSR.primal(d) > 0
    if has_sol
      BeamTracking.exact_solenoid!(i, coords, Kn[1], beta0, gamma0 * gamma0, mass / P0c, nothing, d)
    else
      BeamTracking.exact_drift!(i, coords, 0, beta0, gamma0 * gamma0, mass / P0c, d)
    end
  end
  if sagan_in_window(b, s0, s1, last)
    has_mult && BeamTracking.multipole_kick_with_rad!(i, coords, val_rad_damping_on, val_rad_fluctuations_on,
                                                      q, m_order, Kn, Ks, a_spin, mass, P0c, beta0, len / 2)
    pond && BeamTracking.rf_pondermotive_kick!(i, coords, q_gradient, a_spin, mass, P0c, len / 2)
  end
  return nothing
end

"""
    sagan_slice_kernel!(i, coords, val_rad_damping_on, val_rad_fluctuations_on, val_traveling_wave, mass, q, P0c, dE_ref, t_ref, n_cells, val_has_mult, val_has_sol, m_order, Bn, Bs, a, q_gradient, q_voltage, rf_omega, t_phi0, L_active, L, s0, s1, last)

Kernel: the window [s0, s1] of sagan_cavity_thick! / sagan_cavity_zero_L_active!. Branches only on scalars
(BeamTracking may call it with SIMD lanes). Segment ends use the same expressions as the neighbouring events.
"""
function sagan_slice_kernel!(i, coords, val_rad_damping_on, val_rad_fluctuations_on, val_traveling_wave,
                             mass, q, P0c, dE_ref, t_ref, n_cells, val_has_mult, val_has_sol, m_order, Bn, Bs,
                             a, q_gradient, q_voltage, rf_omega, t_phi0, L_active, L, s0, s1, last)
  c_l = BeamTracking.c_light(eltype(coords.v))
  Lo = (L - L_active) / 2
  qp = q * c_l / P0c
  seg!(qp, P0c, sa, sb, len, inside) = sagan_segment!(i, coords, val_rad_damping_on, val_rad_fluctuations_on,
    val_traveling_wave, val_has_mult, val_has_sol, m_order, Bn .* qp, Bs .* qp, q, q_gradient, a, mass, P0c,
    sa, sb, len, s0, s1, last, inside)

  seg!(qp, P0c, 0.0, Lo, Lo, false)
  if CSR.primal(L_active) > 0 && sagan_in_window(Lo, s0, s1, last)
    BeamTracking.sagan_cavity_fringe!(i, coords, a, q_gradient, rf_omega, t_phi0, t_ref, mass, P0c, +1)
  end

  if n_cells == 0
    h = L_active / 2
    CSR.primal(L_active) > 0 && seg!(qp, P0c, Lo, Lo + h, h, true)
    if sagan_in_window(Lo + h, s0, s1, last)
      BeamTracking.sagan_cavity_kick!(i, coords, a, CSR.primal(L_active) > 0 ? q_gradient * L_active : q_voltage,
                                      rf_omega, t_phi0, t_ref, mass, P0c)
      dP0c = BeamTracking.dpc_given_dE(P0c, dE_ref, mass)
      BeamTracking.reference_momentum_shift!(i, coords, P0c, dP0c, Val{true}())
      P0c += dP0c
      qp = q * c_l / P0c
    end
    CSR.primal(L_active) > 0 && seg!(qp, P0c, Lo + h, Lo + L_active, h, true)
  else
    lc = L_active / n_cells
    for i_step in 0:n_cells
      p = Lo + i_step * lc
      if sagan_in_window(p, s0, s1, last)
        kf = sagan_kick_factor(i_step, n_cells)
        BeamTracking.sagan_cavity_kick!(i, coords, a, q_gradient * L_active / (n_cells * kf), rf_omega, t_phi0,
                                        t_ref, mass, P0c)
        dP0c = BeamTracking.dpc_given_dE(P0c, dE_ref / (n_cells * kf), mass)
        BeamTracking.reference_momentum_shift!(i, coords, P0c, dP0c, Val{true}())
        P0c += dP0c
        qp = q * c_l / P0c
      end
      i_step == n_cells && break
      seg!(qp, P0c, p, Lo + (i_step + 1) * lc, lc, true)
    end
  end

  if CSR.primal(L_active) > 0 && sagan_in_window(Lo + L_active, s0, s1, last)
    BeamTracking.sagan_cavity_fringe!(i, coords, a, q_gradient, rf_omega, t_phi0, t_ref, mass, P0c, -1)
  end
  seg!(qp, P0c, Lo + L_active, L, Lo, false)
  return nothing
end

"""
    slice_sagan(ele, up, s0, L_slice, first, last, context) -> SaganSlice

SaganCavity slice `[s0, s0 + L_slice]`, with the cavity constants computed for the whole element as BeamTracking does
(RF frequency and phase, cells, gradient, multipoles at the entrance energy), alignment per slice and apertures at
the ends. The last slice ends at `L` exactly.
"""
function slice_sagan(ele::LineElement, up, s0, L_slice, first, last, context)
  bte = bt_ext()
  tm = up.tracking_method
  L = float(up.L)
  for (P, name) in ((ele.MapParams, "MapParams"), (ele.PatchParams, "PatchParams"), (ele.BendParams, "BendParams"),
                    (ele.FourPotentialParams, "FourPotentialParams"), (ele.EMultipoleParams, "EMultipoleParams"))
    isactive(deval(P, context)) && error("slice_element: SaganCavity with $name is not supported (as in BeamTracking)")
  end
  rp = deval(ele.RFParams, context)
  isactive(rp) || error("slice_element: SaganCavity without RFParams is not supported (as in BeamTracking)")
  lp = deval(ele.BeamlineParams, context)
  isactive(lp) || error("slice_element: a SaganCavity element must be in a Beamline (reference energy and dE_ref)")
  p_over_q_ref = lp.beamline.p_over_q_ref
  p_over_q_ref isa Number || error("slice_element: time-dependent reference energy is not supported")

  species = lp.beamline.species_ref
  mass = massof(species)
  q = chargeof(species)
  E1 = R_to_E(species, p_over_q_ref)
  dE_ref = float(lp.dE_ref)
  E0 = E1 - dE_ref
  rf_omega = bte.rf_omega_calc(rp, lp)
  t_phi0 = bte.rf_phi0_calc(rp, species) / rf_omega
  n_cells, L_active = bte.rf_step_calc(tm.n_cells, tm.L_active, rf_omega, L)
  CSR.primal(L_active) <= CSR.primal(L) * (1 + eps(CSR.primal(L))) || error("slice_element: SaganCavity L_active ($L_active) is greater than L ($L)")

  bm = deval(ele.BMultipoleParams, context)
  p0q = BeamTracking.E_to_R(species, E0)
  if isactive(bm)
    m_order = bm.order
    Kn, Ks = bte.get_strengths(bm, L, p0q)
    Bn, Bs = Kn .* p0q, Ks .* p0q
    has_mult = length(m_order) > 0
    has_sol = has_mult && m_order[1] == 0
  else
    m_order, Bn, Bs, has_mult, has_sol = SA{Int32}[], SA{Float64}[], SA{Float64}[], false, false
  end

  s = LineElement()
  s.UniversalParams = UniversalParams(up.kind, up.name, L_slice, tm)
  ap = deval(ele.AlignmentParams, context)
  isactive(ap) && (s.AlignmentParams = slice_alignment(ap, nothing, L, s0, L_slice))
  dp = deval(ele.ApertureParams, context)
  if isactive(dp)
    dps = slice_aperture(dp, first, last)
    isnothing(dps) || (s.ApertureParams = dps)
  end

  q_gradient = CSR.primal(L_active) > 0 ? q * rp.voltage / L_active : 0.0
  ve = (float(E0), dE_ref, sqrt(E0^2 - mass^2), float(p_over_q_ref))
  TE = float(promote_type(map(typeof, ve)...))
  vt = (float(rf_omega), float(t_phi0), q_gradient, float(q * rp.voltage), float(L_active), L, float(s0),
        float(last ? L : s0 + L_slice))
  T = float(promote_type(map(typeof, vt)...))
  return SaganSlice(; ele = s, tm, species, mass = float(mass), q = float(q),
                    a = float(gyromagnetic_anomaly(species)), traveling_wave = rp.traveling_wave,
                    E0 = convert(TE, ve[1]), dE_ref = convert(TE, ve[2]), P0c0 = convert(TE, ve[3]),
                    p_over_q_ref = convert(TE, ve[4]), rf_omega = convert(T, vt[1]), t_phi0 = convert(T, vt[2]),
                    q_gradient = convert(T, vt[3]), q_voltage = convert(T, vt[4]), n_cells = float(n_cells),
                    L_active = convert(T, vt[5]), L = convert(T, vt[6]), m_order, Bn, Bs, has_mult, has_sol,
                    s0 = convert(T, vt[7]), s1 = convert(T, vt[8]), first, last)
end

"""
    sagan_exit_P0c(c)

Reference momentum [eV] at the slice exit and the linearly ramped one used for the CSR kick there.
"""
sagan_exit_P0c(c::SaganSlice) = c.last ? sagan_P0c_at(c, c.L, true) : sagan_P0c_at(c, c.s1, false)

"""
    track_slice!(bunch, slice, p_over_q_ref)

Track `bunch` through a slice made by `CSR.slice_element`.
"""
track_slice!(bunch, s::LineElement, p_over_q_ref) = track!(bunch, s; _p_over_q_ref = p_over_q_ref)

"""
    track_slice!(bunch, c::SaganSlice)

Track `bunch` through a SaganCavity slice: entrance alignment/aperture, `sagan_slice_kernel!`, exit
alignment/aperture, with BeamTracking's `KernelChain`/`launch!`. Adds the slice's reference time to `bunch.t_ref`, so
that after the last slice it is the entrance `t_ref` plus the whole-cavity `dt_ref` (as BeamTracking ≥ 0.8.7; to
round-off).
"""
function track_slice!(bunch, c::SaganSlice, p_over_q_ref = c.p_over_q_ref)
  bte = bt_ext()
  bunch.species == c.species || error("track_slice!: bunch species differs from the cavity's reference species")
  coords = bunch.coords
  tm = c.tm
  L_slice = c.s1 - c.s0
  ap = c.ele.AlignmentParams
  dp = c.ele.ApertureParams
  P0c = c.first ? c.P0c0 : sagan_P0c_at(c, c.s0, false)

  t_enter = bunch.t_ref
  bg = R_to_beta_gamma(c.species, c.p_over_q_ref)
  t_exit = t_enter - sagan_dt_ref(c, c.s0) + sagan_dt_ref(c, c.last ? c.L : c.s1)
  kc = BeamTracking.KernelChain(Val{10}(), BeamTracking.RefState{eltype(coords.v)}(; t_enter, beta_gamma_enter = bg,
                                t_exit, beta_gamma_exit = bg, L = L_slice, g = (0, 0), ds_step = L_slice))

  # Entrance aperture and alignment, as universal! for SaganCavity
  kc = sagan_ends(bte, tm, kc, c, ap, dp, L_slice, true)
  body = (Val{tm.radiation_damping_on}(), Val{tm.radiation_fluctuations_on}(), Val{c.traveling_wave}(),
          c.mass, c.q, P0c, c.dE_ref, 0, c.n_cells, Val{c.has_mult}(), Val{c.has_sol}(), c.m_order, c.Bn, c.Bs,
          c.a, c.q_gradient, c.q_voltage, c.rf_omega, c.t_phi0, c.L_active, c.L, c.s0, c.s1, c.last)
  kc = BeamTracking.push(kc, BeamTracking.make_kernel_call(sagan_slice_kernel!, body))
  kc = sagan_ends(bte, tm, kc, c, ap, dp, L_slice, false)
  BeamTracking.launch!(coords, kc)
  bunch.t_ref = t_exit
  return bunch
end

"""
    sagan_ends(bte, tm, kc, c, ap, dp, L_slice, entering)

Alignment and aperture kernels at the slice entrance or exit, in the order BeamTracking uses for SaganCavity.
"""
function sagan_ends(bte, tm, kc, c, ap, dp, L_slice, entering)
  pq = c.p_over_q_ref
  if isactive(ap)
    if isactive(dp)
      if dp.aperture_shifts_with_body == entering
        kc = bte.alignment(tm, kc, pq, nothing, ap, nothing, L_slice, entering)
        kc = bte.aperture(tm, kc, pq, nothing, dp, entering)
      else
        kc = bte.aperture(tm, kc, pq, nothing, dp, entering)
        kc = bte.alignment(tm, kc, pq, nothing, ap, nothing, L_slice, entering)
      end
    else
      kc = bte.alignment(tm, kc, pq, nothing, ap, nothing, L_slice, entering)
    end
  elseif isactive(dp)
    kc = bte.aperture(tm, kc, pq, nothing, dp, entering)
  end
  return kc
end
