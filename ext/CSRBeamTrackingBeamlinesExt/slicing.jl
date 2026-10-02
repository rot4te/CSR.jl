# Element slicing for route (a): track each CSR step with BeamTracking.track! on a detached slice.
# Slice rules follow Bmad's create_element_slice / makeup_super_slave1 (bmad/modules/bookkeeper_mod.f90),
# expressed in Beamlines parameters. Uses BeamTracking internals (quaternion and alignment helpers), checked
# against BeamTracking dev after v0.8.4 (RungeKutta, Beamlines EMFieldParams).

has_fringe_in(f::Fringe.T) = f == Fringe.BothEnds || f == Fringe.EntranceEnd
has_fringe_out(f::Fringe.T) = f == Fringe.BothEnds || f == Fringe.ExitEnd

function fringe_at_ends(fin::Bool, fout::Bool)
  fin && fout && return Fringe.BothEnds
  fin && return Fringe.EntranceEnd
  fout && return Fringe.ExitEnd
  return Fringe.NoEnd
end

# Symplectic order that SciBmadStandard uses for this element
# (mirrors ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl, unchanged since BeamTracking v0.8.4).
function sbs_order(bp, bm, rp)
  isactive(rp) && return isactive(bm) ? 4 : 2              # bmultipole_rf, pure_rf
  isactive(bm) || return 2                                  # drift, bend without field, pure_edipole
  n = length(bm.order)
  if isactive(bp)
    return (n == 1 && bm.order[1] == 1) ? 2 : 4             # bend_pure_bdipole
  else
    return (n == 1 && bm.order[1] in (0, 1)) ? 2 : 4        # pure_bsolenoid, pure_bdipole
  end
end

function slice_tracking_method(tm, frac, first, last, bp, bm, rp)
  if tm isa SciBmadStandard
    tm = Symplectic(; order = sbs_order(bp, bm, rp),
      radiation_damping_on = tm.radiation_damping_on, radiation_fluctuations_on = tm.radiation_fluctuations_on,
      ibs_damping_on = tm.ibs_damping_on, ibs_fluctuations_on = tm.ibs_fluctuations_on)
  end
  if tm isa AbstractSymplectic
    fringe_at = fringe_at_ends(has_fringe_in(tm.fringe_at) && first, has_fringe_out(tm.fringe_at) && last)
    n_steps = tm.n_steps > 0 ? max(1, round(Int, tm.n_steps * frac)) : -1
    return typeof(tm)(; order = tm.order, n_steps, ds_step = tm.ds_step,
      radiation_damping_on = tm.radiation_damping_on, radiation_fluctuations_on = tm.radiation_fluctuations_on,
      fringe_at, ibs_damping_on = tm.ibs_damping_on, ibs_fluctuations_on = tm.ibs_fluctuations_on,
      implicit_use_newton = tm.implicit_use_newton, use_optimized_schemes = tm.use_optimized_schemes)
  elseif tm isa Exact
    return Exact(; fringe_at = fringe_at_ends(has_fringe_in(tm.fringe_at) && first, has_fringe_out(tm.fringe_at) && last))
  elseif tm isa RungeKutta
    # No fringe kernels: only the step count is rescaled (ds_step is kept as is).
    n_steps = tm.n_steps > 0 ? max(1, round(Int, tm.n_steps * frac)) : nothing
    ds_step = tm.n_steps > 0 ? nothing : tm.ds_step
    return RungeKutta(; ds_step, n_steps, ibs_damping_on = tm.ibs_damping_on, ibs_fluctuations_on = tm.ibs_fluctuations_on)
  end
  error("slice_element: tracking method $(typeof(tm)) not supported")
end

function scale_integrated(bm::BMultipoleParams, frac)
  sc = [i ? frac : one(frac) for i in bm.integrated]
  return BMultipoleParams(bm.n .* sc, bm.s .* sc, copy(bm.tilt), bm.order, bm.normalized, bm.integrated)
end

function scale_integrated(em::EMultipoleParams, frac)
  sc = [i ? frac : one(frac) for i in em.integrated]
  return EMultipoleParams(em.n .* sc, em.s .* sc, copy(em.etilt), em.order, em.integrated)
end

function slice_aperture(dp::ApertureParams, first, last)
  at_in = dp.aperture_at == ApertureAt.Entrance || dp.aperture_at == ApertureAt.BothEnds
  at_out = dp.aperture_at == ApertureAt.Exit || dp.aperture_at == ApertureAt.BothEnds
  at_in &= first
  at_out &= last
  (at_in || at_out) || return nothing
  dp.aperture_at = at_in && at_out ? ApertureAt.BothEnds : (at_in ? ApertureAt.Entrance : ApertureAt.Exit)
  return dp
end

# Euler angles (x_rot, y_rot, z_rot) of q = rot_quaternion(x_rot, y_rot, z_rot) = qy ⊗ qx ⊗ qz,
# from the rotated basis vectors (M = Ry Rx Rz). Valid for |x_rot| < π/2.
function rot_angles(q)
  ex = quat_rotate((1.0, 0.0, 0.0), q)
  ey = quat_rotate((0.0, 1.0, 0.0), q)
  ez = quat_rotate((0.0, 0.0, 1.0), q)
  return asin(-ez[2]), atan(ez[1], ez[3]), atan(ex[2], ey[2])
end

# Alignment of the slice [s0, s0 + Ls] such that the slices compose to the misaligned element
# (Bmad: makeup_super_slave1). Straight: same rotation, offsets moved with the slice center.
# Bend: the element's alignment transform at the slice center is the slice's chord-midpoint transform.
function slice_alignment(a::AlignmentParams, bp, L, s0, Ls)
  if !isactive(bp) || (bp.g_ref == 0 && bp.tilt_ref == 0)
    s_del = s0 + Ls / 2 - L / 2
    d = quat_rotate((0, 0, s_del), rot_quaternion(a.x_rot, a.y_rot, a.tilt))
    return AlignmentParams(a.x_offset + d[1], a.y_offset + d[2], a.z_offset + d[3] - s_del, a.x_rot, a.y_rot, a.tilt)
  end
  g, tilt_ref = bp.g_ref, bp.tilt_ref
  mid_r, mid_q, st, ct = coord_alignment_bend_mid(a.x_offset, a.y_offset, a.z_offset, a.x_rot, a.y_rot, a.tilt,
                                                  g, tilt_ref, L)
  R, Q = coord_alignment_bend_at_s(mid_r, mid_q, st, ct, g, L, s0 + Ls / 2)
  # Invert coord_alignment_bend_mid for length Ls.
  x_rot, y_rot, tilt = rot_angles(quat_mul(Q, inv_rot_quaternion(0, 0, tilt_ref)))
  f = -Ls^2 * g * one_cos_norm(Ls * g / 2) / 4
  d = quat_rotate((-f, 0, 0), Q)
  return AlignmentParams(R[1] - f * ct - d[1], R[2] - f * st - d[2], R[3] - d[3], x_rot, y_rot, tilt)
end

# Patch slice (Bmad: makeup_super_slave1). The first slice holds the patch transform with the exit face
# moved back by L - Ls along the new z axis; the other slices are drifts.
function slice_patch(p::PatchParams, L, Ls, first)
  first || return PatchParams(zero(Ls), zero(Ls), zero(Ls), Ls, zero(Ls), zero(Ls), zero(Ls))
  ez = quat_rotate((0, 0, 1), rot_quaternion(p.dx_rot, p.dy_rot, p.dz_rot))
  dl = L - Ls
  return PatchParams(p.dt, p.dx - dl * ez[1], p.dy - dl * ez[2], p.dz - dl * ez[3], p.dx_rot, p.dy_rot, p.dz_rot)
end

# Four-potential or EM field evaluated at element coordinates: the integrator passes s and t relative to the
# slice entrance (t = (s/β₀ - z/β)/c), so shift both by the slice start.
struct ShiftedPotential{F,T} <: Function
  f::F
  s0::T
  t0::T
end
(p::ShiftedPotential)(x, y, s, t, pp = nothing) = p.f(x, y, s + p.s0, t + p.t0, pp)

function shifted_to_slice(f, ele, s0)
  bg = R_to_beta_gamma(ele.species_ref, ele.p_over_q_ref)
  beta_0 = bg / sqrt(1 + bg^2)
  return ShiftedPotential(f, float(s0), s0 / (beta_0 * C_LIGHT))
end

function slice_four_potential(fp::FourPotentialParams, ele, s0)
  f = shifted_to_slice(fp.four_potential, ele, s0)
  return FourPotentialParams(f, fp.four_potential_params, fp.four_potential_normalized)
end

# EM field (used by RungeKutta tracking).
function slice_em_field(ef::EMFieldParams, ele, s0)
  return EMFieldParams(shifted_to_slice(ef.em_field, ele, s0), ef.em_field_params, ef.em_field_normalized)
end

# RF slice: voltage ∝ length (Bmad), rate converted to a frequency (the harmonic number refers to the
# original beamline's circumference).
function slice_rf(rp::RFParams, ele, frac)
  return RFParams(ele.rf_frequency, rp.voltage * frac, rp.phi0, RateMeaning.RFFrequency, rp.zero_phase,
                  rp.traveling_wave, rp.is_crabcavity)
end

function CSR.csr_sliceable(ele::LineElement; context = Beamlines.NULL_CONTEXT)
  return deval(ele.UniversalParams, context).L != 0 && !isactive(ele.MapParams)
end

function CSR.slice_element(ele::LineElement, s0, L_slice, first::Bool, last::Bool;
                           context = Beamlines.NULL_CONTEXT)
  up = deval(ele.UniversalParams, context)
  L = float(up.L)
  L > 0 || error("slice_element: element length must be positive (L = $L)")
  frac = L_slice / L
  ap = deval(ele.AlignmentParams, context)
  bp = deval(ele.BendParams, context)
  bm = deval(ele.BMultipoleParams, context)
  em = deval(ele.EMultipoleParams, context)
  mp = deval(ele.MapParams, context)
  pp = deval(ele.PatchParams, context)
  rp = deval(ele.RFParams, context)
  fp = deval(ele.FourPotentialParams, context)
  ef = deval(ele.EMFieldParams, context)
  dp = deval(ele.ApertureParams, context)
  isactive(rp) && up.tracking_method isa SaganCavity && error("slice_element: SaganCavity tracking is not supported")

  s = LineElement()
  if isactive(mp)
    # A map cannot be split: only the whole element is allowed (tracked without CSR, as Bmad does for Taylor).
    (first && last && L_slice == L) || error("slice_element: map elements cannot be sliced")
    s.UniversalParams = up
    s.MapParams = mp
  else
    s.UniversalParams = UniversalParams(up.kind, up.name, L_slice,
                                        slice_tracking_method(up.tracking_method, frac, first, last, bp, bm, rp))
  end
  isactive(ap) && (s.AlignmentParams = slice_alignment(ap, bp, L, s0, L_slice))
  isactive(bp) && (s.BendParams = bp)
  isactive(bm) && (s.BMultipoleParams = scale_integrated(bm, frac))
  isactive(em) && (s.EMultipoleParams = scale_integrated(em, frac))
  isactive(pp) && (s.PatchParams = slice_patch(pp, L, L_slice, first))
  isactive(fp) && (s.FourPotentialParams = slice_four_potential(fp, ele, s0))
  isactive(ef) && (s.EMFieldParams = slice_em_field(ef, ele, s0))
  if isactive(dp)
    dps = slice_aperture(dp, first, last)
    isnothing(dps) || (s.ApertureParams = dps)
  end
  if isactive(rp)
    # RF tracking reads the reference species from the slice's beamline.
    s.RFParams = slice_rf(rp, ele, frac)
    return Beamline([s]; species_ref = ele.species_ref, p_over_q_ref = ele.p_over_q_ref).line[1]
  end
  return s
end
