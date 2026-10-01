# Element slicing for route (a): track each CSR step with BeamTracking.track! on a detached slice.

has_fringe_in(f::Fringe.T) = f == Fringe.BothEnds || f == Fringe.EntranceEnd
has_fringe_out(f::Fringe.T) = f == Fringe.BothEnds || f == Fringe.ExitEnd

function fringe_at_ends(fin::Bool, fout::Bool)
  fin && fout && return Fringe.BothEnds
  fin && return Fringe.EntranceEnd
  fout && return Fringe.ExitEnd
  return Fringe.NoEnd
end

# Symplectic order that SciBmadStandard uses for this element
# (mirrors ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl, BeamTracking 0.8.4).
function sbs_order(bp, bm)
  isactive(bm) || return 2                                  # drift, bend without field
  n = length(bm.order)
  if isactive(bp)
    return (n == 1 && bm.order[1] == 1) ? 2 : 4             # bend_pure_bdipole
  else
    return (n == 1 && bm.order[1] in (0, 1)) ? 2 : 4        # pure_bsolenoid, pure_bdipole
  end
end

function slice_tracking_method(tm, frac, first, last, bp, bm)
  if tm isa SciBmadStandard
    tm = Symplectic(; order = sbs_order(bp, bm),
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
  end
  error("slice_element: tracking method $(typeof(tm)) not supported")
end

function scale_integrated(bm::BMultipoleParams, frac)
  sc = [i ? frac : one(frac) for i in bm.integrated]
  return BMultipoleParams(bm.n .* sc, bm.s .* sc, copy(bm.tilt), bm.order, bm.normalized, bm.integrated)
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

function CSR.slice_element(ele::LineElement, L_slice, first::Bool, last::Bool;
                           context = Beamlines.NULL_CONTEXT)
  for (pg, name) in ((:AlignmentParams, "alignment"), (:PatchParams, "patch"), (:MapParams, "map"),
                     (:RFParams, "RF"), (:FourPotentialParams, "four-potential"), (:EMultipoleParams, "electric multipole"))
    isactive(getproperty(ele, pg)) && error("slice_element: elements with $name parameters are not supported")
  end
  up = deval(ele.UniversalParams, context)
  L = float(up.L)
  L > 0 || error("slice_element: element length must be positive (L = $L)")
  frac = L_slice / L
  bp = deval(ele.BendParams, context)
  bm = deval(ele.BMultipoleParams, context)
  dp = deval(ele.ApertureParams, context)

  s = LineElement()
  s.UniversalParams = UniversalParams(up.kind, up.name, L_slice,
                                      slice_tracking_method(up.tracking_method, frac, first, last, bp, bm))
  isactive(bp) && (s.BendParams = bp)
  isactive(bm) && (s.BMultipoleParams = scale_integrated(bm, frac))
  if isactive(dp)
    dps = slice_aperture(dp, first, last)
    isnothing(dps) || (s.ApertureParams = dps)
  end
  return s
end
