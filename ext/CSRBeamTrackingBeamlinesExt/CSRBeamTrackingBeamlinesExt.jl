module CSRBeamTrackingBeamlinesExt

using CSR, BeamTracking, Beamlines
using AtomicAndPhysicalConstants: C_LIGHT, chargeof, massof, gyromagnetic_anomaly
using StaticArrays: SA
using BeamTracking: Fringe, AbstractSymplectic, R_to_beta_gamma, R_to_E,
                    quat_rotate, quat_mul, rot_quaternion, inv_rot_quaternion, one_cos_norm,
                    coord_alignment_bend_mid, coord_alignment_bend_at_s
using Beamlines: deval, isactive, SciBmadStandard, ApertureAt, RateMeaning

include("slicing.jl")
include("sagan.jl")
include("track.jl")

end
