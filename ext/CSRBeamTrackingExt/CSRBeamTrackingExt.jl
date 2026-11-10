"""
    CSRBeamTrackingExt

`CSREffect` and `CSRReplay` as effects of BeamTracking's
`track!(bunch, bl; effects)`: reference survey, centroid orbit and
one CSR kick at the entrance of each slice.
"""
module CSRBeamTrackingExt

using CSR, BeamTracking, Beamlines
using CSR: CSREffect, CSRRecord, CSRReplay, FloorPosition
using AtomicAndPhysicalConstants: chargeof, massof
using BeamTracking: Bunch, quat_rotate, rot_quaternion, STATE_ALIVE,
                    ZI, PZI
using Beamlines: deval, isactive, scalarize, R_to_E, SaganCavity

include("effect.jl")

end
