# Translated from Bmad lattice file: lat.bmad

using Beamlines
using BeamTracking

@elements begin
  ben01 = SBend(L =  1.00000000000000006E-001, g_ref = -1.21353548476911799E+000,
     Bn0 =  1.70000000000000012E-001)
  end_b0 = Marker()
end    # @elements
#---------------------------------------------------------------------------------------
# Overlay and Group elements


br0 = Beamline([ ben01, end_b0]; pc_ref =  4.19968913143865168E+007, species_ref = Species("electron"))
