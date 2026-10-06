# Translated from Bmad lattice file: lat.bmad

using Beamlines
using BeamTracking

@elements begin
  mar_csr = Marker()
  ff_pip00b = Drift(L =  5.99999999999999978E-002)
  ff_ben01 = SBend(L =  1.33000000000000007E-001, g_ref = -1.21353548476911799E+000,
     Bn0 =  1.70000000000000040E-001, Bn1 = -1.06209858499315342E+001)
  ff_pip01 = Drift(L =  7.00000000000000067E-002)
  ff_ben02 = SBend(L =  1.21999999999999997E-001, g_ref =  1.99876432785501801E+000,
     Bn0 = -2.80000000000000082E-001, Bn1 =  1.00166517550704306E+001)
  ff_pip02a = Drift(L =  5.99999999999999978E-002)
  mar_end = Marker()
  end_b0 = Marker()
end    # @elements
#---------------------------------------------------------------------------------------
# Overlay and Group elements


lat = Beamline([ mar_csr, ff_pip00b, ff_ben01, ff_pip01, ff_ben02, ff_pip02a, mar_end,
     end_b0]; pc_ref =  4.19968913143865168E+007, species_ref = Species("electron"))
