# Image kick at lag 0 on a drift (analytic 0: the image source is level with the kicked particle, field along y only).
# PLAN Section 7a attributed Bmad's 0.125 there to its retardation L/(2γ²). Default vs bmad_compat.
# julia --project=<env> planning/diagnostics/drift_lag0_image.jl
using CSR
using CSR: CSRKick1, s_source_calc, image_charge_kick, with_fields
for bc in (false, true), gamma in (50.0, 200.0)
  geom = CSRGeometry([10.0], [0.0], zeros(2, 6); bmad_compat = bc)
  kp = CSRKickPoint(geom, 1, 5.0, gamma; y_source = 0.01)
  k, _ = s_source_calc(CSRKick1(; ix_ele_source = 1, dz_particles = 0.0), geom, kp)
  ks, _ = s_source_calc(CSRKick1(; ix_ele_source = 1, dz_particles = 1e-3), geom, kp)
  println("bmad_compat $bc γ $gamma: K(lag 0) $(image_charge_kick(k, geom, kp))   K(lag 1 mm) $(image_charge_kick(ks, geom, kp))")
end
