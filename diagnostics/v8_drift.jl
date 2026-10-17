using CSR, StaticArrays, LinearAlgebra
# Literal port of Bmad image_charge_kick_calc (kick_factor = 1), drift so g_bend = 0 (radiate = 0).
function image_kick(k, kp, theta_s)
  Bp = kp.beta * SVector(sin(theta_s), 0.0, cos(theta_s))
  N = (kp.floor_k.r - k.floor_s.r) / k.L
  B = SVector(sin(kp.floor_k.theta), 0.0, cos(kp.floor_k.theta))
  k3 = (1 - dot(N, Bp))^3
  return dot(B, N - Bp) / (kp.gamma2 * k.L^2 * k3)
end
geom = CSRGeometry([10.0], [0.0], zeros(2, 6))
gamma = 100.0; y = 0.01
kp = CSR.CSRKickPoint(geom, 1, 5.0, gamma; y_source = y)
println(" dz [m]     where   analytic     Bmad theta   true theta")
for dz in (0.03, 0.005, 0.0, -0.005, -0.009, -0.011, -0.03)
  k, s = CSR.s_source_calc(CSR.CSRKick1(; ix_ele_source = 1, dz_particles = dz), geom, kp)
  ana = gamma * dz / (gamma^2 * dz^2 + y^2)^1.5
  where = s > kp.s_chord_kick ? "ahead " : "behind"
  println(rpad(dz, 9), "  ", where, "  ", lpad(round(ana, sigdigits=5), 11), "  ",
          lpad(round(image_kick(k, kp, k.floor_s.theta), sigdigits=5), 11), "  ",
          lpad(round(image_kick(k, kp, 0.0), sigdigits=5), 11))
end
