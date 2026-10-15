# Mutation check for wake_test.jl "One source per lag": the same uniqueness check with bmad_compat = true (Bmad's
# seam formulas) should find lags whose source depends on the start element.
# julia --project=<env> planning/diagnostics/seam_uniqueness.jl
using CSR, StaticArrays
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
ddz(geom, kp, j, s) = last(CSR.ddz_calc_csr(CSR.CSRKick1(; ix_ele_source = j, dz_particles = 0.0), s, geom, kp))
for bc in (false, true), s_kick in (0.03, 0.13, 0.17, 0.29)
  geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
  kp = CSRKickPoint(geom, 2, s_kick, 200.0; y_source = 0.01)
  zb = ddz(geom, kp, 1, geom.ele[1].L_chord)   # boundary lag, element 1 side
  za = ddz(geom, kp, 2, 0.0)                   # boundary lag, element 2 side
  lags = range(min(za, zb) - 1e-9, max(za, zb) + 1e-9; length = 201)
  nbad = count(lags) do u
    r = [first(CSR.s_source_calc(CSR.CSRKick1(; ix_ele_source = j, dz_particles = u), geom, kp)).floor_s.r for j in 0:4]
    !all(x -> maximum(abs, x - r[1]) < 1e-7, r)
  end
  println("bmad_compat $bc s_kick $s_kick: jump $(za - zb), lags with start-dependent source $nbad / 201")
end
