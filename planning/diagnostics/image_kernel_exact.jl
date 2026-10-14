# D24: CSR.jl's image kernel (Bmad's approximate retardation) vs an independent Liénard–Wiechert field with exact
# retardation, for a source on a circle of radius ρ displaced vertically by y (units c = 1, q = 1, Gaussian).
# Observer at s = 0 moving along +x; source at lag u (u > 0: behind) on the same circle, height y.
using CSR, LinearAlgebra, Printf
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

pos(s, ρ, y) = [ρ * sin(s / ρ), -ρ * (1 - cos(s / ρ)), y]     # bends to −x side as planar_survey (g > 0); sign immaterial
tangent(s, ρ) = [cos(s / ρ), -sin(s / ρ), 0.0]
normal(s, ρ) = [-sin(s / ρ), -cos(s / ρ), 0.0]                 # towards the centre

# Exact longitudinal LW field at the observer from the source at lag u.
function E_exact(u, ρ, y, γ)
  β = sqrt(1 - 1 / γ^2)
  f(τ) = τ - norm(pos(0.0, ρ, 0.0) - pos(-β * τ - u, ρ, y))    # τ = −c t_ret
  a, b = 0.0, 1e3
  f(a) < 0 || error("bracket")
  for _ in 1:200
    m = (a + b) / 2
    (f(m) > 0) == (f(a) > 0) ? (a = m) : (b = m)
  end
  τ = (a + b) / 2
  sr = -β * τ - u
  R = pos(0.0, ρ, 0.0) - pos(sr, ρ, y)
  r = norm(R); n = R / r
  βv = β * tangent(sr, ρ)
  βdot = β^2 / ρ * normal(sr, ρ)
  κ = 1 - dot(n, βv)
  E = (n - βv) / (γ^2 * κ^3 * r^2) + cross(n, cross(n - βv, βdot)) / (κ^3 * r)
  return dot(E, tangent(0.0, ρ)), -sr
end

function K_csr(u, geom, kp)
  k = CSR.CSRKick1(); k.ix_ele_source = kp.ix_ele_kick
  k.dz_particles = u
  CSR.s_source_calc!(k, geom, kp)
  return CSR.image_charge_kick(k, geom, kp)
end

ρ, γ, n_ele = 10.0, 1e4, 400
geom = CSRGeometry(circle_lattice(ρ, 0.1, n_ele)...)
for y in (0.003, 0.03, 0.3, 0.768, 1.92)
  kp = CSR.CSRKickPoint(geom, n_ele, 0.05, γ; y_source = y, kick_factor = 1.0)
  println("y = $y")
  println("      u [m]      K_csr          E_exact        ratio      Ls_exact")
  for u in (-6e-3, -3e-3, -1e-3, -1e-4, 0.0, 1e-4, 1e-3, 3e-3, 6e-3)
    Kc = K_csr(u, geom, kp)
    Ee, Ls = E_exact(u, ρ, y, γ)
    @printf("  %+.1e  %+.6e  %+.6e  %+.6f  %7.3f\n", u, Kc, Ee, Kc / Ee, Ls)
  end
end
