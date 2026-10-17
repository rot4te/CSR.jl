# D24 follow-up: why CSR.jl's image kernel is ~14% low near lag 0 at large y (image_kernel_exact.jl).
# Circle ρ = 10 m (40 m arc), γ = 1e4. For each (y, u):
#   Ls_csr, Ls_exact      source arc distance behind the kick point (CSR.jl root vs exact retardation)
#   dLp_csr − dLp_exact   in-plane path excess Ls − chord at CSR.jl's own source point: CSR.jl (dspline_len sum)
#                         minus exact (Ls − 2ρ sin(Ls/2ρ))
#   K_csr / E(Ls_csr)     CSR.jl kernel vs the exact LW field evaluated with the source at CSR.jl's Ls
#   E(Ls_csr) / E_exact   the same field at the exact Ls
using CSR, LinearAlgebra, Printf
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

pos(s, ρ, y) = [ρ * sin(s / ρ), -ρ * (1 - cos(s / ρ)), y]
tangent(s, ρ) = [cos(s / ρ), -sin(s / ρ), 0.0]
normal(s, ρ) = [-sin(s / ρ), -cos(s / ρ), 0.0]

# LW longitudinal field at the observer (s = 0) from a source at arc position sr = −Ls, height y (field formula only).
function E_at(sr, ρ, y, γ)
  β = sqrt(1 - 1 / γ^2)
  R = pos(0.0, ρ, 0.0) - pos(sr, ρ, y)
  r = norm(R); n = R / r
  βv = β * tangent(sr, ρ); βdot = β^2 / ρ * normal(sr, ρ)
  κ = 1 - dot(n, βv)
  E = (n - βv) / (γ^2 * κ^3 * r^2) + cross(n, cross(n - βv, βdot)) / (κ^3 * r)
  return dot(E, tangent(0.0, ρ))
end
# Exact retarded Ls for lag u (bisection on τ = |R|, Ls = βτ + u).
function Ls_exact(u, ρ, y, γ)
  β = sqrt(1 - 1 / γ^2)
  f(τ) = τ - norm(pos(0.0, ρ, 0.0) - pos(-β * τ - u, ρ, y))
  a, b = 0.0, 1e3
  for _ in 1:200
    m = (a + b) / 2
    (f(m) > 0) == (f(a) > 0) ? (a = m) : (b = m)
  end
  return β * (a + b) / 2 + u
end

ρ, γ, n_ele = 10.0, 1e4, 400
geom = CSRGeometry(circle_lattice(ρ, 0.1, n_ele)...)
s_kick = 0.1 * (n_ele - 1) + 0.05
println("     y       u       Ls_csr     Ls_exact   dLp_csr-dLp_ex  K_csr/E(Ls_csr)  E(Ls_csr)/E_ex  K_csr/E_ex")
for y in (0.03, 0.3, 0.768, 1.92), u in (0.0, 1e-3, 6e-3)
  kp = CSR.CSRKickPoint(geom, n_ele, 0.05, γ; y_source = y, kick_factor = 1.0)
  k, sc = CSR.s_source_calc(CSR.CSRKick1(; ix_ele_source = n_ele, dz_particles = u), geom, kp)
  K = CSR.image_charge_kick(k, geom, kp)
  Lsc = s_kick - CSR.source_s(geom, k.ix_ele_source, sc)
  Lp = sqrt(k.L_vec[1]^2 + k.L_vec[3]^2)
  dLp_csr = k.dL + (k.L - Lp)                 # undo the y correction: in-plane excess as CSR.jl computed it
  dLp_ex = Lsc - 2ρ * sin(Lsc / (2ρ))
  Lse = Ls_exact(u, ρ, y, γ)
  Ec, Ee = E_at(-Lsc, ρ, y, γ), E_at(-Lse, ρ, y, γ)
  @printf("  %5.3f  %6.0e  %9.5f  %9.5f  %+.3e      %.6f         %.6f        %.6f\n", y, u, Lsc, Lse,
          dLp_csr - dLp_ex, K / Ec, Ec / Ee, K / Ee)
end
