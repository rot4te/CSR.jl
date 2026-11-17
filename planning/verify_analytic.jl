# Numerical checks of the analytic relations in csr_gpu_ad_design.md and of the Bmad 1D CSR
# formulas (bmad-ecosystem/bmad/space_charge/csr_and_space_charge_mod.f90).
# Base Julia only. Run: julia verify_analytic.jl

using Printf, LinearAlgebra, Random

simpson(f, a, b, n) = (h = (b - a) / n; h / 3 * (f(a) + f(b) + 4 * sum(f(a + (2k - 1) * h) for k in 1:n÷2) + 2 * sum(f(a + 2k * h) for k in 1:n÷2-1)))

function report(name, err, tol)
  @printf("%-70s err = %10.3e  %s\n", name, err, err < tol ? "PASS" : "FAIL")
  return err < tol
end

results = Bool[]

# ---------------------------------------------------------------------------
# C1  Retardation: z = Ls - beta*L (exact), Bmad uses z ~ L/(2 gamma^2) + dL.
#     Error of the approximation should be L*(1 - beta - 1/(2 gamma^2)) ~ L/(8 gamma^4).
let
  for gamma in (10.0, 100.0, 1000.0), L in (0.01, 1.0)
    beta = sqrt(1 - 1 / gamma^2)
    dL = 1e-4 * L
    Ls = L + dL
    z_exact = Ls - beta * L
    z_bmad = L / (2gamma^2) + dL
    pred = L / (8gamma^4)
    push!(results, report(@sprintf("C1 z_exact - z_bmad vs L/(8γ⁴), γ=%g L=%g", gamma, L), abs((z_exact - z_bmad) - pred) / pred, 1e-2))
  end
end

# ---------------------------------------------------------------------------
# C2  Circular arc: chord L = 2ρ sin(φ/2), dL = ρφ - L ≈ ρφ³/24 (next term -ρφ⁵/1920).
let rho = 1.0
  for phi in (1e-1, 1e-2)
    L = 2rho * sin(phi / 2)
    dL = rho * phi - L
    push!(results, report(@sprintf("C2 dL vs ρφ³/24 - ρφ⁵/1920, φ=%g", phi), abs(dL - (rho * phi^3 / 24 - rho * phi^5 / 1920)) / dL, 1e-6))
    push!(results, report(@sprintf("C2 L vs ρφ(1-φ²/24), φ=%g", phi), abs(L - rho * phi * (1 - phi^2 / 24)) / L, phi^4))
  end
end

# ---------------------------------------------------------------------------
# C3  Overtaking length: for γ → ∞, z = dL; Ls = (24 ρ² z)^(1/3) to leading order.
let rho = 2.0
  for phi in (1e-2, 1e-3)
    z = rho * phi - 2rho * sin(phi / 2)
    Ls = rho * phi
    push!(results, report(@sprintf("C3 Ls vs (24ρ²z)^(1/3), φ=%g", phi), abs(Ls - cbrt(24rho^2 * z)) / Ls, phi^2))
  end
end

# ---------------------------------------------------------------------------
# Circle geometry for Bmad's I_csr (source an angle φ behind the kick point, same circle).
# θ_sl = θ_lk = φ/2 (tangent-to-chord angles).
circ_L(rho, phi) = 2rho * sin(phi / 2)
circ_dL(rho, phi) = rho * phi - circ_L(rho, phi)
circ_z(rho, phi, gamma) = circ_L(rho, phi) / (2gamma^2) + circ_dL(rho, phi)
function I_csr_circle(rho, phi, gamma)
  L = circ_L(rho, phi); dL = circ_dL(rho, phi); z = circ_z(rho, phi, gamma)
  th = phi / 2
  return -2 * (dL / z + gamma^2 * th * th / (1 + gamma^2 * th^2)) / L
end
I_ss(rho, z) = -2 / cbrt(3) * rho^(-2/3) * z^(-1/3)   # steady-state kernel implied by doc §2.4

# C4  Bmad I_csr on a circle tends to -2·3^(-1/3) ρ^(-2/3) z^(-1/3) for γφ ≫ 1, φ ≪ 1.
let rho = 1.0, gamma = 1e6
  for phi in (1e-2, 1e-3)
    z = circ_z(rho, phi, gamma)
    push!(results, report(@sprintf("C4 I_csr(circle) vs steady-state kernel, φ=%g", phi), abs(I_csr_circle(rho, phi, gamma) / I_ss(rho, z) - 1), 1e-3))
  end
end

# C5  Bmad first-bin integral, γ → ∞:  -(g Ls/2)²  should equal ∫₀ᶻ I_ss = -3^(2/3) ρ^(-2/3) z^(2/3).
let rho = 1.0
  for phi in (1e-2, 1e-3)
    z = circ_dL(rho, phi); Ls = rho * phi
    lhs = -(Ls / (2rho))^2
    rhs = -3^(2/3) * rho^(-2/3) * z^(2/3)
    push!(results, report(@sprintf("C5 first-bin I_int (γ=∞) vs ∫I_ss, φ=%g", phi), abs(lhs / rhs - 1), phi^2))
  end
end

# C6  Bmad first-bin integral with finite γ:
#     -((g Ls/2)² - ln(2γ² z/Ls)/γ²)  vs  ∫₀ᶻ I_csr(z') dz' computed by quadrature on a circle.
let rho = 1.0
  for gamma in (30.0, 100.0), phi in (0.01, 0.05, 0.2)
    dzdphi(p) = rho * cos(p / 2) / (2gamma^2) + rho - rho * cos(p / 2)
    integral = simpson(p -> (p == 0 ? 0.0 : I_csr_circle(rho, p, gamma)) * dzdphi(p), 0.0, phi, 20000)
    z = circ_z(rho, phi, gamma); Ls = rho * phi
    bmad = -((Ls / (2rho))^2 - log(2gamma^2 * z / Ls) / gamma^2)
    @printf("C6 γ=%5g φ=%5g  γφ=%7.2f  quad=%12.5e  bmad=%12.5e  rel diff=%9.2e\n", gamma, phi, gamma * phi, integral, bmad, abs(bmad / integral - 1))
  end
end

# ---------------------------------------------------------------------------
# C7  Bmad discretization (edge gradients, first-bin + trapezoid I_int, causal sum) converges to
#     W(z) = ∫₀^∞ I_ss(u) λ'(z-u) du  =  -(2/3^(1/3)) ρ^(-2/3) ∫_{-∞}^z λ'(z')/(z-z')^(1/3) dz'
#     (normalised λ, prefactors stripped). Checks the doc §2.4 integral against Bmad's scheme.
let rho = 1.0, sig = 1e-3
  lam(z) = exp(-z^2 / (2sig^2)) / (sqrt(2pi) * sig)
  dlam(z) = -z / sig^2 * lam(z)
  W(z) = -(2 / cbrt(3)) * rho^(-2/3) * 1.5 * simpson(u -> dlam(z - u^1.5), 0.0, (12sig)^(2/3), 4000)
  for nb in (100, 400, 1600)
    zlo, zhi = -6sig, 6sig
    dz = (zhi - zlo) / nb
    edges = zlo .+ dz .* (0:nb)
    Q = [simpson(lam, edges[j], edges[j+1], 20) for j in 1:nb]           # bin charge
    g = [j == 1 ? 0.0 : (Q[j] - Q[j-1]) / dz^2 for j in 1:nb]            # edge gradient
    Iint = similar(Q)
    Iint[1] = -3^(2/3) * rho^(-2/3) * dz^(2/3)                           # γ=∞ first bin
    for m in 2:nb
      Iint[m] = (I_ss(rho, m * dz) + I_ss(rho, (m - 1) * dz)) * dz / 2
    end
    kick = [sum(Iint[k] * g[i+1-k] for k in 1:i) for i in 1:nb]
    zc = edges[1:nb] .+ dz / 2
    idx = [argmin(abs.(zc .- zt)) for zt in (-sig, 0.0, sig)]
    errs = [abs(kick[i] - W(zc[i])) / maximum(abs, W.(zc[idx])) for i in idx]
    push!(results, report(@sprintf("C7 Bmad scheme vs steady-state integral, n_bin=%d", nb), maximum(errs), nb >= 1600 ? 1e-2 : 1.0))
  end
end

# ---------------------------------------------------------------------------
# C8  dspline_len closed form vs quadrature of (x' - dθ)²/2 for x = c0 + c1 s + c2 s² + c3 s³.
#     Bmad's version (bmad = true) has `* 9*c(3)**2` for the s⁴ term; integrating 9c3² s⁴ gives
#     9c3²/5 * (s1⁵ - s0⁵)/ds, so the factor 1/5 appears to be missing. Exact when c3 = 0.
function dspline_len(s0, s1, c, dtheta; bmad = false)
  c1 = c[2] - dtheta; c2 = c[3]; c3 = c[4]; ds = s1 - s0
  return (ds / 2) * (c1^2 + (2s0 + ds) * 2c1 * c2 + (3s0^2 + 3s0 * ds + ds^2) * (6c1 * c3 + 4c2^2) / 3 +
                     (4s0^3 + 6s0^2 * ds + 4s0 * ds^2 + ds^3) * 3c2 * c3 +
                     (5s0^4 + 10s0^3 * ds + 10s0^2 * ds^2 + 5s0 * ds^3 + ds^4) * 9c3^2 / (bmad ? 1 : 5))
end
let rng = MersenneTwister(1)
  for _ in 1:3
    c = randn(rng, 4) .* [0, 1e-2, 1e-2, 1e-2]; s0, s1 = rand(rng), 1 + rand(rng); dth = 1e-3 * randn(rng)
    q = simpson(s -> (c[2] + 2c[3] * s + 3c[4] * s^2 - dth)^2 / 2, s0, s1, 2000)
    push!(results, report("C8 dspline_len (with 9c3²/5) vs quadrature", abs(dspline_len(s0, s1, c, dth) / q - 1), 1e-10))
    @printf("   Bmad form (9c3², no /5): ratio to quadrature = %.4f\n", dspline_len(s0, s1, c, dth; bmad = true) / q)
  end
  # Spline with x(0) = x(Lc) = 0 and end slopes t0, t1 (Bmad's create_a_spline construction): c3 = (t0 + t1)/Lc².
  Lc = 1.0
  for (t0, t1) in ((0.01, -0.01), (0.01, -0.012), (0.01, 0.0))
    c = [0.0, t0, -(2t0 + t1) / Lc, (t0 + t1) / Lc^2]
    q = simpson(s -> (c[2] + 2c[3] * s + 3c[4] * s^2)^2 / 2, 0.0, Lc, 2000)
    @printf("   chord spline t0=%6.3f t1=%7.4f: Bmad/quad = %.4f, corrected/quad = %.6f\n", t0, t1,
            dspline_len(0.0, Lc, c, 0.0; bmad = true) / q, dspline_len(0.0, Lc, c, 0.0) / q)
  end
end

# ---------------------------------------------------------------------------
# C9  Upstream infinite drift: Bmad root ds = -(-b + sqrt(b²-4ac))/(2a), a=1/γ², b=2(Lsz0-dz-β²Lz),
#     c=(Lsz0-dz)²-β²L0², must satisfy  Ls - β L = dz  with Ls = Lsz0 - ds, L = |r_k - r_s(ds)|.
let rng = MersenneTwister(2)
  for _ in 1:3
    gamma = 50 + 500rand(rng); beta2 = 1 - 1 / gamma^2
    th0 = 0.1randn(rng); f0 = [0.0, 0.0]; t = [sin(th0), cos(th0)]           # (x, z) floor plane
    fk = [0.02randn(rng), 1 + rand(rng)]
    L0 = norm(fk - f0); Lz = dot(fk - f0, t)
    Lsz0 = L0 + 1e-4; dz = 1e-3 * rand(rng)
    a = 1 / gamma^2; b = 2 * (Lsz0 - dz - beta2 * Lz); c = (Lsz0 - dz)^2 - beta2 * L0^2
    ds = -(-b + sqrt(b^2 - 4a * c)) / (2a)
    rs = f0 .+ ds .* t
    resid = (Lsz0 - ds) - sqrt(beta2) * norm(fk - rs) - dz
    push!(results, report("C9 upstream-drift quadratic satisfies retardation (ds<0)", abs(resid) / dz + (ds < 0 ? 0.0 : 1.0), 1e-6))
  end
end

# ---------------------------------------------------------------------------
# C10 Triangle shape (Bmad csr_bin_particles): overlap weights sum to 1.
function tri_overlap(zp, dzp, zb0, zb1)
  zp0 = zp - dzp / 2; zp1 = zp + dzp / 2; ov = 0.0
  a = max(zp0, zb0); b = min(zp, zb1)
  b > a && (ov += 2 * ((b - zp0)^2 - (a - zp0)^2) / dzp^2)
  a = max(zp, zb0); b = min(zp1, zb1)
  b > a && (ov += 2 * ((a - zp1)^2 - (b - zp1)^2) / dzp^2)
  return ov
end
let dz = 0.1, span = 2, rng = MersenneTwister(3)
  for _ in 1:5
    zp = 1 + rand(rng)
    s = sum(tri_overlap(zp, span * dz, k * dz, (k + 1) * dz) for k in 0:40)
    push!(results, report("C10 triangle overlap weights sum to 1", abs(s - 1), 1e-12))
  end
end

# ---------------------------------------------------------------------------
# C11 Reverse-mode adjoints of design doc §3.8 vs finite differences (linear hat shape, doc §4.2 kernels).
let rng = MersenneTwister(4), n = 24, np = 50, dz = 0.1, z0 = 0.0, c = 0.7
  hat(u) = (i = floor(Int, u) + 1; f = u - (i - 1); (i, f))
  function forward(z, q, K)
    Q = zeros(n)
    for p in eachindex(z)
      i, f = hat((z[p] - z0) / dz - 0.5); Q[i] += q[p] * (1 - f); Q[i+1] += q[p] * f
    end
    g = [(j == 1 ? Q[1] : Q[j] - Q[j-1]) / dz^2 for j in 1:n]
    k = [c * sum(K[i-j+1] * g[j] for j in 1:i) for i in 1:n]
    d = [(i_f = hat((z[p] - z0) / dz - 0.5); (1 - i_f[2]) * k[i_f[1]] + i_f[2] * k[i_f[1]+1]) for p in eachindex(z)]
    return d, Q, g, k
  end
  z = z0 .+ dz .* (4 .+ (n - 10) .* rand(rng, np)); q = rand(rng, np); K = randn(rng, n); r = randn(rng, np)
  loss(z, q, K) = dot(r, forward(z, q, K)[1])
  _, Q, g, k = forward(z, q, K)
  # adjoints (doc §3.8), δ̄ = r
  kb = zeros(n); zb = zeros(np)
  for p in 1:np
    i, f = hat((z[p] - z0) / dz - 0.5)
    kb[i] += (1 - f) * r[p]; kb[i+1] += f * r[p]
    zb[p] += r[p] * (k[i+1] - k[i]) / dz                         # w' for hat
  end
  gb = [c * sum(K[i-j+1] * kb[i] for i in j:n) for j in 1:n]
  Qb = [(gb[j] - (j < n ? gb[j+1] : 0.0)) / dz^2 for j in 1:n]
  qb = zeros(np)
  for p in 1:np
    i, f = hat((z[p] - z0) / dz - 0.5)
    qb[p] = (1 - f) * Qb[i] + f * Qb[i+1]
    zb[p] += q[p] * (Qb[i+1] - Qb[i]) / dz
  end
  Kb = [c * sum(kb[j+m-1] * g[j] for j in 1:n-m+1) for m in 1:n]
  h = 1e-6
  fd(f, x, j) = (xp = copy(x); xm = copy(x); xp[j] += h; xm[j] -= h; (f(xp) - f(xm)) / 2h)
  eq = maximum(abs(fd(x -> loss(z, x, K), q, j) - qb[j]) for j in 1:np)
  eK = maximum(abs(fd(x -> loss(z, q, x), K, j) - Kb[j]) for j in 1:n)
  ez = maximum(abs(fd(x -> loss(x, q, K), z, j) - zb[j]) for j in 1:np)
  sc = maximum(abs, [qb; Kb; zb])
  push!(results, report("C11 adjoint q̄ (scatter→gradᵀ→corr→deposit gather) vs FD", eq / sc, 1e-6))
  push!(results, report("C11 adjoint K̄_m = c Σ_j k̄_{j+m-1} g_j vs FD", eK / sc, 1e-6))
  push!(results, report("C11 adjoint z̄ (gather terms) vs FD", ez / sc, 1e-6))
end

println()
println(all(results) ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED ($(count(!, results)) of $(length(results)))")
