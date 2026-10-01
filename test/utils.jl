# Test utilities.

using AtomicAndPhysicalConstants

"Composite Simpson rule with `n` (even) intervals."
function simpson(f, a, b, n)
  h = (b - a) / n
  return h / 3 * (f(a) + f(b) + 4 * sum(f(a + (2k - 1) * h) for k in 1:n÷2) + 2 * sum(f(a + 2k * h) for k in 1:n÷2-1))
end

gaussian(z, sigma) = exp(-z^2 / (2sigma^2)) / (sqrt(2pi) * sigma)
dgaussian(z, sigma) = -z / sigma^2 * gaussian(z, sigma)

"""
    steady_state_dpz_ds(z, dlambda, rho, N, gamma; u_max, n = 4000)

Steady-state 1D CSR energy change per unit length, as a relative momentum change for electrons:

    dδ/ds = -(2/3^(1/3)) N r_e / γ ρ^(-2/3) ∫_{-∞}^{z} λ'(z') (z - z')^(-1/3) dz'

with `λ` normalised to 1 and `z > 0` towards the head. The integral is evaluated with
`z - z' = u^(3/2)`, which removes the singularity: `∫ … = (3/2) ∫₀^∞ λ'(z - u^(3/2)) du`.
"""
function steady_state_dpz_ds(z, dlambda, rho, N, gamma; u_max, n = 4000)
  integral = 1.5 * simpson(u -> dlambda(z - u^1.5), 0.0, u_max, n)
  return -(2 / cbrt(3)) * N * R_ELECTRON / gamma * rho^(-2/3) * integral
end

"""
    circle_lattice(rho, L_ele, n_ele) -> (L, g, centroid)

`n_ele` sector bends of length `L_ele` and radius `rho`, with the centroid on the reference orbit.
"""
function circle_lattice(rho, L_ele, n_ele)
  return fill(L_ele, n_ele), fill(1 / rho, n_ele), zeros(n_ele + 1, 6)
end

"""
    gaussian_line_bunch(sigma, Q, n) -> (z, q, alive)

Deterministic line bunch: `n` particles on a uniform grid over ±6σ with Gaussian charges summing to `Q` [C].
"""
function gaussian_line_bunch(sigma, Q, n)
  z = collect(range(-6sigma, 6sigma; length = n))
  w = gaussian.(z, sigma)
  q = Q .* w ./ sum(w)
  return z, q, fill(true, n)
end
