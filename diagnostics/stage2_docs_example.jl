# Runs the collective-TPS example of docs/src/usage.md (not executed by the docs build).
using CSR, GTPSA, AtomicAndPhysicalConstants
e = Species("electron")
L = [0.2, 0.3, 0.2]; centroid = zeros(4, 6)
n = 400; z = collect(range(-3e-3, 3e-3; length = n)); q = 1e-9 .* exp.(-z .^ 2 ./ 2e-6) ./ n; alive = fill(true, n)
pz = zeros(n); params = CSRParams(n_bin = 40, ds_step = 0.1)
x = @vars(Descriptor(2, 2))
g = [0.0, 1.5 + x[1], 0.0]   # bend curvature as a knob
pz_new = csr_kick_pz(pz, z, q .* (1 + x[2]), alive, CSRGeometry(L, g, centroid), 2, 0.17, 200.0, e, params)
J = GTPSA.jacobian(pz_new)   # ∂Δpz/∂g₂ and ∂Δpz/∂(charge scale) for each particle
println(size(J), " ", J[n ÷ 2, :], " scalar ", scalar(pz_new[n ÷ 2]))
