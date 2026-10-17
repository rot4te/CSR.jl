# Stage 2 recon: collective TPS kick (knobs g2, gamma, charge scale) vs ForwardDiff Jacobian/Hessian.
# Usage: julia -t 4 stage2_tps_recon.jl [n_layers]
using CSR, GTPSA, ForwardDiff, StaticArrays, AtomicAndPhysicalConstants, KernelAbstractions
using Random
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
rng = Xoshiro(1); n = 400
z = 1e-3 .* randn(rng, n); q = fill(1e-9 / n, n); alive = trues(n); pz0 = zeros(n)
params = CSRParams(n_bin = 40, ds_step = 0.01, n_shield_images = parse(Int, get(ARGS, 1, "0")), beam_chamber_height = 3e-3)
d = Descriptor(3, 2); x = @vars(d)
# knobs: g2, gamma, bunch charge scale
F(p) = csr_kick_pz(pz0, z, q .* p[3], alive, CSRGeometry(L3, [0.0, p[1], 0.0], c3), 2, 0.17, p[2], electron, params)
p0 = [1.5, 200.0, 1.0]
println("Float64: ", F(p0)[1:3])
tp = [p0[1] + x[1], p0[2] + x[2], p0[3] + x[3]]
t = @time F(tp)
t = @time F(tp)
J = ForwardDiff.jacobian(F, p0)
Jt = [t[i][[j => 1]] for i in eachindex(t), j in 1:3]
println("Jacobian, max abs diff / max|J| per column: ", [maximum(abs.(Jt[:, j] .- J[:, j])) / maximum(abs.(J[:, j])) for j in 1:3])
println("scalar part vs Float64: ", maximum(abs.(scalar.(t) .- F(p0))) / maximum(abs.(F(p0))))
H = ForwardDiff.jacobian(p -> ForwardDiff.jacobian(F, p), p0)   # (n*3) × 3
Ht(i, a, b) = a == b ? 2 * t[i][[a => 2]] : t[i][[a => 1, b => 1]]
err = 0.0; hmax = maximum(abs, H)
for i in eachindex(t), a in 1:3, b in 1:3
  global err = max(err, abs(Ht(i, a, b) - H[(a - 1) * length(t) + i, b]))
end
println("Hessian max abs diff / max|H|: ", err / hmax)
