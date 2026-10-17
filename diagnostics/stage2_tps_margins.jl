# Stage 2a: errors of collective TPS vs ForwardDiff (the cases of test/tps_collective_test.jl), and a mutation check.
# Usage: julia -t 4 stage2_tps_margins.jl [mutate]   (mutate: root_order(::TPS) = 1, one Newton step only)
using CSR, GTPSA, ForwardDiff, StaticArrays, AtomicAndPhysicalConstants
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
function kernel_test_bunch(sigma, Q, n)   # as test/kernels_test.jl
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n); alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end
get(ARGS, 1, "") == "mutate" && (@eval CSR.root_order(t::TPS) = 1)
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
n = 400; z, q, alive = kernel_test_bunch(1e-3, 1e-9, n); pz0 = zeros(n)
v0 = [0.3, 1.5, 1.1e-3, 0.17, 200.0, 0.0]
function F(v, p)
  c = similar(c3, eltype(v)); c .= c3; c[3, 1] = v[3]
  geom = CSRGeometry([L3[1], v[1], L3[3]], [g3[1], v[2], g3[3]], c; bmad_compat = p.bmad_compat)
  return csr_kick_pz(pz0, z .* (1 + 0.2v[6]), q .* (1 + v[6]), alive, geom, 2, v[4], v[5], electron, p)
end
for n_img in (0, 2), bc in (false, true)
  p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0, bmad_compat = bc)
  x = @vars(Descriptor(6, 2)); t = F(v0 .+ x, p); y0 = F(v0, p)
  Jt, Jf = GTPSA.jacobian(t), ForwardDiff.jacobian(v -> F(v, p), v0)
  jerr = [maximum(abs, Jt[:, c] .- Jf[:, c]) / maximum(abs, Jf[:, c]) for c in 1:6]
  Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(v -> F(v, p), u), v0), n, 6, 6)
  Ht = permutedims(cat((GTPSA.hessian(ti) for ti in t)...; dims = 3), (3, 1, 2))
  println("images $n_img compat $bc: scalar ", maximum(abs, scalar.(t) .- y0) / maximum(abs, y0),
          "  J cols ", round.(jerr; sigdigits = 2), "  H ", maximum(abs, Ht .- Hf) / maximum(abs, Hf))
end
for n_img in (0, 2)
  p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0)
  F2(u) = F([v0[1], u[1], v0[3], v0[4], u[2], v0[6]], p); u0 = [v0[2], v0[5]]
  x = @vars(Descriptor(2, 3)); t = F2(u0 .+ x)
  T3f = reshape(ForwardDiff.jacobian(w -> ForwardDiff.jacobian(u -> ForwardDiff.jacobian(F2, u), w), u0), n, 2, 2, 2)
  T3t = [scalar(GTPSA.deriv(t[j], [count(==(i), (a, b, c)) for i in 1:2])) for j in 1:n, a in 1:2, b in 1:2, c in 1:2]
  println("order 3 images $n_img: ", maximum(abs, T3t .- T3f) / maximum(abs, T3f))
end
