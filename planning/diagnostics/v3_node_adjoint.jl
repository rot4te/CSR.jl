# Follow-up of v3_image_entry_adjoint.jl: the mismatch is in the hat interval m = 0 (lags 0, 1). Per node of that
# interval: d K(u) / d gamma from the pullback (nodes_adjoint! with one node, weight 1) vs a Dual gamma through
# image_kernel_at (with the forward tables' PrimalSearch, and without), and the source outcome.
# julia --project=<env> planning/diagnostics/v3_node_adjoint.jl [s_kick] [layer]
using CSR, StaticArrays, ForwardDiff, KernelAbstractions
using CSR: hat_nodes, nodes_adjoint!, N_ELE_PARAMS, csr_image_table, CSRKick1, image_kernel_at, PrimalSearch, s_source_walk, with_fields
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
function kernel_test_bunch(sigma, Q, n)   # copy of test/kernels_test.jl
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n)
  alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end

L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
s_kick = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 0.17
ns = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
geom = CSRGeometry(L3, g3, c3)
p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01)
z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
dz, n_bin = csr_bin_particles(z, q, alive, p).dz, p.n_bin
y = ns * 0.01
γ0 = 200.0
kp(γ) = CSRKickPoint(geom, 2, s_kick, γ; y_source = y, kick_factor = 1.0)
W = zeros(2n_bin + 1); W[n_bin+1] = 1; W[n_bin+2] = 1
nodes = sort!(filter(((u, w),) -> 0 <= u <= dz, hat_nodes(geom, kp(γ0), dz, n_bin, W)))
n_g = 4 + N_ELE_PARAMS * length(geom.ele)
println("s_kick $s_kick layer $ns: $(length(nodes)) nodes in [0, dz]")
for (u, _) in nodes
  grad = zeros(n_g + 2)
  nodes_adjoint!(grad, geom, 2, [(u, 1.0)], true, s_kick, γ0; y_source = y, kick_factor = 1.0)
  fps = ForwardDiff.derivative(γ0) do γ
    k = kp(γ)
    last(image_kernel_at(CSRKick1(geom, k), u, geom, k, PrimalSearch(geom, CSR.primal(k))))
  end
  fnone = ForwardDiff.derivative(γ -> (k = kp(γ); last(image_kernel_at(CSRKick1(geom, k), u, geom, k))), γ0)
  _, _, o = s_source_walk(with_fields(CSRKick1(; ix_ele_source = 2); dz_particles = u), geom, kp(γ0))
  rel = abs(grad[end] - fps) / max(abs(fps), 1e-300)
  (rel > 1e-10 || u == 0) &&
    println("  u $(round(u, sigdigits = 6)) code $(o.code) ele $(o.ix): pullback $(grad[end]) fwd(search) $fps fwd(no search) $fnone rel $rel")
end

# Linear combination over the hat nodes vs the table entries (values in Float64, derivatives with a Dual gamma)
img0 = first(csr_image_table(geom, kp(γ0), dz, n_bin, CPU(); bin_integrate = true))
fwd = ForwardDiff.derivative(γ -> first(csr_image_table(geom, kp(γ), dz, n_bin, CPU(); bin_integrate = true)), γ0)
for j in (n_bin + 1, n_bin + 2)
  Wj = zeros(2n_bin + 1); Wj[j] = 1
  nj = hat_nodes(geom, kp(γ0), dz, n_bin, Wj)
  val = sum(w * last(image_kernel_at(CSRKick1(geom, kp(γ0)), u, geom, kp(γ0))) for (u, w) in nj)
  der = sum(nj) do (u, w)
    w * ForwardDiff.derivative(γ -> (k = kp(γ); last(image_kernel_at(CSRKick1(geom, k), u, geom, k))), γ0)
  end
  println("entry $j: nodes value $val table $(img0[j]) | nodes d/dγ $der table d/dγ $(fwd[j])")
end
