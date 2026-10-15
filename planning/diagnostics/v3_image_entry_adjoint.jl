# Pullback (D30) vs ForwardDiff mismatch in the default image mode at some kick positions (v3_rrule_scan.jl).
# Per image-table entry j: d image[j] / d gamma from the pullback's hat nodes (Wimg = unit vector e_j) vs a Dual gamma
# through csr_image_table. Bins from the ad_rrule_test.jl bunch.
# julia --project=<env> planning/diagnostics/v3_image_entry_adjoint.jl [s_kick] [layer]
using CSR, StaticArrays, ForwardDiff, KernelAbstractions
using CSR: hat_nodes, nodes_adjoint!, csr_image_table, N_ELE_PARAMS
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
ns = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 2
geom = CSRGeometry(L3, g3, c3)
p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01)
z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
bins = csr_bin_particles(z, q, alive, p)
dz, n_bin = bins.dz, p.n_bin
y = ns * 0.01
γ0 = 200.0

kp(γ) = CSRKickPoint(geom, 2, s_kick, γ; y_source = y, kick_factor = 1.0)
fwd = ForwardDiff.derivative(γ -> first(csr_image_table(geom, kp(γ), dz, n_bin, CPU(); bin_integrate = true)), γ0)
n_g = 4 + N_ELE_PARAMS * length(geom.ele)
println("s_kick $s_kick layer $ns dz $dz")
for j in 1:2n_bin+1
  W = zeros(2n_bin + 1); W[j] = 1
  nodes = hat_nodes(geom, kp(γ0), dz, n_bin, W)
  grad = zeros(n_g + 2)
  nodes_adjoint!(grad, geom, 2, nodes, true, s_kick, γ0; y_source = y, kick_factor = 1.0)
  rel = abs(grad[end] - fwd[j]) / maximum(abs, fwd)
  rel > 1e-12 && println("  j $j (lag $(j - n_bin - 1)): pullback $(grad[end])  forward $(fwd[j])  rel to max $rel")
end
