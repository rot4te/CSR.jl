# Follow-up of v3_node_adjoint.jl: hat_nodes' node sum differs from the image table's value for lags 0, 1 (1e-4
# relative), so the hat quadrature of interval m = 0 depends on how the source search is started. Compares the
# moments of interval m from a cold work struct and warm-started through the previous intervals (as hat_nodes),
# and lists lags where the kernel differs between a cold start and the warm state of the sequence.
# julia --project=<env> planning/diagnostics/v3_warm_start.jl [s_kick] [layer]
using CSR, StaticArrays, KernelAbstractions
using CSR: CSRKick1, image_interval_moments, image_u_min, image_hat_tol, csr_image_table, image_kernel_at
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
kp = CSRKickPoint(geom, 2, s_kick, 200.0; y_source = ns * 0.01, kick_factor = 1.0)
img0 = first(csr_image_table(geom, kp, dz, n_bin, CPU(); bin_integrate = true))
u_min = image_u_min(kp)
tol = image_hat_tol(geom, kp, img0, dz, 1e-6, u_min)

kw = CSRKick1(geom, kp)
for m in -n_bin-1:2
  k_in = kw
  global kw, mw = image_interval_moments(kw, geom, kp, m, dz, tol, u_min)
  _, mc = image_interval_moments(CSRKick1(geom, kp), geom, kp, m, dz, tol, u_min)
  d = maximum(abs, mw - mc) / maximum(abs, mc)
  d > 1e-12 && println("m $m: warm (from ele $(k_in.ix_ele_source)) $mw  cold $mc  rel $d")
end

# Kernel at lags in [0, dz]: cold start vs a start from each element
for u in range(0, dz; length = 41)
  K = [last(image_kernel_at(CSRKick1(; ix_ele_source = j), u, geom, kp)) for j in 1:3]
  spread = (maximum(K) - minimum(K)) / maximum(abs, K)
  spread > 1e-9 && println("u $(round(u, sigdigits = 5)): K from ele 1/2/3 start: $K")
end

# The same on the hat nodes of interval m = 0 (down to u_min), and the table entry from the moments
using CSR: hat_nodes
W = zeros(2n_bin + 1); W[n_bin+1] = 1
nodes = sort!(hat_nodes(geom, kp, dz, n_bin, W))
println("$(length(nodes)) nodes for entry $(n_bin + 1); u_min $u_min")
for (u, w) in nodes
  K = [last(image_kernel_at(CSRKick1(; ix_ele_source = j), u, geom, kp)) for j in 1:3]
  spread = (maximum(K) - minimum(K)) / maximum(abs, K)
  spread > 1e-9 && println("  u $(round(u, sigdigits = 5)) w $(round(w, sigdigits = 4)): K from ele 1/2/3 start: $K")
end
_, mm1 = image_interval_moments(CSRKick1(geom, kp), geom, kp, -1, dz, tol, u_min)
_, m0 = image_interval_moments(CSRKick1(geom, kp), geom, kp, 0, dz, tol, u_min)
println("entry $(n_bin + 1): moments $((mm1[1] + m0[2]) / dz)  table $(img0[n_bin+1])  nodes (cold K) ",
        sum(w * last(image_kernel_at(CSRKick1(geom, kp), u, geom, kp)) for (u, w) in nodes))

# The ambiguous lag: search outcomes from each start element, and ddz along the lattice (sign changes = roots)
using CSR: s_source_walk, with_fields, ddz_calc_csr
u = first(nodes[findmin(((v, _),) -> abs(v - 1.5528e-4), nodes)[2]]); println("u = ", repr(u))
for j in 1:3
  _, _, o = s_source_walk(with_fields(CSRKick1(; ix_ele_source = j); dz_particles = u), geom, kp)
  println("start ele $j: code $(o.code) ele $(o.ix) s_root $(o.s_root) s_last $(o.s_last)")
end
for j in 1:2
  e = geom.ele[j]
  ss = range(0, e.L_chord; length = 2001)
  d = [last(ddz_calc_csr(with_fields(CSRKick1(; ix_ele_source = j); dz_particles = u), s, geom, kp)) for s in ss]
  idx = findall(i -> sign(d[i]) != sign(d[i+1]), 1:length(d)-1)
  println("ele $j: ddz sign changes at s_chord ", ss[idx], "  ddz range ", extrema(d))
end

# Work struct after the search from start elements 1 and 2
using CSR: s_source_calc
k1, s1 = s_source_calc(with_fields(CSRKick1(; ix_ele_source = 1); dz_particles = u), geom, kp)
k2, s2 = s_source_calc(with_fields(CSRKick1(; ix_ele_source = 2); dz_particles = u), geom, kp)
println("s_chord: $s1 vs $s2")
for f in fieldnames(typeof(k1))
  a, b = getfield(k1, f), getfield(k2, f)
  isequal(a, b) || println("  $f: $a | $b")
end
