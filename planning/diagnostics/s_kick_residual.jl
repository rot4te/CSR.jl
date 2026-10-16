# s_kick rrule residual (PLAN Section 4): rrule (csr_kick_param_adjoint) vs ForwardDiff for s_kick, free space vs
# images, lattice of ad_rrule_test.jl; then per image hat node, the node's s_kick partial in the pullback
# (node_adjoint!) vs ForwardDiff of image_kernel_at, to find which nodes carry the residual.
# julia --project=<env> planning/diagnostics/s_kick_residual.jl [s_kick, 0.23]
using CSR, StaticArrays, ChainRulesCore, ForwardDiff, Printf
using CSR: CSRKickPoint, CSRKick1, image_kernel_at, hat_nodes, ParamAdjointWork, node_adjoint!, image_layer_weight,
           seeded, SLOT_S_KICK, SLOT_GAMMA, csr_image_table, PrimalSearch
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))

function kernel_test_bunch(sigma, Q, n)   # copy of test/kernels_test.jl
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n)
  alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end

electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
n = 400
z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
pz = 1e-4 .* sin.(1:n)
ybar = cos.(1:n)
s = isempty(ARGS) ? 0.23 : parse(Float64, ARGS[1])
ik = 2
geom = CSRGeometry(L3, g3, c3)

for n_img in (0, 1)
  p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = 0.01)
  f(x) = sum(ybar .* csr_kick_pz(pz, z, q, alive, geom, ik, x, 200.0, electron, p))
  _, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, ik, s, 200.0, electron, p)
  t = back(ybar)[8]
  fs = ForwardDiff.derivative(f, s)
  @printf("n_img %d  rrule %.15e  ForwardDiff %.15e  rel %.1e\n", n_img, t, fs, abs(t - fs) / abs(fs))
end

# Per node, layer 1: pullback partial vs ForwardDiff of the kernel at that lag (both w.r.t. s_kick)
p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 1, beam_chamber_height = 0.01)
y = p.beam_chamber_height
kp0 = CSRKickPoint(geom, ik, s, 200.0; y_source = y)
dz = CSR.csr_bin_particles(z, q, alive, p).dz   # weights set to 1: only the node lags matter here
nodes = hat_nodes(geom, kp0, dz, p.n_bin, ones(2p.n_bin + 1))
sort!(nodes)
search = PrimalSearch(geom, kp0)
wk = ParamAdjointWork(geom, ik, 4 + 26 * 3 + 2)
kp_d = CSRKickPoint(wk.geom_d, ik, seeded(s, SLOT_S_KICK), seeded(200.0, SLOT_GAMMA); y_source = y)
k0 = CSRKick1(; ix_ele_source = ik)
worst = Tuple{Float64,Float64,Float64,Float64,Int}[]
for (u, _) in nodes
  fill!(wk.grad, 0.0)
  global k0 = node_adjoint!(wk, true, 1.0, u, k0, kp0, kp_d)
  a = wk.grad[end-1]
  fd = ForwardDiff.derivative(x -> begin
      kp = CSRKickPoint(geom, ik, x, 200.0; y_source = y)
      last(image_kernel_at(CSRKick1(geom, kp), u, geom, kp, PrimalSearch(geom, kp0)))
    end, s)
  kk, K = image_kernel_at(CSRKick1(geom, kp0), u, geom, kp0)
  push!(worst, (abs(a - fd) / max(abs(fd), 1e-300), u, a, fd, kk.ix_ele_source))
end
sort!(worst; rev = true)
println("layer 1 hat nodes: $(length(nodes)); worst per-node rel diff (pullback vs ForwardDiff of the kernel):")
for (r, u, a, fd, ix) in worst[1:8]
  @printf("  u %+.6e  ele %d  pullback %+.6e  ForwardDiff %+.6e  rel %.1e\n", u, ix, a, fd, r)
end

# Round-off floor of ForwardDiff itself: seed scaled by c (exact in exact arithmetic), spread relative to the rrule
println("ForwardDiff with seed c, (d/dt f(s + c t))/c, relative to the rrule (images, 1 layer):")
let p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 1, beam_chamber_height = 0.01)
  f(x) = sum(ybar .* csr_kick_pz(pz, z, q, alive, geom, ik, x, 200.0, electron, p))
  _, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, ik, s, 200.0, electron, p)
  t = back(ybar)[8]
  for c in (1.0, 2.0, 3.0, 0.7, 1 / 3, 5.0, 0.1)
    d = ForwardDiff.derivative(x -> f(s + c * x), 0.0) / c
    @printf("  c %-6.4g  rel to rrule %+.1e\n", c, (d - t) / abs(t))
  end
end
