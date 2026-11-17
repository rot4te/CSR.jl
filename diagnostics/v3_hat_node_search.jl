# Pullback (D30) vs ForwardDiff mismatch in the default image mode at some kick positions (v3_rrule_scan.jl).
# Hypothesis: the pullback re-searches the source at each hat node with a different warm start (nodes sorted by lag)
# than the forward hat quadrature, and the search outcome differs at some lags. Compares, per node, the source
# outcome and kernel value from a cold start, from the pullback's sorted warm start, and the forward's interval order.
# julia --project=<env> planning/diagnostics/v3_hat_node_search.jl [s_kick] [n_img]
using CSR, StaticArrays
using CSR: CSRKick1, s_source_walk, s_source_calc, image_charge_kick, with_fields, hat_nodes, image_layer_weight

L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
s_kick = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 0.17
n_img = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 2
geom = CSRGeometry(L3, g3, c3)
n_bin = 40
dz = 12e-3 / n_bin   # roughly the test bunch's bin width; the exact value does not matter for the search
Wimg = ones(2n_bin + 1)
println("s_kick $s_kick")
for ns in 1:n_img
  kp = CSRKickPoint(geom, 2, s_kick, 200.0; y_source = ns * 0.01, kick_factor = 1.0)
  nodes = sort!(hat_nodes(geom, kp, dz, n_bin, Wimg))
  kw = CSRKick1(; ix_ele_source = 2)
  nbad = 0
  for (u, _) in nodes
    kc, _, oc = s_source_walk(with_fields(CSRKick1(; ix_ele_source = 2); dz_particles = u), geom, kp)
    kw, _, ow = s_source_walk(with_fields(kw; dz_particles = u), geom, kp)
    kc2, _ = s_source_calc(with_fields(CSRKick1(; ix_ele_source = 2); dz_particles = u), geom, kp)
    kw2, _ = s_source_calc(with_fields(kw; dz_particles = u), geom, kp)
    Kc, Kw = image_charge_kick(kc2, geom, kp), image_charge_kick(kw2, geom, kp)
    if (oc.code, oc.ix) != (ow.code, ow.ix) || abs(Kc - Kw) > 1e-9 * max(abs(Kc), 1e-30)
      nbad += 1
      nbad <= 10 && println("  layer $ns u $(round(u, sigdigits = 6)): cold $(oc.code) ele $(oc.ix) K $Kc | warm $(ow.code) ele $(ow.ix) K $Kw")
    end
  end
  println("layer $ns: $(length(nodes)) nodes, $nbad differ")
end
