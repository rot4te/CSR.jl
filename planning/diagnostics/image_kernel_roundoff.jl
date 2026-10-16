# Round-off of image_charge_kick's formula: Bmad's `1 − N·β′` and `N − β′` vs the rewrite
# `1 − N·β′ = 1/(γ²(1+β)) + β|N − t̂|²/2`, `N − β′ = (N − t̂) + (1 − β)t̂`, `B·(N − β′) = (B − t̂)·d − |d|²/2 + (1 − β)B·t̂`
# (t̂ the source's unit direction, d = N − t̂; D43).
# Inputs are the Float64 sources at the layer-1 hat nodes of the ad_rrule_test.jl lattice (s_kick as argument); each
# formula is evaluated in Float64 and in BigFloat on the same inputs. Also |Δr|/L − 1 (the rewrite assumes |N| = 1).
# julia --project=<env> planning/diagnostics/image_kernel_roundoff.jl [s_kick, 0.23]
using CSR, StaticArrays, Printf
using CSR: CSRKickPoint, CSRKick1, image_kernel_at, hat_nodes, spline_d1, spline_d2
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
setprecision(256)

cross3(a, b) = SVector(a[2]b[3] - a[3]b[2], a[3]b[1] - a[1]b[3], a[1]b[2] - a[2]b[1])
dot3(a, b) = a[1]b[1] + a[2]b[2] + a[3]b[3]

# Bmad's form (image_charge_kick before the rewrite)
function kernel_old(rs, ths, g, L, rk, thk, beta, gamma2)
  Bp = beta * SVector(sin(ths), 0, cos(ths))
  G = beta^2 * g * SVector(-cos(ths), 0, sin(ths))
  N = (rk - rs) / L
  B = SVector(sin(thk), 0, cos(thk))
  OneNBp3 = (1 - dot3(N, Bp))^3
  NBp = N - Bp
  return dot3(B, cross3(N, cross3(NBp, G))) / (L * OneNBp3) + dot3(B, NBp) / (gamma2 * L^2 * OneNBp3)
end

function kernel_new(rs, ths, g, L, rk, thk, beta, gamma2)
  t = SVector(sin(ths), 0, cos(ths))
  G = beta^2 * g * SVector(-cos(ths), 0, sin(ths))
  N = (rk - rs) / L
  B = SVector(sin(thk), 0, cos(thk))
  d = N - t
  dd = dot3(d, d)
  omb = 1 / (gamma2 * (1 + beta))          # 1 − β
  OneNBp3 = (omb + beta * dd / 2)^3
  NBp = d + omb * t
  B_NBp = dot3(B - t, d) - dd / 2 + omb * dot3(B, t)
  return dot3(B, cross3(N, cross3(NBp, G))) / (L * OneNBp3) + B_NBp / (gamma2 * L^2 * OneNBp3)
end

L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom = CSRGeometry(L3, g3, c3)
s = isempty(ARGS) ? 0.23 : parse(Float64, ARGS[1])
kp = CSRKickPoint(geom, 2, s, 200.0; y_source = 0.01)
nodes = sort!(first.(hat_nodes(geom, kp, 3e-4, 40, ones(81))))

worst_old = worst_new = worst_norm = 0.0
k = CSRKick1(geom, kp)
for u in nodes
  global k, _ = image_kernel_at(k, u, geom, kp)
  ix = k.ix_ele_source
  g = 1 <= ix <= length(geom.ele) ?
      (sp = geom.ele[ix].spline; -spline_d2(sp, k.s_chord_source) / sqrt(1 + spline_d1(sp, k.s_chord_source)^2)^3) : 0.0
  args = (k.floor_s.r, k.floor_s.theta, g, k.L, kp.floor_k.r, kp.floor_k.theta, kp.beta, kp.gamma2)
  ref = kernel_old(map(a -> big.(a), args)...)        # same inputs in BigFloat (old and new agree there)
  ref2 = kernel_new(map(a -> big.(a), args)...)
  eo = Float64(abs(kernel_old(args...) - ref) / abs(ref))
  en = Float64(abs(kernel_new(args...) - ref2) / abs(ref2))
  global worst_old = max(worst_old, eo)
  global worst_new = max(worst_new, en)
  global worst_norm = max(worst_norm, abs(sqrt(dot3(kp.floor_k.r - k.floor_s.r, kp.floor_k.r - k.floor_s.r)) / k.L - 1))
end
@printf("s_kick %.3f, %d nodes: max rel round-off  old %.1e  new %.1e;  max | |Δr|/L − 1 | %.1e\n",
        s, length(nodes), worst_old, worst_new, worst_norm)
