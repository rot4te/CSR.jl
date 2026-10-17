using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, ForwardDiff, GTPSA
include("/Users/alexcoxe/.julia/dev/CSR.jl/test/utils.jl")
function kernel_test_bunch(sigma, Q, n)
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n)
  alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end
mutate = get(ENV, "MUTATE", "0") == "1"
if mutate
  CSR.root_order(t::GTPSA.TPS) = 1
end
println("mutate = ", mutate, "  root_order(TPS)=", CSR.root_order(@vars(Descriptor(1,3))[1]))
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
n = 400
z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
pz0 = zeros(n)
v0 = [0.3, 1.5, 1.1e-3, 0.17, 200.0, 0.0]
function F(v, p)
  c = similar(c3, eltype(v)); c .= c3; c[3, 1] = v[3]
  geom = CSRGeometry([L3[1], v[1], L3[3]], [g3[1], v[2], g3[3]], c; bmad_compat = p.bmad_compat)
  return csr_kick_pz(pz0, z .* (1 + 0.2v[6]), q .* (1 + v[6]), alive, geom, 2, v[4], v[5], electron, p)
end
println("== Order 2")
for n_img in (0, 2), bc in (false, true)
  p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0, bmad_compat = bc)
  d = Descriptor(6, 2); x = @vars(d)
  t = F(v0 .+ x, p)
  Hf = reshape(ForwardDiff.jacobian(u -> ForwardDiff.jacobian(v -> F(v, p), u), v0), n, 6, 6)
  Ht = permutedims(cat((GTPSA.hessian(ti) for ti in t)...; dims = 3), (3, 1, 2))
  Hmax = maximum(abs, Hf)
  worst = 0.0; minrel = Inf; nfail = 0; nz = 0; wb = (0,0)
  for a in 1:6, b in a:6
    ref = maximum(abs, Hf[:, a, b]); err = maximum(abs, Ht[:, a, b] .- Hf[:, a, b])
    bound = 1e-10 * ref + 1e-14 * Hmax
    r = err / bound
    if r > worst; worst = r; wb = (a,b); end
    ref > 0 && (minrel = min(minrel, ref / Hmax))
    ref == 0 && (nz += 1)
    err <= bound || (nfail += 1)
  end
  println("images=$n_img bc=$bc: worst err/bound=$(round(worst,sigdigits=4)) at $wb, min nonzero ref/Hmax=$(minrel), zero-ref blocks=$nz, failing blocks=$nfail/21")
end
println("== Order 3")
for n_img in (0, 2)
  p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 3e-3 : 0.0)
  F2(u) = F([v0[1], u[1], v0[3], v0[4], u[2], v0[6]], p)
  d = Descriptor(2, 3); x = @vars(d)
  t = F2([v0[2], v0[5]] .+ x)
  u0 = [v0[2], v0[5]]
  T3f = reshape(ForwardDiff.jacobian(w -> ForwardDiff.jacobian(u -> ForwardDiff.jacobian(F2, u), w), u0), n, 2, 2, 2)
  T3t = [scalar(GTPSA.deriv(t[j], [count(==(i), (a, b, c)) for i in 1:2])) for j in 1:n, a in 1:2, b in 1:2, c in 1:2]
  Tmax = maximum(abs, T3f)
  worst = 0.0; minrel = Inf; nfail = 0; wb = (0,0,0); info=String[]
  for a in 1:2, b in a:2, c in b:2
    ref = maximum(abs, T3f[:, a, b, c]); err = maximum(abs, T3t[:, a, b, c] .- T3f[:, a, b, c])
    bound = 1e-9 * ref + 1e-13 * Tmax
    r = err / bound
    if r > worst; worst = r; wb = (a,b,c); end
    ref > 0 && (minrel = min(minrel, ref / Tmax))
    err <= bound || (nfail += 1)
    push!(info, "($a,$b,$c): err/bound=$(round(r,sigdigits=3)) ref/Tmax=$(round(ref/Tmax,sigdigits=3))")
  end
  println("images=$n_img: worst err/bound=$(round(worst,sigdigits=4)) at $wb, min nonzero ref/Tmax=$(minrel), failing blocks=$nfail/4")
  println("   ", join(info, "; "))
end
