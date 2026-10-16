# D32 flake: FD vs exact Jacobian of the test_rrule wrapper ("images, bmad_compat"), then failure rate over random ȳ
# drawn as ChainRulesTestUtils does (rand(-9:0.01:9) per output), with test_approx's semantics.
# Usage (test env, 1 thread): julia --project=<env> -t 1 d32_flake.jl [compat|free] [Γ] [rtol, 1e-2]
# γ enters as Γ·x (Γ = 1 old test, 1e5 the fix in 2748fb9); s_kick included. Results: PLAN.md Section 4 (D32).
using Test, CSR, ForwardDiff, StaticArrays, Random, LinearAlgebra, Printf
using ChainRulesTestUtils: FiniteDifferences
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
include(expanduser("~/.julia/dev/CSR.jl/test/kernels_test.jl"))   # kernel_test_bunch

electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
kw = (; kick_factor = 0.5, pz_scale = 1.3)
case = get(ARGS, 1, "compat")
n_img, bc, abq = case == "compat" ? (2, true, false) : (0, false, true)
zs, qs, as = kernel_test_bunch(1e-3, 1e-9, 60)
p = CSRParams(n_bin = 24, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
              bmad_compat = bc, all_bin_quadrature = abq)
geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
gv = CSR.geometry_to_vector(geom)
pz0 = 1e-4 .* sin.(1:60)
ng = length(gv)
Γ = parse(Float64, get(ARGS, 2, "1"))
rtol = parse(Float64, get(ARGS, 3, "1e-2"))
blocks = (pz = 1:60, q = 61:120, geom = 121:120+ng, s = 121+ng:121+ng, gamma = 122+ng:122+ng)
x0 = [pz0; qs; gv; 0.17; 200.0 / Γ]
F(x) = csr_kick_pz(x[blocks.pz], zs, x[blocks.q], as, CSR.geometry_from_vector(x[blocks.geom], 3, bc), 2, x[blocks.s[1]],
                   Γ * x[blocks.gamma[1]], electron, p; kw...)
fdm = FiniteDifferences.central_fdm(5, 1; max_range = 1e-6)
Jfd = first(FiniteDifferences.jacobian(fdm, F, x0))
Jad = ForwardDiff.jacobian(F, x0)

rng = Xoshiro(1)
N = 20000
fails = Dict(k => 0 for k in keys(blocks)); anyfail = 0
worst = Dict(k => (0.0, 0) for k in keys(blocks))
elfail = Dict{Tuple{Symbol,Int},Int}()
for t in 1:N
  ȳ = rand(rng, -9:0.01:9, 60)
  failed = false
  for (k, r) in pairs(blocks)
    a, e = Jad[:, r]' * ȳ, Jfd[:, r]' * ȳ
    a, e = vec(a), vec(e)
    isapprox(a, e; rtol, atol = 1e-10) && continue
    bad = findall(i -> !isapprox(a[i], e[i]; rtol, atol = 1e-10), eachindex(a))
    isempty(bad) && continue
    fails[k] += 1; failed = true
    for i in bad; elfail[(k, i)] = get(elfail, (k, i), 0) + 1; end
  end
  global anyfail += failed
end
@printf("case %s, Γ = %g, rtol = %g: P(fail) = %.4f over %d draws\n", case, Γ, rtol, anyfail / N, N)
for k in keys(blocks); @printf("  %-6s fails %d\n", k, fails[k]); end
for ((k, i), c) in sort(collect(elfail); by = last, rev = true)[1:min(end, 10)]
  col = blocks[k][i]
  @printf("  %s[%d]: %d fails; |Jfd-Jad| col max %.3e, |Jad| col max %.3e, rel col err %.3e\n", k, i, c,
          maximum(abs, Jfd[:, col] - Jad[:, col]), maximum(abs, Jad[:, col]),
          norm(Jfd[:, col] - Jad[:, col]) / max(norm(Jad[:, col]), 1e-300))
end
# Columns with the largest relative FD error
relerr = [norm(Jfd[:, c] - Jad[:, c]) / max(norm(Jad[:, c]), 1e-300) for c in axes(Jad, 2)]
println("largest relative column errors:")
for c in sortperm(relerr; rev = true)[1:8]
  k = first(k for (k, r) in pairs(blocks) if c in r)
  @printf("  %s[%d]: rel %.3e, |Jad| %.3e\n", k, c - first(blocks[k]) + 1, relerr[c], norm(Jad[:, c]))
end
