# D32 flake: FD vs exact Jacobian of the test_rrule wrapper ("images, bmad_compat"), then failure rate over random ȳ
# drawn as ChainRulesTestUtils does (rand(-9:0.01:9) per output), with test_approx's semantics.
# Usage (test env, 1 thread): julia --project=<env> -t 1 d32_flake.jl [compat|free] [Γ] [rtol, 1e-2] [tangent|block] [atol, 1e-10]
# Grouping (default tangent): as ChainRulesTestUtils.test_approx compares the cotangents. pz, q: whole vector (norm);
# the CSRGeometry Tangent field by field: SVector fields (r, spline) by norm, scalars alone. `block` compares the whole
# geometry block by norm first (the grouping before 2026-10-16; too lenient: it predicted no free-space failures at
# rtol 1e-7, test_rrule_seeds.jl gives 42 of 100).
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
grouping = get(ARGS, 4, "tangent")
atol = parse(Float64, get(ARGS, 5, "1e-10"))
# test_approx groups of geometry_to_vector: start r, θ; per element L, g, s_start, 4 × (r, θ), theta_chord, L_chord,
# spline, dL_s (element_numbers order).
ele_groups = [1:1, 2:2, 3:3, 4:6, 7:7, 8:10, 11:11, 12:14, 15:15, 16:18, 19:19, 20:20, 21:21, 22:25, 26:26]
geom_groups = [[1:3, 4:4]; [CSR.geometry_base(j) .+ r for j in 1:3 for r in ele_groups]]
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
    if k == :geom && grouping == "tangent"
      bad = reduce(vcat, [collect(gr) for gr in geom_groups if !isapprox(a[gr], e[gr]; rtol, atol)]; init = Int[])
    else
      isapprox(a, e; rtol, atol) && continue
      bad = k == :geom ? findall(i -> !isapprox(a[i], e[i]; rtol, atol), eachindex(a)) : collect(eachindex(a))
    end
    isempty(bad) && continue
    fails[k] += 1; failed = true
    for i in bad; elfail[(k, i)] = get(elfail, (k, i), 0) + 1; end
  end
  global anyfail += failed
end
@printf("case %s, Γ = %g, rtol = %g, atol = %g, grouping %s: P(fail) = %.5f over %d draws\n", case, Γ, rtol, atol, grouping, anyfail / N, N)
for k in keys(blocks); @printf("  %-6s fails %d\n", k, fails[k]); end
for ((k, i), c) in sort(collect(elfail); by = last, rev = true)[1:min(end, 10)]
  col = blocks[k][i]
  @printf("  %s[%d]: %d fails; |Jfd-Jad| col max %.3e, |Jad| col max %.3e, rel col err %.3e\n", k, i, c,
          maximum(abs, Jfd[:, col] - Jad[:, col]), maximum(abs, Jad[:, col]),
          norm(Jfd[:, col] - Jad[:, col]) / max(norm(Jad[:, col]), 1e-300))
end
# atol vs the cotangents it compares: atol/‖a‖ per test_approx group (AD value a) over the same draws; "atol binds"
# where atol > rtol·‖a‖. Groups whose AD columns are all zero are counted apart (atol is their only bound).
groups = [(:pz, blocks.pz), (:q, blocks.q)]
for gr in (grouping == "tangent" ? geom_groups : [1:ng]); push!(groups, (:geom, blocks.geom[gr])); end
push!(groups, (:s, blocks.s), (:gamma, blocks.gamma))
zero_groups = [g for g in groups if all(iszero, Jad[:, g[2]])]
live = [g for g in groups if !all(iszero, Jad[:, g[2]])]
ratios = zeros(N, length(live))
rng = Xoshiro(1)
for t in 1:N
  ȳ = rand(rng, -9:0.01:9, 60)
  for (j, (_, c)) in enumerate(live); ratios[t, j] = atol / norm(Jad[:, c]' * ȳ); end
end
med = [sort(ratios[:, j])[N ÷ 2] for j in axes(ratios, 2)]
jmax = argmax(med)
@printf("atol/‖cotangent‖ over %d groups (%d all-zero groups excluded): median %.2e; group medians %.2e … %.2e (largest: %s cols %s)\n",
        length(live), length(zero_groups), sort(vec(ratios))[length(ratios) ÷ 2], minimum(med), maximum(med),
        live[jmax][1], string(live[jmax][2] .- first(blocks[live[jmax][1]]) .+ 1))
@printf("  atol binds (atol > rtol·‖a‖) in %.2e of comparisons; max atol/‖a‖ %.2e\n", count(>(rtol), ratios) / length(ratios),
        maximum(ratios))
for j in sortperm(med; rev = true)[1:min(end, 6)]
  @printf("  %-5s cols %-8s median atol/‖a‖ %.2e, atol binds in %.3f, max|Jad| %.2e\n", live[j][1],
          string(live[j][2] .- first(blocks[live[j][1]]) .+ 1), med[j], count(>(rtol), ratios[:, j]) / N,
          maximum(abs, Jad[:, live[j][2]]))
end
# Columns with the largest relative FD error
relerr = [norm(Jfd[:, c] - Jad[:, c]) / max(norm(Jad[:, c]), 1e-300) for c in axes(Jad, 2)]
println("largest relative column errors:")
for c in sortperm(relerr; rev = true)[1:8]
  k = first(k for (k, r) in pairs(blocks) if c in r)
  @printf("  %s[%d]: rel %.3e, |Jad| %.3e\n", k, c - first(blocks[k]) + 1, relerr[c], norm(Jad[:, c]))
end
