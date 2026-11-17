# The test_rrule check of ad_rrule_test.jl ("test_rrule: …", csr_kick_pz wrapper only) under many RNG seeds, to
# check its rtol against the actual ChainRulesTestUtils FD (j′vp of ȳ⋅F, not the Jacobian of d32_flake.jl).
# Usage (test env, 1 thread): julia --project=<env> -t 1 test_rrule_seeds.jl [free|compat] [rtol] [n_seeds, 100] [atol, 1e-10] [verbose]
# verbose: print the failure messages (which comparison, values) instead of only the seed numbers.
using Test, CSR, ForwardDiff, StaticArrays, Random, ChainRulesCore, ChainRulesTestUtils, Printf
using ChainRulesTestUtils: FiniteDifferences
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
include(expanduser("~/.julia/dev/CSR.jl/test/kernels_test.jl"))   # kernel_test_bunch
function FiniteDifferences.to_vec(g::CSRGeometry)                # as ad_rrule_test.jl
  n, bc = length(g.ele), g.bmad_compat
  return CSR.geometry_to_vector(g), v -> CSR.geometry_from_vector(v, n, bc)
end
ChainRulesTestUtils.test_approx(a::CSRGeometry, b::CSRGeometry, msg = ""; kw...) =
  ChainRulesTestUtils.test_approx(CSR.geometry_to_vector(a), CSR.geometry_to_vector(b), msg; kw...)

electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
kw = (; kick_factor = 0.5, pz_scale = 1.3)
case = get(ARGS, 1, "free")
n_img, bc, abq = case == "compat" ? (2, true, false) : (0, false, true)
rtol = parse(Float64, get(ARGS, 2, case == "compat" ? "3e-3" : "1e-5"))
n_seeds = parse(Int, get(ARGS, 3, "100"))
atol = parse(Float64, get(ARGS, 4, "1e-10"))
verbose = get(ARGS, 5, "") == "verbose"
fdm = FiniteDifferences.central_fdm(5, 1; max_range = 1e-6)
Γ = 1e5
zs, qs, as = kernel_test_bunch(1e-3, 1e-9, 60)
p = CSRParams(n_bin = 24, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
              bmad_compat = bc, all_bin_quadrature = abq)
geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
h(pz, q, gm, s, w) = csr_kick_pz(pz, zs, q, as, gm, 2, s, Γ * w, electron, p; kw...)
function h_rrule(::RuleConfig, _, pz, q, gm, s, w)
  y, back = ChainRulesCore.rrule(csr_kick_pz, pz, zs, q, as, gm, 2, s, Γ * w, electron, p; kw...)
  h_pullback(ȳ) = (t = back(ȳ); (NoTangent(), t[2], t[4], t[6], t[8], Γ * t[9]))
  return y, h_pullback
end
n_fail = 0
for seed in 1:n_seeds
  Random.seed!(seed)
  failed = try   # an outermost @testset throws on any failure or error
    redirect_stdout(verbose ? stdout : devnull) do
      @testset "seed $seed" begin
        test_rrule(h ⊢ NoTangent(), 1e-4 .* sin.(1:60), qs, geom, 0.17, 200.0 / Γ;
                   rrule_f = h_rrule, fdm, check_inferred = false, rtol, atol)
      end
    end
    false
  catch
    true
  end
  global n_fail += failed
  failed && println("seed $seed failed")
end
@printf("case %s, rtol %g, atol %g: %d of %d seeds failed\n", case, rtol, atol, n_fail, n_seeds)
