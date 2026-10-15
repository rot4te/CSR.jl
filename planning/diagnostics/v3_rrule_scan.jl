# V3: rrule (pullback, D30) vs ForwardDiff for gamma and s_kick over kick positions, images (2 layers, h = 1 cm),
# lattice of ad_rrule_test.jl. Run with the old and the new code to see whether a pullback error is pre-existing.
# julia --project=<env> planning/diagnostics/v3_rrule_scan.jl
using CSR, StaticArrays, ChainRulesCore, ForwardDiff
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
println("bmad_compat = false: ", length(ARGS) > 0 ? ARGS[1] : "")
for (ik, svals) in ((2, 0.03:0.02:0.29), (3, (0.05, 0.1, 0.15)))
  for n_img in (1, 2), s in svals
    p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = 0.01)
    geom = CSRGeometry(L3, g3, c3)
    f(s, γ) = sum(ybar .* csr_kick_pz(pz, z, q, alive, geom, ik, s, γ, electron, p))
    _, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, ik, s, 200.0, electron, p)
    t = back(ybar)
    fs = ForwardDiff.derivative(x -> f(x, 200.0), s)
    fg = ForwardDiff.derivative(γ -> f(s, γ), 200.0)
    kp = CSRKickPoint(geom, ik, s, 200.0)
    println("ele $ik n_img $n_img s $s  s_chord $(round(kp.s_chord_kick, digits = 6))  rel err s_kick ",
            round(abs(t[8] - fs) / abs(fs), sigdigits = 2), "  gamma ", round(abs(t[9] - fg) / abs(fg), sigdigits = 2))
  end
end
