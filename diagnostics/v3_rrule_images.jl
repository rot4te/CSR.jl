# V3 (exact s_ref_to_s_chord): the "images" case of ad_rrule_test.jl failed rrule == ForwardDiff after the change.
# Compares the rrule (pullback, D30), ForwardDiff and central differences for s_kick and gamma.
# Run from test/: julia --project=<env> ../planning/diagnostics/v3_rrule_images.jl
using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, ForwardDiff, GTPSA
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
include(expanduser("~/.julia/dev/CSR.jl/test/kernels_test.jl"))   # kernel_test_bunch

electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
n = 400
z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
pz = 1e-4 .* sin.(1:n)
kw = (kick_factor = 0.5, pz_scale = 1.3)
ybar = cos.(1:n)
for (name, n_img, abq) in (("free", 0, true), ("images", 2, true), ("images, n_img 1", 1, true))
  p = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                all_bin_quadrature = abq)
  geom = CSRGeometry(L3, g3, c3)
  f(s, γ) = sum(ybar .* csr_kick_pz(pz, z, q, alive, geom, 2, s, γ, electron, p; kw...))
  _, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, 2, 0.17, 200.0, electron, p; kw...)
  t = back(ybar)
  fs = ForwardDiff.derivative(s -> f(s, 200.0), 0.17)
  fg = ForwardDiff.derivative(γ -> f(0.17, γ), 200.0)
  println(name)
  println("  s_kick: rrule $(t[8])  FD-AD $fs  rel $(abs(t[8] - fs) / abs(fs))")
  for h in (1e-3, 1e-4, 1e-5)
    println("    central h=$h: ", (f(0.17 + h, 200.0) - f(0.17 - h, 200.0)) / 2h)
  end
  println("  gamma:  rrule $(t[9])  FD-AD $fg  rel $(abs(t[9] - fg) / abs(fg))")
  for h in (1.0, 0.1, 0.01)
    println("    central h=$h: ", (f(0.17, 200.0 + h) - f(0.17, 200.0 - h)) / 2h)
  end
end
