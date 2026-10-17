# FD error of the γ and s_kick columns vs FD max_range (5-point central, adaptive), relative to ForwardDiff
# Usage (test env): julia --project=<env> -t 1 d32_steps.jl. Results: PLAN.md Section 4 (D32).
using Test, CSR, ForwardDiff, StaticArrays, LinearAlgebra, Printf
using ChainRulesTestUtils: FiniteDifferences
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
include(expanduser("~/.julia/dev/CSR.jl/test/kernels_test.jl"))
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
kw = (; kick_factor = 0.5, pz_scale = 1.3)
zs, qs, as = kernel_test_bunch(1e-3, 1e-9, 60)
pz0 = 1e-4 .* sin.(1:60)
for (case, n_img, bc, abq) in (("free", 0, false, true), ("compat", 2, true, false))
  p = CSRParams(n_bin = 24, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                bmad_compat = bc, all_bin_quadrature = abq)
  geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
  fγ(γ) = csr_kick_pz(pz0, zs, qs, as, geom, 2, 0.17, γ, electron, p; kw...)
  fs(s) = csr_kick_pz(pz0, zs, qs, as, geom, 2, s, 200.0, electron, p; kw...)
  for (name, f, x0) in (("gamma", fγ, 200.0), ("s_kick", fs, 0.17))
    d = ForwardDiff.derivative(f, x0)
    for r in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
      fdm = FiniteDifferences.central_fdm(5, 1; max_range = r)
      e = norm(fdm(f, x0) - d) / norm(d)
      @printf("%-6s %-6s max_range %.0e: rel err %.2e\n", case, name, r, e)
    end
  end
end
