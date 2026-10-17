# Free-space test_rrule FD flake (after D43): FD error of each geometry column vs FD max_range (5-point central,
# adaptive), relative to ForwardDiff, for the test_rrule wrapper of ad_rrule_test.jl ("free space"). Columns are
# indices of geometry_to_vector (element j's offset o at 4 + 26(j - 1) + o).
# Usage (test env): julia --project=<env> -t 1 flake_geom_steps.jl [compat]. Results: PLAN.md Section 4.
using Test, CSR, ForwardDiff, StaticArrays, LinearAlgebra, Printf
using ChainRulesTestUtils: FiniteDifferences
include(expanduser("~/.julia/dev/CSR.jl/test/utils.jl"))
include(expanduser("~/.julia/dev/CSR.jl/test/kernels_test.jl"))
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
kw = (; kick_factor = 0.5, pz_scale = 1.3)
n_img, bc, abq = get(ARGS, 1, "free") == "compat" ? (2, true, false) : (0, false, true)
zs, qs, as = kernel_test_bunch(1e-3, 1e-9, 60)
pz0 = 1e-4 .* sin.(1:60)
p = CSRParams(n_bin = 24, ds_step = 0.1, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
              bmad_compat = bc, all_bin_quadrature = abq)
gv = CSR.geometry_to_vector(CSRGeometry(L3, g3, c3; bmad_compat = bc))
F(v) = csr_kick_pz(pz0, zs, qs, as, CSR.geometry_from_vector(v, 3, bc), 2, 0.17, 200.0, electron, p; kw...)
J = ForwardDiff.jacobian(F, gv)
active = findall(c -> norm(J[:, c]) > 0, axes(J, 2))
name(c) = c <= 4 ? "start[$c]" : "ele$(cld(c - 4, 26)) o$(mod1(c - 4, 26))"
steps = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
err = zeros(length(active), length(steps))
for (k, r) in enumerate(steps)
  fdm = FiniteDifferences.central_fdm(5, 1; max_range = r)
  for (i, c) in enumerate(active)
    f(x) = F(setindex!(copy(gv), x, c))
    err[i, k] = norm(fdm(f, gv[c]) - J[:, c]) / norm(J[:, c])
  end
end
@printf("%-12s %10s  %s\n", "column", "value", join((@sprintf("%9.0e", r) for r in steps), " "))
for i in sortperm(err[:, 2]; rev = true)
  @printf("%-12s %+10.3e  %s\n", name(active[i]), gv[active[i]], join((@sprintf("%9.1e", e) for e in err[i, :]), " "))
end
