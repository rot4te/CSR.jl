# Timing of the geometry/s_kick/gamma pullback (csr_kick_param_adjoint) vs the kick itself: steady-state bend on a
# rho = 10 m arc of 0.1 m elements, kick in the last element, n_bin 200 (free space) and 4 image layers (n_bin 120).
# Minimum over repeats, seconds. Usage: julia --project=<env> bench_pullback.jl [n_ele ...]
using CSR, Printf
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
electron = Species("electron")
N = 20001
z, q, alive = gaussian_line_bunch(1e-3, 1e-9, N)
n_eles = isempty(ARGS) ? [10, 80] : parse.(Int, ARGS)
cases = [("free space, n_bin 200", CSRParams(n_bin = 200, ds_step = 0.1)),
         ("4 layers h = 1 cm, n_bin 120", CSRParams(n_bin = 120, ds_step = 0.1, n_shield_images = 4,
                                                    beam_chamber_height = 0.01))]
for n_ele in n_eles, (name, p) in cases
  geom = CSRGeometry(circle_lattice(10.0, 0.1, n_ele)...)
  st = CSR.csr_binned_kick(z, q, alive, geom, n_ele, 0.05, 1e4, electron, p)
  kbar = cos.(1:p.n_bin)
  f() = CSR.csr_binned_kick(z, q, alive, geom, n_ele, 0.05, 1e4, electron, p)
  g() = CSR.csr_kick_param_adjoint(kbar, st.bins, geom, n_ele, 0.05, 1e4, electron, p)
  f(); g()
  tk = minimum(@elapsed(f()) for _ in 1:5)
  tp = minimum(@elapsed(g()) for _ in 1:2)
  n_act = length(CSR.geometry_active_indices(geom, n_ele, 0.05, 1e4, p, st.bins.dz))
  @printf("%3d elements, %-30s kick %8.2f ms  pullback %9.1f ms  (%.0f×, %d active)\n", n_ele, name, 1e3tk, 1e3tp,
          tp / tk, n_act)
end
