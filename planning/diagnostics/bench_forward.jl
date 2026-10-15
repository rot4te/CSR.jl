# Timing of forward-mode derivatives through the kick tables (csr_bins_kick with a Dual geometry, s_kick and gamma)
# vs the Float64 kick: steady-state bend on a rho = 10 m arc of 0.1 m elements, kick in the last element, n_bin 200
# (free space) and 4 image layers (n_bin 120). One pass with 1 and with 12 partials (random directions over all
# geometry numbers, s_kick, gamma). Minimum over repeats, seconds. Usage: julia --project=<env> bench_forward.jl [n_ele ...]
using CSR, ForwardDiff, Printf, Random
using ForwardDiff: Dual
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
  bins = CSR.csr_binned_kick(z, q, alive, geom, n_ele, 0.05, 1e4, electron, p).bins
  x0 = [CSR.geometry_to_vector(geom); 0.05; 1e4]
  n_g = length(x0) - 2
  kick(x) = CSR.csr_bins_kick(bins, CSR.CPU(), CSR.geometry_from_vector(view(x, 1:n_g), n_ele, false), n_ele,
                              x[n_g+1], x[n_g+2], electron, p).kick
  rng = Xoshiro(1)
  xd(np) = [Dual(x, ntuple(_ -> 1e-6 * randn(rng), np)...) for x in x0]
  x1, x12 = xd(1), xd(12)
  kick(x0); kick(x1); kick(x12)
  t0 = minimum(@elapsed(kick(x0)) for _ in 1:5)
  t1 = minimum(@elapsed(kick(x1)) for _ in 1:3)
  t12 = minimum(@elapsed(kick(x12)) for _ in 1:3)
  @printf("%3d elements, %-30s kick %8.2f ms  1 partial %8.1f ms (%.1f×)  12 partials %8.1f ms (%.1f×)\n", n_ele, name,
          1e3t0, 1e3t1, t1 / t0, 1e3t12, t12 / t0)
end
