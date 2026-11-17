# Timing of csr_kick! (default mode) for the path-excess change: steady-state bend (free space, n_bin 200,
# all-bin quadrature) and the same with 16 image layers (n_bin 120). Minimum over repeats, seconds.
using CSR, Printf
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
electron = Species("electron")
N = 20001
z, q, alive = gaussian_line_bunch(1e-3, 1e-9, N)
geom = CSRGeometry(circle_lattice(10.0, 0.1, 80)...)
cases = [("free space, n_bin 200", CSRParams(n_bin = 200, ds_step = 0.1), 40),
         ("16 layers h = 1 cm, n_bin 120", CSRParams(n_bin = 120, ds_step = 0.1, n_shield_images = 16,
                                                     beam_chamber_height = 0.01), 5)]
for (name, p, n) in cases
  pz = zeros(N)
  csr_kick!(pz, z, q, alive, geom, 80, 0.05, 1e4, electron, p)
  t = minimum(@elapsed(csr_kick!(pz, z, q, alive, geom, 80, 0.05, 1e4, electron, p)) for _ in 1:n)
  @printf("%-32s %.3f ms\n", name, 1e3t)
end
