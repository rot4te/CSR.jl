# D24: per-bin kick convergence of the image-layer sum with k-fold averaging (same weights as layer_sum_methods.jl).
# Steady-state bend on a 40 m arc (ρ = 10 m, σ = 1 mm, γ = 1e4, n_bin 240). Reference: avg(8) over N layers.
# Error = max over bins with |z| ≤ 3σ of |kick_M − kick_ref| / max |kick_ref|.
using CSR, Printf
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
electron = Species("electron")
zb, qb, aliveb = gaussian_line_bunch(1e-3, 1e-9, 40001)
n_ele, nb, gamma = 400, 240, 1e4
N = parse(Int, get(ARGS, 1, "512"))
geom = CSRGeometry(circle_lattice(10.0, 0.1, n_ele)...)
function avg_weights(M, k)
  w = ones(M)
  for n in M-k+1:M
    w[n] = sum(binomial(k, j) for j in (n - (M - k)):k) / 2^k
  end
  return w
end
for h in (0.003, 0.01)
  st = CSR.csr_binned_kick(zb, qb, aliveb, geom, n_ele, 0.05, gamma, electron, CSRParams(n_bin = nb, ds_step = 0.1))
  K = zeros(nb, N)                       # per-layer kick contributions (no free-space part)
  for ns in 1:N
    kp = CSR.CSRKickPoint(geom, n_ele, 0.05, gamma; y_source = ns * h, kick_factor = (-1)^ns * 2)
    img, _ = CSR.csr_image_table(geom, kp, st.bins.dz, nb, CSR.CPU(); bin_integrate = true)
    CSR.csr_image_convolve!(view(K, :, ns), img, st.bins.charge, st.coef)
  end
  kick(M, k) = st.kick .+ K[:, 1:M] * avg_weights(M, k)
  ref = kick(N, 8)
  core = abs.(st.bins.z_center) .<= 3e-3
  err(x) = maximum(abs.(x[core] .- ref[core])) / maximum(abs.(ref[core]))
  @printf("h = %g, N = %d; reference vs avg(4) at N: %.2e; max|ref|/max|free| = %.3e\n", h, N, err(kick(N, 4)),
          maximum(abs.(ref[core])) / maximum(abs.(st.kick[core])))
  println("     M   plain      avg(1)     avg(2)     avg(4)     avg(8)")
  for M in (16, 32, 64, 128, 256)
    @printf("  %4d  %.2e   %.2e   %.2e   %.2e   %.2e\n", M, err(kick(M, 0)), err(kick(M, 1)), err(kick(M, 2)),
            err(kick(M, 4)), err(kick(M, 8)))
  end
  flush(stdout)
end
