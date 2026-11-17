# D24: shape of the per-layer image terms a_n (steady-state bend ρ = 10 m, σ = 1 mm, γ = 1e4, hat tables).
# For selected layers: a_n / free-space loss on an 8 m and a 40 m arc (kick 5 cm before the end), the source s at
# lag 0 and at the extreme lags (s_kick − s_source), and the image table at lag 0 vs its maximum.
using CSR, Printf
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
electron = Species("electron")
zb, qb, aliveb = gaussian_line_bunch(1e-3, 1e-9, 40001)
gamma = 1e4
nb = parse(Int, get(ARGS, 1, "240"))
layers = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512)
for h in (0.003, 0.01), n_ele in (80, 400)
  geom = CSRGeometry(circle_lattice(10.0, 0.1, n_ele)...)
  s_kick = 0.1 * (n_ele - 1) + 0.05
  st = CSR.csr_binned_kick(zb, qb, aliveb, geom, n_ele, 0.05, gamma, electron, CSRParams(n_bin = nb, ds_step = 0.1))
  e0 = sum(st.kick .* st.bins.charge)
  println("h = $h, arc = $(0.1n_ele) m, n_bin = $nb")
  println("    n   a_n/E0         Ls(l=0)  Ls(l=-nb)  Ls(l=+nb)  K(0)/max|K|  sym/total")
  for ns in layers
    kp = CSR.CSRKickPoint(geom, n_ele, 0.05, gamma; y_source = ns * h, kick_factor = (-1)^ns * 2)
    img, s_src = CSR.csr_image_table(geom, kp, st.bins.dz, nb, CSR.CPU(); bin_integrate = true)
    kick = zeros(nb)
    CSR.csr_image_convolve!(kick, img, st.bins.charge, st.coef)
    a = sum(kick .* st.bins.charge) / e0
    i0 = nb + 1                                     # lag 0 (lags −nb:nb)
    sym = (img .+ reverse(img)) ./ 2
    ks = zeros(nb); CSR.csr_image_convolve!(ks, sym, st.bins.charge, st.coef)
    @printf("  %3d  %+.4e  %7.3f  %8.3f  %8.3f  %+.3e  %.4f\n", ns, a, s_kick - s_src[i0], s_kick - s_src[1],
            s_kick - s_src[end], img[i0] / maximum(abs, img), sum(ks .* st.bins.charge) / e0 / a)
    flush(stdout)
  end
end
