# Per-layer image contributions a_n to the energy loss (steady-state bend ρ = 10 m, σ = 1 mm, γ = 1e4),
# hat-integrated tables. Writes layer_terms_<h>_<n_bin>.txt (n, a_n / free-space loss).
using CSR, DelimitedFiles
include("/Users/alexcoxe/.julia/dev/CSR.jl/test/utils.jl")
electron = Species("electron")
N = 40001
zb, qb, aliveb = gaussian_line_bunch(1e-3, 1e-9, N)
geom = CSRGeometry(circle_lattice(10.0, 0.1, 80)...)
gamma = 1e4
nl = parse(Int, get(ARGS, 1, "256"))
for nb in (240,), h in (0.003, 0.01)
  st = CSR.csr_binned_kick(zb, qb, aliveb, geom, 80, 0.05, gamma, electron, CSRParams(n_bin = nb, ds_step = 0.1))
  e0 = sum(st.kick .* st.bins.charge)
  a = zeros(nl)
  t = @elapsed for ns in 1:nl
    kp = CSR.CSRKickPoint(geom, 80, 0.05, gamma; y_source = ns * h, kick_factor = (-1)^ns * 2)
    img, _ = CSR.csr_image_table(geom, kp, st.bins.dz, nb, CSR.CPU(); bin_integrate = true)
    kick = zeros(nb)
    CSR.csr_image_convolve!(kick, img, st.bins.charge, st.coef)
    a[ns] = sum(kick .* st.bins.charge) / e0
  end
  writedlm(joinpath(@__DIR__, "layer_terms_$(h)_$(nb).txt"), [1:nl a])
  println("h=$h n_bin=$nb  $(round(t, digits = 1)) s  partial sum = $(1 + sum(a))"); flush(stdout)
end
