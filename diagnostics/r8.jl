using CSR, GTPSA, ForwardDiff, AtomicAndPhysicalConstants
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom0 = CSRGeometry([0.2, 0.3, 0.2], [0.0, 1.5, 0.0], c3)
z = collect(range(-6e-3, 6e-3; length = 400)) .+ 0.37 * (12e-3 / 400) .* sin.(1:400)
w = exp.(-z.^2 ./ 2e-6) .* (1 .+ 0.2 .* cos.(3 .* (1:400))); q = 1e-9 .* w ./ sum(w)
alive = fill(true, 400); sp = Species("electron"); params = CSRParams(n_bin = 40, ds_step = 0.1)
function t(name, f)
  print(rpad(name,52))
  try; r = f(); println("OK   ", typeof(r)); catch e; println("FAIL ", first(split(sprint(showerror,e),'\n'))); end
end
dd = Descriptor(2,2)
for (lab, x) in (("static", @vars(dd)), ("dynamic", @vars(dd, dynamic=true)))
  println("== ", lab)
  t("csr_kick_pz  gamma TPS",  () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17, 200.0 + x[1], sp, params))
  t("csr_kick_pz  s_kick TPS", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, sp, params))
  t("csr_kick_pz  kick_factor TPS", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17, 200.0, sp, params; kick_factor = 1.0 + x[1]))
  t("csr_kick!    Float64 pz, gamma TPS",  () -> csr_kick!(zeros(400), z, q, alive, geom0, 2, 0.17, 200.0 + x[1], sp, params))
  t("csr_kick!    Float64 pz, s_kick TPS", () -> csr_kick!(zeros(400), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, sp, params))
  t("csr_kick!    Float64 pz, q TPS", () -> csr_kick!(zeros(400), z, q .* (1 + x[2]), alive, geom0, 2, 0.17, 200.0, sp, params))
  t("csr_kick!    TPS pz, s_kick TPS", () -> csr_kick!([zero(x[1]) for _ in 1:400], z, q, alive, geom0, 2, 0.17 + x[1], 200.0, sp, params))
end
println("== ForwardDiff")
D(v) = ForwardDiff.Dual{:t}(v, 1.0)
t("csr_kick_pz  gamma Dual", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, 0.17, D(200.0), sp, params))
t("csr_kick_pz  s_kick Dual", () -> csr_kick_pz(zeros(400), z, q, alive, geom0, 2, D(0.17), 200.0, sp, params))
t("csr_kick!    Float64 pz, gamma Dual", () -> csr_kick!(zeros(400), z, q, alive, geom0, 2, 0.17, D(200.0), sp, params))
t("csr_kick!    Float64 pz, s_kick Dual", () -> csr_kick!(zeros(400), z, q, alive, geom0, 2, D(0.17), 200.0, sp, params))
