using CSR, GTPSA, AtomicAndPhysicalConstants
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom0 = CSRGeometry([0.2, 0.3, 0.2], [0.0, 1.5, 0.0], c3)
z = collect(range(-6e-3, 6e-3; length = 400)) .+ 0.37 * (12e-3 / 400) .* sin.(1:400)
w = exp.(-z.^2 ./ 2e-6) .* (1 .+ 0.2 .* cos.(3 .* (1:400))); q = 1e-9 .* w ./ sum(w)
alive = fill(true, 400); sp = Species("electron"); params = CSRParams(n_bin = 40, ds_step = 0.1)
d = Descriptor(2,2); x = @vars(d)
try
  csr_kick!(zeros(400), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, sp, params)
catch e
  Base.showerror(stdout, e, catch_backtrace()); println()
end
println("---- gamma")
try
  csr_kick!(zeros(400), z, q, alive, geom0, 2, 0.17, 200.0 + x[1], sp, params)
catch e
  println(first(split(sprint(showerror, e), '\n')))
end
println("---- TPS pz")
pz = [zero(x[1]) for _ in 1:400]
csr_kick!(pz, z, q, alive, geom0, 2, 0.17 + x[1], 200.0, sp, params); println("OK")
