using CSR, Metal, StaticArrays, AtomicAndPhysicalConstants, ChainRulesCore
println("Metal functional: ", Metal.functional())
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
n = 2000
sigma, Q = 1e-3, 1e-9
z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
w = exp.(-z .^ 2 ./ (2sigma^2)) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
q = Q .* w ./ sum(w)
alive = fill(true, n); alive[3:97:end] .= false
z32, q32 = Float32.(z), Float32.(q)
params = CSRParams(n_bin = 40, ds_step = 0.1)
for (name, geom) in (("drift-bend-drift", CSRGeometry(L3, g3, c3)), ("free space", CSRGeometry([1.0], [0.0], zeros(2, 6))))
  ix = name == "free space" ? 1 : 2
  sk = name == "free space" ? 0.5 : 0.17
  pz_cpu = zeros(Float32, n)
  csr_kick!(pz_cpu, z32, q32, alive, geom, ix, sk, 200.0, electron, params)
  zg, qg, ag, pzg = MtlArray(z32), MtlArray(q32), MtlArray(alive), MtlArray(zeros(Float32, n))
  out = csr_kick_pz(pzg, zg, qg, ag, geom, ix, sk, 200.0, electron, params)
  println(name, ": csr_kick_pz type ", typeof(out), " pz unchanged: ", all(Array(pzg) .== 0))
  pzc = MtlArray(zeros(Float32, n)); csr_kick!(pzc, zg, qg, ag, geom, ix, sk, 200.0, electron, params)
  println("  == csr_kick! on copies (exact): ", Array(out) == Array(pzc))
  a, b = Array(out), pz_cpu
  println("  vs CPU Float32 max rel: ", maximum(abs, a .- b) / maximum(abs, b))
  # rrule
  res = rrule(csr_kick_pz, pzg, zg, qg, ag, geom, ix, sk, 200.0, electron, params)
  println("  rrule forward == csr_kick_pz: ", Array(res[1]) == Array(out), " type ", typeof(res[1]))
  ybar = MtlArray(ones(Float32, n))
  try
    pb = res[2](ybar)
    println("  pullback ran; types: ", typeof.(pb[2:4]), " geom/s_kick/gamma: ", typeof(pb[6]), " ", typeof(pb[8]), " ", typeof(pb[9]))
  catch e
    println("  pullback error: ", sprint(showerror, e)[1:min(end, 600)])
  end
  # particle-only adjoint
  try
    st = CSR.csr_binned_kick(zg, qg, ag, geom, ix, sk, 200.0, electron, params)
    r = CSR.csr_kick_adjoint(ybar, zg, qg, ag, st, params; pz_scale = 1f0)
    println("  csr_kick_adjoint (particle cotangents) ran: ", typeof.(r))
  catch e
    println("  csr_kick_adjoint error: ", sprint(showerror, e)[1:min(end, 600)])
  end
end
