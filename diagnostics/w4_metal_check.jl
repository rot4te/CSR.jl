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
# Max relative difference: max |a - b| / max |b| (b must be nonzero, else the comparison is vacuous)
function maxrel(a, b)
  m = maximum(abs, b)
  m > 0 || error("reference kick is identically zero: vacuous comparison")
  return maximum(abs, a .- b) / m
end
check(label, val, tol) = println("  ", label, ": ", val, tol === nothing ? "" : (val < tol ? "  PASS (< $tol)" : "  FAIL (>= $tol)"))
geom = CSRGeometry(L3, g3, c3)
# Kick points with a bend: in the bend and in the last drift (Metal's atomic deposit order varies, so no exact equality)
for (name, ix, sk) in (("drift-bend-drift, kick in bend", 2, 0.17), ("drift-bend-drift, kick in last drift", 3, 0.1))
  pz_cpu = zeros(Float32, n)
  csr_kick!(pz_cpu, z32, q32, alive, geom, ix, sk, 200.0, electron, params)
  println(name, ": max |CPU Float32 kick| = ", maximum(abs, pz_cpu))
  zg, qg, ag, pzg = MtlArray(z32), MtlArray(q32), MtlArray(alive), MtlArray(zeros(Float32, n))
  out = csr_kick_pz(pzg, zg, qg, ag, geom, ix, sk, 200.0, electron, params)
  println("  csr_kick_pz type ", typeof(out), " pz unchanged: ", all(Array(pzg) .== 0))
  pzc = MtlArray(zeros(Float32, n)); csr_kick!(pzc, zg, qg, ag, geom, ix, sk, 200.0, electron, params)
  check("csr_kick_pz vs csr_kick! on copies, max rel", maxrel(Array(out), Array(pzc)), 2e-6)
  check("csr_kick_pz vs CPU Float32, max rel", maxrel(Array(out), pz_cpu), 1e-5)
  # rrule
  res = rrule(csr_kick_pz, pzg, zg, qg, ag, geom, ix, sk, 200.0, electron, params)
  println("  rrule forward type ", typeof(res[1]))
  check("rrule forward vs csr_kick_pz, max rel", maxrel(Array(res[1]), Array(out)), 2e-6)
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
