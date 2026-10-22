using CSR, Metal, StaticArrays, AtomicAndPhysicalConstants
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
n = 2000
z = collect(range(-6e-3, 6e-3; length = n)) .+ 1e-5 .* sin.(1:n); q = (1 .+ 0.2 .* cos.(1:n)) .* 1e-9/n; alive = fill(true, n)
zg, qg, ag = MtlArray(Float32.(z)), MtlArray(Float32.(q)), MtlArray(alive)
params = CSRParams(n_bin = 40, ds_step = 0.1)
geom = CSRGeometry(L3, g3, c3)
outs = map(1:4) do _
  p = MtlArray(zeros(Float32, n)); csr_kick!(p, zg, qg, ag, geom, 2, 0.17, 200.0, electron, params); Array(p)
end
println("csr_kick! repeated on Metal identical: ", all(==(outs[1]), outs), "  max rel spread: ", maximum(abs, outs[1] .- outs[2]) / maximum(abs, outs[1]))
o2 = Array(csr_kick_pz(MtlArray(zeros(Float32, n)), zg, qg, ag, geom, 2, 0.17, 200.0, electron, params))
println("csr_kick_pz vs csr_kick! max rel: ", maximum(abs, o2 .- outs[1]) / maximum(abs, outs[1]))
# default kick_factor / pz_scale, Float64 geometry: no Float64 reaches the device
println(typeof(csr_kick_pz(MtlArray(zeros(Float32, n)), zg, qg, ag, geom, 2, 0.17, 200.0, electron, params)))
