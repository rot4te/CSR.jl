using CSR, Metal, StaticArrays, AtomicAndPhysicalConstants
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
n = 2000
z = collect(range(-6e-3, 6e-3; length = n)); q = fill(1e-9/n, n); alive = fill(true, n)
zg, qg, ag = MtlArray(Float32.(z)), MtlArray(Float32.(q)), MtlArray(alive)
params = CSRParams(n_bin = 40, ds_step = 0.1)
geom = CSRGeometry(L3, g3, c3)
st = CSR.csr_binned_kick(zg, qg, ag, geom, 2, 0.17, 200.0, electron, params)
try
  CSR.csr_kick_adjoint(MtlArray(ones(Float32, n)), zg, qg, ag, st, params; pz_scale = 1f0)
catch e
  Base.showerror(stdout, e, catch_backtrace())
end
