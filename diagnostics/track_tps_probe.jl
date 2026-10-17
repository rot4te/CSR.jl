# Phase 7 step 5 probe: track_csr! with a TPS / Dual test bunch. TPS Jacobian and Hessian of the final coordinates
# vs ForwardDiff of the same tracking with a Float64 test bunch; Float64 test particles vs bunch particles.
# Run: julia --project=<test env> track_tps_probe.jl [sagan]
using CSR, BeamTracking, Beamlines, GTPSA, ForwardDiff, LinearAlgebra
using AtomicAndPhysicalConstants: massof
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

e = Species("electron")
gamma = 200.0
pq = BeamTracking.E_to_R(e, gamma * massof(e))
line = [Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5, Kn0 = 1.5), Quadrupole(L = 0.2, Kn1 = 3.0),
        SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
bl = Beamline(line; p_over_q_ref = pq, species_ref = e)
params = CSRParams(n_bin = 40, ds_step = 0.1)

function make_bunch(n = 400)
  sigma = 1e-3
  zz = collect(range(-6sigma, 6sigma; length = n))
  w = gaussian.(zz, sigma)
  w .*= (1e-9 / E_CHARGE) / sum(w)
  v = zeros(n, 6)
  v[:, 1] .= 1e-4
  v[:, 5] .= zz
  return Bunch(v, nothing, w; species = e, p_over_q_ref = pq)
end

v_ref = [2e-4, -1e-4, 0.0, 0.0, 3e-4, 1e-4]

# Final coordinates of one Float64/Dual test particle starting at v_ref + v
function f(v)
  tb = Bunch(reshape(v_ref .+ v, 1, 6); species = e, p_over_q_ref = pq)
  track_csr!(make_bunch(), bl; params, test_bunch = tb)
  return tb.coords.v[1, :]
end

d = Descriptor(6, 2)
x = @vars(d)
tb = Bunch(reshape(v_ref .+ x, 1, 6); species = e, p_over_q_ref = pq)
track_csr!(make_bunch(), bl; params, test_bunch = tb)
vt = tb.coords.v[1, :]
y0 = f(zeros(6))
println("scalar parts vs Float64: ", maximum(abs.(scalar.(vt) .- y0)))
Jt, Jf = GTPSA.jacobian(vt), ForwardDiff.jacobian(f, zeros(6))
println("Jacobian max abs diff: ", maximum(abs.(Jt .- Jf)), "  max |J|: ", maximum(abs.(Jf)))
for c in 1:6
  println("  column $c relative: ", norm(Jt[:, c] - Jf[:, c]) / norm(Jf[:, c]))
end
for j in (1, 5, 6)
  Ht, Hf = GTPSA.hessian(vt[j]), ForwardDiff.hessian(v -> f(v)[j], zeros(6))
  println("Hessian coord $j: max abs diff ", maximum(abs.(Ht .- Hf)), "  max |H| ", maximum(abs.(Hf)))
end

# CSR effect on the map (sanity: differs from tracking without CSR)
tb0 = Bunch(reshape(v_ref .+ x, 1, 6); species = e, p_over_q_ref = pq)
track!(tb0, bl)
println("map change from CSR, jacobian: ", maximum(abs.(GTPSA.jacobian(tb0.coords.v[1, :]) .- Jt)))

# Float64 test particles at bunch particles' coordinates
b = make_bunch()
idx = 1:37:400
tb64 = Bunch(b.coords.v[idx, :], nothing, nothing; species = e, p_over_q_ref = pq)
track_csr!(b, bl; params, test_bunch = tb64)
println("Float64 test == bunch: ", tb64.coords.v == b.coords.v[idx, :], " max diff ",
        maximum(abs.(tb64.coords.v .- b.coords.v[idx, :])))

# Is the non-bitwise difference BeamTracking's (subset vs whole bunch, no CSR)?
b = make_bunch()
track!(b, bl)   # move off the initial grid so all coordinates are nonzero
bs = Bunch(b.coords.v[idx, :], nothing, nothing; species = e, p_over_q_ref = pq)
track!(b, bl)
track!(bs, bl)
println("track! subset == whole bunch: ", bs.coords.v == b.coords.v[idx, :], " max diff ",
        maximum(abs.(bs.coords.v .- b.coords.v[idx, :])))
for k in (1, 2, 3, 7, 8, 16)
  b = make_bunch(); track!(b, bl)
  ix = 1:k
  bs = Bunch(b.coords.v[ix, :], nothing, nothing; species = e, p_over_q_ref = pq)
  track!(b, bl); track!(bs, bl)
  println("  first $k particles: max diff ", maximum(abs.(bs.coords.v .- b.coords.v[ix, :])))
end

# Where do Float64 test particles and bunch particles first differ? (after each kick, per coordinate)
let b = make_bunch(), tb = Bunch(b.coords.v[idx, :], nothing, nothing; species = e, p_over_q_ref = pq)
  first_diff = Ref(true)
  diag(i, i_step, s, r) = begin
    dv = maximum(abs.(tb.coords.v .- b.coords.v[idx, :]); dims = 1)
    if first_diff[] && any(!iszero, dv)
      println("first difference after the kick at element $i step $i_step: ", vec(dv))
      first_diff[] = false
    end
  end
  track_csr!(b, bl; params, test_bunch = tb, diagnostics = diag)
end

# BeamTracking alone with pz != 0: subset vs whole bunch through one bend slice and through the whole line
let b = make_bunch()
  b.coords.v[:, 6] .= range(-1e-3, 1e-3; length = 400)
  for k in (1, 2, 4, 11)
    ix = 1:37:(1 + 37 * (k - 1))
    for (name, ele) in (("bend slice", CSR.slice_element(bl.line[2], 0.0, 0.1, true, false)), ("whole line", bl))
      bb = Bunch(copy(b.coords.v), nothing, nothing; species = e, p_over_q_ref = pq)
      bs = Bunch(b.coords.v[ix, :], nothing, nothing; species = e, p_over_q_ref = pq)
      kw = ele isa Beamline ? (;) : (; _p_over_q_ref = pq)
      track!(bb, ele; kw...); track!(bs, ele; kw...)
      println("pz != 0, $k particles, $name: max diff ", maximum(abs.(bs.coords.v .- bb.coords.v[ix, :])))
    end
  end
end

# Weighted whole bunch vs unweighted subset (track_csr!'s bunch has weights, the test bunch none)
let b = make_bunch()
  b.coords.v[:, 6] .= range(-1e-3, 1e-3; length = 400)
  for (name, ele) in (("bend slice", CSR.slice_element(bl.line[2], 0.0, 0.1, true, false)), ("whole line", bl))
    kw = ele isa Beamline ? (;) : (; _p_over_q_ref = pq)
    for wsub in (false, true)
      bb = Bunch(copy(b.coords.v), nothing, copy(b.coords.weight); species = e, p_over_q_ref = pq)
      bs = Bunch(b.coords.v[idx, :], nothing, wsub ? b.coords.weight[idx] : nothing; species = e, p_over_q_ref = pq)
      track!(bb, ele; kw...); track!(bs, ele; kw...)
      println("weighted bunch, subset weighted = $wsub, $name: max diff ", maximum(abs.(bs.coords.v .- bb.coords.v[idx, :])))
    end
  end
end

# Snapshot after the kick at element 2 step 0, then track the slice of step 1 outside track_csr!
let b = make_bunch(), tb = Bunch(b.coords.v[idx, :], nothing, nothing; species = e, p_over_q_ref = pq)
  snap = Ref{Any}()
  diag(i, i_step, s, r) = (i, i_step) == (2, 0) && (snap[] = (copy(b.coords.v), copy(tb.coords.v)))
  track_csr!(b, bl; params, test_bunch = tb, diagnostics = diag)
  vb, vt = snap[]
  println("snapshot equal: ", vt == vb[idx, :])
  sl = CSR.slice_element(bl.line[2], 0.0, 0.1, true, false)
  for (name, vb_in) in (("whole", vb), ("subset only", vb[idx, :]))
    bb = Bunch(copy(vb_in), nothing, nothing; species = e, p_over_q_ref = pq)
    bs = Bunch(copy(vt), nothing, nothing; species = e, p_over_q_ref = pq)
    track!(bb, sl; _p_over_q_ref = pq); track!(bs, sl; _p_over_q_ref = pq)
    vbi = name == "whole" ? bb.coords.v[idx, :] : bb.coords.v
    dv = vec(maximum(abs.(bs.coords.v .- vbi); dims = 1))
    println("slice from snapshot ($name): ", dv, "  rows differing: ", findall(r -> any(bs.coords.v[r, :] .!= vbi[r, :]), 1:length(idx)))
  end
end

# Which particles of the 400 differ when tracked alone through that slice? (lane/remainder pattern)
let b = make_bunch(), tb = Bunch(b.coords.v[idx, :], nothing, nothing; species = e, p_over_q_ref = pq)
  snap = Ref{Any}()
  diag(i, i_step, s, r) = (i, i_step) == (2, 0) && (snap[] = copy(b.coords.v))
  track_csr!(b, bl; params, test_bunch = tb, diagnostics = diag)
  vb = snap[]
  sl = CSR.slice_element(bl.line[2], 0.0, 0.1, true, false)
  for N in (400, 397, 64)
    bb = Bunch(copy(vb[1:N, :]), nothing, nothing; species = e, p_over_q_ref = pq)
    track!(bb, sl; _p_over_q_ref = pq)
    diffrows = Int[]
    for r in 1:N
      b1 = Bunch(vb[r:r, :], nothing, nothing; species = e, p_over_q_ref = pq)
      track!(b1, sl; _p_over_q_ref = pq)
      b1.coords.v[1, :] == bb.coords.v[r, :] || push!(diffrows, r)
    end
    println("N = $N: rows differing from single-particle tracking: ", length(diffrows), " ",
            isempty(diffrows) ? "" : (extrema(diffrows), diffrows[1:min(end, 12)]))
  end
end
