# Phase 7 step 5 probe: TPS test bunch through a Branch with bends and an lcavity (SaganCavity slices, dE_ref),
# vs ForwardDiff of the same tracking with a Float64 test particle.
# Run: julia --project=<test env> track_tps_lcavity_probe.jl
using CSR, BeamTracking, Beamlines, GTPSA, ForwardDiff, LinearAlgebra
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

const LC_E = Species("electron")
const LC_E0 = 1e8
pq = BeamTracking.E_to_R(LC_E, LC_E0)

function lc_line_bunch()
  n = 4001
  z = collect(range(-6e-4, 6e-4; length = n))
  w = gaussian.(z, 1e-4)
  w .*= (1e-9 / E_CHARGE) / sum(w)
  v = zeros(n, 6)
  v[:, 5] .= z
  return Bunch(v, nothing, w; species = LC_E, p_over_q_ref = pq)
end

function lc_lattice(n_cells, dE)
  bends = [SBend(L = 0.1, g_ref = 0.5, Kn0 = 0.5) for _ in 1:10]
  cav = RFCavity(L = 1.0, voltage = 2e8, rf_frequency = 1.3e9, dE_ref = dE, phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = n_cells),
                 traveling_wave = true)
  return Branch([Marker(E_ref = LC_E0, species_ref = LC_E), bends..., cav, Drift(L = 0.5)])
end

br = lc_lattice(4, 1e8)
params = CSRParams(n_bin = 80, ds_step = 0.05)
v_ref = [1e-4, -2e-5, 0.0, 0.0, 1e-4, 2e-4]

function f(v)
  tb = Bunch(reshape(v_ref .+ v, 1, 6); species = LC_E, p_over_q_ref = pq)
  track_csr!(lc_line_bunch(), br; params, test_bunch = tb)
  return tb.coords.v[1, :]
end

d = Descriptor(6, 2)
x = @vars(d)
tb = Bunch(reshape(v_ref .+ x, 1, 6); species = LC_E, p_over_q_ref = pq)
track_csr!(lc_line_bunch(), br; params, test_bunch = tb)
vt = tb.coords.v[1, :]
y0 = f(zeros(6))
println("scalar parts vs Float64: ", maximum(abs.(scalar.(vt) .- y0)))
Jt, Jf = GTPSA.jacobian(vt), ForwardDiff.jacobian(f, zeros(6))
for c in 1:6
  println("  column $c relative: ", norm(Jt[:, c] - Jf[:, c]) / norm(Jf[:, c]))
end
for j in (1, 5, 6)
  Ht, Hf = GTPSA.hessian(vt[j]), ForwardDiff.hessian(v -> f(v)[j], zeros(6))
  println("Hessian coord $j: max abs diff ", maximum(abs.(Ht .- Hf)), "  max |H| ", maximum(abs.(Hf)))
end
