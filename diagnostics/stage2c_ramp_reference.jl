# Stage 2c: Float64 reference for the lcavity ramp rule change (D49: ramp for every SaganCavity, not E_start != E_end).
# Generated 2026-10-17 with the code at 2fdbe77 (before Stage 2c): a Branch with bends and a SaganCavity with
# dE_ref = 0, tracked by track_csr!; writes the final coordinates (400×6) and t_ref with full precision.
# Stage 2c acceptance: the same script's lattice tracked by the new code agrees to ≤ 1e-15 relative (max abs diff /
# max abs of the reference coordinates). Run: julia --project=<test env> stage2c_ramp_reference.jl [outfile]
using CSR, BeamTracking, Beamlines, DelimitedFiles
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

const E = Species("electron")
const E0 = 1e8

function ramp_lattice()
  bends = [SBend(L = 0.1, g_ref = 0.5, Kn0 = 0.5) for _ in 1:4]
  cav = RFCavity(L = 1.0, voltage = 2e7, rf_frequency = 1.3e9, dE_ref = 0.0, phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  return Branch([Marker(E_ref = E0, species_ref = E), bends..., cav, Drift(L = 0.5)])
end

function ramp_bunch()
  n = 400
  z = collect(range(-6e-4, 6e-4; length = n))
  w = gaussian.(z, 1e-4)
  w .*= (1e-9 / E_CHARGE) / sum(w)
  v = zeros(n, 6)
  v[:, 1] .= 1e-4
  v[:, 5] .= z
  return Bunch(v, nothing, w; species = E, p_over_q_ref = BeamTracking.E_to_R(E, E0))
end

br = ramp_lattice()
println("SaganCavity elements: ", count(ele -> ele.tracking_method isa SaganCavity, reduce(vcat, [bl.line for bl in br.beamlines])))
b = ramp_bunch()
track_csr!(b, br; params = CSRParams(n_bin = 40, ds_step = 0.05))
out = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "stage2c_ramp_dE0_reference.txt")
writedlm(out, vcat(b.coords.v, fill(b.t_ref, 1, 6)))
println("wrote ", out, "  max|v| = ", maximum(abs, b.coords.v), "  t_ref = ", b.t_ref)
