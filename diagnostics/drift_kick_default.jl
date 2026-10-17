# Default-mode wake kick on the first drift of the Tao csr_beam_tracking example (all sources on the upstream
# infinite drift), relative to the bend kicks. After D41 the cross-check showed a nonzero kick there (Bmad: 0).
#   julia --project=<env> drift_kick_default.jl [csr_beam_tracking|csr_beam_tracking_patch] [last element printed, 2]
using CSR, BeamTracking, Beamlines, AtomicAndPhysicalConstants, Printf
const XC = joinpath(homedir(), ".julia/dev/CSR.jl/crosscheck")
include(joinpath(XC, "bmad_io.jl"))
case = isempty(ARGS) ? "csr_beam_tracking" : ARGS[1]
ix_last = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
include(joinpath(XC, case, "lat.jl"))
bl = lat

v, charge, _, _, _ = read_bmad_beam(joinpath(XC, case, "beam0.dat"))
for compat in (true, false)
  b = Bunch(copy(v), nothing, abs.(charge) ./ E_CHARGE; species = bl.line[1].species_ref,
            p_over_q_ref = bl.line[1].p_over_q_ref)
  p = CSRParams(; n_bin = 40, particle_bin_span = 2, ds_step = 0.01, bmad_compat = compat)
  rows = Tuple{Int,Int,Float64,Float64,Float64}[]
  track_csr!(b, bl; params = p, diagnostics = (i, i_step, s, r) ->
    push!(rows, (i, i_step, maximum(abs, r.kick_per_meter), maximum(abs, r.I_csr_per_meter),
                 maximum(abs, r.kick_per_meter) == 0 ? 0.0 : r.z_center[argmax(abs.(r.kick_per_meter))])))
  kmax = maximum(r[3] for r in rows)
  println("bmad_compat = $compat: max |kick/m| over all steps = $(@sprintf("%.3e", kmax))")
  println("  ele step    max|kick/m|   /max_all    max|I_csr/m|   z at max")
  for (i, j, k, I, z) in rows
    i <= ix_last && @printf("  %3d %4d   %.3e   %.2e   %.3e   %+.3e\n", i, j, k, k / kmax, I, z)
  end
end
