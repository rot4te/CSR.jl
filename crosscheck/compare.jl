# Phase 3: compare CSR.jl (track_csr! on BeamTracking) with Bmad/Tao on identical initial particles.

using CSR, BeamTracking, Beamlines, AtomicAndPhysicalConstants, Printf

include("bmad_io.jl")
include("tao.jl")

relmax(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), floatmin())

# Bunch from a Bmad beam file. weight = real particles per macroparticle (CSR.jl convention).
function bunch_from_bmad(file, species, p_over_q_ref)
  v, charge, _, alive, _ = read_bmad_beam(file)
  all(alive) || error("bunch_from_bmad: dead particles in $file")
  return Bunch(copy(v), nothing, abs.(charge) ./ E_CHARGE; species, p_over_q_ref)
end

"""
    compare_case(dir, bl; name, csr_params, centroid_start, beam0, coords_off = nothing)

Track `beam0` through `bl` with Tao (CSR on, Bmad parameters from `csr_params`) and with `track_csr!`
(`csr_params`, which should have `bmad_compat = true`). Print the per-step wake comparison and the comparison of
the CSR-induced change of the final coordinates. `coords_off`: final coordinates with CSR off from Bmad and
BeamTracking (from `compare_off`).
"""
function compare_case(dir, bl; name, csr_params::CSRParams, centroid_start, beam0, coords_off)
  p = csr_params
  out = run_tao(dir; tag = name, csr = true, n_bin = p.n_bin, particle_bin_span = p.particle_bin_span,
                ds_step = p.ds_step, n_shield_images = p.n_shield_images, beam_chamber_height = p.beam_chamber_height,
                beam_file = beam0)
  wake_bmad = read_bmad_wake(out["wake"])
  v_bmad, _ = read_bmad_beam(out["END"])

  species = bl.line[1].species_ref
  b = bunch_from_bmad(joinpath(dir, beam0), species, bl.line[1].p_over_q_ref)
  wake_jl = Dict{Tuple{Int,Int},Any}()
  track_csr!(b, bl; params = p, centroid_start,
             diagnostics = (i, i_step, s, r) -> (wake_jl[(i, i_step)] = (s = s, r = r)))

  # Bmad's diagnostic divides kick and I_csr by the kick_factor of the last image layer, (-1)^n·2 times the
  # free-space one, and prints the source points of that layer (csr_and_space_charge_mod.f90:385-391).
  n = p.n_shield_images
  diag_scale = n == 0 ? 1.0 : (-1)^n * 2.0

  println("\n=== $name: per-step wake, max |CSR.jl − Bmad| / max |Bmad| per column")
  n > 0 && println("(Bmad kick/m and I_csr/m multiplied by $diag_scale; s_source not comparable with images)")
  @printf("%5s %5s %9s %11s %11s %11s %11s %11s\n", "ele", "step", "s", "z", "charge/m", "kick/m", "I_csr/m", "s_source")
  worst = zeros(5)
  for w in wake_bmad
    j = wake_jl[(w.ix_ele, w.i_step)]
    r = j.r
    t = w.table
    # s_source of bins with no kick (I_csr = 0: source ahead) is not meaningful in either code
    src = findall(!iszero, t[:, 4])
    cols = (relmax(r.z_center, t[:, 1]), relmax(r.charge_per_meter, t[:, 2]),
            relmax(r.kick_per_meter, diag_scale .* t[:, 3]), relmax(r.I_csr_per_meter, diag_scale .* t[:, 4]),
            (isempty(src) || n > 0) ? 0.0 : relmax(r.s_source[src], t[src, 5]))
    worst .= max.(worst, cols)
    @printf("%5d %5d %9.5f %11.2e %11.2e %11.2e %11.2e %11.2e\n", w.ix_ele, w.i_step, w.s, cols...)
  end
  @printf("%-21s %11.2e %11.2e %11.2e %11.2e %11.2e\n", "worst", worst...)
  length(wake_bmad) == length(wake_jl) || println("NOTE: $(length(wake_bmad)) Bmad steps vs $(length(wake_jl)) CSR.jl steps")

  v_jl = b.coords.v
  d_bmad = v_bmad .- coords_off.bmad
  d_jl = v_jl .- coords_off.jl
  println("CSR-induced change of the final coordinates (on − off), max |CSR.jl − Bmad| / max |Bmad|:")
  for (k, c) in enumerate(("x", "px", "y", "py", "z", "pz"))
    @printf("  %-3s max|Δ_Bmad| = %.3e   rel. diff = %.2e\n", c, maximum(abs, d_bmad[:, k]), relmax(d_jl[:, k], d_bmad[:, k]))
  end
  return (; worst, wake_bmad, wake_jl, d_bmad, d_jl)
end

"""
    compare_off(dir, bl; beam0, csr_params, centroid_start) -> (bmad, jl)

Final coordinates without CSR from Tao, and from `track_csr!` with the same slicing and charges scaled by 1e-30
(BeamTracking's integrators depend on the slicing: the default Symplectic takes one step per element, so slices
refine the integration). Prints both against Bmad, and against unsliced `track!`.
"""
function compare_off(dir, bl; beam0, csr_params::CSRParams, centroid_start)
  p = csr_params
  out = run_tao(dir; tag = "off", csr = false, n_bin = p.n_bin, particle_bin_span = p.particle_bin_span,
                ds_step = p.ds_step, beam_file = beam0)
  v_bmad, _ = read_bmad_beam(out["END"])
  b = bunch_from_bmad(joinpath(dir, beam0), bl.line[1].species_ref, bl.line[1].p_over_q_ref)
  track!(b, bl)
  v_whole = copy(b.coords.v)
  b = bunch_from_bmad(joinpath(dir, beam0), bl.line[1].species_ref, bl.line[1].p_over_q_ref)
  b.coords.weight .*= 1e-30
  track_csr!(b, bl; params = p, centroid_start)
  v_jl = copy(b.coords.v)
  println("\n=== CSR off: tracking baseline, max |BeamTracking − Bmad| per coordinate")
  @printf("  %-3s %11s %11s %11s\n", "", "sliced", "unsliced", "max|Bmad|")
  for (k, c) in enumerate(("x", "px", "y", "py", "z", "pz"))
    @printf("  %-3s %11.3e %11.3e %11.3e\n", c, maximum(abs, v_jl[:, k] .- v_bmad[:, k]),
            maximum(abs, v_whole[:, k] .- v_bmad[:, k]), maximum(abs, v_bmad[:, k]))
  end
  return (bmad = v_bmad, jl = v_jl)
end
