# Phase 3 cross-check driver. From crosscheck/:
#   julia --project=<env with CSR, BeamTracking, Beamlines> run_case.jl <case dir> <beamline name> [cx cy]
# <case dir> holds lat.bmad, tao.init and lat.jl (`bmad_to_scibmad lat.bmad`, names with '.' replaced by '_').
# cx, cy: start of the centroid orbit (Tao uses its single-particle orbit, which starts at particle_start).
include("compare.jl")
const DIR = abspath(ARGS[1])
include(joinpath(DIR, "lat.jl"))
bl = getfield(Main, Symbol(ARGS[2]))
centroid_start = length(ARGS) >= 4 ? [parse(Float64, ARGS[3]), 0, parse(Float64, ARGS[4]), 0, 0, 0] : zeros(6)

params(; kw...) = CSRParams(; n_bin = 40, particle_bin_span = 2, ds_step = 0.01, bmad_compat = true, kw...)

# Initial particles: Tao's beam_init (quasi-random, deterministic), written once and used by both codes
gen = run_tao(DIR; tag = "gen", csr = false, n_bin = 40, particle_bin_span = 2, ds_step = 0.01, saved_at = ["BEGINNING"])
beam0 = "beam0.dat"
mv(gen["BEGINNING"], joinpath(DIR, beam0); force = true)

off = compare_off(DIR, bl; beam0, csr_params = params(), centroid_start)
compare_case(DIR, bl; name = "free_space", csr_params = params(), centroid_start, beam0, coords_off = off)
compare_case(DIR, bl; name = "shielded", csr_params = params(n_shield_images = 4, beam_chamber_height = 0.01),
             centroid_start, beam0, coords_off = off)

# CSR.jl's default mode (Bmad fixes V1, V5, V7, V8 and the downstream source search on), for information:
# how far the default results are from Bmad.
compare_case(DIR, bl; name = "free_space_default", csr_params = params(bmad_compat = false), centroid_start, beam0,
             coords_off = off)
# Default mode with images: also hat-integrated image kernel (D20) and averaged layer sum (D24).
compare_case(DIR, bl; name = "shielded_default",
             csr_params = params(n_shield_images = 4, beam_chamber_height = 0.01, bmad_compat = false), centroid_start,
             beam0, coords_off = off)
