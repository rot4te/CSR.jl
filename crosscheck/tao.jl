# Run Tao (Bmad) for the phase 3 cross-check. Needs the `tao` executable: set ENV["TAO"] or put it in PATH
# (conda env `bmad`: ~/miniforge3/envs/bmad/bin/tao).

const TAO = get(ENV, "TAO", joinpath(homedir(), "miniforge3", "envs", "bmad", "bin", "tao"))

"""
    run_tao(dir; tag, csr, n_bin, particle_bin_span, ds_step, n_shield_images = 0, beam_chamber_height = 0.0,
            beam_file = nothing, saved_at = ["BEGINNING", "END"]) -> Dict of output paths

Track the beam of `dir/tao.init` through `dir/lat.bmad` with 1D CSR on (`csr = true`) or off. Writes
`<tag>_wake.dat` (CSR on) and `<tag>_beam_<ele>.dat` for each element in `saved_at`. `beam_file`: initial
particles (else Tao's beam_init).
"""
function run_tao(dir; tag, csr, n_bin, particle_bin_span, ds_step, n_shield_images = 0, beam_chamber_height = 0.0,
                 beam_file = nothing, saved_at = ["BEGINNING", "END"])
  wake = joinpath(dir, "$(tag)_wake.dat")
  rm(wake; force = true)
  beams = Dict(e => "$(tag)_beam_$(lowercase(e)).dat" for e in saved_at)
  cmd = """
    set space_charge_com ds_track_step = $ds_step
    set space_charge_com n_bin = $n_bin
    set space_charge_com particle_bin_span = $particle_bin_span
    set space_charge_com n_shield_images = $n_shield_images
    set space_charge_com beam_chamber_height = $beam_chamber_height
    set space_charge_com diagnostic_output_file = "$(basename(wake))"
    set ele * csr_method = $(csr ? "1_dim" : "off")
    set ele * space_charge_method = off
    set bmad_com csr_and_space_charge_on = $(csr ? "T" : "F")
    set global track_type = beam
    """ * join(["write beam -ascii -at $e $(beams[e])" for e in saved_at], "\n") * "\nexit\n"
  run_file = joinpath(dir, "$(tag)_run.tao")
  write(run_file, cmd)
  args = ["-noplot", "-init", "tao.init", "-startup_file", basename(run_file)]
  isnothing(beam_file) || append!(args, ["-beam_init_position_file", beam_file])
  log = read(Cmd(`$TAO $args`; dir), String)
  occursin("[ERROR", log) && error("Tao reported an error:\n$log")
  return Dict("wake" => wake, "log" => log, (e => joinpath(dir, f) for (e, f) in beams)...)
end
