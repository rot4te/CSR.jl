# Readers for Bmad/Tao output used by the phase 3 cross-check.

"""
    read_bmad_beam(file) -> (v, charge, p0c, alive, s)

Bmad ASCII beam file (version 4, `write beam -ascii`). `v` is the N×6 phase-space matrix, `charge` the macroparticle
charge [C] per particle, `p0c` the reference momentum [eV], `alive` the live mask, `s` the s-position.
"""
function read_bmad_beam(file)
  header = Dict{String,String}()
  cols = String[]
  rows = Vector{Vector{String}}()
  for line in eachline(file)
    if startswith(line, "#!")
      cols = split(line[3:end])
    elseif startswith(line, "#")
      k, v = strip.(split(line[2:end], "="; limit = 2))
      header[k] = v
    elseif !isempty(strip(line))
      push!(rows, split(line))
    end
  end
  isempty(cols) && error("read_bmad_beam: no column header in $file")
  col(name) = findfirst(==(name), cols)
  n = length(rows)
  v = zeros(n, 6)
  for (j, name) in enumerate(("x", "px", "y", "py", "z", "pz"))
    v[:, j] = [parse(Float64, r[col(name)]) for r in rows]
  end
  charge = isnothing(col("charge")) ? fill(parse(Float64, header["charge"]), n) : [parse(Float64, r[col("charge")]) for r in rows]
  p0c = [parse(Float64, r[col("p0c")]) for r in rows]
  alive = [r[col("state")] == "Alive" for r in rows]
  s = [parse(Float64, r[col("s_position")]) for r in rows]
  return v, charge, p0c, alive, s
end

"""
    read_bmad_wake(file) -> Vector of (ix_ele, name, i_step, s, table)

Bmad's `space_charge_com%diagnostic_output_file`. `table` columns: z, charge/m, CSR kick/m, I_csr/m, s_source.
Printed with 7 significant digits.
"""
function read_bmad_wake(file)
  out = []
  ix_ele = 0; name = ""; i_step = -1; s = NaN
  rows = Vector{Vector{Float64}}()
  flush!() = (i_step >= 0 && push!(out, (ix_ele = ix_ele, name = name, i_step = i_step, s = s, table = permutedims(reduce(hcat, rows)))); empty!(rows))
  for line in eachline(file)
    if startswith(line, "!#---")
      flush!()
    elseif (m = match(r"^!\s+(\d+)\s+(\S+)\s*$", line)) !== nothing
      flush!(); i_step = -1
      ix_ele = parse(Int, m[1]); name = m[2]
    elseif (m = match(r"Step index:\s*(\d+)", line)) !== nothing
      i_step = parse(Int, m[1])
    elseif (m = match(r"S-position:\s*(\S+)", line)) !== nothing
      s = parse(Float64, m[1])
    elseif !startswith(line, "!") && !isempty(strip(line))
      push!(rows, parse.(Float64, split(line)))
    end
  end
  flush!()
  return out
end
