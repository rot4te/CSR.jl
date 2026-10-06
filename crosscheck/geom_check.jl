# CSR alone, independent of tracking differences: build CSR.jl's geometry from Bmad's design (centroid) orbit and
# compare (1) CSR.jl's reference floor positions with Bmad's, (2) I_csr and s_source per step with Bmad's diagnostic
# (dz from Bmad's z grid). Run after run_case.jl (needs <dir>/free_space_wake.dat). From crosscheck/:
#   julia --project=<env> geom_check.jl <case dir> <beamline name>
include("compare.jl")
const DIR = abspath(ARGS[1])
include(joinpath(DIR, "lat.jl"))
bl = getfield(Main, Symbol(ARGS[2]))
ext = Base.get_extension(CSR, :CSRBeamTrackingBeamlinesExt)
n = length(bl.line)

# Bmad orbit and reference floor at the exit of elements 0..n
# (Tao truncates a long -command string: one call per quantity and element block)
tao_cmd(cmds) = read(Cmd(`$TAO -noplot -init tao.init -nostartup -command $(join(cmds, "; ") * "; exit")`; dir = DIR), String)
log = join([tao_cmd(["python ele:$what $i|model" for i in blk]) for what in ("orbit", "floor") for blk in Iterators.partition(0:n, 5)])
vals(line) = parse.(Float64, split(line, ';')[4:end])
orbit = zeros(n + 1, 6); floor = zeros(n + 1, 6)
io = [0, 0]
for line in split(log, '\n')
  m = match(r"^(x|px|y|py|z|pz);REAL;", line)
  if m !== nothing
    io[1] += 1
    orbit[(io[1] - 1) ÷ 6 + 1, (io[1] - 1) % 6 + 1] = vals(line)[1]
  elseif startswith(line, "Reference;REAL_ARR")
    io[2] += 1
    floor[io[2], :] = vals(line)
  end
end
io == [6(n + 1), n + 1] || error("could not parse Tao output: $io, n = $n\n$(log[end-2000:end])")

# CSR.jl's reference geometry (as track_csr!)
L = [float(e.L) for e in bl.line]
g = [Beamlines.isactive(e.PatchParams) ? 0.0 : ext.csr_ref_curvature(e.BendParams, i) for (i, e) in enumerate(bl.line)]
ref_exit = FloorPosition[]
f = FloorPosition()
for (i, e) in enumerate(bl.line)
  global f = ext.csr_ref_exit(f, L[i], g[i], e.PatchParams)
  push!(ref_exit, f)
end
dr = maximum(i -> maximum(abs, ref_exit[i].r .- floor[i+1, 1:3]), 1:n)
dth = maximum(i -> abs(ref_exit[i].theta - floor[i+1, 4]), 1:n)
@printf("Reference floor, CSR.jl − Bmad: max |Δr| = %.2e m, max |Δθ| = %.2e rad\n", dr, dth)

e = bl.line[1].species_ref
gamma = sqrt(float(bl.line[1].pc_ref)^2 + massof(e)^2) / massof(e)
p = CSRParams(n_bin = 40, ds_step = 0.01, bmad_compat = get(ENV, "COMPAT", "1") == "1")
compat = get(ENV, "COMPAT", "1") == "1"
geom = CSRGeometry(L, g, ref_exit, orbit; bmad_compat = compat)
max_dI = 0.0; max_I = 0.0; worst_s = 0.0
for w in read_bmad_wake(joinpath(DIR, "free_space_wake.dat"))
  w.i_step == 0 && w.s == 0 && all(iszero, w.table[:, 4]) && continue
  n_step, ds = csr_n_step(L[w.ix_ele], p)
  kf = (w.i_step == 0 || w.i_step == n_step) ? 0.5 : 1.0
  dz = w.table[2, 1] - w.table[1, 1]
  # Bmad prints the kick position with 6 decimals; use the exact step position
  abs(w.i_step * ds - w.s) < 1e-6 || error("step position mismatch")
  kp = CSR.CSRKickPoint(geom, w.ix_ele, w.i_step * ds, gamma; kick_factor = kf)
  I_csr, _, s_source = CSR.csr_wake_table(geom, kp, dz, p.n_bin, CSR.CPU())
  el = geom.ele[w.ix_ele]
  I = I_csr ./ (kf * ds * el.L_chord / el.L)
  src = findall(!iszero, w.table[:, 4])
  d, j = findmax(abs.(I .- w.table[:, 4]))
  get(ENV, "VERBOSE", "") == "1" && @printf("STEP ele %2d step %2d s %.5f  max|ΔI| %.3e at bin %2d (I = %.6e, Bmad %.6e, source ele %s)\n", w.ix_ele, w.i_step, w.s, d, j, I[j], w.table[j, 4], w.table[j, 5])
  global max_dI = max(max_dI, d)
  global max_I = max(max_I, maximum(abs, w.table[:, 4]))
  isempty(src) || (global worst_s = max(worst_s, relmax(s_source[src], w.table[src, 5])))
end
# I_csr relative to its largest value over all steps (some steps have I_csr ≈ 0: drifts far from a bend)
@printf("With Bmad's orbit: max over steps |I_csr − Bmad| / max over steps |Bmad| = %.2e; s_source (per step): %.2e\n",
        max_dI / max_I, worst_s)
