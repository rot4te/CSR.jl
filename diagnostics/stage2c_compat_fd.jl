# Stage 2c (2026-10-18): why does lattice A in bmad_compat mode (with or without images) disagree with central FD
# (knobs 2 = bend L, 3 = Kn1) and with the Float64 scalar part? Step scan h = 1e-3 … 1e-6 (a jump in the map gives an
# FD error ∝ 1/h; a smooth disagreement does not), grid pinned by ghosts at ±1 cm, per configuration.
# Run: julia --project=<test env> -t 1 stage2c_compat_fd.jl
using CSR, BeamTracking, Beamlines, GTPSA
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
const E = Species("electron")
const PQ = BeamTracking.E_to_R(E, 200 * massof(E))
const ROWS = collect(1:37:400)
nrm(x) = sqrt(sum(abs2, x))

function build_a(p, params; ghosts = 1e-2)
  c(x) = x + zero(p[1]); pq = PQ * (1 + p[6])
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[2], g_ref = 1.5 + p[1], Kn0 = 1.5 + p[1]),
          Quadrupole(L = 0.2, Kn1 = 2.0 + p[3]), Patch(dx = 1e-4 * (1 + p[4])),
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = E)
  z = collect(range(-6e-3, 6e-3; length = 400)); w = gaussian.(z, 1e-3); w .*= (1e-9 / E_CHARGE) / sum(w)
  v0 = zeros(400, 6); v0[:, 1] .= 1e-4; v0[:, 5] .= z
  if ghosts > 0
    v0 = vcat(v0, [1e-4 0 0 0 -ghosts 0; 1e-4 0 0 0 ghosts 0]); w = vcat(w, [0.0, 0.0])
  end
  v = c.(v0); v[1:400, 6] .+= p[7] .* v0[1:400, 5]
  b = Bunch(v, nothing, w .* (1 + p[5]); species = E, p_over_q_ref = c(pq), t_ref = c(0.0))
  return b, bl, params
end
function run(p, params)
  b, bl, prm = build_a(p, params); track_csr!(b, bl; params = prm)
  return vcat(vec(b.coords.v[ROWS, :]), [b.t_ref])
end
k = collect(@vars(Descriptor(7, 1)))
if false; for (name, prm) in (("default", CSRParams(n_bin = 40, ds_step = 0.1)),
                    ("compat", CSRParams(n_bin = 40, ds_step = 0.1, bmad_compat = true)),
                    ("images+compat", CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01,
                                                bmad_compat = true)))
  vt = run(k, prm); v0 = run(zeros(7), prm); m = length(v0)
  println("== ", name, ": scalar rel diff ", round(maximum(abs, scalar.(vt[1:m-1]) .- v0[1:m-1]) / maximum(abs, v0[1:m-1]); sigdigits = 3))
  Jt = GTPSA.jacobian(vt)
  for j in (2, 3), h in (1e-3, 1e-4, 1e-5, 1e-6)
    e = zeros(7); e[j] = h
    fd = (run(e, prm) .- run(-e, prm)) ./ (2h)
    println("   knob ", j, "  h=", h, "  rel diff ", round(nrm(Jt[1:m-1, j] .- fd[1:m-1]) / nrm(fd[1:m-1]); sigdigits = 3),
            "  |FD| ", round(nrm(fd[1:m-1]); sigdigits = 4), "  |TPS| ", round(nrm(Jt[1:m-1, j]); sigdigits = 4))
  end
end
end  # if false: part 1 results in the 2026-10-18 run

# Part 2: is the 6.J zero-horizontal-distance case structural? Count guard hits in Float64 runs (images + compat) at
# p = 0 and at p = ±h on knob 2 (bend L): a copy of ddz_calc_csr with a counter in the guard branch is evaluated into
# CSR (this process only). Also FD for images in default mode.
const HITS = Ref(0)
let src = read(joinpath(pkgdir(CSR), "src", "kernels", "source_point.jl"), String)
  i = findfirst("@inline function ddz_calc_csr", src)
  j = findnext("\nend\n", src, last(i))
  f = src[first(i):last(j)]
  @assert occursin("if primal(Lp2) == 0\n", f)
  f = replace(f, "if primal(Lp2) == 0\n" => "if primal(Lp2) == 0\n      Main.HITS[] += 1\n")
  include_string(CSR, f)
end
pic = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01, bmad_compat = true)
if false; for (label, p) in (("p = 0", zeros(7)), ("knob 2 +1e-4", [0, 1e-4, 0, 0, 0, 0, 0]), ("knob 2 -1e-4", [0, -1e-4, 0, 0, 0, 0, 0]),
                   ("knob 3 +1e-4", [0, 0, 1e-4, 0, 0, 0, 0]))
  HITS[] = 0; run(p, pic)
  println("images+compat Float64 ", rpad(label, 14), ": zero-horizontal guard hits ", HITS[])
end
end  # if false: part 2a
pim = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01)
if false; vt = run(k, pim); Jt = GTPSA.jacobian(vt); m = length(vt)
for h in (1e-3, 1e-4)
  e = zeros(7); e[2] = h
  fd = (run(e, pim) .- run(-e, pim)) ./ (2h)
  println("images default  knob 2  h=", h, "  rel diff ", round(nrm(Jt[1:m-1, 2] .- fd[1:m-1]) / nrm(fd[1:m-1]); sigdigits = 3))
end
end  # if false: part 2b

# Part 3 (run 2026-10-18; disabled): images (default mode), knob 2: sample the Float64 map at 81 points p₂ ∈ [-2e-4, 2e-4] for the output where
# TPS and FD differ most; residuals from the TPS tangent line show a staircase (dense jumps) or a smooth slope error.
if false
  vt = run(k, pim); Jt = GTPSA.jacobian(vt); m = length(vt)
  e = zeros(7); e[2] = 1e-4
  fd = (run(e, pim) .- run(-e, pim)) ./ 2e-4
  jo = argmax(abs.(Jt[1:m-1, 2] .- fd[1:m-1]))
  ps = range(-2e-4, 2e-4; length = 81)
  ys = [run([0, p, 0, 0, 0, 0, 0], pim)[jo] for p in ps]
  y0 = scalar(vt[jo])
  println("output ", jo, ": TPS slope ", Jt[jo, 2], ", FD slope ", fd[jo])
  res = ys .- (y0 .+ Jt[jo, 2] .* ps)
  d = diff(ys) ./ step(ps)
  println("finite slopes between samples: min ", minimum(d), " max ", maximum(d), " median ", sort(d)[40])
  for (p, r, y) in zip(ps[1:8:end], res[1:8:end], ys[1:8:end])
    println("  p=", round(p; sigdigits = 3), "  y=", y, "  residual vs TPS tangent ", r)
  end
end

# Part 4: images in default mode, knob 2: TPS vs central FD with the image table's adaptive-Simpson tolerance at its
# default (rtol = 1e-6) and tightened to 1e-10 (call site in src/kick.jl patched in this process only). If the AD
# derivative is right, the FD disagreement must shrink with the tolerance.
const IMG_RTOL = Ref(1e-6)
let src = read(joinpath(pkgdir(CSR), "src", "kick.jl"), String)
  pat = "bin_integrate = !params.bmad_compat)"
  @assert count(pat, src) == 1
  i = findfirst(pat, src)
  fstart = findprev("\nfunction ", src, first(i))
  fend = findnext("\nend\n", src, last(i))
  f = src[first(fstart)+1:last(fend)]
  f = replace(f, pat => "bin_integrate = !params.bmad_compat, rtol = Main.IMG_RTOL[])")
  include_string(CSR, f)
end
for rt in (1e-6, 1e-10)
  IMG_RTOL[] = rt
  vt = run(k, pim); Jt = GTPSA.jacobian(vt); m = length(vt)
  for j in (2, 3), h in (1e-3, 1e-4)
    e = zeros(7); e[j] = h
    fd = (run(e, pim) .- run(-e, pim)) ./ (2h)
    println("images default, image rtol ", rt, ": knob ", j, "  h=", h, "  rel diff ",
            round(nrm(Jt[1:m-1, j] .- fd[1:m-1]) / nrm(fd[1:m-1]); sigdigits = 3), "  |TPS| ", round(nrm(Jt[1:m-1, j]); sigdigits = 4))
  end
end
