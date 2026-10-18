# Stage 2c (2026-10-18): images, knob 2 (first bend's L) disagrees with central FD by ~25% through track_csr!.
# (1) Does the image-table tolerance patch take effect (Float64 outputs at rtol 1e-6 vs 1e-10)?
# (2) Localize: per-kick bin kicks (track_csr! diagnostics callback) of the TPS run vs central FD of the Float64 runs,
#     kick by kick; the first kick that disagrees localizes the error. Grid pinned by ghosts (±1 cm).
# Run: julia --project=<test env> -t 1 stage2c_images_dL.jl
using CSR, BeamTracking, Beamlines, GTPSA
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
const E = Species("electron")
const PQ = BeamTracking.E_to_R(E, 200 * massof(E))
nrm(x) = sqrt(sum(abs2, x))
function build_a(p; ghosts = 1e-2)
  c(x) = x + zero(p[1]); pq = PQ * (1 + p[6])
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[2], g_ref = 1.5 + p[1], Kn0 = 1.5 + p[1]),
          Quadrupole(L = 0.2, Kn1 = 2.0 + p[3]), Patch(dx = 1e-4 * (1 + p[4])),
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = E)
  z = collect(range(-6e-3, 6e-3; length = 400)); w = gaussian.(z, 1e-3); w .*= (1e-9 / E_CHARGE) / sum(w)
  v0 = zeros(400, 6); v0[:, 1] .= 1e-4; v0[:, 5] .= z
  v0 = vcat(v0, [1e-4 0 0 0 -ghosts 0; 1e-4 0 0 0 ghosts 0]); w = vcat(w, [0.0, 0.0])
  v = c.(v0); v[1:400, 6] .+= p[7] .* v0[1:400, 5]
  return Bunch(v, nothing, w .* (1 + p[5]); species = E, p_over_q_ref = c(pq), t_ref = c(0.0)), bl
end
const IMG_RTOL = Ref(1e-6)
let src = read(joinpath(pkgdir(CSR), "src", "kick.jl"), String)
  pat = "bin_integrate = !params.bmad_compat)"
  i = findfirst(pat, src); fstart = findprev("\nfunction ", src, first(i)); fend = findnext("\nend\n", src, last(i))
  f = replace(src[first(fstart)+1:last(fend)], pat => "bin_integrate = !params.bmad_compat, rtol = Main.IMG_RTOL[])")
  include_string(CSR, f)
end
pim = CSRParams(n_bin = 40, ds_step = 0.1, n_shield_images = 2, beam_chamber_height = 0.01)
pfree = CSRParams(n_bin = 40, ds_step = 0.1)
function run(p, prm)
  b, bl = build_a(p); kicks = Dict{Tuple{Int,Int},Any}()
  track_csr!(b, bl; params = prm, diagnostics = (i, j, s, r) -> (kicks[(i, j)] = copy(r.kick_per_meter)))
  return b.coords.v[1:400, :], kicks
end
if false  # (1), (2): run 2026-10-18
IMG_RTOL[] = 1e-6; v6, _ = run(zeros(7), pim)
IMG_RTOL[] = 1e-10; v10, _ = run(zeros(7), pim)
println("(1) Float64 coords, image rtol 1e-6 vs 1e-10: max abs diff ", maximum(abs, v6 .- v10))
IMG_RTOL[] = 1e-6
# (2)
k = collect(@vars(Descriptor(7, 1)))
for (name, prm) in (("free space", pfree), ("images", pim))
  _, kt = run(k, prm)
  h = 1e-4; e = zeros(7); e[2] = h
  _, kp = run(e, prm); _, km = run(-e, prm)
  println("(2) ", name, ": per-kick bin kick, knob 2, TPS vs FD (h = 1e-4)")
  for key in sort(collect(keys(kt)))
    jt = [GTPSA.jacobian([x])[1, 2] for x in kt[key]]
    fd = (kp[key] .- km[key]) ./ (2h)
    println("   element ", key[1], " step ", key[2], ":  |FD| ", round(nrm(fd); sigdigits = 4), "  |TPS| ",
            round(nrm(jt); sigdigits = 4), "  rel diff ", round(nrm(jt .- fd) / max(nrm(fd), 1e-300); sigdigits = 3))
  end
end
end

# (3) Image table at the first kick in bend 1 (element 2, s_kick = L₂/4), outside tracking: centroid on the reference
# orbit, lattice [Drift 0.2, SBend L₂ (g 1.5), Drift 0.2, SBend 0.3 (g -1), Drift 0.2], γ = 200, y = 1 cm, dz = 5e-4.
# Knob p on (a) L₂ and s_kick together, (b) L₂ only (s_kick = 0.1), (c) s_kick only; TPS vs central FD, h = 1e-3 … 1e-6.
const CPUB = CSR.CPU()
function img(pL, ps)
  L = [0.2, 0.4 + pL, 0.2, 0.3, 0.2]; g = [0.0, 1.5, 0.0, -1.0, 0.0]
  geom = CSRGeometry(L, g, zeros(6, 6) .+ zero(pL))
  kp = CSRKickPoint(geom, 2, 0.1 + ps, 200.0; y_source = 0.01)
  return first(CSR.csr_image_table(geom, kp, 5e-4, 40, CPUB; bin_integrate = true))
end
kk = @vars(Descriptor(1, 1))
for (name, f) in (("(a) L and s_kick", p -> img(p, p / 4)), ("(b) L only", p -> img(p, zero(p))), ("(c) s_kick only", p -> img(zero(p), p / 4)))
  t = f(kk[1]); jt = [GTPSA.jacobian([x])[1, 1] for x in t]
  print("(3) ", rpad(name, 18), " |TPS| ", round(nrm(jt); sigdigits = 4))
  for h in (1e-3, 1e-4, 1e-5, 1e-6)
    fd = (f(h) .- f(-h)) ./ (2h)
    print("   h=", h, ": rel diff ", round(nrm(jt .- fd) / nrm(fd); sigdigits = 3))
  end
  println()
end
