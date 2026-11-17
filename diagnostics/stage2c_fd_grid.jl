# Stage 2c (2026-10-17): why do the 6.E central-FD checks fail while TPS == ForwardDiff?
# (1) Bin grid: csr_bin_particles builds the grid from primal min/max z, so AD holds it fixed; FD moves it with the
#     extreme particles. Test: add two zero-weight "ghost" particles far outside the bunch (no chirp/offset knob on
#     them), so the grid is the same in all FD runs; compare TPS vs FD per knob with and without ghosts.
# (2) dE_ref knob at primal 0 (factor 2): compare TPS vs FD with CSR on and off.
# Run: julia --project=<test env> -t 1 stage2c_fd_grid.jl
using CSR, BeamTracking, Beamlines, GTPSA
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
const E = Species("electron")
pq_of(gamma) = BeamTracking.E_to_R(E, gamma * massof(E))
const PQ = pq_of(200)
const LC_E0 = 1e8
const ROWS = collect(1:37:400)
nrm(x) = sqrt(sum(abs2, x))

# Line bunch, optionally with two ghosts (weight 0) at ±zg appended (rows 401, 402)
function line_bunch(sigma, Q, n; x = 0.0, ghosts = 0.0)
  z = collect(range(-6sigma, 6sigma; length = n)); w = gaussian.(z, sigma); w .*= (Q / E_CHARGE) / sum(w)
  v = zeros(n, 6); v[:, 1] .= x; v[:, 5] .= z
  if ghosts > 0
    v = vcat(v, [x 0 0 0 -ghosts 0; x 0 0 0 ghosts 0]); w = vcat(w, [0.0, 0.0])
  end
  return v, w
end

function build_a(p; csr = true, ghosts = 0.0)
  c(x) = x + zero(p[1]); pq = PQ * (1 + p[6])
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[2], g_ref = 1.5 + p[1], Kn0 = 1.5 + p[1]),
          Quadrupole(L = 0.2, Kn1 = 2.0 + p[3]), Patch(dx = 1e-4 * (1 + p[4])),
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = E)
  v0, w0 = line_bunch(1e-3, 1e-9, 400; x = 1e-4, ghosts)
  v = c.(v0); v[1:400, 6] .+= p[7] .* v0[1:400, 5]
  b = Bunch(v, nothing, w0 .* (1 + p[5]); species = E, p_over_q_ref = c(pq), t_ref = c(0.0))
  return b, bl, (; params = CSRParams(n_bin = 40, ds_step = 0.1), csr_on = csr ? (ele -> true) : (ele -> false))
end
function build_b(p; dE0 = false, csr = true, ghosts = 0.0, tail = true)
  c(x) = x + zero(p[1]); E0 = LC_E0 * (1 + p[4])
  bends = [SBend(L = 0.1, g_ref = 0.5 + p[1], Kn0 = 0.5 + p[1]) for _ in 1:10]
  cav = RFCavity(L = 1.0 + p[7], voltage = 2e8 * (1 + p[2]), rf_frequency = 1.3e9,
                 dE_ref = dE0 ? 1e7 * p[3] : 1e8 * (1 + p[3]), phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  br = Branch(tail ? [Marker(E_ref = E0, species_ref = E), bends..., cav, Drift(L = 0.5)] : [Marker(E_ref = E0, species_ref = E), bends..., cav])
  v0, w0 = line_bunch(1e-4, 1e-9, 400; ghosts)
  v = c.(v0); v[1:400, 5] .+= 1e-4 * p[6]
  b = Bunch(v, nothing, w0 .* (1 + p[5]); species = E, p_over_q_ref = c(BeamTracking.E_to_R(E, E0)), t_ref = c(0.0))
  return b, br, (; params = CSRParams(n_bin = 40, ds_step = 0.05), csr_on = csr ? (ele -> true) : (ele -> false))
end
function run(build, p)
  b, lat, kw = build(p); track_csr!(b, lat; kw...)
  return vcat(vec(b.coords.v[ROWS, :]), [b.t_ref])
end

k = collect(@vars(Descriptor(7, 1)))
function compare(name, build; knobs = 1:7, hs = (1e-5, 1e-4))
  Jt = GTPSA.jacobian(run(build, k))
  m = size(Jt, 1)
  for j in knobs, h in hs
    e = zeros(7); e[j] = h
    fd = (run(build, e) .- run(build, -e)) ./ (2h)
    r = nrm(Jt[1:m-1, j] .- fd[1:m-1]) / max(nrm(fd[1:m-1]), 1e-300)
    println(rpad(name, 34), " knob ", j, "  h=", h, "  |TPS|=", round(nrm(Jt[1:m-1, j]); sigdigits = 4),
            "  |FD|=", round(nrm(fd[1:m-1]); sigdigits = 4), "  rel diff=", round(r; sigdigits = 3))
  end
end

# compare("A, grid free", build_a)  (results in the 2026-10-17 run; uncomment to rerun)
# compare("A, ghosts at ±1 cm", p -> build_a(p; ghosts = 1e-2))
# compare("B, grid free", build_b; knobs = (1, 4, 5, 6, 7))
# compare("B, ghosts at ±1 mm", p -> build_b(p; ghosts = 1e-3); knobs = (1, 4, 5, 6, 7))
# compare("B dE0, CSR on", p -> build_b(p; dE0 = true); knobs = (3,), hs = (1e-5,))
# compare("B dE0, CSR off", p -> build_b(p; dE0 = true, csr = false); knobs = (3,), hs = (1e-5,))
# compare("B dE0, CSR on, ghosts", p -> build_b(p; dE0 = true, ghosts = 1e-3); knobs = (3,), hs = (1e-5,))
compare("B dE0, CSR on, cavity last", p -> build_b(p; dE0 = true, tail = false); knobs = (3,), hs = (1e-5, 1e-4))
compare("B dE0, CSR off, cavity last", p -> build_b(p; dE0 = true, csr = false, tail = false); knobs = (3,), hs = (1e-5,))
