# Stage 2c (2026-10-17): find every place where track_csr! with a TPS bunch in knobs calls a GTPSA function that aborts
# the process at a zero scalar part (sqrt, log, atan2, div, inv). GTPSA's in-place functions are overwritten in this
# process only: at a zero argument they record the CSR.jl call site, shift the scalar part by 1e-200 and continue.
# Lattices as specs/stage2c.md 6.E (A: Beamline, B: Branch with an lcavity, B with dE_ref knob at 0).
# Run: julia --project=<test env> stage2c_tps_zero_hazards.jl
using CSR, BeamTracking, Beamlines, GTPSA
using GTPSA: RealTPS
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

const HAZ = Dict{String,Int}()
function record(kind)
  fr = [f for f in stacktrace() if occursin(r"(CSR|BeamTracking)\.jl/(src|ext)", string(f.file))]
  key = kind * "  " * join(["$(f.func) $(basename(string(f.file))):$(f.line)" for f in fr[1:min(3, end)]], " <- ")
  HAZ[key] = get(HAZ, key, 0) + 1
end
# At a hazard: copy the argument into the output buffer c, shift its scalar part off zero, and apply the operation in
# place on c (GTPSA itself calls e.g. sqrt!(d, d)).
sc(t) = GTPSA.geti(t, 0)
function shift_into!(c, a)
  c === a || GTPSA.copy!(c, a)
  GTPSA.seti!(c, 0, 0.0, sc(c) + 1e-200)
  return c
end
@eval GTPSA begin
  sqrt!(t::RealTPS, t1::RealTPS) = (Main.sc(t1) == 0 ? (Main.record("sqrt"); mad_tpsa_sqrt!(Main.shift_into!(t, t1), t)) : mad_tpsa_sqrt!(t1, t))
  log!(t::RealTPS, t1::RealTPS) = (Main.sc(t1) == 0 ? (Main.record("log"); mad_tpsa_log!(Main.shift_into!(t, t1), t)) : mad_tpsa_log!(t1, t))
  function atan!(c::RealTPS, a::RealTPS, b::RealTPS)
    if Main.sc(a) == 0 && Main.sc(b) == 0
      Main.record("atan2")
      c === a && error("atan2 hazard with aliased output")
      return mad_tpsa_atan2!(a, Main.shift_into!(c, b), c)
    end
    return mad_tpsa_atan2!(a, b, c)
  end
  function div!(c::RealTPS, a::RealTPS, b::RealTPS)
    if Main.sc(b) == 0
      Main.record("div")
      c === a && error("div hazard with aliased output")
      return mad_tpsa_div!(a, Main.shift_into!(c, b), c)
    end
    return mad_tpsa_div!(a, b, c)
  end
  inv!(c::RealTPS, a::RealTPS, v = 1) = (Main.sc(a) == 0 ? (Main.record("inv"); mad_tpsa_inv!(Main.shift_into!(c, a), Float64(v), c)) : mad_tpsa_inv!(a, Float64(v), c))
end

const TC_E = Species("electron")
tc_pq(gamma) = BeamTracking.E_to_R(TC_E, gamma * massof(TC_E))
function tc_line_bunch(sigma, Q, n, pq; x = 0.0)
  z = collect(range(-6sigma, 6sigma; length = n)); w = gaussian.(z, sigma); w .*= (Q / E_CHARGE) / sum(w)
  v = zeros(n, 6); v[:, 1] .= x; v[:, 5] .= z
  return Bunch(v, nothing, w; species = TC_E, p_over_q_ref = pq)
end
const LC_E0 = 1e8
const KT_PQ = tc_pq(200)

function kt_build_a(p)
  c(x) = x + zero(p[1])
  pq = KT_PQ * (1 + p[6])
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[2], g_ref = 1.5 + p[1], Kn0 = 1.5 + p[1]),
          Quadrupole(L = 0.2, Kn1 = 2.0 + p[3]), Patch(dx = 1e-4 * (1 + p[4])),
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = TC_E)
  b0 = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)
  v = c.(b0.coords.v); v[:, 6] .+= p[7] .* b0.coords.v[:, 5]
  b = Bunch(v, nothing, b0.coords.weight .* (1 + p[5]); species = TC_E, p_over_q_ref = c(pq), t_ref = c(0.0))
  return b, bl, (; params = CSRParams(n_bin = 40, ds_step = 0.1))
end
function kt_build_b(p; dE0 = false)
  c(x) = x + zero(p[1])
  E0 = LC_E0 * (1 + p[4])
  bends = [SBend(L = 0.1, g_ref = 0.5 + p[1], Kn0 = 0.5 + p[1]) for _ in 1:10]
  cav = RFCavity(L = 1.0 + p[7], voltage = 2e8 * (1 + p[2]), rf_frequency = 1.3e9,
                 dE_ref = dE0 ? 1e7 * p[3] : 1e8 * (1 + p[3]), phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  br = Branch([Marker(E_ref = E0, species_ref = TC_E), bends..., cav, Drift(L = 0.5)])
  n = 400; z = collect(range(-6e-4, 6e-4; length = n)); w = gaussian.(z, 1e-4); w .*= (1e-9 / E_CHARGE) / sum(w)
  v0 = zeros(n, 6); v0[:, 5] .= z; v = c.(v0); v[:, 5] .+= 1e-4 * p[6]
  b = Bunch(v, nothing, w .* (1 + p[5]); species = TC_E, p_over_q_ref = c(BeamTracking.E_to_R(TC_E, E0)), t_ref = c(0.0))
  return b, br, (; params = CSRParams(n_bin = 40, ds_step = 0.05))
end

k = collect(@vars(Descriptor(7, 2)))
for (name, build) in (("A", kt_build_a), ("B", kt_build_b), ("B dE0", p -> kt_build_b(p; dE0 = true)))
  empty!(HAZ)
  b, lat, kw = build(k)
  t = @elapsed track_csr!(b, lat; kw...)
  println("== lattice ", name, " (", round(t; digits = 1), " s): ", length(HAZ), " hazard sites")
  for (key, n) in sort(collect(HAZ); by = first)
    println("  ", n, "×  ", key)
  end
end
