# Stage 2c probe (2026-10-17): do Beamlines/BeamTracking track a TPS bunch through elements whose parameters are TPS
# knobs (no CSR), and where does CSR.jl's track_csr! / slicing fail today with TPS knobs?
# Each case: run with TPS knobs (Descriptor(nk, 1)), compare the knob gradient of the final coordinates with central
# finite differences of the same Float64 tracking. Prints OK/FAIL and the first line of any error.
# Run: julia --project=<test env> stage2c_tps_params_probe.jl
using CSR, BeamTracking, Beamlines, GTPSA, LinearAlgebra
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))

const E = Species("electron")
const E0 = 1e8
pq0 = BeamTracking.E_to_R(E, E0)
v0 = [1e-3 -2e-4 5e-4 3e-4 1e-3 2e-3; -2e-3 1e-4 -1e-3 -2e-4 -5e-4 -1e-3]

const NK = 4
d = Descriptor(NK, 1)
k = @vars(d)
tps_const(c) = zero(k[1]) + c

# f(p, T): final coords for knob values p (vector of length NK, element type T); the lattice is built inside f.
function check(name, f; h = 1e-7, rtol = 1e-5)
  try
    vt = f(collect(k), :tps)
    J = GTPSA.jacobian(vec(vt))   # (n coords) × NK
    Jfd = similar(J)
    for j in 1:NK
      e = zeros(NK); e[j] = h
      Jfd[:, j] = (vec(f(e, :float)) .- vec(f(-e, :float))) ./ (2h)
    end
    val = maximum(abs, scalar.(vec(vt)) .- vec(f(zeros(NK), :float)))
    err = maximum(abs, J .- Jfd) / max(maximum(abs, Jfd), 1e-300)
    println(rpad(name, 48), err < rtol && val < 1e-12 ? "OK  " : "FAIL",
            "  values Δ=", round(val; sigdigits = 2), "  ∂/∂knob rel err=", round(err; sigdigits = 2),
            "  max|J|=", round(maximum(abs, Jfd); sigdigits = 3))
  catch ex
    println(rpad(name, 48), "ERROR ", first(split(sprint(showerror, ex), '\n')))
    bt = stacktrace(catch_backtrace())
    fr = [f for f in bt if occursin(r"Beamlines|BeamTracking|CSR|GTPSA", string(f.file))]
    for f in fr[1:min(4, end)]
      println("    at ", f.func, " ", basename(string(f.file)), ":", f.line, "  (", string(f.file), ")")
    end
  end
end

# BeamTracking writes t_ref (and p_over_q_ref) back into the Bunch: with TPS knobs these must be TPS fields
const DUAL_T = Ref{Any}(nothing)   # element type of the knob vector in :dual runs (set by bunch_of's callers)
cst(T, c) = T === :tps ? tps_const(c) : T === :dual ? zero(DUAL_T[]) + c : c
bunch_of(T, pq) = T === :float ? Bunch(copy(v0); species = E, p_over_q_ref = pq) :
                  Bunch(cst.(T, v0); species = E, p_over_q_ref = cst(T, pq), t_ref = cst(T, 0.0))

# 1. Field knob (quad Kn1), bend g_ref/Kn0 knob, bend L knob, patch dx knob; Float64 reference energy.
function f_elements(p, T)
  DUAL_T[] = eltype(p)
  line = [Drift(L = 0.2 + p[3]), Quadrupole(L = 0.3, Kn1 = 2.0 + p[1]),
          SBend(L = 0.4 + p[3], g_ref = 1.5 + p[2], Kn0 = 1.5 + p[2]), Patch(dx = 1e-4 + p[4]), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq0, species_ref = E)
  b = bunch_of(T, pq0)
  track!(b, bl)
  return b.coords.v
end
check("BT: Kn1, g_ref/Kn0, L, patch dx", f_elements)

# 2. Reference energy knob: Beamline p_over_q_ref TPS (bunch p_over_q_ref TPS too).
function f_pq(p, T)
  DUAL_T[] = eltype(p)
  pq = pq0 * (1 + p[1])
  line = [Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5, Kn0 = 1.5), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = E)
  b = bunch_of(T, pq)
  track!(b, bl)
  return b.coords.v
end
check("BT: p_over_q_ref", f_pq)

# 3. lcavity (SaganCavity) with voltage and dE_ref knobs, whole element.
function f_cav(p, T)
  DUAL_T[] = eltype(p)
  cav = RFCavity(L = 1.0, voltage = 2e8 * (1 + p[1]), rf_frequency = 1.3e9, dE_ref = 1e8 * (1 + p[2]), phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  br = Branch([Marker(E_ref = E0, species_ref = E), cav])
  b = bunch_of(T, pq0)
  for bl in br.beamlines, ele in bl.line
    track!(b, ele; context = bl.context)
  end
  return b.coords.v
end
check("BT: SaganCavity voltage, dE_ref (whole)", f_cav)

# 4. Initial-coordinate knob and TPS weights in a Bunch (construction only + tracking).
function f_coords(p, T)
  DUAL_T[] = eltype(p)
  line = [Drift(L = 0.2), SBend(L = 0.4, g_ref = 1.5, Kn0 = 1.5)]
  bl = Beamline(line; p_over_q_ref = pq0, species_ref = E)
  v = cst.(T, v0) .+ 0 * p[1]
  v[:, 5] .+= p[1]; v[:, 6] .+= p[2] .* v0[:, 5]
  w = [1e9, 2e9] .* (1 + p[3])
  b = T === :float ? Bunch(v, nothing, w; species = E, p_over_q_ref = pq0) :
                     Bunch(v, nothing, w; species = E, p_over_q_ref = cst(T, pq0), t_ref = cst(T, 0.0))
  track!(b, bl)
  return b.coords.v
end
check("BT: initial z/chirp knobs, TPS weights", f_coords)

# 5. CSR.jl slicing with TPS parameters: element sliced into 4 (CSR.slice_element + BeamTracking track_slice!).
const EXT = Base.get_extension(CSR, :CSRBeamTrackingBeamlinesExt)
function f_slices(p, T)
  ele = SBend(L = 0.4 + p[3], g_ref = 1.5 + p[2], Kn0 = 1.5 + p[2], Kn1 = 0.5 + p[1])
  bl = Beamline([ele]; p_over_q_ref = pq0, species_ref = E)
  ele = bl.line[1]
  b = bunch_of(T, pq0)
  L = ele.L; n = 4; ds = L / n
  for i in 1:n
    s0 = (i - 1) * ds
    EXT.track_slice!(b, CSR.slice_element(ele, s0, i * ds - s0, i == 1, i == n), pq0)
  end
  return b.coords.v
end
check("CSR: slice_element with TPS L, g, Kn1", f_slices)

# 6. CSR.jl SaganCavity slices with TPS voltage/dE_ref.
function f_sagan_slices(p, T)
  cav = RFCavity(L = 1.0, voltage = 2e8 * (1 + p[1]), rf_frequency = 1.3e9, dE_ref = 1e8 * (1 + p[2]), phi0 = 0.0,
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  ele = Branch([Marker(E_ref = E0, species_ref = E), cav]).beamlines[2].line[1]
  b = bunch_of(T, pq0)
  n = 4; ds = 1.0 / n
  for i in 1:n
    s0 = (i - 1) * ds
    EXT.track_slice!(b, CSR.slice_element(ele, s0, i * ds - s0, i == 1, i == n))
  end
  return b.coords.v
end
check("CSR: SaganCavity slices, TPS voltage/dE_ref", f_sagan_slices)

# 7. track_csr! today with a TPS bunch and TPS knobs (expected to fail somewhere: Float64 arrays).
function f_track_csr(p, T)
  pq = pq0
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[3], g_ref = 1.5 + p[2], Kn0 = 1.5 + p[2]), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = E)
  n = 400
  z = collect(range(-6e-4, 6e-4; length = n))
  w = gaussian.(z, 1e-4); w = w .* ((1e-9 / E_CHARGE) / sum(w) * (1 + p[1]))
  v = zeros(n, 6); v[:, 5] .= z
  b = T === :tps ? Bunch(tps_const.(v), nothing, w; species = E, p_over_q_ref = tps_const(pq), t_ref = tps_const(0.0)) :
                   Bunch(v, nothing, w; species = E, p_over_q_ref = pq)
  track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.1))
  return b.coords.v[1:37:end, :]
end
check("CSR: track_csr! TPS bunch, knobs", f_track_csr; h = 1e-6, rtol = 1e-3)

# p_over_q_ref knob: finite-difference step scan (is the 0.6% mismatch FD error?)
let J = GTPSA.jacobian(vec(f_pq(collect(k), :tps)))[:, 1]
  for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    e = zeros(NK); e[1] = h
    Jfd = (vec(f_pq(e, :float)) .- vec(f_pq(-e, :float))) ./ (2h)
    println("p_over_q_ref FD h=", h, "  rel err=", round(maximum(abs, J .- Jfd) / maximum(abs, Jfd); sigdigits = 2))
  end
end

# Dual knobs (ForwardDiff) through the same BeamTracking cases: Jacobian vs the TPS one. The bunch is built with
# Dual coordinates, t_ref and p_over_q_ref (constants zero(p[1]) + c).
using ForwardDiff
const BUNCH_OF_TPS = bunch_of
for (name, f) in (("Dual: Kn1, g_ref/Kn0, L, patch dx", f_elements), ("Dual: p_over_q_ref", f_pq),
                  ("Dual: SaganCavity voltage, dE_ref", f_cav), ("Dual: initial z/chirp, weights", f_coords))
  try
    Jt = GTPSA.jacobian(vec(f(collect(k), :tps)))
    Jd = ForwardDiff.jacobian(p -> vec(f(p, :dual)), zeros(NK))
    println(rpad(name, 48), "Dual vs TPS rel err=", round(maximum(abs, Jd .- Jt) / maximum(abs, Jt); sigdigits = 2))
  catch ex
    println(rpad(name, 48), "ERROR ", first(split(sprint(showerror, ex), '\n')))
    bt = stacktrace(catch_backtrace())
    fr = [f for f in bt if occursin(r"Beamlines|BeamTracking|CSR|ForwardDiff", string(f.file))]
    for f in fr[1:min(3, end)]
      println("    at ", f.func, " ", basename(string(f.file)), ":", f.line)
    end
  end
end
