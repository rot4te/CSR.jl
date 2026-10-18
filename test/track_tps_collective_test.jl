# Stage 2c: track_csr! with a TPS bunch in knobs (element parameters, bunch charge, reference energy, initial
# coordinates) vs ForwardDiff and central finite differences (order 1), nested Duals (order 2), and entry checks.
# Uses tc_pq, tc_line_bunch, TC_E, TC_EXT (track_csr_test.jl); LC_E, LC_E0 (lcavity_test.jl); gaussian, E_CHARGE
# (utils.jl). The last testset leaves another GTPSA Descriptor current. File-scope names start with KT_, kt_ or Knob.

const KT_NK = 7
const KT_PQ = tc_pq(200)
const KT_PARAMS_A = CSRParams(n_bin = 40, ds_step = 0.1)
const KT_PARAMS_B = CSRParams(n_bin = 40, ds_step = 0.05)
const KT_ROWS = collect(1:37:400)   # 11 sample particles
kt_norm(x) = sqrt(sum(abs2, x))     # 2-norm (LinearAlgebra is not loaded by runtests.jl)

# Final sampled coordinates (11 particles × 6, column-major) and t_ref for knob values p (element type: Float64,
# Dual or TPS). `build(p)` returns (bunch, lattice, keyword NamedTuple for track_csr!).
struct KnobTrack
  build::Any
end
function (f::KnobTrack)(p)
  b, lat, kw = f.build(p)
  track_csr!(b, lat; kw...)
  return vcat(vec(b.coords.v[KT_ROWS, :]), [b.t_ref])
end
struct KnobJacobian
  f::KnobTrack
end
(j::KnobJacobian)(p) = ForwardDiff.jacobian(j.f, p)

# Lattice A (Beamline), knobs p₁…p₇ (all with nominal value 0). ghosts > 0: two zero-weight ghost particles (rows 401,
# 402) at z = ∓ghosts fix the bin grid for the FD check (E.3).
function kt_build_a(p; csr = true, ghosts = 0.0)
  c(x) = x + zero(p[1])
  pq = KT_PQ * (1 + p[6])                                            # p6: reference energy
  line = [Drift(L = 0.2), SBend(L = 0.4 + p[2], g_ref = 1.5 + p[1], Kn0 = 1.5 + p[1]),   # p1: g, p2: L
          Quadrupole(L = 0.2, Kn1 = 2.0 + p[3]), Patch(dx = 1e-4 * (1 + p[4])),          # p3: Kn1, p4: patch dx
          SBend(L = 0.3, g_ref = -1.0, Kn0 = -1.0), Drift(L = 0.2)]
  bl = Beamline(line; p_over_q_ref = pq, species_ref = TC_E)
  b0 = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)
  v = c.(b0.coords.v)
  v[:, 6] .+= p[7] .* b0.coords.v[:, 5]                              # p7: chirp (rows 1-400)
  w = b0.coords.weight
  if ghosts > 0
    g = zeros(2, 6)
    g[:, 1] .= 1e-4
    g[:, 5] .= [-ghosts, ghosts]
    v = vcat(v, c.(g))
    w = vcat(w, zeros(2))
  end
  w = w .* (1 + p[5])                                                # p5: bunch charge
  b = Bunch(v, nothing, w; species = TC_E, p_over_q_ref = c(pq), t_ref = c(0.0))
  return b, bl, (; params = KT_PARAMS_A, csr_on = csr ? (ele -> true) : (ele -> false))
end

# Lattice B (Branch with an lcavity). ghosts > 0: as in kt_build_a, with x = 0.
function kt_build_b(p; dE0 = false, ghosts = 0.0)
  c(x) = x + zero(p[1])
  E0 = LC_E0 * (1 + p[4])                                            # p4: reference energy
  bends = [SBend(L = 0.1, g_ref = 0.5 + p[1], Kn0 = 0.5 + p[1]) for _ in 1:10]   # p1: g of all bends
  cav = RFCavity(L = 1.0 + p[7], voltage = 2e8 * (1 + p[2]), rf_frequency = 1.3e9,  # p7: cavity L, p2: voltage
                 dE_ref = dE0 ? 1e7 * p[3] : 1e8 * (1 + p[3]), phi0 = 0.0,          # p3: dE_ref
                 zero_phase = PhaseRef.Accelerating, tracking_method = SaganCavity(n_cells = 4), traveling_wave = true)
  # dE0: the cavity is the last element; the final Drift is left out because of the BeamTracking 0.8.7 double pz shift
  # after a SaganCavity (6.H.4)
  drift = dE0 ? () : (Drift(L = 0.5),)
  br = Branch([Marker(E_ref = E0, species_ref = LC_E), bends..., cav, drift...])
  n = 400
  z = collect(range(-6e-4, 6e-4; length = n))
  w = gaussian.(z, 1e-4)
  w .*= (1e-9 / E_CHARGE) / sum(w)
  v0 = zeros(n, 6)
  v0[:, 5] .= z
  v = c.(v0)
  v[:, 5] .+= 1e-4 * p[6]                                            # p6: z offset of all particles (rows 1-400)
  if ghosts > 0
    g = zeros(2, 6)
    g[:, 5] .= [-ghosts, ghosts]
    v = vcat(v, c.(g))
    w = vcat(w, zeros(2))
  end
  b = Bunch(v, nothing, w .* (1 + p[5]); species = LC_E,             # p5: bunch charge
            p_over_q_ref = c(BeamTracking.E_to_R(LC_E, E0)), t_ref = c(0.0))
  return b, br, (; params = KT_PARAMS_B)
end

# Error raised by f (nothing if f returns)
function kt_error(f)
  try
    f()
    return nothing
  catch err
    return err
  end
end

@testset "track_csr! with a TPS bunch in knobs" begin
  d = Descriptor(KT_NK, 2)
  k = collect(@vars(d))
  k1 = collect(@vars(Descriptor(KT_NK, 1)))   # order 1, for the E.3 ghost runs

  for (name, f, fg) in (("Beamline", KnobTrack(kt_build_a), KnobTrack(p -> kt_build_a(p; ghosts = 1e-2))),
                        ("Branch with an lcavity", KnobTrack(kt_build_b), KnobTrack(p -> kt_build_b(p; ghosts = 1e-3))))
    @testset "$name" begin
      vt = f(k)
      v0 = f(zeros(KT_NK))
      m = length(v0)

      # E.1 Scalar part
      vs = scalar.(vt)
      @test maximum(abs, vs[1:m-1] .- v0[1:m-1]) <= 1e-12 * maximum(abs, v0[1:m-1])
      @test abs(vs[m] - v0[m]) <= 1e-14 * abs(v0[m])

      # E.2 First order vs ForwardDiff
      Jt = GTPSA.jacobian(vt)
      Jf = ForwardDiff.jacobian(f, zeros(KT_NK))
      for j in 1:KT_NK
        @test kt_norm(Jf[1:m-1, j]) > 0
        @test kt_norm(Jt[1:m-1, j] - Jf[1:m-1, j]) <= 1e-10 * kt_norm(Jf[1:m-1, j])
        @test abs(Jt[m, j] - Jf[m, j]) <= 1e-10 * maximum(abs, Jf[m, :])
      end
      @test maximum(abs, Jf[m, :]) > 0

      # E.3 First order vs central finite differences, on the bin grid pinned by the ghosts. AD holds the grid fixed,
      # FD would move it; the ghosts fix it, but knobs that move the ghosts (g, L, Kn1, patch) keep a 0.1-10%
      # difference (measured 2026-10-17, worst 0.097 for Kn1), so 0.15 only catches gross errors.
      Jg = GTPSA.jacobian(fg(k1))
      h = 1e-4
      Jfd = zeros(m, KT_NK)
      for j in 1:KT_NK
        ej = zeros(KT_NK)
        ej[j] = 1.0
        Jfd[:, j] = (fg(h .* ej) .- fg(-h .* ej)) ./ (2h)
      end
      for j in 1:KT_NK
        @test kt_norm(Jg[1:m-1, j] - Jfd[1:m-1, j]) <= 0.15 * kt_norm(Jfd[1:m-1, j])
        @test kt_norm(Jfd[1:m-1, j]) > 0
        @test abs(Jg[m, j] - Jfd[m, j]) <= 1e-6 * maximum(abs, Jg[m, :])
      end

      # E.4 Second order vs nested Duals
      H = reshape(ForwardDiff.jacobian(KnobJacobian(f), zeros(KT_NK)), m, KT_NK, KT_NK)
      Hmax = maximum(abs, H[1:m-1, :, :])
      @test Hmax > 0
      for j in 1:m-1
        @test isapprox(GTPSA.hessian(vt[j]), H[j, :, :]; rtol = 1e-9, atol = 1e-13 * Hmax)
      end
      @test isapprox(GTPSA.hessian(vt[m]), H[m, :, :]; rtol = 1e-10, atol = 1e-13 * maximum(abs, H[m, :, :]))
      @test maximum(abs, H[m, :, :]) > 0
    end
  end

  @testset "CSR changes the knob map" begin
    Jon = GTPSA.jacobian(KnobTrack(kt_build_a)(k))
    Joff = GTPSA.jacobian(KnobTrack(p -> kt_build_a(p; csr = false))(k))
    @test all(iszero, Joff[:, 5])   # charge has no effect without CSR
    @test kt_norm(Jon[1:end-1, 5]) > 0
    @test maximum(abs, Jon .- Joff) > 0
  end

  @testset "dE_ref knob at 0 (lcavity ramp)" begin
    fb = KnobTrack(p -> kt_build_b(p; dE0 = true))
    Jt = GTPSA.jacobian(fb(k))
    h = 1e-5
    e3 = zeros(KT_NK)
    e3[3] = 1.0
    Jfd3 = (fb(h .* e3) .- fb(-h .* e3)) ./ (2h)   # the -h run has a negative dE_ref (allowed)
    @test kt_norm(Jt[1:end-1, 3]) > 0
    @test kt_norm(Jt[1:end-1, 3] - Jfd3[1:end-1]) <= 1e-5 * kt_norm(Jt[1:end-1, 3])
  end

  @testset "Test bunch in 6D + knobs beside a TPS bunch" begin
    d13 = Descriptor(13, 1)
    x = @vars(d13)
    b, bl, kw = kt_build_a(collect(x[7:13]))   # bunch in knobs 7-13
    r = [101, 301]
    tb = Bunch(vcat([permutedims([b.coords.v[ri, j] + x[j] for j in 1:6]) for ri in r]...); species = TC_E,
               p_over_q_ref = b.p_over_q_ref, t_ref = b.t_ref)
    track_csr!(b, bl; kw..., test_bunch = tb)
    for i in 1:2
      Jt = GTPSA.jacobian(tb.coords.v[i, :])
      Jb = GTPSA.jacobian(b.coords.v[r[i], :])   # 6 × 13
      @test isapprox(Jt[:, 7:13], Jb[:, 7:13]; rtol = 1e-12)
      @test kt_norm(Jt[:, 7:13]) > 0
      @test maximum(abs, scalar.(tb.coords.v[i, :]) .- scalar.(b.coords.v[r[i], :])) <= 1e-15
    end

    # Frozen-wake 6D block vs step 5
    y = @vars(Descriptor(6, 1))
    b5, bl5, kw5 = kt_build_a(zeros(KT_NK))
    tb5 = Bunch(vcat([permutedims([b5.coords.v[ri, j] + y[j] for j in 1:6]) for ri in r]...); species = TC_E,
                p_over_q_ref = b5.p_over_q_ref)
    track_csr!(b5, bl5; kw5..., test_bunch = tb5)
    for i in 1:2
      @test isapprox(GTPSA.jacobian(tb.coords.v[i, :])[:, 1:6], GTPSA.jacobian(tb5.coords.v[i, :]); rtol = 1e-10)
    end
  end

  @testset "csr_ref_curvature with knobs" begin
    @test TC_EXT.csr_ref_curvature(BendParams(g_ref = 0.0), 1) === 0.0
    gk = TC_EXT.csr_ref_curvature(BendParams(g_ref = k[1]), 1)
    @test scalar(gk) == 0
    @test GTPSA.jacobian([gk])[1, 1] == 1
    gk2 = TC_EXT.csr_ref_curvature(BendParams(g_ref = k[1], tilt_ref = Float64(π)), 1)
    @test GTPSA.jacobian([gk2])[1, 1] == -1
  end

  @testset "Entry checks" begin
    bl = kt_build_a(k)[2]

    # Float64 bunch with a TPS lattice: Float64 t_ref
    b = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)
    v_before = copy(b.coords.v)
    err = kt_error(() -> track_csr!(b, bl; params = KT_PARAMS_A))
    @test err isa ErrorException && occursin("track_csr!: bunch.t_ref has type", err.msg)
    @test b.coords.v == v_before

    # TPS bunch with a Float64 test bunch t_ref
    b = kt_build_a(k)[1]
    tb = Bunch(reshape([zero(k[1]) for _ in 1:6], 1, 6); species = TC_E, p_over_q_ref = b.p_over_q_ref)
    err = kt_error(() -> track_csr!(b, bl; params = KT_PARAMS_A, test_bunch = tb))
    @test err isa ErrorException && occursin("track_csr!: test_bunch.t_ref has type", err.msg)

    # Float64 p_over_q_ref with TPS elements
    b1 = kt_build_a(k)[1]
    b = Bunch(b1.coords.v, nothing, b1.coords.weight; species = TC_E, p_over_q_ref = KT_PQ, t_ref = b1.t_ref)
    err = kt_error(() -> track_csr!(b, kt_build_a(k)[2]; params = KT_PARAMS_A))
    @test err isa ErrorException && occursin("track_csr!: bunch.p_over_q_ref has type", err.msg)

    # Float64 coordinates with TPS t_ref and p_over_q_ref
    b0 = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ; x = 1e-4)
    b = Bunch(b0.coords.v, nothing, b0.coords.weight; species = TC_E, p_over_q_ref = KT_PQ + zero(k[1]),
              t_ref = zero(k[1]))
    err = kt_error(() -> track_csr!(b, kt_build_a(k)[2]; params = KT_PARAMS_A))
    @test err isa ErrorException && occursin("track_csr!: bunch coordinates have element type", err.msg)

    # Hardware floats are never checked
    @test TC_EXT.check_ref_types(Bunch(zeros(Float32, 2, 6); species = TC_E, p_over_q_ref = KT_PQ), nothing,
                                 Float64) === nothing
  end

  @testset "Descriptor check" begin
    # Last testset of the file: leaves another Descriptor current
    d1 = Descriptor(7, 1)
    kd = collect(@vars(d1, dynamic = true))
    b, bl, kw = kt_build_a(kd)
    Descriptor(3, 1)
    err = kt_error(() -> track_csr!(b, bl; kw...))
    @test err isa ErrorException && occursin("track_csr!: TPS in argument `coords`", err.msg)

    d2 = Descriptor(4, 1)
    kL = @vars(d2, dynamic = true)[1]
    bl2 = Beamline([Drift(L = 0.2), SBend(L = 0.4 + kL, g_ref = 1.5, Kn0 = 1.5), Drift(L = 0.2)];
                   p_over_q_ref = KT_PQ, species_ref = TC_E)
    d3 = Descriptor(5, 1)
    kb = @vars(d3, dynamic = true)[1]
    b0 = tc_line_bunch(1e-3, 1e-9, 400, KT_PQ)
    b2 = Bunch(reshape([x + zero(kb) for x in b0.coords.v], size(b0.coords.v)), nothing, b0.coords.weight;
               species = TC_E, p_over_q_ref = KT_PQ + zero(kb), t_ref = zero(kb))
    err = kt_error(() -> track_csr!(b2, bl2; params = KT_PARAMS_A))
    @test err isa ErrorException && occursin("track_csr!: TPS in argument `L`", err.msg)
  end
end
