# Number types of the kick's inputs: Float32, Dual and TPS (was Phase
# 7 Stage 2 (D37), collective TPS). S2f removed the derivative
# testsets and the element-type and descriptor checks (test/MOVED.md);
# the Float64 and Float32 assertions are kept.

@testset "Number types (Float32, Dual, TPS)" begin
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  pz0 = zeros(n)
  geom0 = CSRGeometry(L3, g3, c3)
  ds0 = L3[2] / max(1, round(Int, L3[2] / 0.1, RoundNearestTiesAway))

  @testset "csr_kick! pz element type" begin
    p = CSRParams(n_bin = 40)
    geom0 = CSRGeometry(L3, g3, c3)
    @test csr_kick!(zeros(Float32, n), z, q, alive, geom0, 2, ds0,
                    0.17, 200.0, electron, p) isa CSRStepResult
    @test csr_kick!(zeros(n), z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                    electron, p) isa CSRStepResult
  end

  @testset "Test particles: pz_t element type" begin
    p = CSRParams(n_bin = 40)
    geom0 = CSRGeometry(L3, g3, c3)
    zt3 = [0.0, 1e-3, -1e-3]
    st = CSR.csr_binned_kick(z, q, alive, geom0, 2, ds0, 0.17,
                             200.0, electron, p)
    # Float32 pz_t with a Float64 kick still runs
    pz32 = CSR.csr_apply_test_kicks!(zeros(Float32, 3), zt3,
                                     fill(true, 3), st.bins, st.kick)
    @test pz32 isa Vector{Float32}
    @test pz32 == Float32.(CSR.csr_apply_test_kicks!(zeros(3), zt3,
        fill(true, 3), st.bins, st.kick))
    @test any(!iszero, pz32)
  end

  @testset "Zero-length element: L_chord == 0" begin
    L = [0.3, 0.0, 0.3]
    g64 = [1.0, 0.0, 1.0]
    c64 = zeros(4, 6)
    # rows 2 and 3 equal: element 2's chord is exactly zero
    c64[:, 1] .= [1e-4, 2e-4, 2e-4, 3e-4]
    g0 = CSRGeometry(L, g64, c64)
    @test g0.ele[2].L_chord == 0
  end

  # S2f: Dual / nested Dual / TPS inputs give the Float64 kick
  # bitwise; the kick leaves the partials / TPS coefficients of pz
  @testset "Dual and TPS inputs: values and zero derivative" begin
    D = ForwardDiff.Dual
    d1 = Descriptor(2, 1)
    v1 = @vars(d1)
    wraps = (
      dual = x -> D{:a}(x, 1.0, 0.5),
      nested = x -> D{:b}(D{:a}(x, 1.0), D{:a}(0.25, 0.5)),
      tps = x -> x + 0.1 * v1[1] + 0.2 * v1[2])
    # nested-Dual / TPS parts of pz: everything but the scalar value
    flat(x::Real) = (x,)
    flat(x::D) = (flat(ForwardDiff.value(x))...,
                  (flat(t) for t in ForwardDiff.partials(x))...)
    rest(x::D) = (Iterators.flatten(flat.(
        ForwardDiff.partials(x)))..., rest(ForwardDiff.value(x))...)
    rest(x::Real) = ()
    rest(x::TPS) = x - scalar(x)
    same(a, b) = all(i -> isequal(rest(a[i]), rest(b[i])),
                     eachindex(a))
    bits(x) = reinterpret(UInt64, Float64.(x))
    sc(x) = Beamlines.scalarize.(x)
    p = CSRParams(n_bin = 40)
    nn = 300
    zb, qb, ab = kernel_test_bunch(1e-3, 1e-9, nn)
    pzb = 1e-4 .* sin.(1:nn)
    idx = findall(ab)[1:30:end]
    zt, at = zb[idx], fill(true, length(idx))
    pzt = 1e-4 .* cos.(1:length(idx))
    ref = csr_kick_pz(pzb, zb, qb, ab, geom0, 2, ds0, 0.17, 200.0,
                      electron, p)
    pzt_ref = copy(pzt)
    csr_kick!(copy(pzb), zb, qb, ab, geom0, 2, ds0, 0.17, 200.0,
              electron, p; test_particles = (pzt_ref, zt, at))
    @test ref != pzb
    @test pzt_ref != pzt
    for w in wraps, which in (:pz, :z, :q, :geom, :all)
      on(k) = which in (k, :all)
      pzv = on(:pz) ? w.(pzb) : pzb
      zv = on(:z) ? w.(zb) : zb
      qv = on(:q) ? w.(qb) : qb
      gv = on(:geom) ? CSRGeometry(w.(L3), w.(g3), c3) : geom0
      # csr_kick_pz (bunch)
      r = csr_kick_pz(pzv, zv, qv, ab, gv, 2, ds0, 0.17, 200.0,
                      electron, p)
      @test eltype(r) == eltype(pzv)
      @test bits(sc(r)) == bits(ref)
      @test same(r, pzv)
      # csr_kick! with test particles (bunch and test, same bits)
      pzt_v = on(:pz) ? w.(pzt) : copy(pzt)
      pzk = copy(pzv)
      csr_kick!(pzk, zv, qv, ab, gv, 2, ds0, 0.17, 200.0, electron,
                p; test_particles = (pzt_v, on(:z) ? w.(zt) : zt,
                                     at))
      @test bits(sc(pzk)) == bits(ref)
      @test bits(sc(pzt_v)) == bits(pzt_ref)
      @test same(pzk, pzv)
      @test same(pzt_v, on(:pz) ? w.(pzt) : pzt)
    end
  end

  # S2f: Dual number inputs (scalars, centroid, ref_exit) give the
  # Float64 kick bitwise; inference; Integer pz; bins or kick that do
  # not hold floating-point numbers
  @testset "Dual number inputs, inference, input errors" begin
    dd(x) = ForwardDiff.Dual{:a}(x, 1.0)
    p = CSRParams(n_bin = 40)
    pzb = 1e-4 .* sin.(1:n)
    ref = csr_kick_pz(pzb, z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                      electron, p; kick_factor = 0.5, pz_scale = 1.3)
    @test !any(isnan, ref) && ref != pzb
    kD = csr_kick_pz(pzb, z, q, alive, geom0, 2, dd(ds0), dd(0.17),
                     dd(200.0), electron, p; kick_factor = dd(0.5),
                     pz_scale = dd(1.3))
    @test kD == ref
    pzk = copy(pzb)
    csr_kick!(pzk, z, q, alive, geom0, 2, dd(ds0), dd(0.17),
              dd(200.0), electron, p; kick_factor = dd(0.5),
              pz_scale = dd(1.3))
    @test pzk == ref
    # centroid and ref_exit (4-argument constructor)
    re = CSR.planar_survey(L3, g3)
    reD = [CSR.FloorPosition(dd.(f.r), dd(f.theta)) for f in re]
    gc = CSRGeometry(L3, g3, dd.(c3))
    g4 = CSRGeometry(L3, g3, reD, dd.(c3))
    for gv in (gc, g4)
      kg = csr_kick_pz(pzb, z, q, alive, gv, 2, ds0, 0.17, 200.0,
                       electron, p; kick_factor = 0.5, pz_scale = 1.3)
      @test kg == ref
    end
    # inference with Dual inputs
    @inferred csr_kick!(copy(pzb), z, q, alive, geom0, 2, dd(ds0),
                        dd(0.17), dd(200.0), electron, p;
                        kick_factor = dd(0.5), pz_scale = dd(1.3))
    @inferred csr_kick_pz(pzb, z, q, alive, geom0, 2, dd(ds0),
                          dd(0.17), dd(200.0), electron, p;
                          kick_factor = dd(0.5), pz_scale = dd(1.3))
    @inferred CSRGeometry(dd.(L3), dd.(g3), dd.(c3))
    @inferred CSRGeometry(L3, g3, reD, dd.(c3))
    @inferred CSR.CSRKickPoint(gc, 2, dd(0.17), dd(200.0);
                               kick_factor = dd(0.5))
    # Integer pz: error before any kick
    pzi = zeros(Int, n)
    @test_throws ErrorException csr_kick!(pzi, z, q, alive, geom0, 2,
        ds0, 0.17, 200.0, electron, p)
    @test_throws ErrorException csr_kick_pz(pzi, z, q, alive, geom0,
        2, ds0, 0.17, 200.0, electron, p)
    @test_throws ErrorException CSR.csr_kick_reference!(pzi, z, q,
        alive, geom0, 2, ds0, 0.17, 200.0, electron, p)
    pzb2, pzti = copy(pzb), zeros(Int, 3)
    @test_throws ErrorException csr_kick!(pzb2, z, q, alive, geom0, 2,
        ds0, 0.17, 200.0, electron, p; test_particles = (pzti,
        [0.0, 1e-3, -1e-3], fill(true, 3)))
    @test pzb2 == pzb && all(iszero, pzti)
    st = CSR.csr_binned_kick(z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                             electron, p)
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pzti,
        [0.0, 1e-3, -1e-3], fill(true, 3), st.bins, st.kick)
    @test all(iszero, pzti)
    # bins or kick not floating-point (complex, TPS, Dual)
    b = st.bins
    bc = CSR.CSRBins(b.dz, b.z0_edge, b.z_center, complex.(b.charge),
                     b.n_particle,
                     complex.(b.edge_dcharge_density_dz))
    v1 = @vars(Descriptor(1, 1))
    kt = [v1[1] for _ in st.kick]
    be = CSR.get_backend(z)
    @test_throws ErrorException CSR.csr_bins_kick(bc, be, geom0, 2,
        ds0, 0.17, 200.0, electron, p)
    pzt = [0.0]
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pzt, [0.0],
        [true], bc, st.kick)
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pzt, [0.0],
        [true], b, kt)
    kd = dd.(st.kick)
    pztd = [dd(0.0)]
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pztd, [0.0],
        [true], b, kd)
    @test pztd[1] === dd(0.0)
    @test_throws ErrorException CSR.csr_apply_test_kicks!(pzt, [0.0],
        [true], b, kd)
    @test_throws ErrorException CSR.csr_apply_kicks!(pzt, [0.0],
        [true], b, kd)
    bd = CSR.CSRBins(dd(b.dz), b.z0_edge, b.z_center, b.charge,
                     b.n_particle, b.edge_dcharge_density_dz)
    @test_throws ErrorException CSR.csr_bins_kick(bd, be, geom0, 2,
        ds0, 0.17, 200.0, electron, p)
    @test_throws "equal lengths" CSR.csr_apply_test_kicks!(pzt,
        [0.0], [true], b, st.kick[1:end-1])
    @test pzt == [0.0]
  end

  # S2f decision 24: kick scales exactly with pz_scale (pz = 0: the
  # added term is pz_scale * (...), the same product as 1 * (...)
  # times s, so equality is bitwise)
  @testset "pz_scale is linear" begin
    p = CSRParams(n_bin = 40)
    k1 = csr_kick_pz(pz0, z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                     electron, p)
    @test any(!iszero, k1)
    for s in (0.3, 2.0, -1.7)
      ks = csr_kick_pz(pz0, z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                       electron, p; pz_scale = s)
      @test ks == s .* k1
    end
  end

  @testset "csr_kick_pz returns eltype(pz)" begin
    p = CSRParams(n_bin = 40)
    D = ForwardDiff.Dual{:a, Float64, 1}
    for pzT in (Float32, D)
      pzv = pzT.(pz0)
      r = csr_kick_pz(pzv, z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                      electron, p)
      @test r isa Vector{pzT}
    end
    @test Float64.(csr_kick_pz(Float32.(pz0), z, q, alive, geom0, 2,
        ds0, 0.17, 200.0, electron, p)) ==
        Float64.(Float32.(csr_kick_pz(pz0, z, q, alive, geom0, 2,
        ds0, 0.17, 200.0, electron, p)))
    # TPS pz: the result shares no TPS object with pz (alive has dead
    # particles, which get no kick)
    vt = @vars(Descriptor(1, 1))
    pzt = pz0 .+ vt[1]
    rt = csr_kick_pz(pzt, z, q, alive, geom0, 2, ds0, 0.17, 200.0,
                     electron, p)
    @test !any(i -> rt[i] === pzt[i], eachindex(pzt))
  end

  # S2f: TPS pz with more than 1024 live particles runs the kick on
  # one workgroup (GTPSA scratch is not thread safe); values = Float64
  @testset "TPS pz, > 1024 particles" begin
    p = CSRParams(n_bin = 40)
    nl = 1030
    zl, ql, al = gaussian_line_bunch(1e-3, 1e-9, nl)
    dl = Descriptor(1, 1)
    vl = @vars(dl)
    pzl = 1e-4 .* sin.(1:nl)
    ref = csr_kick_pz(pzl, zl, ql, al, geom0, 2, ds0, 0.17, 200.0,
                      electron, p)
    pzv = pzl .+ 0.1 .* vl[1]
    csr_kick!(pzv, zl, ql, al, geom0, 2, ds0, 0.17, 200.0, electron,
              p)
    @test reinterpret(UInt64, scalar.(pzv)) ==
        reinterpret(UInt64, ref)
    @test all(i -> isequal(pzv[i] - scalar(pzv[i]), 0.1 * vl[1]),
              eachindex(pzv))
  end
end
