# Stage 04 review fixes (2026-10-18): sub-stage B (W3, W4, W5), sub-stage C (S2), sub-stage D (S7, S9, S10, D.10) and
# the final review fixes (input lengths and non-finite z, SaganCavity entrance energy). Sub-stage B: shifted_to_slice
# number types, device number types (backends without Float64), gamma > 1 checks; C: track_csr! entry checks.
# Uses TC_EXT, TC_E, tc_pq (track_csr_test.jl), kernel_test_bunch (kernels_test.jl) and NoFloat64Backend
# (binning_test.jl), included before this file. NoFloat64Backend is not defined here (binning_test.jl defines it
# with supertype CSR.KernelAbstractions.GPU).

import KernelAbstractions

@testset "Review fixes 2026-10-18" begin
  L3 = [0.2, 0.3, 0.2]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]

  @testset "W3 shifted_to_slice" begin
    f(x, y, s, t, pp = nothing) = (s, t)
    ele = (; species_ref = TC_E, p_over_q_ref = ForwardDiff.Dual(tc_pq(200), 1.0))
    sp = TC_EXT.shifted_to_slice(f, ele, 0.25)
    @test sp.s0 isa ForwardDiff.Dual
    @test typeof(sp.s0) == typeof(sp.t0)
    @test ForwardDiff.value(sp.s0) == 0.25
    @test ForwardDiff.value(sp(0.0, 0.0, 0.1, 0.0)[1]) ≈ 0.35
    bg = TC_EXT.R_to_beta_gamma(TC_E, tc_pq(200))
    beta0 = bg / sqrt(1 + bg^2)
    @test ForwardDiff.value(sp.t0) ≈ 0.25 / (beta0 * TC_EXT.C_LIGHT)
    @test ForwardDiff.partials(sp.t0)[1] != 0
    @test ForwardDiff.partials(sp.s0)[1] == 0

    ele64 = (; species_ref = TC_E, p_over_q_ref = tc_pq(200))
    sp64 = TC_EXT.shifted_to_slice(f, ele64, 0.25)
    @test sp64.s0 === 0.25
    @test sp64.t0 isa Float64
    @test sp64.t0 ≈ 0.25 / (beta0 * TC_EXT.C_LIGHT)
  end

  @testset "W4 device_number_type" begin
    @test (@inferred CSR.device_number_type(NoFloat64Backend(), Float64)) === Float32
    @test (@inferred CSR.device_number_type(NoFloat64Backend(), Float32)) === Float32
    @test (@inferred CSR.device_number_type(NoFloat64Backend(), ForwardDiff.Dual{Nothing,Float64,1})) ===
          ForwardDiff.Dual{Nothing,Float64,1}
    @test (@inferred CSR.device_number_type(KernelAbstractions.CPU(), Float64)) === Float64

    # CPU: csr_kick_pz keeps Float64 output for a Float32 pz
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
    params = CSRParams(n_bin = 40, ds_step = 0.05)
    pz = zeros(Float32, length(z))
    pz_new = csr_kick_pz(pz, z, q, alive, geom, 3, 0.1, 200.0, Species("electron"), params)
    @test eltype(pz_new) == Float64

    # CPU: the rrule forward pass gives the same kick as csr_kick_pz (pz is zero, so pz_new is the kick)
    y, _ = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, 3, 0.1, 200.0, Species("electron"), params)
    @test eltype(y) == Float64
    @test maximum(abs, y .- pz_new) <= 1e-6 * maximum(abs, pz_new)
  end

  @testset "W5 gamma check" begin
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    electron = Species("electron")
    for g in (1.0, 0.5, NaN)
      @test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1, g)
      @test_throws "gamma must be > 1" csr_kick_coef(electron, g, 0.01)
      @test_throws "gamma must be > 1" csr_kick_coef(electron, g, 0.01; bmad_compat = true)
    end
    @test CSRKickPoint(geom, 3, 0.1, 1.0001) isa CSRKickPoint
    @test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1, ForwardDiff.Dual(1.0, 1.0))
    @test CSRKickPoint(geom, 3, 0.1, ForwardDiff.Dual(200.0, 1.0)) isa CSRKickPoint
    @test csr_kick_coef(electron, ForwardDiff.Dual(200.0, 1.0), 0.01) isa ForwardDiff.Dual
  end

  @testset "S2 track_csr! entry checks" begin
    pq = tc_pq(200)
    bl = Beamline([Drift(L = 0.3), SBend(L = 0.5, g_ref = 0.4, Kn0 = 0.4)]; p_over_q_ref = pq, species_ref = TC_E)
    b = tc_line_bunch(1e-3, 1e-9, 101, pq)
    v0 = copy(b.coords.v)

    @test_throws ErrorException track_csr!(b, bl; params = CSRParams(n_bin = 40))
    @test b.coords.v == v0
    err = try
      track_csr!(b, bl; params = CSRParams(n_bin = 40))
      nothing
    catch e
      e
    end
    @test err isa ErrorException && occursin("ds_step", sprint(showerror, err))
    @test err isa ErrorException && occursin("must be > 0", sprint(showerror, err))

    @test_throws ErrorException track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.05),
                                           centroid_start = zeros(5))
    @test b.coords.v == v0
    err = try
      track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.05), centroid_start = zeros(5))
      nothing
    catch e
      e
    end
    @test err isa ErrorException && occursin("centroid_start", sprint(showerror, err))

    @test track_csr!(b, bl; params = CSRParams(n_bin = 40, ds_step = 0.05), centroid_start = zeros(6)) === b
  end

  @testset "S10 last SaganCavity slice" begin
    ele = lc_in_branch(lc_cavity(L = 0.3, tracking_method = SaganCavity()))
    @test 0.2 + 0.1 != 0.3
    sl = CSR.slice_element(ele, 0.2, 0.1, false, true)
    @test sl.s1 == 0.3
    @test sl.s1 == sl.L
    sm = CSR.slice_element(ele, 0.2, 0.1, false, false)
    @test sm.s1 == 0.2 + 0.1
  end

  @testset "S10 reference range test" begin
    z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
    params = CSRParams(n_bin = 40)
    bins = csr_bin_particles(z, q, alive, params)
    kick = zeros(40)
    for zf in (bins.z_center[end] + 10bins.dz, bins.z_center[1] - 10bins.dz)
      @test_throws ErrorException CSR.csr_apply_kicks_reference!([0.0], [zf], [true], bins, kick)
    end
  end

  @testset "S7 SaganSlice keywords" begin
    ele = lc_in_branch(lc_cavity(L = 0.3, tracking_method = SaganCavity()))
    sl = CSR.slice_element(ele, 0.2, 0.1, false, true)
    sl2 = TC_EXT.SaganSlice(; (f => getfield(sl, f) for f in fieldnames(typeof(sl)))...)
    @test all(f -> getfield(sl, f) == getfield(sl2, f), filter(f -> f ∉ (:ele, :tm), fieldnames(typeof(sl))))
  end

  @testset "S9 no boxed state" begin
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    params = CSRParams(n_bin = 40, ds_step = 0.05, n_shield_images = 2, beam_chamber_height = 0.02)
    @test (@inferred CSR.geometry_active_indices(geom, 3, 0.1, 200.0, params, 1e-4)) isa Vector{Int}
    @test CSR.widen_source_range(3, 3, false, false, 0, 4) == (1, 3, true, false)
    @test CSR.widen_source_range(3, 3, false, false, 5, 4) == (3, 4, false, true)
    @test CSR.widen_source_range(3, 3, false, false, 2, 4) == (2, 3, false, false)
  end

  @testset "D.10 slice_end types" begin
    @test TC_EXT.slice_end(0.3, 0.2, 0.1, true) === 0.3
    @test TC_EXT.slice_end(0.3, 0.2, 0.1, false) === 0.2 + 0.1
    r = @inferred TC_EXT.slice_end(0.3, ForwardDiff.Dual(0.2, 1.0), 0.1, true)
    @test r isa ForwardDiff.Dual && ForwardDiff.value(r) == 0.3
    r2 = @inferred TC_EXT.slice_end(ForwardDiff.Dual(0.3, 1.0), 0.2, 0.1, false)
    @test r2 isa ForwardDiff.Dual && ForwardDiff.value(r2) == 0.2 + 0.1
  end

  @testset "Final review fixes" begin
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    z, q, alive = kernel_test_bunch(1e-3, 1e-9, 400)
    params = CSRParams(n_bin = 40, ds_step = 0.05)
    e = Species("electron")
    @test_throws "equal lengths" csr_bin_particles(z, q[1:end-1], alive, params)
    @test_throws "equal lengths" csr_kick!(zeros(length(z) - 1), z, q, alive, geom, 3, 0.1, 200.0, e, params)
    z2 = copy(z)
    z2[findfirst(alive)] = NaN
    @test_throws "non-finite" csr_bin_particles(z2, q, alive, params)
    z2[findfirst(alive)] = Inf
    @test_throws "non-finite" csr_bin_particles(z2, q, alive, params)
    z3 = copy(z)
    a3 = copy(alive)
    z3[1] = NaN
    a3[1] = false
    @test csr_bin_particles(z3, q, a3, params) isa CSR.CSRBins

    ele = Branch([Marker(E_ref = 1e5, species_ref = LC_E), lc_cavity(L = 0.3, tracking_method = SaganCavity(),
                  dE_ref = 2LC_E0)]).beamlines[2].line[1]
    @test_throws "not above the rest energy" CSR.slice_element(ele, 0.0, 0.1, true, false)
    sl = CSR.slice_element(lc_in_branch(lc_cavity(L = 0.3, tracking_method = SaganCavity())), 0.2, 0.1, false, true)
    @test sl.q_gradient isa Float64
  end

  # Keep last: Descriptor(...) changes GTPSA.desc_current.
  @testset "TPS cases" begin
    d = Descriptor(1, 1)
    x = @vars(d)
    geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
    @test_throws "gamma must be > 1" CSRKickPoint(geom, 3, 0.1, 1.0 + x[1])
    @test CSRKickPoint(geom, 3, 0.1, 200.0 + x[1]) isa CSRKickPoint
    f(x, y, s, t, pp = nothing) = (s, t)
    spt = TC_EXT.shifted_to_slice(f, (; species_ref = TC_E, p_over_q_ref = tc_pq(200) + x[1]), 0.25)
    @test typeof(spt.s0) == typeof(spt.t0)
    @test scalar(spt.s0) == 0.25
    @test (@inferred TC_EXT.slice_end(0.3, 0.2 + x[1], 0.1, true)) isa TPS
    @test scalar(TC_EXT.slice_end(0.3, 0.2 + x[1], 0.1, true)) == 0.3
  end
end
