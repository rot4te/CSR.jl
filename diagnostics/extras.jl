using CSR, Test, StaticArrays, AbstractFFTs, BeamTracking, Beamlines, ChainRulesCore, ForwardDiff, GTPSA
include("/Users/alexcoxe/.julia/dev/CSR.jl/test/utils.jl")
function kernel_test_bunch(sigma, Q, n)
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n); alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end
electron = Species("electron")
L3 = [0.2, 0.3, 0.2]; g3 = [0.0, 1.5, 0.0]
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
n = 400
z, q, alive = kernel_test_bunch(1e-3, 1e-9, n); pz0 = zeros(n)
p = CSRParams(n_bin = 40, ds_step = 0.1)
geom0 = CSRGeometry(L3, g3, c3)
errmsg(f) = try f(); "NOERROR" catch err; sprint(showerror, err) end
tmpl(fname, name) = "$fname: TPS in argument `$name` is not in GTPSA's current Descriptor (GTPSA.desc_current). CSR.jl creates TPS constants in the current Descriptor; use one Descriptor for all TPS inputs and make it current (the last Descriptor created is current)."

@testset "extras" begin
  d1 = Descriptor(2, 2)
  x = @vars(d1, dynamic = true)
  xs = @vars(d1)
  # same-shape second descriptor: is it a different Descriptor?
  d1b = Descriptor(2, 2)
  println("d1 == d1b: ", d1 == d1b, "; current==d1b: ", GTPSA.desc_current == d1b)
  geom1 = CSRGeometry(L3, [0.0, 1.5 + x[1], 0.0], c3)
  d2 = Descriptor(3, 2)
  y = @vars(d2, dynamic = true)

  @testset "exact message" begin
    @test errmsg(() -> csr_kick_pz(pz0, z, q .* (1 + x[2]), alive, geom0, 2, 0.17, 200.0, electron, p)) == tmpl("csr_kick_pz", "q")
    @test errmsg(() -> csr_bin_particles(z, q .* (1 + x[2]), alive, p)) == tmpl("csr_bin_particles", "q")
    @test errmsg(() -> csr_bin_particles(z .+ x[1], q, alive, p)) == tmpl("csr_bin_particles", "z")
    @test errmsg(() -> planar_survey(L3, [0.0, 1.5 + x[1], 0.0])) == tmpl("planar_survey", "g")
    @test errmsg(() -> planar_survey([0.2 + x[1], 0.3, 0.2], g3)) == tmpl("planar_survey", "L")
    @test errmsg(() -> planar_survey(L3, g3; start = FloorPosition([x[1], 0.0, 0.0], 0.0))) |> x -> occursin("`start`", x)
  end
  @testset "CSRGeometry arguments" begin
    @test occursin("`L`", errmsg(() -> CSRGeometry([0.2 + x[1], 0.3, 0.2], g3, c3)))
    cxm = Matrix{typeof(x[1])}(c3 .+ 0.0 .* x[1])
    @test occursin("`centroid`", errmsg(() -> CSRGeometry(L3, g3, cxm)))
    @test occursin("`ref_start`", errmsg(() -> CSRGeometry(L3, g3, c3; ref_start = FloorPosition([x[1], 0.0, 0.0], 0.0))))
    ref_exit = planar_survey(L3, g3)
    @test occursin("`ref_exit`", errmsg(() -> CSRGeometry(L3, g3, [FloorPosition([x[1], 0.0, 0.0], 0.0) for _ in 1:3], c3)))
    @test occursin("`ref_start`", errmsg(() -> CSRGeometry(L3, g3, ref_exit, c3; ref_start = FloorPosition([x[1], 0.0, 0.0], 0.0))))
    @test occursin("`g`", errmsg(() -> CSRGeometry(L3, [0.0, 1.5 + x[1], 0.0], ref_exit, c3)))
    # check precedes existing arg checks: wrong length plus mismatch -> descriptor error first
    @test occursin("GTPSA's current", errmsg(() -> CSRGeometry(L3, [1.5 + x[1]], ref_exit, c3)))
    @test occursin("GTPSA's current", errmsg(() -> planar_survey(L3, [1.5 + x[1]])))
    @test occursin("must have the same length", errmsg(() -> planar_survey(L3, [1.5])))
  end
  @testset "kick args" begin
    for (nm, f) in [
        ("z", () -> csr_kick_pz(pz0, z .+ x[1], q, alive, geom0, 2, 0.17, 200.0, electron, p)),
        ("pz", () -> csr_kick_pz(pz0 .+ x[1], z, q, alive, geom0, 2, 0.17, 200.0, electron, p)),
        ("gamma", () -> csr_kick_pz(pz0, z, q, alive, geom0, 2, 0.17, 200.0 + x[1], electron, p)),
        ("kick_factor", () -> csr_kick_pz(pz0, z, q, alive, geom0, 2, 0.17, 200.0, electron, p; kick_factor = 1 + x[1])),
        ("pz_scale", () -> csr_kick_pz(pz0, z, q, alive, geom0, 2, 0.17, 200.0, electron, p; pz_scale = 1 + x[1])),
        ("pz", () -> csr_kick!(pz0 .+ x[1], z, q, alive, geom0, 2, 0.17, 200.0, electron, p)),
        ("z", () -> csr_kick!(copy(pz0), z .+ x[1], q, alive, geom0, 2, 0.17, 200.0, electron, p)),
        ("q", () -> csr_kick!(copy(pz0), z, q .* (1 + x[1]), alive, geom0, 2, 0.17, 200.0, electron, p)),
        ("geom", () -> csr_kick!(copy(pz0), z, q, alive, geom1, 2, 0.17, 200.0, electron, p)),
        ("gamma", () -> csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17, 200.0 + x[1], electron, p)),
        ("kick_factor", () -> csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17, 200.0, electron, p; kick_factor = 1 + x[1])),
        ("pz_scale", () -> csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17, 200.0, electron, p; pz_scale = 1 + x[1])),
        ("y_source", () -> CSRKickPoint(geom0, 2, 0.17, 200.0; y_source = x[1])),
        ("kick_factor", () -> CSRKickPoint(geom0, 2, 0.17, 200.0; kick_factor = 1 + x[1])),
        ("s_kick", () -> CSRKickPoint(geom0, 2, 0.17 + x[1], 200.0)),
        ("geom", () -> CSRKickPoint(geom1, 2, 0.17, 200.0)),
      ]
      @test errmsg(f) |> m -> occursin("`$nm`", m) && occursin("GTPSA's current Descriptor", m)
    end
    # ordering: descriptor mismatch first, 6.E second
    @test occursin("GTPSA's current", errmsg(() -> csr_kick!(zeros(n), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, electron, p)))
    # fname in message
    @test startswith(errmsg(() -> csr_kick!(copy(pz0), z, q, alive, geom0, 2, 0.17 + x[1], 200.0, electron, p)), "csr_kick!: TPS in argument `s_kick`")
  end
  @testset "no-error paths" begin
    # test_particles not checked; static not checked; ComplexTPS not checked
    @test !CSR.descriptor_mismatch(xs[1])
    @test !CSR.descriptor_mismatch([xs[1]])
    @test CSR.descriptor_mismatch([x[1], x[2]])
    @test !CSR.descriptor_mismatch(y[1])
    @test !CSR.descriptor_mismatch(ComplexF64(1.0))
    @test !CSR.descriptor_mismatch(1.0 + 0im + 0 * GTPSA.ComplexTPS64(xs[1]))
    @test !CSR.descriptor_mismatch(Float32[1])
    @test !CSR.descriptor_mismatch(ForwardDiff.Dual{Nothing}(1.0, 1.0))
    @test !CSR.descriptor_mismatch(FloorPosition())
    @test CSR.descriptor_mismatch(FloorPosition([x[1], 0.0, 0.0], 0.0))
    @test !CSR.descriptor_mismatch(planar_survey(L3, [0.0, 1.5 + y[1], 0.0]))
  end
  @testset "dynamic same descriptor works with csr_kick!" begin
    # d2 current; pz dynamic TPS in d2
    pzt = [zero(y[1]) for _ in 1:n]
    r = csr_kick!(pzt, z, q .* (1 + y[1]), alive, geom0, 2, 0.17, 200.0, electron, p)
    @test r isa CSRStepResult
    ref = csr_kick_pz(pz0, z, q .* (1 + y[1]), alive, geom0, 2, 0.17, 200.0, electron, p)
    @test scalar.(pzt) == scalar.(ref)
    @test GTPSA.jacobian(pzt) == GTPSA.jacobian(ref)
  end
  @testset "6.E message and variants" begin
    geom0b = CSRGeometry(L3, g3, c3)
    m = errmsg(() -> csr_kick!(zeros(n), z, q, alive, geom0b, 2, 0.17, ForwardDiff.Dual{Nothing}(200.0, 1.0), electron, p))
    println(m)
    Tk = typeof(ForwardDiff.Dual{Nothing}(200.0, 1.0))
    @test m == "csr_kick!: pz has element type Float64, which cannot hold the kick in number type $Tk; use csr_kick_pz, or a pz with element type $(promote_type(Float64, Tk))."
    @test csr_kick!(zeros(BigFloat, n), z, q, alive, geom0b, 2, 0.17, 200.0, electron, p) isa CSRStepResult
    @test csr_kick!(zeros(n), z, q, alive, geom0b, 2, 0.17, 200.0, electron, p; pz_scale = 2.0) isa CSRStepResult
    @test occursin("cannot hold", errmsg(() -> csr_kick!(zeros(Float32, n), z, q, alive, geom0b, 2, 0.17, 200.0, electron, p; pz_scale = ForwardDiff.Dual{Nothing}(1.0, 1.0))))
    @test occursin("cannot hold", errmsg(() -> csr_kick!(zeros(n), z, q, alive, geom0b, 2, 0.17, 200.0, electron, p; kick_factor = ForwardDiff.Dual{Nothing}(1.0, 1.0))))
    # Dual geometry
    gd = CSRGeometry(L3, [0.0, ForwardDiff.Dual{Nothing}(1.5, 1.0), 0.0], c3)
    @test occursin("cannot hold", errmsg(() -> csr_kick!(zeros(n), z, q, alive, gd, 2, 0.17, 200.0, electron, p)))
    # Dual pz works
    pzd = zeros(typeof(ForwardDiff.Dual{Nothing}(1.0, 1.0)), n)
    @test csr_kick!(pzd, z, q, alive, gd, 2, 0.17, 200.0, electron, p) isa CSRStepResult
    # Float64 pz from dual-free test particles path unaffected
    # test_particles with dual do not matter
    # Integer pz? eltype Int is not AbstractFloat -> no 6.E error (spec)
  end
end
