using CSR: triangle_overlap

@testset "Binning" begin
  @testset "triangle_overlap" begin
    dz = 0.1
    for zp in (1.0, 1.03, 1.05, 1.099)
      @test sum(triangle_overlap(zp, 2dz, k * dz, (k + 1) * dz) for k in 0:40) ≈ 1 atol = 1e-14
      @test sum(triangle_overlap(zp, 3dz, k * dz, (k + 1) * dz) for k in 0:40) ≈ 1 atol = 1e-14
    end
    @test triangle_overlap(1.0, 0.2, 2.0, 3.0) == 0
  end

  @testset "csr_bin_particles" begin
    params = CSRParams(n_bin = 30, particle_bin_span = 2, ds_step = 0.1)
    z, q, alive = gaussian_line_bunch(1e-3, 1e-9, 1001)
    alive[1:10] .= false
    bins = csr_bin_particles(z, q, alive, params)
    @test sum(bins.charge) ≈ sum(q[alive]) rtol = 1e-12
    @test sum(bins.n_particle) ≈ count(alive) rtol = 1e-12
    @test bins.charge[1] == 0
    @test bins.charge[end] == 0
    @test bins.z_center ≈ bins.z0_edge .+ bins.dz / 2
    zl, zr = extrema(z[alive])
    @test bins.dz ≈ 1.0000001 * (zr - zl) / (30 - 2 - 3)
    @test (bins.z0_edge[1] + bins.z0_edge[end] + bins.dz) / 2 ≈ (zl + zr) / 2
    @test bins.edge_dcharge_density_dz[1] == 0
    @test bins.edge_dcharge_density_dz[5] ≈ (bins.charge[5] - bins.charge[4]) / bins.dz^2

    @test_throws ErrorException csr_bin_particles(zeros(5), ones(5), fill(true, 5), params)

    # Float32 particles keep a Float32 grid (GPUs without Float64, e.g. Metal)
    b32 = csr_bin_particles(Float32.(z), Float32.(q), alive, params)
    @test b32.dz isa Float32
    @test eltype(b32.z0_edge) == eltype(b32.charge) == Float32
    @test b32.z0_edge ≈ bins.z0_edge rtol = 1e-5
    @test sum(b32.charge) ≈ sum(q[alive]) rtol = 1e-5
  end

  @testset "CSRParams" begin
    @test_throws ErrorException CSRParams(n_bin = 5, particle_bin_span = 2)
    @test CSRParams(n_bin = 6, particle_bin_span = 2).n_bin == 6
  end
end
