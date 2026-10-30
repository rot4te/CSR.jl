# Phase 6: derivatives of one CSR kick with respect to the particles (pz, z, q). ForwardDiff through csr_kick_pz,
# the rrule (explicit adjoint kernels) against ForwardDiff J'ȳ, and both against central finite differences.
# The bin grid is held constant, so finite differences in z match only for particles that are not the extreme
# live particles (which set the grid).

@testset "Automatic differentiation" begin
  electron = Species("electron")
  L3 = [0.2, 0.3, 0.2]
  g3 = [0.0, 1.5, 0.0]
  c3 = zeros(4, 6)
  c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
  c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
  n = 400
  z, q, alive = kernel_test_bunch(1e-3, 1e-9, n)
  pz = 1e-4 .* sin.(1:n)
  dead = findall(.!alive)
  i_zmin = argmin(ifelse.(alive, z, Inf))
  i_zmax = argmax(ifelse.(alive, z, -Inf))
  interior = filter(i -> alive[i] && i ∉ (i_zmin, i_zmax), [97, 151, 200, 263, 333])

  cases = (
    ("kick in bend", 2, 0.17, 0, false, false),
    ("kick in bend, all-bin quadrature", 2, 0.17, 0, false, true),
    ("kick in bend, bmad_compat", 2, 0.17, 0, true, false),
    ("kick at element entrance, images", 3, 0.0, 2, false, false),
    ("kick in bend, images, all-bin quadrature", 2, 0.17, 2, false, true),
    ("kick in bend, images, bmad_compat", 2, 0.17, 2, true, false),
  )
  @testset "$name" for (name, ix, s_kick, n_img, bc, abq) in cases
    p = CSRParams(n_bin = 40, n_shield_images = n_img, beam_chamber_height = n_img > 0 ? 0.01 : 0.0,
                  bmad_compat = bc, all_bin_quadrature = abq)
    geom = CSRGeometry(L3, g3, c3; bmad_compat = bc)
    ds = L3[ix] / max(1, round(Int, Beamlines.scalarize_all(L3[ix]) / 0.1, RoundNearestTiesAway))
    kw = (kick_factor = 0.5, pz_scale = 1.3)
    f(pp, zz, qq) = csr_kick_pz(pp, zz, qq, alive, geom, ix, ds, s_kick, 200.0, electron, p; kw...)

    # Non-mutating version == csr_kick!
    pz0 = copy(pz)
    y = f(pz, z, q)
    @test pz == pz0
    pz1 = copy(pz)
    csr_kick!(pz1, z, q, alive, geom, ix, ds, s_kick, 200.0, electron, p; kw...)
    @test y == pz1
    @test maximum(abs, y .- pz) > 0

    # ForwardDiff Jacobians
    Jpz = ForwardDiff.jacobian(pp -> f(pp, z, q), pz)
    Jz = ForwardDiff.jacobian(zz -> f(pz, zz, q), z)
    Jq = ForwardDiff.jacobian(qq -> f(pz, z, qq), q)
    @test Jpz == ((1:n) .== (1:n)')
    @test all(iszero, Jz[dead, :]) && all(iszero, Jz[:, dead])
    @test all(iszero, Jq[dead, :]) && all(iszero, Jq[:, dead])
    @test maximum(abs, Jz) > 0 && maximum(abs, Jq) > 0

    # Central finite differences on core particles (tail charges are ~1e-10 of the core, below the rounding and
    # atomic-order noise of the kick). The deposit is piecewise quadratic in z (h_z ≪ dz) and the kick linear in q
    # (h_q = q is exact). pz = 0 avoids cancellation against pz.
    pz_0 = zero(pz)
    for i in interior
      hz = 1e-8
      zp = copy(z); zp[i] += hz
      zm = copy(z); zm[i] -= hz
      @test maxrel((f(pz_0, zp, q) .- f(pz_0, zm, q)) ./ 2hz, Jz[:, i]) < 1e-7
      qp = copy(q); qp[i] *= 2
      qm = copy(q); qm[i] = 0
      @test maxrel((f(pz_0, z, qp) .- f(pz_0, z, qm)) ./ 2q[i], Jq[:, i]) < 1e-9
    end
    # The grid depends on z only, so ∂/∂q is exact for the extreme particles too (given core-sized charges)
    qx = copy(q); qx[[i_zmin, i_zmax]] .= maximum(q)
    Jqx = ForwardDiff.jacobian(qq -> f(pz_0, z, qq), qx)
    for i in (i_zmin, i_zmax)
      qp = copy(qx); qp[i] *= 2
      qm = copy(qx); qm[i] = 0
      @test maxrel((f(pz_0, z, qp) .- f(pz_0, z, qm)) ./ 2qx[i], Jqx[:, i]) < 1e-9
    end

    # rrule pullback == ForwardDiff J'ȳ
    y2, back = ChainRulesCore.rrule(csr_kick_pz, pz, z, q, alive, geom, ix, ds, s_kick, 200.0, electron, p; kw...)
    @test y2 == y
    for seed in 1:2
      ybar = sin.(seed .* (1:n) .+ 0.3)
      t = back(ybar)
      @test length(t) == 12
      @test t[2] == ybar
      @test maxrel(t[3], Jz' * ybar) < 1e-12
      @test maxrel(t[4], Jq' * ybar) < 1e-12
      @test all(iszero, t[3][dead]) && all(iszero, t[4][dead])
      @test all(x -> x isa NoTangent, t[[1, 5, 7, 11, 12]])
      @test t[6] isa Tangent{typeof(geom)} && t[8] isa Real && t[9] isa Real && t[10] isa Real   # checked in ad_rrule_test.jl
    end
    # Thunked and zero cotangents
    ybar = cos.(1:n)
    @test back(@thunk(ybar))[3] == back(ybar)[3]
    tz = back(ZeroTangent())
    @test all(x -> x isa AbstractZero, tz)
  end

  @testset "Dual derivative of the charge in q; grid constant" begin
    p = CSRParams(n_bin = 40)
    b = csr_bin_particles(z, q, alive, p)
    bd = csr_bin_particles(ForwardDiff.Dual.(z, 1.0), ForwardDiff.Dual.(q, 0.0), alive, p)
    @test bd.dz == b.dz && bd.z0_edge == b.z0_edge && bd.n_particle == b.n_particle
    @test ForwardDiff.value.(bd.charge) == b.charge
    # Moving all particles rigidly moves no charge in a fixed grid's sum
    @test abs(sum(ForwardDiff.partials.(bd.charge, 1))) < 1e-12 * sum(abs, q) / b.dz
  end
end
