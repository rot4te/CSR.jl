using CSR
include(joinpath(homedir(), ".julia/dev/CSR.jl/test/utils.jl"))
@eval CSR begin
# K(u) at lag u
function image_kernel_at!(k, u, geom, kp)
  k.dz_particles = u
  s_source_calc!(k, geom, kp)
  return image_charge_kick(k, geom, kp)
end
# Adaptive Simpson of (∫K, ∫K t) over [a, b], t = (u - u0)/dz. fa, fm, fb at a, mid, b.
function image_simpson!(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, depth, cnt)
  m = (a + b) / 2
  h = b - a
  fl = image_kernel_at!(k, (a + m) / 2, geom, kp); fr = image_kernel_at!(k, (m + b) / 2, geom, kp); cnt[] += 2
  t(u) = (u - u0) / dz
  w = SVector(1.0, 0.0)
  g(u, f) = SVector(f, f * t(u))
  S1 = h / 6 * (g(a, fa) + 4g(m, fm) + g(b, fb))
  S2 = h / 12 * (g(a, fa) + 4g((a + m) / 2, fl) + 2g(m, fm) + 4g((m + b) / 2, fr) + g(b, fb))
  if depth <= 0 || maximum(abs, S2 - S1) <= 15tol
    return S2 + (S2 - S1) / 15
  end
  return image_simpson!(k, geom, kp, a, m, fa, fl, fm, u0, dz, tol / 2, depth - 1, cnt) +
         image_simpson!(k, geom, kp, m, b, fm, fr, fb, u0, dz, tol / 2, depth - 1, cnt)
end
function image_interval!(k, geom, kp, a, b, u0, dz, tol, cnt)
  fa = image_kernel_at!(k, a, geom, kp); fm = image_kernel_at!(k, (a + b) / 2, geom, kp)
  fb = image_kernel_at!(k, b, geom, kp); cnt[] += 3
  return image_simpson!(k, geom, kp, a, b, fa, fm, fb, u0, dz, tol, 40, cnt)
end
# Hat-weighted table, lags -n_bin:n_bin
function image_table_hat(geom, kp, dz, n_bin; rtol = 1e-6)
  k = CSRKick1(); k.ix_ele_source = kp.ix_ele_kick
  Kmax = maximum(abs(image_kernel_at!(k, m * dz, geom, kp)) for m in -n_bin-1:n_bin+1)
  tol = rtol * Kmax * dz
  u_min = 0.01 * kp.y_source / kp.gamma
  cnt = Ref(0)
  A = zeros(2n_bin + 2); B = zeros(2n_bin + 2)   # interval m = -n_bin-1:n_bin at index m + n_bin + 2
  for m in -n_bin-1:n_bin
    a, b = m * dz, (m + 1) * dz
    # graded split towards u = 0 for the two intervals touching it
    cuts = [a, b]
    if m == 0 || m == -1
      x = dz / 2
      while x > u_min
        push!(cuts, m == 0 ? x : -x); x /= 2
      end
      push!(cuts, 0.0); sort!(unique!(cuts))
    end
    acc = SVector(0.0, 0.0)
    for c in 1:length(cuts)-1
      acc += image_interval!(k, geom, kp, cuts[c], cuts[c+1], a, dz, tol * (cuts[c+1] - cuts[c]) / dz, cnt)
    end
    i = m + n_bin + 2
    A[i] = acc[2]; B[i] = acc[1] - acc[2]   # rising weight t, falling 1 - t
  end
  image = [(A[l + n_bin + 1] + B[l + n_bin + 2]) / dz for l in -n_bin:n_bin]
  return image, cnt[]
end
end

electron = Species("electron")
eloss(r) = sum(r.kick_per_meter .* r.charge_per_meter) * (r.z_center[2] - r.z_center[1])
N = 40001
zb, qb, aliveb = gaussian_line_bunch(1e-3, 1e-9, N)
geom = CSRGeometry(circle_lattice(10.0, 0.1, 80)...)
gamma = 1e4
function run(p; hat)
  st = CSR.csr_binned_kick(zb, qb, aliveb, geom, 80, 0.05, gamma, electron,
                           CSRParams(n_bin = p.n_bin, ds_step = 0.1, all_bin_quadrature = true))
  kick = copy(st.kick); n = 0
  for ns in 1:p.n_shield_images
    kp = CSR.CSRKickPoint(geom, 80, 0.05, gamma; y_source = ns * p.beam_chamber_height, kick_factor = (-1)^ns * 2)
    if hat
      img, c = CSR.image_table_hat(geom, kp, st.bins.dz, p.n_bin); n += c
    else
      img, _ = CSR.csr_image_table(geom, kp, st.bins.dz, p.n_bin, CSR.CPU()); n += 2p.n_bin + 1
    end
    CSR.csr_image_convolve!(kick, img, st.bins.charge, st.coef)
  end
  return sum(kick .* st.bins.charge), n
end
for h in (0.003, 0.01), nb in (240, 960)
  r0, _ = run((n_bin = nb, n_shield_images = 0, beam_chamber_height = 0.0); hat = false)
  for nl in (15, 16, 31, 32, 63, 64)
    p = (n_bin = nb, n_shield_images = nl, beam_chamber_height = h)
    rp, _ = run(p; hat = false); rh, _ = run(p; hat = true)
    println("h=$h n_bin=$nb layers=$nl  point: ", round(rp / r0, sigdigits = 4), "  hat: ", round(rh / r0, sigdigits = 4))
  end
end
