# Kick tables per lag (host side): free-space wake table and image-charge tables.

"""
    table_type(geom, kp, dz)

Number type of the kick tables: promoted from the geometry, the kick point and the bin width.
"""
table_type(geom::CSRGeometry, kp::CSRKickPoint, dz) = float(promote_type(number_type(geom), number_type(kp), typeof(dz)))

"""
    csr_wake_table(geom, kp, dz, n_bin, backend; n_block, all_bin_quadrature = false) -> (I_csr, I_int, s_source)

Kick integral `I_csr`, its integral over each bin `I_int`, and the reference s of the source point, for lags
`i * dz`, `i = 1:n_bin`, at kick point `kp`. The lags are split into `n_block` blocks (one work item and
workgroup each); a block reuses the previous lag's source element as the start of the next search.
Default: 4 blocks per thread; always one block for TPS tables (`thread_safe_number`). CPU backend only: `CSRGeometry`
is not isbits (its element vector), and the source search raises errors and recurses (adaptive quadrature). The tables
are in `table_type(geom, kp, dz)` (`Float64`, or e.g. `Dual`s or TPS for derivatives with respect to the geometry,
`s_kick` or `gamma`; the sources are then searched in primal values, `primal_search`).

`I_int` for bins `i ≥ 2`: trapezoid in `I_csr`, or with `all_bin_quadrature` Simpson in `u^(1/3)` with 8 nodes per
bin (`I_csr_quadrature`; 8 more source solves per bin, in blocks as above).
"""
function csr_wake_table(geom::CSRGeometry, kp::CSRKickPoint, dz, n_bin, backend;
                        n_block = min(n_bin, 4 * Threads.nthreads()), all_bin_quadrature::Bool = false)
  T = table_type(geom, kp, dz)
  thread_safe_number(T) || (n_block = 1)
  I_csr = KernelAbstractions.zeros(backend, T, n_bin)
  I_int = KernelAbstractions.zeros(backend, T, n_bin)
  s_source = KernelAbstractions.zeros(backend, T, n_bin)
  n_lag = cld(n_bin, n_block)
  search = primal_search(geom, kp, T)
  # Workgroup size passed at launch: a static size (`csr_wake_kernel!(backend, 1)`) is not inferred.
  csr_wake_kernel!(backend)(I_csr, I_int, s_source, geom, kp, dz, n_lag, search; ndrange = cld(n_bin, n_lag), workgroupsize = 1)
  KernelAbstractions.synchronize(backend)
  if n_bin > 1
    if all_bin_quadrature
      n_q = cld(n_bin - 1, min(n_bin - 1, n_block))
      csr_bin_quadrature_kernel!(backend)(I_int, geom, kp, dz, n_q, 8, search; ndrange = cld(n_bin - 1, n_q), workgroupsize = 1)
    else
      csr_trapezoid_kernel!(backend)(I_int, I_csr, dz; ndrange = n_bin - 1, workgroupsize = cpu_workgroupsize(n_bin - 1, T))
    end
    KernelAbstractions.synchronize(backend)
  end
  return I_csr, I_int, s_source
end

"""
    csr_image_table(geom, kp, dz, n_bin, backend; n_block, bin_integrate = false, rtol = 1e-6) -> (image, s_source)

Image-charge kick (`image_charge_kick`) and reference s of the source point for lags `i * dz`,
`i = -n_bin:n_bin` (index `i + n_bin + 1`), at kick point `kp` with `kp.y_source ≠ 0`. Blocks of lags as in
`csr_wake_table`. CPU backend only.

`bin_integrate = false` samples the kernel at the lags (Bmad). `bin_integrate = true` integrates it against the hat
function of width `2dz` centred on each lag (the charge density interpolated linearly between bin centres), by
adaptive Simpson with absolute tolerance `rtol * dz * max|K|` per interval (`image_hat_tol`), down to lag widths of
`0.01 y/γ`. The kernel's structure can be much finer than a bin (`~y/γ` on a drift, `~Ls³/24ρ²` in a bend), which
point sampling does not resolve (docs, "Physics decisions", D20). `s_source` is at the lags in both cases.
"""
function csr_image_table(geom::CSRGeometry, kp::CSRKickPoint, dz, n_bin, backend;
                         n_block = min(2n_bin + 1, 4 * Threads.nthreads()), bin_integrate::Bool = false,
                         rtol = 1e-6)
  n = 2n_bin + 1
  T = table_type(geom, kp, dz)
  thread_safe_number(T) || (n_block = 1)
  image = KernelAbstractions.zeros(backend, T, n)
  s_source = KernelAbstractions.zeros(backend, T, n)
  n_lag = cld(n, n_block)
  search = primal_search(geom, kp, T)
  csr_image_kernel!(backend)(image, s_source, geom, kp, dz, n_lag, search; ndrange = cld(n, n_lag), workgroupsize = 1)
  KernelAbstractions.synchronize(backend)
  bin_integrate || return image, s_source

  u_min = image_u_min(kp)
  tol = search === nothing ? image_hat_tol(geom, kp, image, dz, rtol, u_min) :
                             image_hat_tol(search.geom, search.kp, primal.(image), primal(dz), rtol, u_min)
  rise = KernelAbstractions.zeros(backend, T, n + 1)
  fall = KernelAbstractions.zeros(backend, T, n + 1)
  n_int = cld(n + 1, min(n + 1, n_block))
  csr_image_hat_kernel!(backend)(rise, fall, geom, kp, dz, tol, u_min, n_int, search; ndrange = cld(n + 1, n_int),
                                 workgroupsize = 1)
  KernelAbstractions.synchronize(backend)
  csr_image_hat_combine_kernel!(backend)(image, rise, fall, dz; ndrange = n, workgroupsize = cpu_workgroupsize(n, T))
  KernelAbstractions.synchronize(backend)
  return image, s_source
end
