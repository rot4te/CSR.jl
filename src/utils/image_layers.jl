# Image layers: averaging weights of the layer sum and tolerances of the hat-weighted image integral.

# Weight of image layer `ns` in the averaged layer sum (see `image_sum_order`); 1 with `bmad_compat`.
"""
    image_layer_weight(ns, params) -> Float64

Weight of image layer `ns` in the averaged layer sum (`CSRParams` `image_sum_order`): with `M = n_shield_images` and
`k = min(image_sum_order, M)`, `2^-k Σ_{j ≥ ns-M+k} C(k, j)`, which is 1 for `ns ≤ M - k`. The weighted sum equals the
binomial average of the partial sums `S_{M-k} … S_M`. Always 1 with `bmad_compat` (Bmad's plain sum).
"""
function image_layer_weight(ns, params::CSRParams)
  params.bmad_compat && return 1.0
  M = params.n_shield_images
  k = min(params.image_sum_order, M)
  ns <= M - k && return 1.0
  return sum(binomial(k, j) for j in ns-(M-k):k) / 2^k
end

"""
    image_u_min(kp)

Finest lag scale resolved by the hat-weighted image table: 1% of y/γ, the width of the drift kernel's peak.
"""
image_u_min(kp::CSRKickPoint) = 0.01 * abs(primal(kp.y_source)) / primal(kp.gamma)

"""
    image_hat_tol(geom, kp, image, dz, rtol, u_min)

Absolute tolerance per lag interval for the hat-weighted image table: `rtol * dz` times the largest |K| at the
lags (`image`, point-sampled) and at the nodes ±dz/2^j > u_min. The lags alone miss the peak near u = 0 when
dz ≫ y/γ, which would give a tolerance below the kernel's round-off noise.
"""
function image_hat_tol(geom::CSRGeometry, kp::CSRKickPoint, image, dz, rtol, u_min)
  k = CSRKick1(geom, kp)
  K = maximum(abs, image)
  x = dz / 2
  while x > u_min
    k, Kp = image_kernel_at(k, x, geom, kp)
    k, Km = image_kernel_at(k, -x, geom, kp)
    K = max(K, abs(Kp), abs(Km))
    x /= 2
  end
  return rtol * primal(K) * dz
end
