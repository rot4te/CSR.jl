# Convolution kernels: kick table with the charge-density gradient
# (free space) or the bin charge (images). Bmad (csr_bin_kicks)
# evaluates the same sums with a zero-padded FFT; the direct sum is
# exact and works for any element type.

"""
    csr_convolve_kernel!(kick, I_int, edge_grad, coef)

One work item per bin `i`: `kick[i] = coef Σ_{k=1}^{i} I_int[k]
edge_grad[i+1-k]` (causal sum).
"""
@kernel function csr_convolve_kernel!(kick, @Const(I_int),
                                      @Const(edge_grad), coef)
  i = @index(Global)
  acc = zero(eltype(kick))
  for k in 1:i
    @inbounds acc += I_int[k] * edge_grad[i+1-k]
  end
  @inbounds kick[i] = coef * acc
end

"""
    csr_image_convolve_kernel!(kick, image, charge, coef)

One work item per bin `i`: `kick[i] += coef Σ_j image(i - j)
charge[j]`, lags `-n_bin:n_bin` stored at `image[l + n_bin + 1]`.
"""
@kernel function csr_image_convolve_kernel!(kick, @Const(image),
    @Const(charge), coef)
  i = @index(Global)
  n_bin = length(kick)
  acc = zero(eltype(kick))
  for j in 1:n_bin
    @inbounds acc += image[i-j+n_bin+1] * charge[j]
  end
  @inbounds kick[i] += coef * acc
end
