# Convolution kernels: kick table with the charge-density gradient
# (free space) or the bin charge (images). Bmad (csr_bin_kicks)
# evaluates the same sums with a zero-padded FFT; the direct sum is
# exact and works for any element type.

"""
    csr_convolve_kernel!(kick, I_int, edge_grad, coef)

One work item per bin `i`: `kick[i] = coef Σ_{k=1}^{i} I_int[k]
edge_grad[i+1-k]` (causal sum). Launched by `csr_convolve!`.

# Arguments
- `kick`: kick per bin (overwritten), length `n_bin`.
- `I_int`: integrated kick integral per bin, `I_int` of
  `csr_wake_table`.
- `edge_grad`: charge-density gradient at the left bin edges,
  `bins.edge_dcharge_density_dz` (`CSRBins`).
- `coef`: kick coefficient, from `csr_kick_coef`.
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
Launched by `csr_image_convolve!`.

# Arguments
- `kick`: kick per bin (modified: the image kick is added),
  `n_bin = length(kick)`.
- `image`: image-charge table of one layer, length `2n_bin + 1`,
  from `csr_image_table`.
- `charge`: bin charges [C], `bins.charge` (`CSRBins`).
- `coef`: kick coefficient, from `csr_kick_coef`.
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
