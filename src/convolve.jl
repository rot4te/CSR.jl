# Causal convolution of the integrated kick table with the edge charge-density gradient.
# Bmad (csr_bin_kicks) evaluates the same sum with a zero-padded FFT; the direct sum is exact and works
# for any element type.

@kernel function csr_convolve_kernel!(kick, @Const(I_int), @Const(edge_grad), coef)
  i = @index(Global)
  acc = zero(eltype(kick))
  for k in 1:i
    @inbounds acc += I_int[k] * edge_grad[i+1-k]
  end
  @inbounds kick[i] = coef * acc
end

"""
    csr_convolve!(kick, I_int, edge_grad, coef)

`kick[i] = coef * Σ_{k=1}^{i} I_int[k] * edge_grad[i+1-k]`. One work item per bin, on the backend of `kick`.
"""
function csr_convolve!(kick::AbstractVector, I_int::AbstractVector, edge_grad::AbstractVector, coef)
  backend = get_backend(kick)
  csr_convolve_kernel!(backend)(kick, I_int, edge_grad, coef; ndrange = length(kick))
  KernelAbstractions.synchronize(backend)
  return kick
end

@kernel function csr_image_convolve_kernel!(kick, @Const(image), @Const(charge), coef)
  i = @index(Global)
  n_bin = length(kick)
  acc = zero(eltype(kick))
  for j in 1:n_bin
    @inbounds acc += image[i-j+n_bin+1] * charge[j]
  end
  @inbounds kick[i] += coef * acc
end

"""
    csr_image_convolve!(kick, image, charge, coef)

`kick[i] += coef * Σ_{j=1}^{n_bin} image(i - j) * charge[j]`, with the image kick for lag `l` stored at
`image[l + n_bin + 1]` (`csr_image_table`). Bmad's image-charge sum (all lags, bin charge rather than its
gradient). One work item per bin.
"""
function csr_image_convolve!(kick::AbstractVector, image::AbstractVector, charge::AbstractVector, coef)
  length(image) == 2length(kick) + 1 || error("csr_image_convolve!: image must have 2n_bin + 1 entries")
  backend = get_backend(kick)
  csr_image_convolve_kernel!(backend)(kick, image, charge, coef; ndrange = length(kick))
  KernelAbstractions.synchronize(backend)
  return kick
end
