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
