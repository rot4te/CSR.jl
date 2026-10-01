# Causal convolution of the integrated kick table with the edge charge-density gradient.
# Bmad (csr_bin_kicks) evaluates the same sum with a zero-padded FFT; the direct sum is exact and works
# for any element type.

"""
    csr_convolve!(kick, I_int, edge_grad, coef)

`kick[i] = coef * Σ_{k=1}^{i} I_int[k] * edge_grad[i+1-k]`.
"""
function csr_convolve!(kick::AbstractVector, I_int::AbstractVector, edge_grad::AbstractVector, coef)
  n = length(kick)
  for i in 1:n
    acc = zero(eltype(kick))
    for k in 1:i
      acc += I_int[k] * edge_grad[i+1-k]
    end
    kick[i] = coef * acc
  end
  return kick
end
