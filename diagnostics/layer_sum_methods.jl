# D24: summation methods for the image-layer series E/E0 = 1 + Σ a_n, from the first M terms, vs a reference from
# all N terms. Reads layer_terms_<h>_240_400.txt (layer_terms.jl 1024 400 <h>).
# Methods (all but Wynn are linear in a_n, i.e. fixed per-layer weights w_n):
#   plain      S_M
#   avg(k)     k-fold repeated averaging of consecutive partial sums S_{M-k} … S_M (binomial weights 2^-k C(k, j))
#   wynn       Wynn ε on S_1 … S_M (nonlinear; reference only)
using DelimitedFiles, Printf

partial(a) = cumsum(a)
# k-fold averaging of partial sums ending at M: Σ_j 2^-k C(k,j) S_{M-k+j}
avgsum(S, M, k) = sum(binomial(k, j) * S[M-k+j] for j in 0:k) / 2^k
# Per-layer weights of avg(k): w_n = 1 for n ≤ M-k, then the tail of the binomial CDF.
function avg_weights(M, k)
  w = ones(M)
  for n in M-k+1:M
    w[n] = sum(binomial(k, j) for j in (n - (M - k)):k) / 2^k
  end
  return w
end

function wynn(S)
  n = length(S)
  e0 = zeros(n + 1); e1 = copy(S)
  best = S[end]
  prev, cur = e0[1:n], e1
  for k in 1:n-1
    nxt = [prev[i+1] + 1 / (cur[i+1] - cur[i]) for i in 1:length(cur)-1]
    iseven(k) && !isempty(nxt) && all(isfinite, nxt) && (best = nxt[end])
    prev, cur = cur, nxt
  end
  return best
end

for h in (0.003, 0.01, 0.03)
  f = joinpath(@__DIR__, "layer_terms_$(h)_240_400.txt")
  isfile(f) || continue
  a = readdlm(f)[:, 2]
  N = length(a)
  S = 1 .+ partial(a)
  ref = avgsum(S, N, 8)
  @printf("h = %g: reference avg(8) at N = %d: %.6e  (avg(4): %.6e, avg(1): %.6e, plain: %.6e)\n", h, N, ref,
          avgsum(S, N, 4), avgsum(S, N, 1), S[N])
  println("     M   plain        avg(1)       avg(2)       avg(4)       avg(8)       wynn          (relative error)")
  for M in (8, 16, 32, 64, 128, 256, 512)
    r(x) = (x - ref) / ref
    @printf("  %4d  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e\n", M, r(S[M]), r(avgsum(S, M, 1)), r(avgsum(S, M, 2)),
            r(avgsum(S, M, 4)), M > 8 ? r(avgsum(S, M, 8)) : NaN, r(wynn(S[1:M])))
  end
  # check: weights reproduce avgsum
  w = avg_weights(64, 4)
  @assert abs(1 + sum(w .* a[1:64]) - avgsum(S, 64, 4)) < 1e-12
end
