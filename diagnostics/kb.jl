gaussian(z, sigma) = exp(-z^2 / (2sigma^2)) / (sqrt(2pi) * sigma)
function kernel_test_bunch(sigma, Q, n)
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n)
  alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end
