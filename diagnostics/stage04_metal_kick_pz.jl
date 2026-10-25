# Stage 04 sub-stage B, acceptance criterion 2 (Metal check, outside the test suite; spec 2026-10-24, amended 2026-10-25).
# Checks on a Metal GPU (Float32 MtlArrays): csr_kick_pz and the rrule forward output equal csr_kick! on a copy of pz,
# within maximum(abs, a .- b) <= 1e-6 * maximum(abs, kick); the output type is MtlArray{Float32}.
# Sensitivity mode (arg "sens") redefines CSR.device_number_type = T; csr_kick_pz on Metal then errors (expected).
# Run (needs Apple Metal and an env with Metal, CSR (dev path), AtomicAndPhysicalConstants, StaticArrays,
# ChainRulesCore, KernelAbstractions):
#   julia --project=<metal_env> diagnostics/stage04_metal_kick_pz.jl          (fix)
#   julia --project=<metal_env> diagnostics/stage04_metal_kick_pz.jl sens     (sensitivity)
using Metal, CSR, AtomicAndPhysicalConstants, StaticArrays, ChainRulesCore, KernelAbstractions, Test
using CSR: csr_kick!, csr_kick_pz

# Helpers copied from test/utils.jl (gaussian) and test/kernels_test.jl (kernel_test_bunch).
gaussian(z, sigma) = exp(-z^2 / (2sigma^2)) / (sqrt(2pi) * sigma)
function kernel_test_bunch(sigma, Q, n)
  z = collect(range(-6sigma, 6sigma; length = n)) .+ 0.37 * (12sigma / n) .* sin.(1:n)
  w = gaussian.(z, sigma) .* (1 .+ 0.2 .* cos.(3 .* (1:n)))
  alive = fill(true, n)
  alive[3:97:end] .= false
  return z, Q .* w ./ sum(w), alive
end

# Ratio max|a - b| / max|kick| (amended criterion 2 uses the bound 1e-6 on this ratio).
ratio(a, b, kick) = maximum(abs, a .- b) / maximum(abs, kick)

if get(ARGS, 1, "") == "sens"
  @eval CSR.device_number_type(backend, ::Type{T}) where {T} = T
  println("SENSITIVITY: CSR.device_number_type(backend, ::Type{T}) = T")
end

# Geometry of test/kick_test.jl style (6.B.4 item 2).
L3 = [0.2, 0.3, 0.2]
c3 = zeros(4, 6)
c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]
c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]
c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom = CSRGeometry(L3, [0.0, 1.5, 0.0], c3)
z64, q64, alive64 = kernel_test_bunch(1e-3, 1e-9, 400)
params = CSRParams(n_bin = 40, ds_step = 0.05)
e = Species("electron")

# Float32 device arrays (pz, z, q) and a Bool mask on the Metal GPU.
n = length(z64)
pz_d = MtlArray(Float32.(1e-4 .* sin.(1:n)))
z_d = MtlArray(Float32.(z64))
q_d = MtlArray(Float32.(q64))
alive_d = MtlArray(alive64)
println("backend: ", KernelAbstractions.get_backend(pz_d), " supports_float64 = ",
        KernelAbstractions.supports_float64(KernelAbstractions.get_backend(pz_d)))

@testset "Metal csr_kick_pz (Stage 04 B, W4)" begin
  pz_in = Array(pz_d)  # snapshot of the input before any call

  pz_new = csr_kick_pz(pz_d, z_d, q_d, alive_d, geom, 3, 0.1, 200.0, e, params; pz_scale = 1.3)
  @test pz_new isa MtlArray{Float32}
  println("typeof(pz_new) = ", typeof(pz_new))

  # Reference: csr_kick! (in place) on a copy of pz_d with the same arguments.
  pz_ref = copy(pz_d)
  csr_kick!(pz_ref, z_d, q_d, alive_d, geom, 3, 0.1, 200.0, e, params; pz_scale = 1.3)
  pz_out = Array(pz_ref)
  kick = pz_out .- pz_in
  @test maximum(abs, kick) > 0
  println("max|csr_kick! kick| = ", maximum(abs, kick))

  r_fwd = ratio(Array(pz_new), pz_out, kick)
  println("ratio csr_kick_pz vs csr_kick!  = ", r_fwd)
  @test r_fwd <= 1e-6

  y_rr, _ = ChainRulesCore.rrule(csr_kick_pz, pz_d, z_d, q_d, alive_d, geom, 3, 0.1, 200.0, e, params; pz_scale = 1.3)
  @test y_rr isa MtlArray{Float32}
  r_rr = ratio(Array(y_rr), pz_out, kick)
  println("ratio rrule forward vs csr_kick! = ", r_rr)
  @test r_rr <= 1e-6
end
