# Number types and table placement per KernelAbstractions backend.

"""
    csr_float_type(backend)

Widest float type the kernels use on `backend`: `Float64`, or `Float32` where the backend has no `Float64`
(`KernelAbstractions.supports_float64`, e.g. Metal).
"""
csr_float_type(backend) = KernelAbstractions.supports_float64(backend) ? Float64 : Float32

# Scalar passed to kernels on `backend`: unchanged unless the backend lacks Float64.
device_scalar(backend, x) = KernelAbstractions.supports_float64(backend) ? x : csr_float_type(backend)(x)

# Backend the kick tables are computed on. The table kernels need the CPU (mutable work struct, non-isbits geometry).
table_backend(backend) = backend isa CPU ? backend : CPU()

# Copy a CPU table to `backend` in `csr_float_type(backend)`; no copy on the CPU.
function to_backend(backend, A::AbstractArray)
  backend isa CPU && return A
  B = KernelAbstractions.allocate(backend, csr_float_type(backend), size(A))
  copyto!(B, csr_float_type(backend).(A))
  return B
end
