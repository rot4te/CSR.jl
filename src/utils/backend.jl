# Number types, atomic adds and table placement per KernelAbstractions
# backend.

"""
    csr_float_type(backend)

Widest float type the kernels use on `backend`: `Float64`, or
`Float32` where the backend has no `Float64`
(`KernelAbstractions.supports_float64`, e.g. Metal).

# Arguments
- `backend`: KernelAbstractions backend (e.g. `CPU()`, or
  `get_backend(pz)` for the particle arrays).

# Returns
- `Float64` or `Float32` (a type).
"""
csr_float_type(backend) = KernelAbstractions.supports_float64(
    backend) ? Float64 : Float32

"""
    device_scalar(backend, x)

Scalar passed to kernels on `backend`: unchanged unless the backend
lacks Float64.

# Arguments
- `backend`: KernelAbstractions backend (e.g. `get_backend(pz)`).
- `x`: number to pass to a kernel (e.g. the kick coefficient from
  `csr_kick_coef`, or `pz_scale`).

# Returns
- `x`, or `csr_float_type(backend)(x)` on a backend without
  `Float64`.
"""
device_scalar(backend,
              x) = KernelAbstractions.supports_float64(
                  backend) ? x : csr_float_type(backend)(x)

"""
    table_backend(backend)

Backend the kick tables are computed on. The table kernels need the
CPU (non-isbits geometry; see `csr_wake_table`).

# Arguments
- `backend`: KernelAbstractions backend of the particle arrays.

# Returns
- `backend` if it is a `CPU`, else `CPU()`.
"""
table_backend(backend) = backend isa CPU ? backend : CPU()

"""
    to_backend(backend, A)

Copy a CPU table to `backend` in `csr_float_type(backend)`; no copy on
the CPU.

# Arguments
- `backend`: KernelAbstractions backend of the particle arrays.
- `A`: array on the CPU (e.g. a table from `csr_wake_table` or
  `csr_image_table`, computed on `table_backend(backend)`).

# Returns
- `A` itself on a `CPU` backend, else a new array on `backend`.
"""
function to_backend(backend, A::AbstractArray)
  backend isa CPU && return A
  B = KernelAbstractions.allocate(backend, csr_float_type(backend),
                                  size(A))
  copyto!(B, csr_float_type(backend).(A))
  return B
end

# Scatter-add used by the deposit: atomic for hardware floats, plain
# otherwise (serial path).
"""
    AtomicAdd()
    PlainAdd()

Scatter-add functors `add!(A, i, v)` for the deposit: `AtomicAdd` uses
`Atomix.@atomic A[i] += v` (kernels on any backend), `PlainAdd` a
plain `A[i] += v` (serial CPU loop for element types without atomics).
"""
struct AtomicAdd end

(::AtomicAdd)(A, i, v) = (Atomix.@atomic A[i] += v; nothing)

"Plain scatter-add functor `add!(A, i, v)` (see `AtomicAdd`)."
struct PlainAdd end

(::PlainAdd)(A, i, v) = (A[i] += v; nothing)

"""
    supports_atomic_add(A)

`true` if the deposit can scatter into `A` with Atomix atomic adds
(element type `Float32` or `Float64`). Other element types (e.g.
`BigFloat`) use the serial CPU deposit.

# Arguments
- `A`: array to scatter into (e.g. `bins.charge` or
  `bins.n_particle`).

# Returns
- `Bool`.
"""
supports_atomic_add(A) = eltype(A) <: Union{Float32,Float64}
