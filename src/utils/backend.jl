# Number types, atomic adds and table placement per KernelAbstractions
# backend.

"""
    csr_float_type(backend)

Widest float type the kernels use on `backend`: `Float64`, or
`Float32` where the backend has no `Float64`
(`KernelAbstractions.supports_float64`, e.g. Metal).
"""
csr_float_type(backend) = KernelAbstractions.supports_float64(
    backend) ? Float64 : Float32

"""
    device_scalar(backend, x)

Scalar passed to kernels on `backend`: unchanged unless the backend
lacks Float64.
"""
device_scalar(backend,
              x) = KernelAbstractions.supports_float64(
                  backend) ? x : csr_float_type(backend)(x)

"""
    device_number_type(backend, T)

Element type for a particle array on `backend` that must hold numbers
of type `T`: `csr_float_type(backend)` when `T <: AbstractFloat` and
the backend lacks `Float64` (e.g. Metal), else `T`. `Dual` and TPS
types are returned unchanged (not supported on such backends).
"""
device_number_type(backend, ::Type{T}) where {T} =
  (T <: AbstractFloat &&
   !KernelAbstractions.supports_float64(backend)) ? csr_float_type(
       backend) : T

"""
    table_backend(backend)

Backend the kick tables are computed on. The table kernels need the
CPU (non-isbits geometry; see `csr_wake_table`).
"""
table_backend(backend) = backend isa CPU ? backend : CPU()

"""
    thread_safe_number(T...)

`true` if arithmetic in all number types `T` may run on several
threads at once: plain and `Dual` numbers (`Real` types). `false` for
GTPSA TPS (non-`Real` types): GTPSA's C library takes scratch TPSs for
nonlinear operations (`/`, `sqrt`, `sin`, ...) from a per-thread stack
indexed by the OpenMP thread number, which is 0 on every Julia thread,
so concurrent TPS operations corrupt memory.
"""
thread_safe_number(T::Type...) = all(t -> t <: Real, T)

"""
    cpu_workgroupsize(n, T...)

Workgroup size for a kernel launch over `n` work items in number types
`T`: `nothing` (KernelAbstractions' default) if
`thread_safe_number(T...)`, else `n`, a single workgroup, which the
KernelAbstractions CPU backend runs on the calling thread.
"""
cpu_workgroupsize(n,
                  T::Type...) = thread_safe_number(T...) ? nothing : n

"""
    descriptor_mismatch(x) -> Bool

`true` if `x` (a number, an array, or a `FloorPosition`,
`CSRElementInfo`, `CSRGeometry`, `CSRKickPoint`) holds a GTPSA TPS
whose Descriptor is not GTPSA's current one. CSR.jl creates TPS
constants from types (e.g. `zero(T)`), which GTPSA puts in the current
Descriptor, so mixed Descriptors would reach GTPSA's C library.
`false` for all other number types; ModularCSRGTPSAExt adds the method
for dynamic-descriptor TPS.
"""
descriptor_mismatch(x) = false
descriptor_mismatch(A::AbstractArray) = eltype(
    A) <: Real ? false : any(descriptor_mismatch, A)
descriptor_mismatch(f::FloorPosition) = !(number_type(f) <: Real) &&
    (descriptor_mismatch(f.theta) || descriptor_mismatch(f.r))
descriptor_mismatch(e::CSRElementInfo) =
  !(number_type(e) <: Real) &&
      any(i -> descriptor_mismatch(getfield(e, i)),
          1:fieldcount(CSRElementInfo))
descriptor_mismatch(g::CSRGeometry) = !(number_type(g) <: Real) &&
    (descriptor_mismatch(g.start) || descriptor_mismatch(g.ele))
descriptor_mismatch(kp::CSRKickPoint) =
  !(number_type(kp) <: Real) &&
      any(i -> descriptor_mismatch(getfield(kp, i)),
          1:fieldcount(CSRKickPoint))

"""
    check_descriptors(fname; args...)

Raise an error naming function `fname` and the first keyword argument
`name = value` for which `descriptor_mismatch(value)` is `true`. No-op
without GTPSA TPS inputs.
"""
function check_descriptors(fname; args...)
  for (name, x) in pairs(args)
    descriptor_mismatch(x) && error("$fname: TPS in argument `$name` is not in GTPSA's current Descriptor (GTPSA.desc_current). CSR.jl creates TPS constants in the current Descriptor; use one Descriptor for all TPS inputs and make it current (the last Descriptor created is current).")
  end
  return nothing
end

"""
    to_backend(backend, A)

Copy a CPU table to `backend` in `csr_float_type(backend)`; no copy on
the CPU.
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
(element type `Float32` or `Float64`). Other element types
(ForwardDiff `Dual`s, `BigFloat`, TPS) use the serial CPU deposit.
"""
supports_atomic_add(A) = eltype(A) <: Union{Float32,Float64}
