# Internals

Non-exported functions, by source file. The layout follows BeamTracking.jl: `src/kernels/` holds the
KernelAbstractions kernels and the per-work-item physics they call, `src/utils/` the helpers, and the top-level files
the types and host-side functions that launch the kernels.

## Types and geometry

```@autodocs
Modules = [CSR]
Public = false
Pages = ["src/CSR.jl", "src/types.jl", "src/geometry.jl"]
```

## Host side: binning, tables, kick

```@autodocs
Modules = [CSR]
Public = false
Pages = ["src/binning.jl", "src/tables.jl", "src/kick.jl"]
```

## Kernels

### Binning

```@autodocs
Modules = [CSR]
Public = false
Pages = ["kernels/binning.jl"]
```

### Source point (retardation)

```@autodocs
Modules = [CSR]
Public = false
Pages = ["kernels/source_point.jl"]
```

### Free-space kick integral and wake table

```@autodocs
Modules = [CSR]
Public = false
Pages = ["kernels/wake.jl"]
```

### Image charges

```@autodocs
Modules = [CSR]
Public = false
Pages = ["kernels/image_charges.jl"]
```

### Convolution and kick

```@autodocs
Modules = [CSR]
Public = false
Pages = ["kernels/convolve.jl", "kernels/kick.jl"]
```

## Utilities

```@autodocs
Modules = [CSR]
Public = false
Pages = ["utils/math.jl", "utils/backend.jl", "utils/spline.jl", "utils/floor.jl", "utils/chord_coordinates.jl",
         "utils/root_finding.jl", "utils/image_layers.jl", "utils/kick_setup.jl"]
```

## Serial reference implementation

```@autodocs
Modules = [CSR]
Public = false
Pages = ["src/reference.jl"]
```

## BeamTracking extension

`CSRBeamTrackingExt`, loaded with BeamTracking: the methods of `BeamTracking.effect_setup` and
`BeamTracking.effect_kick!` for [`CSREffect`](@ref) and [`CSRReplay`](@ref), and their helpers.

```@autodocs
Modules = [Base.get_extension(CSR, :CSRBeamTrackingExt)]
```
