"""
    CSR

1D (line-charge) coherent synchrotron radiation for the SciBmad ecosystem, ported from Bmad's
`csr_method = one_dim`, with optional shielding by parallel plates (image charges). The core works on plain
particle arrays with KernelAbstractions kernels; `track_csr!` (BeamTracking + Beamlines extension) tracks a bunch
through a beamline with CSR kicks between element slices.
"""
module CSR

using StaticArrays
using AtomicAndPhysicalConstants
using KernelAbstractions
import Atomix

export CSRParams, FloorPosition, CSRGeometry, CSRElementInfo, CSRBins, CSRKickPoint, CSRStepResult
export planar_survey, csr_bin_particles, csr_kick!, csr_kick_pz, csr_kick_coef, csr_n_step, dspline_len
export track_csr!

include("types.jl")

include("utils/math.jl")
include("utils/backend.jl")
include("utils/spline.jl")
include("utils/floor.jl")
include("utils/chord_coordinates.jl")
include("utils/root_finding.jl")
include("utils/image_layers.jl")
include("utils/kick_setup.jl")

include("geometry.jl")
include("binning.jl")
include("tables.jl")
include("kick.jl")
include("adjoint.jl")
include("reference.jl")
include("interface.jl")

include("kernels/binning.jl")
include("kernels/source_point.jl")
include("kernels/wake.jl")
include("kernels/image_charges.jl")
include("kernels/convolve.jl")
include("kernels/kick.jl")
include("kernels/adjoint.jl")

end
