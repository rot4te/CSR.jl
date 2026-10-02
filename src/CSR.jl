# CSR.jl

module CSR

using StaticArrays
using AtomicAndPhysicalConstants
using KernelAbstractions
import Atomix

export CSRParams, FloorPosition, CSRGeometry, CSRElementInfo, CSRBins, CSRKickPoint, CSRStepResult
export planar_survey, csr_bin_particles, csr_kick!, csr_kick_coef, csr_n_step, dspline_len
export track_csr!

include("params.jl")
include("geometry.jl")
include("binning.jl")
include("source_point.jl")
include("wake.jl")
include("convolve.jl")
include("kick.jl")
include("step.jl")
include("reference.jl")
include("interface.jl")

end
