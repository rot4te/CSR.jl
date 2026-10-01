# CSR.jl

module CSR

using StaticArrays
using AtomicAndPhysicalConstants

export CSRParams, FloorPosition, CSRGeometry, CSRElementInfo, CSRBins, CSRKickPoint, CSRStepResult
export planar_survey, csr_bin_particles, csr_kick!, csr_kick_coef, csr_n_step, dspline_len

include("params.jl")
include("geometry.jl")
include("binning.jl")
include("source_point.jl")
include("wake.jl")
include("convolve.jl")
include("kick.jl")
include("step.jl")
include("interface.jl")

end
