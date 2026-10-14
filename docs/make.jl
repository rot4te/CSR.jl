using CSR
using BeamTracking, Beamlines, ForwardDiff, ChainRulesCore   # load the package extensions
using Documenter

const CSRBeamTrackingBeamlinesExt = Base.get_extension(CSR, :CSRBeamTrackingBeamlinesExt)

DocMeta.setdocmeta!(CSR, :DocTestSetup, :(using CSR); recursive = true)

makedocs(;
  modules = [CSR, CSRBeamTrackingBeamlinesExt],
  authors = "Alexander M. Coxe and contributors",
  sitename = "CSR.jl",
  checkdocs = :all,
  format = Documenter.HTML(;
    canonical = "https://rot4te.github.io/CSR.jl",
    edit_link = "main",
    assets = String[],
    size_threshold = nothing,
  ),
  pages = [
    "Home" => "index.md",
    "Usage" => "usage.md",
    "Physics model" => "physics.md",
    "Physics decisions" => "decisions.md",
    "Validation" => "validation.md",
    "API" => "api.md",
    "Internals" => "internals.md",
  ],
)

deploydocs(; repo = "github.com/rot4te/CSR.jl", devbranch = "main", push_preview = false)
