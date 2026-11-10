using CSR
using BeamTracking, Beamlines   # load the package extension
using Documenter

const CSRBeamTrackingExt = Base.get_extension(CSR, :CSRBeamTrackingExt)

DocMeta.setdocmeta!(CSR, :DocTestSetup, :(using CSR); recursive = true)

makedocs(;
  modules = [CSR, CSRBeamTrackingExt],
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

deploydocs(; repo = "github.com/rot4te/CSR.jl", devbranch = "main",
           push_preview = false)
