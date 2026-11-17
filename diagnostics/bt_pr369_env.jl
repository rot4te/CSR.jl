# BeamTracking PR #369 check (2026-10-18): build a scratch env with dev CSR.jl, Beamlines.jl and BeamTracking from
# the PR branch ("pr") or the local ~/.julia/dev/BeamTracking.jl ("base"), plus the test dependencies.
# Run: julia bt_pr369_env.jl pr|base <env dir>
using Pkg
pr = ARGS[1] == "pr"
Pkg.activate(ARGS[2])
Pkg.develop([PackageSpec(path=expanduser("~/.julia/dev/CSR.jl")), PackageSpec(path=expanduser("~/.julia/dev/Beamlines.jl"))])
if pr
  Pkg.add(PackageSpec(url="https://github.com/bmad-sim/BeamTracking.jl", rev="fix-sagan-ref-momentum"))
else
  Pkg.develop(PackageSpec(path=expanduser("~/.julia/dev/BeamTracking.jl")))
end
Pkg.add(["Test","StaticArrays","AbstractFFTs","ChainRulesCore","ChainRulesTestUtils","GTPSA","AtomicAndPhysicalConstants","KernelAbstractions","Atomix","ForwardDiff","DelimitedFiles"])
Pkg.resolve(); Pkg.status()
