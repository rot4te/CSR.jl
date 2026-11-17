cd /Users/alexcoxe/.julia/dev/CSR.jl/test
for t in 4 1; do
  /usr/bin/time -p julia --project=/private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/527ad108-7ac8-4e74-b71d-2ccf3df02e7b/scratchpad/env -t $t -e 'include("runtests.jl")' > /private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/527ad108-7ac8-4e74-b71d-2ccf3df02e7b/scratchpad/accept/suite_t$t.log 2>&1
done
echo done > /private/tmp/claude-501/-Users-alexcoxe--julia-dev-CSR-jl/527ad108-7ac8-4e74-b71d-2ccf3df02e7b/scratchpad/accept/suite_done
