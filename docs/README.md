# CSR.jl documentation

Built with [Documenter.jl](https://documenter.juliadocs.org). From the repository root:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once (Julia ≥ 1.11 picks up CSR from `..` via [sources])
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. The build fails if a docstring is not included in a page (`checkdocs = :all`): new
functions need a docstring, and `api.md`/`internals.md` pick them up through `@autodocs` by file.

`.github/workflows/Documentation.yml` builds the pages on pushes and pull requests and deploys them to the
`gh-pages` branch (GitHub Pages) from `main` and tags.
