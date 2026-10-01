# Functions whose methods are defined in package extensions.

"""
    slice_element(ele, L_slice, first::Bool, last::Bool; context) -> LineElement

Detached copy of the Beamlines `LineElement` `ele` with length `L_slice`, for tracking one slice with
`BeamTracking.track!` (pass `_p_over_q_ref`, since the copy is not in a `Beamline`).
`first`/`last` mark the slice at the element entrance/exit. Requires BeamTracking and Beamlines.

- Parameter groups are evaluated with `deval(…, context)`.
- Integrated multipole strengths are scaled by `L_slice / L`.
- Edges and fringe fields stay on only at the element ends (`fringe_at`).
  `SciBmadStandard` becomes the `Symplectic` method BeamTracking would use for this element
  (order 2 or 4, mirroring `scibmadstandard_bl.jl` in BeamTracking 0.8.4), so `fringe_at` can be set.
- `n_steps` of a `Symplectic`-family method is scaled to `max(1, round(n_steps * L_slice / L))`;
  `ds_step` is kept.
- Apertures are kept only on the end slices, at the ends where the element has them.

Unsupported (errors): alignment, patch, map, RF, four-potential and electric multipole parameters,
and tracking methods other than `SciBmadStandard`, `Symplectic`-family and `Exact`.
"""
function slice_element end
