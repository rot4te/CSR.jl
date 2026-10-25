# Functions implemented in package extensions (CSRBeamTrackingBeamlinesExt).

"""
    slice_element(ele, s0, L_slice, first::Bool, last::Bool; context) -> LineElement

Copy of the Beamlines `LineElement` `ele` restricted to `[s0, s0 + L_slice]` (relative to the element
entrance), for tracking one slice with `BeamTracking.track!` (pass `_p_over_q_ref`).
`first`/`last` mark the slice at the element entrance/exit. Requires BeamTracking and Beamlines.
Slice rules follow Bmad's `create_element_slice` / `makeup_super_slave1`:

- Parameter groups are evaluated with `deval(…, context)`.
- Integrated magnetic and electric multipole strengths are scaled by `L_slice / L`.
- Edges and fringe fields stay on only at the element ends (`fringe_at`).
  `SciBmadStandard` becomes the `Symplectic` method BeamTracking would use for this element
  (order 2 or 4, mirroring `scibmadstandard_bl.jl` in BeamTracking 0.8.4), so `fringe_at` can be set.
- `n_steps` of a `Symplectic`-family method is scaled to `max(1, round(n_steps * L_slice / L))`;
  `ds_step` is kept.
- Apertures are kept only on the end slices, at the ends where the element has them.
- Alignment: each slice gets offsets/rotations such that the slices compose to the misaligned element.
- Patch: the first slice holds the patch transform (exit face moved back by `L - L_slice`); the other
  slices are drifts.
- Four-potential: evaluated at `s + s0` and `t + s0/(β₀c)`, so the slice sees the element's field.
  Needs `ele` in a `Beamline` (reference species and energy).
- RF: voltage scaled by `L_slice / L`; the rate is converted to a frequency. The returned slice sits in
  a one-element `Beamline` that carries the reference species. Needs `ele` in a `Beamline`.
  A `SaganCavity` element gives a CSR.jl `SaganSlice` instead, with the cavity constants of the whole element; the
  slices compose to the whole cavity to round-off.
- Map elements (`MapParams`) cannot be split: only the whole element (`first && last`, `L_slice == L`)
  is allowed. See `csr_sliceable`.
"""
function slice_element end

"""
    csr_sliceable(ele; context) -> Bool

`false` for elements that are tracked whole with no CSR kicks: zero length, or a map element
(`MapParams`; Bmad does the same for Taylor elements). Requires BeamTracking and Beamlines.
"""
function csr_sliceable end

"""
    track_csr!(bunch, bl; params::CSRParams, csr_on = ele -> true, centroid_start = nothing,
               sliced_centroid = true, diagnostics = nothing, test_bunch = nothing) -> bunch

Track a BeamTracking `Bunch` through a Beamlines `Beamline` with 1D CSR (Bmad `csr_method = one_dim`).
Requires BeamTracking and Beamlines; see the method docstring for details.
"""
function track_csr! end
