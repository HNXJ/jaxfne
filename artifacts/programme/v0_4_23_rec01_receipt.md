# 23-REC-01 receipt — IMPROVEMENT (streaming capture + preflight + toggle fix)

**Branch:** `dev` (on top of `01d7c47`)
**Verdict:** IMPROVEMENT. W17.2 preflight + streaming delivered; one
recording-control defect fixed.

## Delivered (`jaxfne/_pipeline.py`, additive only)
- `run_continuation_strided(step_fn, state, schedule, *, stride)`:
  segmented decimated capture (segments of `stride` steps through the
  proven `run_continuation`, keeping each segment's final frame).
  Transient O(stride) per segment instead of O(T). `kept[j]` equals the
  uninterrupted frame at `kept_indices[j]` exactly; final state equals
  the uninterrupted final state (dynamic leaves, `prng_key`,
  `step_index`, `delay_state`) — holds with noise, delays, HDP, on all
  paths, since draws come from the same carried key sequence.
- `memory_report(model, simulation=None, recorder=None)`: pure
  shape/dtype arithmetic — persistent / dynamic / delay ring /
  recording (V/spikes/sources iff recorded-or-field-projected; H/w/aux
  iff HDP engaged; `w` iff `record_weight_trace`) / transient draws —
  plus total, dominant component, and actionable advice. Signature
  follows the W17 provisional surface (kept in `_pipeline`; root export
  deferred to the release surface sweep so the pinned v034/contract
  snapshots don't churn mid-programme).
- Toggle fix: one `rwt` owner for `record_weight_trace` in
  `compile_step_fn`'s HDP branch. Before: registered + False crashed
  (`None[0]`); legacy + False silently stacked-then-dropped w. After:
  4-tuple outputs, identical dynamics and final states on both.

## Evidence
- `tests/test_rec01_streaming_preflight.py` (12): strided exactness
  ×3 strides, delayed+registered+noisy strided exactness, bad-stride
  rejection, estimator-vs-measured byte equality, HDP/stride scaling,
  toggle arity+equivalence × legacy/registered.
- Regression gates: continuation contract, HDP dispatch, JOM-01,
  delayed-registrable (8), HDP-01, finite-delay — all PASS (60 total
  across the 7 files run this increment).
