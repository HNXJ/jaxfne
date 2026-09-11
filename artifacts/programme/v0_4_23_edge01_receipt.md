# 23-EDGE-01 receipt — IMPROVEMENT (defect repair, not new surface)

**Branch:** `dev` (on top of `e959637`)
**Verdict:** IMPROVEMENT. Scope ruling first: **EDGE_OWNED**.

## Scope ruling (the h_dim IndexError)
- Reproduced on HEAD: `_model_with_edge_parameter` indexes size-0
  `edges.weight` with a 90-edge mask → `IndexError`.
- Root cause: compact `weight_storage="magnitude_times_presynaptic_sign"`
  keeps `weight` a size-0 placeholder; the tune path read it directly
  instead of resolving. Masks (`pre`/`post`) are full-length — selection
  was always correct; only weight read/apply assumed `per_edge`.
- Neither H schema nor HDP rule dims are involved (fails before HDP
  engages) → not H_STATE/HDP_OWNED. The failing layer IS the selector +
  immutable transform layer → EDGE_OWNED, repaired here per instruction 4.

## Repair (`jaxfne/_model_tune.py`, minimal)
- `_resolved_edge_weights_host(model)`: read via canonical
  `resolve_edge_weight` (same helper kernels consume) + emitter sign.
- `_model_with_edge_parameter`: read resolved; write back
  `weight_storage="per_edge"` (tuned per-class magnitudes are not
  derivable from scalar metadata); tau/delay compactness untouched.
- `_initial_parameter_values`: same resolved read (no write).
- Precedent: mirrors `_scale_edge_list_weights` else-branch and the
  matrix path's `materialize_edge_list_arrays` usage in-file.

## Evidence
- `tests/test_edge01_selectors_transforms.py` (7): determinism,
  unselected bit-exact + input immutability, parallel-edge co-selection,
  no dense-W involvement, compact fixture end-to-end (weight expands,
  tau/delay stay compact), compact `_initial_parameter_values`,
  post-transform chunked continuation exact.
- Repaired: HP-07 closure green; 3/4 population tests green
  (IndexError → gone).

## Owned residuals (not hidden, not in EDGE)
- Etude scalar metric: `0.381` vs frozen `0.434±0.05` (deterministic;
  HDP-disabled `off` case passes with identical tuned weights, so the
  repair is value-identical and the drift sits in the HDP-active branch
  — unobservable at HEAD due to the crash; slow marker, excluded from
  broad gate). Frozen bundle untouched; tolerance unwidened. Separate
  TODO 23-HDP-02 gates the release (slow markers run at VERIFY).
- Disconnected-null `diag is None`: unchanged, still needs the human
  semantic decision (identity routing vs forced engagement).
