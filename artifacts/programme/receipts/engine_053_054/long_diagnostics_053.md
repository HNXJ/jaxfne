# 0.5.3 ENGINE item 7 receipt — long-horizon diagnostics (W2)

Branch `w2-53` (after item-6 commit `586930a`). Authority: H5
(adversarial), H8 (`d_H=1`; bounded != stable: RBS may sit inside bounds
without returning, and the two are measured independently, never inferred
one from the other).

## Measurements (separate by construction)

Test-local, JSON-safe, over recorded 2000-step twin trajectories
(3-neuron/2-edge fixture, legacy HDP kernel, noise-free, `d_H=1`):

- `boundedness_report`: H within `[H_min, H_max]`, `|w| <= w_ceiling`;
  reports `{within_bounds, H/W_violations, observed min/max}`.
- `stability_report`: twin trajectories differing only in initial H
  (delta 0.05); peak-distance late/early ratio; tiers RETURNING (< 0.5),
  PERSISTING (<= 2), DIVERGING (> 2). Zero early distance is a vacuous
  assay and is refused, not scored.

## Results

- Case A (fixed W, `K_ctrl=5` restoring): bounded PASS, RETURNING
  (ratio 1.5e-4).
- Case B (fixed W, no restoring): bounded PASS, PERSISTING (ratio 1.0)
  — the non-inference proof: bounded AND non-returning coexist.
- Plastic long run (responsive gains, W moves): bounded PASS.
- Synthetic adversarial: injected violations caught (1+1 counts);
  manufactured drift verdicts DIVERGING; degenerate/shape-mismatched
  assays refused; reports `json.dumps`-clean for the future Y vector.

## Verification

- New `tests/test_long_diagnostics_053.py`: 11 passed (~7 s).
- No source changes (measurement-only item); kernel paths already
  covered by item-5 suites. Invariants: K_HDP=0 null, H!=HDP, `d_H=1`
  untouched.

## Carried

- Promotion of these helpers into a shared diagnostics module belongs to
  the 0.5.5 measurement-vector item, not here. Long AT-scale horizons
  (20-area, 0.5.1 envelope) are seal-level, owned by the dispatcher.
