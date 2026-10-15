# TFNE-PARAM-04 — CLOSED. Declared relative geometry reaches execution.

**Branch:** `w1-52` (0.5.2 ENGINE item 1)
**Entry state:** `e6d3483` + adopted item-1 diff (6 files, dispatcher-verified).
**Authorization:** 0.5.2 decision 0a (human, DECIDED 2026-09-23): a declared
range is a pair of fractions of the area's extent within [0,1]; a range
outside [0,1] is refused. Correctness repair, not optimization.

## Semantics (decision 0a, exactly)

- `G = [z0 = a; z1 = b]` (likewise `x`, `y`) declares a fractional domain
  of the sampled (area, layer) block's extent. A declared sub-range of
  [0,1] changes executed positions; absent `G` (or a full [0,1] axis) takes
  the historical sampling path bit-identically.
- Outside [0,1], half-declared, degenerate (`hi <= lo`) or non-finite
  bounds are refused at resolve time (`E_GEOMETRY_OUT_OF_RANGE`, via
  `TFNEInvalidProportion`), not rescaled. Relative coordinates never
  acquire physical (mm/um/conductivity/distance) semantics.
- One sampled block cannot honour two different declared domains on one
  axis: refused (`E_GEOMETRY_AMBIGUOUS`), not averaged. A single
  declaration wins for the whole block; undeclared leaves inherit it.

## Chain

```
TFNE G -> resolve (_validate_geometry_body) -> realize (s["geometry"])
        -> to_configuration (metadata["tfne_geometry"]: declared + domains)
        -> construct (_neuron_population_from_config samples the sub-range)
        -> manifest ("tfne_geometry": value_tag "relative", declared,
            realized_domains; key present only when a sub-range declared)
```

`to_neuronal_tensor` is untouched (structural role only, per PARAM-01).
JDNA completion already sampled declared `G`; it needed no change — only
its test fixture moved from the now-refused `z1 = 4.0` to `z1 = 0.75`.

## Tests

- `test_declared_geometry_does_not_reach_the_executed_positions` inverted
  to `test_declared_geometry_reaches_the_executed_positions`: sub-range
  [0.2, 0.5] changes executed z at equal seed and bounds it; bare ==
  full-[0,1] bit-identically; (10,20) and (-5,-4) refuse with
  `E_GEOMETRY_OUT_OF_RANGE`.
- `test_conflicting_geometry_in_one_sampled_block_is_refused`:
  `E_GEOMETRY_AMBIGUOUS` from `to_configuration`.
- `tests/test_tfne_ctx01.py`: fixture `z1 = 4.0 -> 0.75`; inertness test
  replaced by `test_geometry_outside_unit_interval_refused` (4.0/40.0 refuse).
- `tests/test_jdna_completion.py`: fixture `z1 = 4.0 -> 0.75` (same reason).

```
tests/test_tfne_parameter_transfer.py + test_tfne_ctx01.py
  + test_jdna_completion.py -> 55 passed
+ test_tfne_algebra.py + test_tfne_execution.py
  + test_tfne_global_assumptions.py + test_construct_golden_snapshot.py
  + test_manifest_readout_compat.py + test_jdna_truth_gate.py
  -> 166 passed (no regressions)
```

## Changed canonical outputs

**None.** Verified, not regenerated:

- No committed canonical spec declares `G` (only the updated test
  fixtures did; historical receipts quote them as prose, untouched).
- Absent-`G` construction is the historical call unchanged
  (`test_construct_golden_snapshot.py` passes); absent-`G` manifests gain
  no key (`tfne_geometry` recorded only when a sub-range is declared).
- The only behavior changes are opt-in (a declared sub-range now moves
  executed positions) and fail-closed (outside-[0,1] `G` now refuses at
  resolve instead of executing inertly).

## Status

- TFNE-PARAM-04: **CLOSED** (configured + realized + executed).
- Geometry-dependent parameters row in the PARAM-01 table: now "yes —
  fractional domain of the sampled block", bounded by [0,1] refusal.
