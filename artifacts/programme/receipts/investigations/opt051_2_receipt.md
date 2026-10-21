# opt051_2 receipt — neuron-table selection index in compile_connection_rules (B2)

Status: SPEC-FROZEN (code untouched).

## Target
`compile_connection_rules` calls `_select_indices` 2× per rule, each a full
O(N) scan with per-row `str()` coercions (bottlenecks_051 B2; P4: 109/240s
= 45% of AT-10 construct; 1960 calls over 20 identical areas recompute the
same local selections with different global offsets).

## Change
`jaxfne/connectivity.py` only:
- Build one `_SelectionIndex` per `compile_connection_rules` call over the
  union of selector fields used by that call's rules: field → str(value) →
  ordered nid list (encounter order) + membership sets.
- `_select_indices` gains an optional index path (same signature + private
  `_index=None` kw): when the index covers the selector's fields, resolve
  by smallest-list intersection (encounter order preserved); otherwise the
  original scan runs. Error messages, `ids` filtering, `str()` coercion
  semantics, and empty-match behavior unchanged by construction.
- No caller changes; `compile_connection_rules` builds the index once and
  passes it to its two per-rule calls.

## Expected gain
−35..−45% of AT-10 construct (~30–38s of 84s); small-model constructs move
by ms (absolute noise). Warm/simulate paths untouched (0% there by
construction).

## Gate (bit-exact; ANY tolerance need → STOP, revert, report)
- G1 indexed-vs-reference equality: every (selector → id list) pair produced
  while constructing (a) single canonical column, (b) merged AT-10 ring
  (all intra + 20 ring rules), (c) two-area-feedforward — recorded under the
  reference implementation and replayed against the indexed path,
  `array_equal` on each pair; plus an adversarial battery (ids filter, None
  values, missing keys, empty→raise exact message, non-mapping→raise exact
  message, int-valued fields).
- G2 realized artifact equality: edge (pre,post,weight,mech) multisets +
  connection tables identical before/after on (a)(b)(c); canonical
  trajectory bit-identity (base-100 + tensor column spikes/V exact).
- G3 affected suites PASS unmodified: test_connection_rule_compile_v0330,
  test_connection_rule_mechanisms_v0330, test_connection_weight_modes_v0330,
  test_mechanism_aware_connection_compiler, test_construct_golden_snapshot,
  test_v0341_kernels.
- G4 before/after AT-10 construct wall (matrix cell re-run) + small-column
  construct sanity.

## Before (matrix_051 post-opt051_1)
at10 construct 84.0s (first 3.5s, warm 2491ms — simulate untouched).

## After
AT-10 cell re-run post-change (matrix script, worktree code):
construct 84.0s → 43.8s (−48%, beats the −35..−45% estimate).
Realized graph identical: n=20000, edges=4,747,700, inter fraction 9.099%
(unchanged). Simulate-path phases moved within cross-process drift
(first 3.5→2.7s, warm 2491→1848ms; path untouched by this change).

## Gate results
- G1 oracle (indexed vs scan, same process): 4154/4154 selector comparisons
  identical — all harvested real selectors (column 96, two-area 2, AT-10
  ring 1960, on 1k/120/20k-row tables) + 19-case adversarial battery
  (ids/None/missing/int-coercion/non-mapping/empty) + uncovered-field
  fallback. PASS.
- G2 end-to-end: base-100 spikes 173/173, n1000 1713/1713 vs committed
  post-opt051_1 checksums; tensor column runs. PASS.
- G3 (6 affected suites): 55 passed, 0 failed.
- G4 (AT-10 construct): −48% measured gain, identical edges.

## Verdict
PASS — landed. Files: `jaxfne/connectivity.py` (+70/−2),
refreshed `artifacts/perf/matrix_051.json` (at10 cell only).

## STOP-rule state after opt051_2
Remaining ranked candidates: B3 dense execution (no equivalent-preserving
fix), B4 HDP-exec proper (unattributed, no concrete change), B5 one-time
first-call, B6/B7 below 25% thresholds, B8 excluded (not bit-exact as run).
No remaining concrete change carries an expected gain ≥10% of its cell
total → STOP item 3 here per 3d; all remainder carried (report only).
