# 24-TESTMIN-01 receipt — conservative test minimization

**Branch:** `dev`.

## Classification applied
Failure classes inventoried per file family (identity/sign/scale/shape/
boundary/state/composition/numerical/JIT/continuation/RNG/release):
dense vs edge backends, null vs engaged HDP, zero vs delayed, bulk vs
continuation, eager vs jit vs vmap — each pair guards a distinct class.
Expensive tests (equivalence gate, fig06, notebook runners, 1000n loads)
are publication/contract evidence; shrinking inputs would weaken the
evidence they exist to provide. Largest parametrize blocks are ≤8 cases
over distinct partitions.

## Accepted (single trim)
- `test_jax01_profile.test_jit_eager_bit_exact_baseline_and_hdp` removed:
  mechanically covered by the stronger successor
  `test_equiv01_table` (baseline + 3 HDP regimes incl. registered rules,
  vs JAX-01's baseline + 1). Retained in JAX-01: vmap/loop exactness,
  single-compile guard, byte agreement (distinct classes).

## Fault probes (both directions)
- Product fault (1.5× HDP rule basis): boundary suite trips via
  genuine spike-train divergence; regime suites stay green because both
  modes share the faulted code — documented limit of equivalence tests
  (they compare modes, not absolute truth). Restored byte-identical.
- Bound tightening (eps → 1e-9/1e-12): all 3 regime tests fail —
  assertions are load-bearing at the claimed sensitivity, not vacuous.
  Restored byte-identical.

## Verdict
No further safe reduction: the 60k lines encode distinct failure
classes; historical regressions stay unless a stronger successor exists
(none found beyond the one trim). Per MIN-01 gate this is a correct
near-NO_CHANGE with one measured improvement.

## Evidence
9/9 green (`test_jax01_profile` trimmed + `test_equiv01_table`);
fault-probe transcripts in this receipt.
