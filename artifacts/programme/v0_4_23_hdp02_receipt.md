# 23-HDP-02 receipt — DIAGNOSIS COMPLETE, human protocol decision required

**Branch:** `dev` (on top of `02ce8a5`)
**Verdict:** corrected behavior exposing stale frozen expectation.
No implementation defect. No code changed. TODO stays OPEN (gates VERIFY).

## Reproduction (independent)
- `off` R_EI = 0.3811 == frozen `off` 0.38110494571266207 (exact).
- `scalar` R_EI = 0.38110494571266207 == frozen `off`, bit-identical spike
  trains (1961 spikes each); seed 18/19 give 0.4042/0.3807 (chaotic
  sensitivity confirmed — tolerance-scale spread).
- `vec` R_EI = 0.9715 vs frozen 0.9599 (within abs 0.05 — passes).
- Null-noise scalar ≡ null-noise off (0.3811 both).

## First-divergence localization
No trajectory divergence to localize: with `{"K_HDP": 0.01,
"h_state_dim": 1}` every H-driver is 0, so H≡1 and dw≡0 — the HDP is
exactly null. The ONLY historical difference between the scalar and off
runs was the RNG stream (HDP kernel vs baseline kernel). `use_hdp` is now
False for these params (`hdp_params_are_identity` ignores the
non-H-driving `K_HDP`), so scalar routes to the baseline kernel with the
identical key → bit-identity is THEOREM, not accident (HDP-01 qualified
`test_disabled_identity_routes_to_baseline` asserts exactly this).

## Timeline (git)
- Bundle frozen 2026-08-11 (v0.4.14 era): scalar ran the HDP kernel with
  null dynamics but a distinct noise stream → 0.4338 (RNG-stream artifact).
- HDP-01, 2026-09-11 (22dda4d): identity routing introduced → null-HDP ≡
  baseline, bit-exact → scalar now 0.3811 ≡ off.

## Classification
Corrected behavior exposing stale frozen expectation. Reverting would
break a qualified invariant. Frozen bundle and tolerance untouched per
mandate.

## Exact human decision required (pick one)
1. Re-baseline frozen `scalar.R_EI` to the `off` value (principled:
   null-HDP ≡ off by the identity theorem), or
2. Redesign the scalar protocol arm with non-null H drivers (protocol
   change under publication authority).
Until then the slow gate (`test_population_restoring_etude_regression_metrics`)
fails and v0.4.23 cannot seal.

## Closure (human pick 1 executed)
- Independent check on the exact scalar arm: predicate True, baseline
  route taken (diag None), spikes/V bit-identical to HDP-off, metric
  anchored at 0.38110494571266207.
- New `artifacts/etudes/hdp_controllability_reachability/metrics_v0423.json`
  carries scalar == off with full provenance; frozen `metrics.json`
  byte-unchanged; off/vector arms still assert against frozen.
- Test wires scalar assert to the v0.4.23 baseline (explicit comment).
- Slow gate: all 4 population tests PASS. HDP-02 CLOSED.
