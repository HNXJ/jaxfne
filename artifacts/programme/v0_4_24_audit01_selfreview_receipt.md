# 24-AUDIT-01 — adversarial self-review (NOT an independent audit)

**Branch:** `dev`. No product code changed by this battery.
**Status:** item stays OPEN — it requires implementer ≠ sole auditor, and
the author of the v0.4.23/v0.4.24 deltas cannot satisfy that role. What
follows is the falsification battery a second party should repeat.

## Battery (all executed, probes in temp dir)
- **A. Seeded-uniform engagement** (attacks `hdp_is_engaged`): kernel-level
  null-HDP vs baseline with noise 0 → spikes exact, V d≤3.1e-05. So
  routing seeded runs to the HDP kernel moves V within the HDP-vs-baseline
  1e-4 class (EQUIV-01) instead of dropping user state. Model-level
  seeded-vs-unseeded comparison is confounded by the baseline path's
  fixed 0.5 noise (pre-existing design, no noise knob when HDP is off) —
  recorded as a probe artifact, not a finding.
- **B. per_neuron aux layout** (unexercised path): registers, cold-starts
  to (N,) via `dynamic_state_from_model`. Holds.
- **C. Frozen bundle / re-baseline consistency**: frozen arms intact;
  v0423 scalar == frozen off; off/vector unchanged. Holds.
- **D. Aux rule under explicit jit=True**: spikes exact, V bit-exact
  (0.0) on the tested config — registered rules are not inherently
  epsilon-bound; the 1e-4 bound stays as the pre-declared ceiling. Holds.

## Further finding (qualifies HDP-01's identity claim)
"Disabled identity bit-exact" is a *routing* property (both sides execute
the baseline kernel), not kernel equivalence: the HDP kernel with null
gains differs from baseline by ≤3.1e-05 on V (XLA fusion, same class as
EQUIV-01 jit findings). No action — the routing invariant is what is
tested and sealed — but the stronger reading must not be claimed.

## Required external audit (flagged, not executed here)
Independent party must re-run: identity-routing matrix, seeded
engagement, aux-layout paths, etude re-baseline provenance, and the
EQUIV-01 corner bounds, and must challenge the closed NO_CHANGE items
(SIMP-02/03/04/05, STOCH-01, FIELD-01) for missed reductions.
