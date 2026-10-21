# 23-LAW-01 receipt — IMPROVEMENT (structured-law exercise)

**Branch:** `dev` (on top of `c39be87`)
**Verdict:** IMPROVEMENT. No kernel numerics changed; primitive extended
along its designed axis (aux carry) + one structured rule + aux-shape
contracts.

## Delivered
- `HDPRuleDescriptor.aux_layout` ("none"|"per_neuron"|"per_edge",
  default "none"; validated at registration) + `expected_aux_shape`
  (`jaxfne/hdp_rule.py`).
- Kernel (`_hdp_registrable_kernel.py`, both paths): aux cold start sized
  from the descriptor; provided `aux_final` shape-checked (loud reject
  on cross-rule chaining mistakes). Zero-delay `synthetic_presyn_gain`
  path byte-unchanged (`(0,)` default preserved).
- `dynamic_state_from_model` (`_pipeline.py`): cold-start `aux` sized
  from the registered descriptor (non-population locality); legacy and
  population paths unchanged.
- Rule `eligibility_trace_gain` (per-edge eligibility E: coincidence
  integration with `tau_e` decay; weight drive `k_w·H_post·E·|w|`; H keeps
  event coupling so the delayed chain holds). Explicitly an expressivity
  probe, not an STDP mechanism claim; no Jomission content.

## Evidence
- `tests/test_law01_structured_rule.py` (7): surface/layout validation +
  bogus-layout rejection; analytic H/E/w vs hand Euler; gain divergence
  (Θ≠ ⇒ I≠); engineered delayed coincidence (E onset 16→19 exactly,
  zero before arrival, H arrival shifts likewise); aux chunked
  continuation exact (V/S bit-exact, aux_final 1e-5); aux shape-mismatch
  rejection; Model dispatch + continuation with non-empty aux.
- Regression: HDP-01 (6), delayed-registrable (8), continuation
  contract, dispatch (10), JOM-01 (4), audit01, finite-delay, REC-01 —
  all PASS.

## Observed, out of scope
- Compact-weight + Model-continuation `w0` in `dynamic_state_from_model`
  reads placeholder `edges.weight` directly (same pre-EDGE-01 assumption
  class as the repaired tune path). Untouched: no coverage, needs its own
  probe before repair. Candidate follow-up if continuation-on-compact
  matters to the programme.
