# 24-HDP-GEN-01 receipt — general finite-state HDP expressivity

**Branch:** `dev`. Bounded closure claim: any JAX-compatible finite-state
HDP rule expressible using the supported state layouts, event/history
inputs, RNG semantics, and declared mutable targets registers without a
mechanism-specific simulation branch. Zero ENGINE_GAP within that grammar.

## Repairs (all generic dimensions, no per-mechanism branches)
- `h_shape` descriptor field → H carry `(N,*h_shape)` (`hdp_rule.py`,
  `_hdp_registrable_kernel.py` with strict `_checked_h0`; Model-level H0
  override in `continuation_state_from_model`).
- `aux_layout="scalar"` → `()` global coordinate; multi-coord trailing-dim
  widening `(n,k)`; `none`+coords and unknown layouts rejected at register.
- `drive_bias` second Theta target (`b_bounds`, kernel `b` carry,
  `DynamicState.b`, continuation/diag plumbing mirroring `aux`); undeclared
  `d_theta` keys raise (was silently ignored).
- Per-step rule key `fold_in(rule_base_key, global_t)`; `rule_base` reuses
  the otherwise-unused split half → membrane/noise streams bit-identical;
  `step_indices` threaded Model-level (global `t_idx`) and validated
  kernel-level with continuation-offset fallback.
- Bounds validation (`h/w/b_bounds` min≤max) at register.
- Incidents: carry variable `b` shadowed Izhikevich `params.b` (caught by
  LAW-01 arrival-shift test, renamed to `bias`); homeostatic tail-scan
  fall-through (caught by broad, restored if/else exclusion); test file
  order-dependence (every probe now self-`_ensure`s; dup name removed).

## Expressivity matrix (all in `tests/test_hdp_gen01_expressivity.py`, 20 tests)
- STATE: scalar H (prior suites), vector d_H=2 analytic, per-neuron aux
  analytic, per-edge aux analytic, per-edge×2 cascade analytic, scalar
  global analytic.
- EVENT/TIME: decay, event, pre/post, finite delay (prior + delayed
  stochastic), mixed.
- THETA: edge_weight analytic/causal, drive_bias
  analytic/causal/saturating/continuation/jit/Model-identity, logistic
  saturation monotone + strictly-bounded, BCM sliding-threshold analytic
  (threshold-as-state, sign structure), strict undeclared/unknown rejection.
- RNG: same-seed deterministic, diff-seed sensitive, kernel-chunked,
  Model-chunked, null-stochastic membrane isolation (V bit-identical,
  aux differs).
- COMPOSITION: vector H + per-edge aux; delayed + stochastic.
- IDENTITY: zero-gain Model-level == static baseline (matched backends).
- MALFORMED: H/aux shapes, layout, undeclared key, unsupported target,
  inverted bounds, step_indices, jit-invalid rule (trace-time rejection).

## Counterexamples (two independent critic rounds, same-model-family qualified)
Round 1: 2 EXPRESSIBLE, 1 OUTSIDE (structural plasticity), 5 GAPs →
  repaired multi-coord aux + per-step key, BCM proven EXPRESSIBLE by probe,
  novel-target registry TRADEOFF-deferred, multi-tap REJECTED
  (trace-subsumption).
Round 2 (declared-grammar only): 5 EXPRESSIBLE (triple-product,
  delayed+instant+trace, endogenous reward-modulation, exc/inh via
  `exc_mask`, relative-timing ISI via timer state), 1 OUTSIDE
  (threshold/multiplicative-gain targets — matches deferred boundary;
  exogenous reward ports and absolute wall-clock gating explicitly
  unclaimed).

## Remaining declared boundaries (not gaps)
Targets are a finite registry (`edge_weight`, `drive_bias`); threshold/gain
targets, exogenous time-series ports, raw multi-tap windows, structural
plasticity: OUTSIDE. Rule sees state + arrivals + key, not raw drive port.

## Gates
GEN 20/20 (order-independent); batteries 165 + 77 green; ruff gate clean
(E731 fixed; test_mcc F841 pre-existing, gate covers `jaxfne/` only);
broad: **3875 passed** (3855 + 20), 75 skipped, 37 deselected, 4 xfailed;
vocab pass; no orphan docs.

## Numerical delta
No change to any existing path: new carry/traces/diagnostics gated behind
declaration; legacy expressions statically branched bit-exact (suites +
frozen hashes green).
