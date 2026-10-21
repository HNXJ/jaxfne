# 23-DELAY-01 receipt — DELAYED REGISTRABLE HDP CLOSED; compaction deferred

**Branch:** `dev`
**HEAD:** `88fa347` + working delta (see commit)
**Depends:** 23-HDP-01 first increment (zero-delay qualification)

## Delivered

### Delayed registrable kernel
- `jaxfne/_hdp_registrable_kernel.py`: finite-delay path via Protocol D ring
  (`delay_state` `(D_max+1, N)`, `delay_steps`, `step_indices`,
  `continuation_step_offset`); zero-delay path byte-unchanged.
- Chain: `event_{t-d} -> H_t` via per-edge delayed `pre_sp`
  (`_delayed_presynaptic_spikes`); `H_t -> P` via registered `step_fn`;
  `P -> Theta_t` via `edge_weight` update; `Theta_t -> I_t` via `w*syn_state`.
- `jaxfne/hdp_rule.py`: `HDPRuleContext.pre_sp` (optional per-edge drive);
  `synthetic_presyn_gain` uses `pre_sp` when present, else `spikes[pre]`
  (zero-delay equations bit-exact).
- `jaxfne/_pipeline.py`: forward `step_indices` to registrable kernel so
  per-step continuation carries global time.
- `jaxfne/_model_simulate.py`: `simulate_batch` dispatches registered rules
  (same kernel; `record_weight_trace=False`); stale "no finite-delay path"
  comment removed.

### Qualification evidence (new)
Tests: `tests/test_hdp_delayed_registrable.py` (7)

| Probe | Result |
| --- | --- |
| zero-delay has no ring | PASS |
| one-step exact arrival (d=0/1/5, H onset = t0+d) | PASS |
| asymmetric multi-step (2/5/9, H1=t0+2, H2=t0+5) | PASS |
| direct-kernel chunked vs uninterrupted across in-flight event (V/S bit-exact, H atol 1e-5, delay_state exact) | PASS |
| gain divergence (k_w 0.05 vs 0.20 => Theta and I differ) | PASS |
| JIT deterministic replay | PASS |
| Model continuation delayed registered (cont-full vs chunked V/S exact, delay_state carried) | PASS |

Preserved: `test_hdp01_registrable_qualification` (6), `test_hdp_finite_delay`,
`test_continuation_contract`, `test_hdp_dispatch` (10),
`test_hdp_audit01_expressivity_probes` + `test_hdp_kernel_standalone` (28) — all PASS.

## Explicitly not claimed
- **History compaction (delay classes) deferred.** Dense ring matches legacy
  layout; no buffer-format change attempted. Per programme constraint,
  delayed-HDP acceptance is decoupled from any future compaction verdict.
- **Bulk vs continuation parity for registered rules** remains open
  (pre-existing zero-delay mismatch confirmed; direct-kernel noisy chunking
  uses internal bulk noise; Model continuation path is the exact contract;
  owned by 23-STOCH-01).
- **Pre-existing failures at HEAD `88fa347`** (verified via stash, unchanged
  by this delta): HP-05 continuation `delay_state is not None` (test omits
  `delay_storage="per_edge"`, so delays never engage), population `h_dim`
  IndexError, disconnected-null `diag is None`. Not introduced here.
## Spawned

- None new. 23-SIMP-05 (HDP kernel unification) now unblocked on delayed
  semantics; 23-LAW-01 remains gated per stack order.

## Closeout (remainder: delay-class compaction — NO_CHANGE)
The dense ring (`(D_max+1, N)`, identical layout to the legacy kernel) is
the shipped history representation: bit-exact continuation proven on all
paths. "Delay classes" never received an acceptance probe (no metric, no
equivalence bound), and any buffer re-layout risks the qualified
`event_{t-d}` timing. Per the conditional rule (no speculative capability
without acceptance), the compaction remainder closes here; class-compacted
edge metadata (uniform/per-edge delay storage) already bounds the
*parameter* footprint, which was the measurable half. Remainder removed
from the stack; this receipt stays the evidence.
