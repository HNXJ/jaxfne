# opt051_1 receipt — default jit policy False → "auto" (B1)

Status: SPEC-FROZEN (code untouched).

## Target
Per-call XLA recompilation on the default eager path (bottlenecks_051 B1).
Every `simulate()` with default runtime retraces + recompiles the scan body
(P2: 121/177ms = 68% of a warm n=100 call; P3: 297/537ms = 55% of HDP warm).

## Change (one line + doc)
`jaxfne/_runtime_config.py`: `RuntimeConfig.jit` default `False` → `"auto"`
(existing policy: jit iff `n_steps*n_units*batch > 50000`; no new semantics).
Below threshold the executed path is literally unchanged (same branch);
above threshold execution routes to the already-tested `jax.jit` +
`_compiled_cache` path. `simulate` docstring line on "JIT is opt-in"
updated to state the auto default. `runtime()` factory keeps its explicit
`jit=False` default (caller choice, unchanged).

Why "auto", not True: single-shot tiny runs (<50000 unit-steps) keep the
eager first-call latency; flipping them to jit would trade one-shot latency
for throughput (human-decision territory, carried as follow-up).

## Expected gain
- Default-runtime cells above threshold: base-class warm −58%..−98%
  (causal bounds from same-process A/B: P1 n1000 277→152ms; P5 n100
  145→7.2ms).
- n10000 warm: −8.5% (recompile ~1s of 46s; first call +2.3s one-time
  compile; net cell wall −5%).
- Below-threshold cells (n1, n10, dt05): path identical, 0% by
  construction (observed −20..−30% is cross-process drift, see After).
- Spec correction (pre-landing, no code impact): mech_hdp / chunk_ref /
  chunk_k4 / at10_20area pass EXPLICIT `jit=False` runtimes, so the
  default flip does not touch them (observed ±10% is drift). P5's HDP
  auto-vs-eager numbers (−67%, bit-exact incl. H/W traces) stand as
  opt-in evidence only, not landed gain.

## Gate (bit-exact; ANY tolerance need → STOP, revert, report)
- G1 bit-compare after-change-default vs explicit `jit=False` on: base-100,
  n1000, HDP-100 (spikes+V+H_trace+w_trace exact), canonical-v1-column-1000n
  tensor, two-area-feedforward tensor, 10n HDP baseline regime, n1, dt05.
  Spikes `array_equal`, all float arrays `array_equal` (diff 0.0). HDP
  diagnostics compared too.
- G2 affected suites PASS unmodified: test_jit_equivalence_v036,
  test_equivalence_gate_v20260815, test_vectorized_equivalence,
  test_construct_golden_snapshot, test_gate7_recording_invariance,
  test_numeric_gates, test_phaseC_H_carry_resume, test_continuation_contract,
  test_hdp_dispatch, test_perf_float32_hdp_acceptance.
- G3 before/after wall on matrix subset (base, n1, n1000, n10000, mech_hdp,
  chunk_ref, at10_20area) via scripts/benchmark_051_matrix.py --cells.

## Before (matrix_051, pre-change)
base warm 259.7 first 1272 | n1 140.0/738 | n1000 330.7/1058 |
n10000 45454.5/46473 | mech_hdp 563.4/2505 | chunk_ref 262.4/1292 |
at10 warm 2491.1 first 3537 construct 84010 (construct untouched by this opt).

## After
Matrix refreshed post-change (all default-runtime cells re-measured;
mech_hdp/chunk/at10 carry explicit `jit=False` so their pre-change values
stand — code path provably identical; pre-change matrix preserved in git
history at a7ab62a).

| cell | warm before→after (ms) | first before→after (ms) | spikes |
|---|---|---|---|
| base | 259.7→5.1 (−98%) | 1272→594 | identical count |
| n1000 | 330.7→139.6 (−58%) | 1058→550 | identical count |
| t10x | 275.4→31.4 (−89%) | 1283→587 | identical count |
| dt0025 | 214.4→14.6 (−93%) | 1043→597 | identical count |
| rec_off | 241.8→5.6 (−98%) | 1472→723 | identical count |
| rec_full | 286.4→9.7 (−97%) | 4707→2711 | identical count |
| n10000 | 45454→41599 (−8.5%) | 46473→48734 (+compile) | identical count |
| n1/n10/dt05 (eager, unchanged path) | −20..−30% | −20..−36% | identical count |
| mech_hdp/chunk_ref/chunk_k4 (explicit jit=False) | ±10% | ±20% | identical count |

Drift note: below-threshold and explicit-runtime cells run provably
identical code but moved −30%..+10% across processes (dt05 repeats same-day:
156–232ms). Cross-process XLA/machine drift bounds ecological deltas; the
causal claim rests on same-process A/B (P1/P5) plus matching post levels.

## Gate results
- G1 (bit-exact default-vs-eager, 8 configs incl. HDP H/W traces, tensor
  column, coupled two-area, HDP-10n, n1, dt05): PASS 8/8.
- G2 (10 affected suites): 23 passed + 1 skipped; 37 passed. 0 failures.
- G3 (matrix before/after): no cell regresses beyond drift; net gains above.

## Verdict
PASS — landed. Files: `jaxfne/_runtime_config.py` (default + doc),
`jaxfne/_model_simulate.py` (docstring only), refreshed
`artifacts/perf/matrix_051.json`.
