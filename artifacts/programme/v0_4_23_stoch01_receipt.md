# 23-STOCH-01 receipt — NO_CHANGE (audit; ownership verified, one lock-in test)

**Branch:** `dev` (on top of `f6d40ff`)
**Verdict:** no RNG redesign. One test added (composition lock-in, no
production-code change).

## RNG domain map (execution)
- Dynamics noise, single domain per run: kernels draw
  `split(key) -> normal(noise_key, (T, N))`; legacy HDP also accepts a
  pre-generated `noise_schedule` into the same domain.
- Continuation ownership: `ContinuationState` carries `prng_key` (K_t),
  `step_index`, and `delay_state`; `run_continuation` advances keys via
  `_advance_prng_key`; per-step kernel calls consume `key_t`. K_t ∈ state ✓.
- Registrable kernel: internal bulk draws only (no `noise_schedule`
  param) — same ownership shape as baseline kernels.
- Poisson drive: pre-generated from `sim.seed`; continuation path rejects
  it loudly (cursor boundary declared in code, not silent).
- Construction RNG (connectivity/geometry/jdna/tensor) uses separate user
  seeds — independent domains by construction.
- Batch: `split(base, n_seeds)`-derived keys per seed.

## Probe battery (all on current tree; script temp, results here)
| property | result |
| --- | --- |
| legacy-HDP noisy cont-vs-cont | PASS (exact) |
| recording on/off dynamics | PASS (V/S bit-exact) |
| JIT-vs-eager noisy HDP | PASS (V/S exact) |
| batch deterministic + key-independent | PASS |
| delayed builtin-HDP noisy cont-vs-cont | PASS (exact) |
| delayed registered-HDP noisy cont-vs-cont | PASS (exact) |
| legacy / registered noisy bulk-vs-cont | NOT exact (both; see boundary) |

## Declared boundary (not a defect)
Bulk-vs-continuation with active noise is exact in NO kernel (legacy
included): bulk consumes one `(T, N)` draw block while continuation
consumes per-step `(1, N)` draws from the advanced key sequence. The
promised contract — cont-vs-cont replay with K_t carried, as tested by
`test_continuation_contract` (C2/C3/C4) — holds everywhere, now
including delay + registered-HDP + noise. Unifying the two streams
would be an RNG redesign for an unpromised property: out of scope.

## Delta
- `tests/test_hdp_delayed_registrable.py`:
  `test_model_continuation_delayed_registered_noisy` (locks probe 7).
