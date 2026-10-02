# Receipt — 0.5.3 ENGINE item 2: H/W recording budgets (w1-53)

Parent commit: `01fea5b` (item 1).

## 0.5.1 opt-in selective/downsampled recording API: does not exist

Verified by grep over `jaxfne/` for
`record_stride|record_subset|selective` and over 0.5.1 reports: the only
hits are vis display downsampling, the legacy STDP streaming
`downsample_factor`, and the pre-0.5.1 `run_continuation_strided` /
`memory_report` stride (23-REC-01). 0.5.1 B6 (`bottlenecks_051.md`)
explicitly excluded item 2c since recording stayed below threshold
(15.9% < 25%). The budgets below are therefore implemented fresh.

## Declared API (opt-in; full recording stays the default)

Via `hdp_params` (same transport as `record_weight_trace`):

- `record_stride`: positive int, default 1. Kept H/W frame j equals the
  full-trace frame at step j*stride exactly.
- `record_h_subset` / `record_w_subset`: 1D integer index arrays,
  default None (= all neurons/edges).
- Fail closed: non-integer/zero/negative stride; empty, non-integer,
  or out-of-range subsets; `record_w_subset` with
  `record_weight_trace=False`; `record_h_subset` under population H
  locality (no neuron axis).

Dynamics are untouched: decimation is a post-scan slice of the
returned traces. `V_m`/spikes/sources are never strided.

## Code

- `jaxfne/emitters.py`: `_validate_recording_budget` +
  `_decimate_hw_traces` (small, named); legacy HDP kernel takes the
  three params and decimates at all three return sites
  (standard/boundary/population). Defaults return traces unchanged
  (same objects, zero overhead).
- `jaxfne/_hdp_registrable_kernel.py`: same three params, both return
  sites (zero-delay + finite-delay).
- `jaxfne/_model_simulate.py`: `_hdp_kernel_kwargs` carries the three
  keys (plain path, legacy + registered); `_simulate_continuation_arrays`
  applies subset selection per step (kernel) and stride post-stack over
  H/W only, and records the budget (`record_stride`, subsets) in the
  HDP diag store.
- `jaxfne/_pipeline.py`: `compile_step_fn` named passthrough
  (hdp_kwargs-wins pattern, as `record_weight_trace`).
- Untouched: `hdp_rule.py`, validators (`public_surface.py` — new keys
  are warnings-only "unrecognized" there, same as any future key; no
  simulate-path enforcement), `fields/**`, `vis/**`, benchmarks.

## Peak-transient scope (honest)

Stride/subset bound the *returned* H/W recording volume
(O(T/k) frames). The scan-time transient is unchanged on the plain
path; long-horizon peak reduction composes with chunked continuation
(one segment per chunk, decimated frames kept). `memory_report`
stride estimates already model this.

## Tests

`tests/test_recording_budgets_053.py`: 24 passed (31.2 s, CPU).

- Default path bit-identical (absent keys vs explicit defaults: V,
  spikes, sources, H/w traces + finals).
- Budgets leave dynamics untouched (V/spikes/sources equal).
- Stride kept-frames == full frames (incl. uneven tail, incl.
  `record_weight_trace=False` H-only).
- H/W subsets == full frames at indices; stride+subset compose.
- Registered rule (`synthetic_presyn_gain`): stride/subset exact +
  default bit-identical.
- Continuation: strided chunks concatenate to continuous strided
  frames; subsets apply per step; V never strided.
- Fail-closed battery (bad stride ×5, bad subsets ×5, w-subset
  without weight trace).
- Recorded volume scales with the declared budget.

Regressions: `test_continuation_contract` + `test_state_ownership_053`
+ `test_checkpoint_persists_every_array_field` (36 passed);
`test_hdp01_registrable_qualification` + `test_phaseC_H_carry_resume` +
`test_hdp_kernel_standalone` (27 passed).

## Invariants held

- Full recording stays the default; default-path outputs bit-identical
  (tested, not assumed).
- `K_HDP=0` null behavior unchanged; H!=HDP separation preserved.
