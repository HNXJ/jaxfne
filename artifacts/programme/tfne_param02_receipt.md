# TFNE-PARAM-02 — CLOSED. Declared delay reaches execution.

**Branch:** `w1-52` (0.5.2 ENGINE item 2)
**Entry state:** `d22b05e` (item 1) on top of `e6d3483`.
**Authorization:** 0.5.2 decision 0b (human, DECIDED 2026-09-23): delay
declared in ms; realized as `delay_steps = round(delay_ms / dt_ms)`
(round half up); configured ms and realized steps both recorded in the
manifest; a positive delay rounding to 0 steps refused. Correctness
repair, not optimization. No kernel change: the finite-delay kernels
already consumed `EdgeList.delay_steps` (Protocol D/C2); only the
compiler chain was missing.

## Semantics (decision 0b, exactly)

- `delay` declared in ms on the connection rule (`O[k] := [...; delay = 4.0]`,
  `Configuration.connections(delay_ms=)`, `Inter/AreaConnection(delay_ms=)`).
  `None` (undeclared) is the pre-0.5.2 execution bit-identically.
- Realized to integer steps at the construction timestep
  (`cfg.metadata["dt_ms"]`, set from `to_configuration(dt_ms=)` on the TFNE
  path). A declared delay with no known dt fails closed rather than zeroing.
- Refusals: negative/non-numeric at rule validation (`E_DELAY_OUT_OF_RANGE`
  on the TFNE path, `ValueError` on the config/tensor path); positive
  rounding to 0 steps (`E_DELAY_ROUNDS_TO_ZERO` / `ValueError`).
- Deliberately NOT grid-alignment: `emitters.edge_delay_steps_from_ms`
  (Protocol D, hand-set EdgeList surface) still rejects non-grid-aligned
  values; the compiler chain follows the authorized 0b round rule. The two
  contracts are documented at both definitions.
- Rule-level only. Per-statement delay in S12 rule bodies is 0.5.4 item 1c.

## Chain

```
rule delay_ms -> compile_connection_rules (validate, fan out per-edge
  edge_delay_ms, record on connection_table) -> to_edge_list(dt_ms)
  -> EdgeList.delay_steps
InterConnection.delay_ms / AreaConnection.delay_ms -> _wire_connection
  -> connections(delay_ms=)  [inspection + hand-built tensor path]
TFNE: rule delay -> realize (s["edge_delay_ms"], connection_specs)
  -> to_configuration (rules carry delay_ms; metadata["tfne_delay"]:
     declared_ms + realized_steps + dt_ms; E_DELAY_ROUNDS_TO_ZERO)
  -> construct (above) -> manifest ("tfne_delay" + "executed_delay"
     summary, both present only when delay is declared/positive)
```

`compose_delay_metadata` (in `jaxfne.tfne`) is the 0.5.4 composition hook:
pure per-area record merge, disjointness + dt-agreement enforced, declared
and importable only — nothing in 0.5.2 calls it.

## Tests

New `tests/test_tfne_delay_transfer.py` (7 tests, semantic-class pattern):
zero-delay limit bit-identical (trajectories, edges, storage; manifest
absent-stays-absent vs declared-zero-records-0); arrival timing (delayed
arrival shifts by exactly 4 steps against an uncoupled baseline);
continuation across a chunk boundary with spikes in flight (segmented ==
continuous bit-exactly: V_m, spikes, delay_state); composition hook
(importable, pure, refusals, not wired into execution); 0b conversion
units + `E_DELAY_ROUNDS_TO_ZERO`.

Updated: `test_c_delay_is_refused_not_dropped` -> transfer + out-of-range
refusal; `test_tensor_bridge_still_cannot_carry_parameters` (bridge now
carries `delay_ms` for inspection, weight/probability still absent,
execution still via specs); CTX-01 refusal test -> delay transfers
(2.0 ms -> 20 steps); doctrine `tfne_algebra.md` delay paragraphs.

```
test_tfne_delay_transfer.py + test_tfne_parameter_transfer.py
  + test_tfne_ctx01.py -> 47 passed
+ jdna_completion + tfne_algebra + tfne_execution + tfne_global_assumptions
  + construct_golden_snapshot + manifest_readout_compat + manifest_v005
  -> 165 passed
+ mechanism_aware_compiler + delay_boundary + c2_delay_continuation
  + edge01_selectors -> 32 passed
+ closure_hp + checkpoint_persist + optim_manifests + jdna_truth_gate
  + jdna_scenarios -> 54 passed, 1 skipped
ruff check on all touched files -> pass
```

## Changed canonical outputs

**None.** Verified, not regenerated:

- Absent delay builds exactly the pre-0.5.2 EdgeList (`to_edge_list`
  without dt omits `delay_steps` as before; sign-only compiler emits
  identical zeros; `test_construct_golden_snapshot.py` passes).
- Manifest gains `tfne_delay` / `executed_delay` keys only when delay is
  declared / positive; delay-free manifests are unchanged.
- `connection_table` rows gain `delay_ms: None` when undeclared — no
  golden/manifest surface compares it (snapshot suites green).
- No kernel, emitter, probe, vis, or atlas change.

## Status

- TFNE-PARAM-02: **CLOSED** (configured + realized + executed).
- Delay row in the PARAM-01 table: now "yes — ms configured, steps
  realized at dt, kernel-consumed", with the zero-delay limit pinned.
- Open (not this item): per-statement delay (0.5.4 item 1c), `X[k]`-rule
  frontier override, cross-area composition semantics (hook ready).
