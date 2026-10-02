# Capability record: delay (worked example, 0.5.2 item 6)

First record under agent-native step 4: equation, code, docs, skill,
tests and inspection for one capability in one place. TFNE says how delay
is specified, code how it is executed, docs the equation, units and
limitations, the skill when and how to configure and test it, tests the
zero-delay limit, arrival timing, continuation and composition, and
inspection the configured, realized and executed value.

## Equation

Configured delay `D` (ms, per connection rule) realizes to integer steps
at the construction timestep `dt` (0.5.2 decision 0b):

```text
d = round(D / dt)            # round half up, integer >= 0
```

Edge `e` with steps `d_e` delivers its presynaptic spikes through the
spike-history ring buffer (`_delayed_presynaptic_spikes`): zero steps
reads the current step (pre-0.5.2 path unchanged), `d_e > 0` reads
`spikes_{t-d_e}`. Measured end to end: a presynaptic spike computed at
step `s` first moves the postsynaptic membrane at step `s + 1 + d_e`
(the `+1` is the kernel's apply-next-step convention, identical for
`d_e = 0`). Units are ms in, steps out — never physical lengths,
conductivities, or distances.

Limitations: rule-level only (per-statement delay in S12 rule bodies is
0.5.4 item 1c); steps are realized for the construction `dt` (the record
carries that `dt`, so a mismatched simulation timestep is detectable, not
silent); the hand-set `EdgeList` surface keeps its own grid-alignment
contract (`edge_delay_steps_from_ms`, Protocol D) and does not round.

## Code

- `jaxfne/connectivity.py` — `delay_steps_from_ms` (the 0b conversion,
  single implementation), `_validate_rule_delay_ms`,
  `compile_connection_rules` (per-rule validation, per-edge `edge_delay_ms`,
  `connection_table` record), `ConnectionCompileResult.to_edge_list(dt_ms=)`
  (steps endpoint; `dt_ms=None` is the pre-0.5.2 EdgeList exactly).
- `jaxfne/_config.py` — `Configuration.connections(delay_ms=)` (rule
  surface; finite `>= 0`, else `ValueError`).
- `jaxfne/neuronal_tensor.py` — `InterConnection.delay_ms` /
  `AreaConnection.delay_ms` (inspection carriers, validated) and
  `_wire_connection` (forwards into the rule).
- `jaxfne/_construct_connectivity.py` + `jaxfne/_construct_core.py` —
  both compilers realize steps at `cfg.metadata["dt_ms"]`; a declared
  delay with no known dt fails closed.
- `jaxfne/tfne.py` — `_validate_tfne_delay_ms` (`E_DELAY_OUT_OF_RANGE`),
  `realize` (`s["edge_delay_ms"]`, `connection_specs`), `to_configuration`
  (rules carry `delay_ms`, `metadata["tfne_delay"]`, `E_DELAY_ROUNDS_TO_ZERO`),
  `compose_delay_metadata` (0.5.4 hook, declared + importable only).
- `jaxfne/emitters.py` — kernel consumption (`resolve_edge_delay_steps`,
  `_delayed_presynaptic_spikes`, finite-delay kernels, `delay_state`
  continuation). Unchanged by 0.5.2.
- `jaxfne/_model_manifest.py` — `tfne_delay` (configured ms + realized
  steps + dt) and `executed_delay` (max steps, delayed-edge count); both
  present only when delay is declared/positive.

## Docs

- `docs/doctrine/tfne_algebra.md` — Pipeline (delay ownership) and the
  tensor-bridge paragraph (bridge carries `delay_ms` for inspection;
  execution reads the specs).
- Receipt: `artifacts/programme/tfne_param02_receipt.md`.

## Skill

No new skill file (0.5.2 item 6 rule). Route through existing skills:
`jaxfne-science` for delay experiments (declare protocol + falsification
before running; reuse the arrival-timing and continuation patterns below),
`jaxfne-core` for authority routing, `jaxfne-repo` for implementation.

## Procedure (text, not a skill file)

To configure a delayed projection: declare `delay = <ms>` on the TFNE
rule (or `delay_ms=` on `connections()` / the tensor connection), build
with `to_configuration(..., dt_ms=<dt>)`, construct, and simulate at the
same `dt`. To test it: assert the zero-delay limit (absent vs `0.0`
bit-identical trajectories/edges), assert arrival shift against an
uncoupled baseline (first postsynaptic deviation moves by exactly the
realized steps), assert chunked continuation with spikes in flight
(segmented == continuous bit-exactly, `delay_state` included), and assert
the manifest records configured ms beside realized steps. Refusals to
cover: negative/non-numeric at validation, positive-rounding-to-zero at
configuration. Never infer a physical distance from a delay value.

## Tests

- `tests/test_tfne_delay_transfer.py` — the delay class battery
  (conversion units, refusal, limit identity, arrival timing,
  in-flight continuation, composition hook, edge_table visibility).
- `tests/test_tfne_parameter_transfer.py` — `test_c_*` transfer +
  refusal (equivalence-by-semantic-class pattern).
- `tests/test_tfne_ctx01.py` — integrated chain with delay.
- Kernel/continuation: `test_v0417_c2_delay_continuation.py`,
  `test_delay_boundary_v0417.py`, `test_closure_hp_reconciliation.py`
  (HDP + delay), `test_checkpoint_persists_every_array_field.py`.

## Inspection (configured -> realized -> executed)

| Stage | Surface |
|---|---|
| configured | rule `delay_ms` / `I["connection_specs"][*]["delay_ms"]` / `relation_origin(key)["params"]["delay"]` |
| realized | `s["edge_delay_ms"]` (per edge) / `cfg.metadata["tfne_delay"]` (`declared_ms`, `realized_steps`, `dt_ms`) / tensor `delay_ms` |
| executed | `EdgeList.delay_steps` via `edge_table()[*]["delay_steps"]` / `resolve_edge_delay_steps` (the kernel's own resolver) |
| manifest | `tfne_delay` + `executed_delay` |
| dynamics proof | arrival-shift vs uncoupled baseline; trajectory change vs zero-delay |

Epistemic status: RELATIVE_PROXY throughout — native current-based
dynamics, uncalibrated amplitudes. No distance/conductivity claim.
