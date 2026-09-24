# Receipt — 0.5.3 ENGINE item 1: H/W/K ownership (w1-53)

HEAD: `5d679a19373648624b69c1f94a5e5c02e3f29a2d` (branch `w1-53`).
Doctrine: Atlas source 8 (`artifacts/project_sources/8_atlas.md`) +
`docs/doctrine/rbs_rbd_hdp.md`. H = RBS; HDP = hidden-state dependent
plasticity; H != HDP; attenuation != adaptation; bounded != stable.

## Inspection surfaces (configured -> realized -> executed)

| State | Configured | Realized | Executed |
|---|---|---|---|
| H | `RuntimeConfig.hdp_params` (gains, bounds, `h_state_dim/locality/readout/coupling`) | `model.params["hdp_initial_H"]`, `DynamicState.H` via `dynamic_state_from_model` | `last_hdp_diagnostics()["H_trace"/"H_final"]`, `ContinuationState.dynamic.H` |
| W | edge weights (connections/TFNE realize `s`), plasticity identity (`hdp_rule`, `K_HDP`, `K_w_ctrl`) | `params["edge_list"].weight`, `params["hdp_initial_w"]`, `DynamicState.w`, `edge_table()[*]["weight"]` | `last_hdp_diagnostics()["w_final"/"w_trace"]`, `ContinuationState.dynamic.w` |
| K | `Simulation.seed` | `ContinuationState.prng_key/step_index` (`continuation_state_from_model(seed=…)`), per-step chain via `_advance_prng_key` | noise draws (`continuation_noise_schedule` on the HDP Model path); same seed + same inputs -> bit-identical (item 4 owns domains) |

Which rule mutates which state:

- Legacy kernel (`simulate_edge_recurrent_izhikevich_hdp`, default
  `signed_linear` family): `K_HDP` gates the H-difference weight term;
  `K_w_ctrl` an independent weight-restoration term; neither gates the H
  equation (H evolves with `K_HDP=0` while W is untouched — the H!=HDP proof).
- Registered rules: mutated targets declared in
  `HDPRuleDescriptor.theta_targets` (`edge_weight` and/or `drive_bias`),
  read via `get_hdp_rule` (read-only; `hdp_rule.py` untouched).
- Routing fact: identity HDP params disengage HDP at Model level
  (`hdp_is_engaged` False -> baseline kernel, diagnostics None, run ==
  fixed-W path). The `K_HDP=0` null pins H at 1.0 at the kernel level
  (phaseC pattern); at Model level identity params take the baseline path.

## Code change

None in `jaxfne/`. No gap required new inspection code: every H/W/K
stage is readable from existing surfaces. The small named read-only
helper `ownership_snapshot` lives in the test file (pure reads, no
mutation, no container restructure).

## Tests

`tests/test_state_ownership_053.py`: 15 passed (17.1 s, CPU).

- H chain (configured gains -> dynamic shape -> trace shape; nonzero moves H).
- Kernel-level null pins H at 1.0, w untouched.
- Identity-params routing: diagnostics None + V/spikes == fixed-W run.
- H!=HDP: H evolves with `K_HDP=0` while `w_final == edge_list.weight`.
- W chain; plasticity-off leaves weights untouched; `K_HDP=0` null is fixed-W.
- `edge_table` exposes realized W per edge.
- K: seed -> `prng_key`/`step_index`; same-seed bit-identity; key advances
  across chained segments.
- Legacy mutation map; registered-rule `theta_targets` declaration check;
  `DynamicState` full mutable set; `with_hdp_initial_state` seeds realized H/W.

## Invariants held

- `K_HDP=0` null behavior unchanged (kernel-level pin verified).
- H!=HDP separation preserved and tested.
- No state containers restructured; no other paths touched.
