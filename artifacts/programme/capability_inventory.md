# Capability inventory: agent-safe operations (0.5.2 item 6, agent-native step 3)

Existing JaxFNE capabilities inventoried as agent-safe operations — task-shaped
entry points an agent uses instead of the 266 classified root symbols. Each
row names the operation, its entry function(s), how to inspect the result,
and its standing. Epistemic levels are NOT assigned here (0.5.2 item 4 owns
them): everything below is RELATIVE_PROXY or uncalibrated native dynamics
unless its row says otherwise. Candidate rows (`candidate: true` in
`artifacts/programme/atlas_coverage.json`) are out of scope until human
authorization; they are listed under Refused/Omitted, never as operations.

## Specify

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Specify a network in TFNE | `jaxfne.tfne.parse` -> `resolve` (grammar `tfne/2`, project source 7) | `ExplicitModel` (NF, digest, `rule_params`, boundaries) | canonical spec language |
| Declare a Configuration directly | `Configuration().areas/column/population/cell_types/connections/mechanisms/probes/field/runtime/...` | `cfg.metadata["circuit"]` (rules carry `status: declared_not_compiled`) | hand-built path; TFNE preferred for new models |
| Declare a delay | rule `delay = <ms>` / `connections(delay_ms=)` / `Inter/AreaConnection(delay_ms=)` | rule record; refusals fail closed (App. A) | 0.5.2 item 2, carried |
| Declare relative geometry | leaf `G = [z0; z1]` (likewise `x`, `y`, fractions in [0,1]) | `s["geometry"]`; outside-[0,1] refused | 0.5.2 item 1, carried |
| Declare a mechanism | rule `mechanism = AMPA/GABA_A/NMDA/GABA_B` or custom `+ tau_ms` | `resolve_mechanism` classes (CANONICAL/CUSTOM_DEFINED/UNRESOLVED/NOT_PERMITTED) | TFNE2-07 vocabulary; `GABA` refused (ambiguous) |

## Complete / realize

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Complete TFNE under JDNA | `jaxfne.jdna.completion.complete_tfne` / `develop` (explicit `K_D`) | positions + `value_origins` per leaf | TFNE-JDNA boundary doctrine |
| Realize to (s, h0, I) | `jaxfne.tfne.realize` / `flatten` | `s` (edges, weights, `edge_delay_ms`, geometry), `h0`, `I` (slices, `connection_specs`, origins) | single resolved representation (PARAM-01) |
| Bridge structure to tensor | `to_neuronal_tensor` | areas/layers/types; `delay_ms` inspection only | structural role; execution reads specs |

## Compile / construct

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Build executable Configuration (TFNE) | `to_configuration(r, duration_ms, dt_ms, emitter, dtype)` | circuit rules + mechanisms; `tfne_delay` / `tfne_geometry` metadata | replaces circuit connectivity from specs |
| Bridge tensor to Configuration | `neuronal_tensor_to_configuration` | circuit rules via `_wire_connection` | hand-built tensor path |
| Compile rules to edges | `compile_connection_rules` / `to_edge_list(dt_ms=)` | `ConnectionCompileResult` (edge arrays, tables) | host-side, deterministic |
| Construct runnable Model | `jaxfne.construct(cfg or tensor, runtime)` | `model.params` (edge_list, positions, emitter), `edge_table()`, `neuron_table()` | canonical lifecycle entry |

## Simulate (+ continuation)

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Run a simulation | `jaxfne.simulate(model, duration_ms, dt_ms, seed)` / `model.simulate(sim, paradigm, continuation, return_state)` | `Signals` (V_m, spikes, sources, `delay_state`, `continuation_step_offset`) | edge_list finite-delay kernels; dense backend rejects nonzero delay |
| Continue across chunks | `continuation=<state>` + `return_state=True` (full `init_state` incl. `delay_state` when delays > 0) | segmented == continuous bit-exact (C2 pattern) | B8 seed-harness divergence owned by 0.5.3 item 3 |
| Batch over seeds | `model.simulate_batch(sim, n_seeds)` | stacked outputs | smoke-covered |

## Observe (source / field / probe — current state, items 3–5 own the repairs)

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Read spikes / voltage / source | `spk_probe`, `vm_probe`, `source_probe` -> `ProbeReadout` | readout object + metadata | proxy status; unified Q is item 3 |
| Project sources to field | `project_laminar_sources` / `project_sources_to_laminar_field` -> `FieldOutput` | field array + diagnostics | proxy; epistemic levels are item 4 |
| LFP/CSD/EEG/MEG proxies | `lfp_proxy_probe`, `csd_proxy_probe`, `eeg/meg/emm_proxy_probe`, `csd_tensor` | probe readouts | proxy; calibration transform is item 4 |
| Declare probes/electrodes | `create_probe`, `Configuration.probes(...)` / `.field(...)` | `cfg.probes`, manifest | declared semantics are item 5 |

## Mutable state / plasticity (current state, 0.5.3 owns controls)

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Declare plastic rule identity | rule `plasticity = <name>` | `rule_params` / `relation_origin` provenance; `h0["w"]` separate from `s["edge_weight"]` | provenance only, not an edge parameter |
| Enable HDP | `Configuration.hdp(...)` (+ registrable HDP surface) | HDP controller state | enable/disable/clamp per rule is 0.5.3 item 5 |
| Null plasticity | `K_HDP=0` | fixed-W path | template for disable; bit-identity vs fixed-W is 0.5.3 acceptance |

## Verify / inspect

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Model manifest | `model.manifest(signals)` | JSON-safe run summary (`tfne_delay`, `tfne_geometry`, `executed_delay`, backend/receptor metadata) | record-only keys; delay/geometry keys are 0.5.2 |
| Run receipt | `model.run_receipt(...)` / release receipts | deterministic receipt ID | evidence pipeline |
| Test gates | `scripts/run_test_gate.py dev/broad/slow` | PASS + timing ledger (`test_timing_051.md`) | 0.5.1 accelerated; full suite at merge only |
| Protocol validations | `jaxfne/protocol_c`, `protocol_e_integration` (C2/E2/E3) | bit-exact continuation receipts | delay continuation covered |
| TFNE-vs-handwritten check (PARAM-03) | `test_tfne_matches_handwritten_canonical_construction` | tau equality {2.0} | PRESENT — grep-verified 0.5.2 item 6, no implementation needed |

## Compose (0.5.4 owns semantics; hook ready)

| Operation | Entry | Inspect | Standing |
|---|---|---|---|
| Merge delay records | `jaxfne.tfne.compose_delay_metadata` | pure merge; overlap/dt-mismatch refused | declared + importable only |
| Area composition `N_A ⊕_C N_B` | — (0.5.4 item 1) | hierarchical ≡ flattened bit-identical (future) | not yet an operation |

## Refused / omitted (fail closed, not operations)

`delay` with negative/non-numeric value; positive delay rounding to 0
steps; geometry outside [0,1] / half-declared / degenerate; unresolvable
mechanism (`GABA`, unknown names); per-statement delay (0.5.4 item 1c);
`X[k]`-rule frontier override (language decision); Φ → X feedback;
magnetic-field-as-input; calibrated HH/field rows (`candidate: true` —
need independent evidence + human authorization before any engine work).

## Appendix A — refusal codes (0.5.2)

`E_GEOMETRY_OUT_OF_RANGE`, `E_GEOMETRY_AMBIGUOUS`, `E_DELAY_OUT_OF_RANGE`,
`E_DELAY_ROUNDS_TO_ZERO`, `E_DELAY_COMPOSITION_AMBIGUOUS` (hook only),
`E_MECHANISM_UNRESOLVED` / `E_MECHANISM_NOT_PERMITTED`,
`E_PARAM_UNSUPPORTED` (retired by items 1–2; no code references remain).
