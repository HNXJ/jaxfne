---
name: jaxfne-modeling-optimization-schema
summary: >-
  Deep schema/truth-gate reference for jaxfne Config, connectivity, selectors,
  and objective/trainer conventions. Condensed 2026-06-30 — Model/Signals/tune
  content moved to jaxfne-neural-network, NeuronalTensor/HDP content moved to
  jaxfne-neural-tensor (both to avoid duplication). CORRECTED 2026-06-30: the
  "Canonical Config structure" (cfg.circuit/.paradigm/.objective/.optimizer
  typed sub-specs) and "Welding semantics" (jtfne.weld()) sections below were
  found to be aspirational/fictional on verification — Configuration's real
  dataclass fields are networks/emitters/fields/probes/metadata
  (dataclasses.fields(Configuration)), and jtfne.weld does not exist. Kept
  here as an explicit warning, not removed silently, so nobody re-adds them
  from memory.
trigger: >-
  Use for connectivity rule schemas, node-identity/selector conventions, or
  objective/trainer path conventions. For Model/Signals/tune, use
  jaxfne-neural-network. For NeuronalTensor/HDP, use jaxfne-neural-tensor.
---

# jaxfne Modeling and Optimization Schema

## FICTIONAL — do not use (verified false 2026-06-30)

```python
jtfne.weld(cfg_a, cfg_b, duplicate_policy="suffix")   # jtfne.weld does not exist
cfg.circuit; cfg.paradigm; cfg.objective; cfg.optimizer  # not real fields
```

`Configuration`'s real fields (verified via `dataclasses.fields`):
`networks, emitters, fields, probes, metadata`. See `jaxfne-config`.

## Identity and selector rules (verified: `quartet`/id selector exists in `jaxfne/connectivity.py`)

Canonical node identity:

```text
area_id:local_id:layer:cell_type
```

Six-digit local IDs, e.g. `V1:000042:L4:PV`. Every node row (from
`model.neuron_table()`) carries `neuron_id, area, layer, cell_type, x, y, z`
— verify exact field names against `neuron_table()` output before assuming
`global_id`/`area_id`/`quartet` are literal dict keys; `quartet` is the
selector-resolution concept in `connectivity.py`, not necessarily a dict key
name. Empty selectors fail unless `allow_empty=True` is explicit
(`compile_connection_rules(..., allow_empty=False)`).

## Connectivity rule schema (real: `compile_connection_rules(neurons, connections, mechanisms, ...)`)

```python
{
  "name": "V1_L4_E_to_V4_L2_E",
  "pre": {"area": "V1", "layer": "L4", "cell_type": "E"},
  "post": {"area": "V4", "layer": "L2", "cell_type": "E"},
  "mechanism": "AMPA",
  "pattern": {"mode": "bernoulli", "probability": 0.10, "seed": 101},
  "weight": {"mode": "random_uniform", "low": 0.0, "high": 1.0, "scale": 0.25, "seed": 102},
  "plasticity": {"enabled": False, "rule": "none", "rate": 0.0},
}
```

Required pattern modes: `all_to_all`, `bernoulli`, `fixed_indegree`,
`fixed_outdegree`, `matrix`, `artifact_ref`. Weight artifact paths resolve
relative to the config file or a declared `artifact_root`; missing artifact
files fail at compile time unless `lazy=True` is explicit. Verify exact kwarg
names against `jaxfne/connectivity.py::compile_connection_rules` before
trusting this shape verbatim — it documents the concept, not a frozen contract.

## Schema rules

- Include `schema_version` in serialized configs.
- Add migration helpers (`jtfne.migrate_schema`) before changing schema shape.
- All config/manifest/training result data must be strict JSON-safe (`allow_nan=False`).
- Raw arrays inside config must be encoded or referenced through `artifact_ref` with path, array_name, and sha256.
- Negative noise and nonfinite drive/a/b/c/d must fail loudly.

## Objective and Trainer path conventions

Trainable parameter paths:

```text
cell.E.drive
cell.PV.noise
conn.feedforward_gain
mechanism.AMPA.g
```

Objective outputs declare source, metric, target/gate, weight, and selector.
Training results should save `best_parameters`, `best_score`, `history`,
`metrics`/`validation`, `truth_gates`, `seed`/search budget — see
`jaxfne-neural-network` for the real `Model.tune()`/`TuneResult` call shape,
including the caveat that the differentiable-optimizer branch is currently a
no-op guard.

## Simulation truth gate (default status, do not escalate)

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

Use one source mode per run. Do not double-count synaptic current:

```text
Forbidden: q = chi*(I_cap + I_ion + I_syn) + q_syn + q_ext
```

Required checks: finite arrays, explicit PRNG, runtime dtype respected,
readout shapes declared, source calibration status exported, field solver
status exported, mean firing rate in declared range unless null/instability
lesson.

## Homeostasis / HDP / plasticity — see the detailed skills

Full behavior and wiring status: `jaxfne-config` (`.homeostasis`/`.plasticity`
fluent methods) and `jaxfne-neural-tensor` (HDP module, tau-law, `RuntimeConfig`).

## Stop conditions

```text
invented public API (weld/circuit/paradigm/objective/optimizer typed sub-specs — see above)
hidden local scientific engine
NaN/Inf export
proxy path described as solved field
uncalibrated source described as physical amplitude
silent placeholder success
connection rule silently selects zero neurons
raw ndarray stored in JSON config without encoding/artifact_ref
```
