---
name: jaxfne-modeling-optimization-schema
description: >-
  Audit jaxfne connectivity rules, selectors, serialized configuration,
  trainable parameter paths, objectives, and optimization reports. Use when
  changing schema or model-selection contracts.
---

# jaxfne modeling and optimization procedure

Read `catalog-glossary-jaxfne` and the mathematical project source first. This
skill defines checks and routing, not objective or HDP mathematics.

## Public configuration boundary

Verify `Configuration` fields and methods with the live dataclass and public
import. Do not invent typed `.circuit`, `.paradigm`, `.objective`, or
`.optimizer` sub-specs, or a `jtfne.weld(...)` helper.

## Identity and selectors

Use the fields emitted by `model.neuron_table()` and the selector contract in
`jaxfne/connectivity.py`. Do not assume a key name such as `global_id` or
`quartet` without checking the current return value.

Empty selector behavior must be explicit. A rule that selects zero nodes should
raise or require an explicit `allow_empty=True` policy.

## Connection rule checks

Before changing a serialized connection rule:

1. Verify the current `compile_connection_rules` signature.
2. Verify pre/post selector scope and mechanism names.
3. Verify pattern and weight modes in code/tests.
4. Check deterministic seed handling.
5. Check edge count and finite weights after compilation.
6. Check sparse/dense semantic equivalence on a small case when both paths
   exist.

Keep raw arrays out of JSON unless the current artifact-reference contract
encodes path, array name, and content identity.

## Objective and tuning checks

Objective output should declare its source, metric, target/gate, weight, and
selector. Tuning reports should preserve parameters, score, history, validation,
seed/search budget, and current status metadata.

Verify the actual `Model.tune()` path before describing an optimizer as
differentiable or gradient-based.

## Truth and serialization

Preserve current conservative fields:

```text
claim_level
field_solver_status
field_claim_level
physical_amplitude_calibrated
```

`field_solver_status="linear_solver"` is compatibility metadata for the current
proxy path. It does not establish a solved field.

## Connectivity rule schema

Use the live `compile_connection_rules` signature and tests before relying on
this conceptual shape:

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

## Field metadata contract

Every field solver output must carry the canonical trio:

| Key | Value |
|-----|-------|
| `field_claim_level` | `"proxy_readout"` |
| `field_solver_status` | `"linear_solver"` (proxy projection) or `"experimental_pde_solver"` (both Poisson entry points) |
| `physical_amplitude_calibrated` | `False` |

Verify the metadata surface for each selected solver/projection path in the
active implementation and its tests; do not assume a unified output object.

## Homeostasis / HDP / plasticity — see the detailed skills

Full behavior and wiring status: `jaxfne-config` (`.homeostasis`/`.plasticity`
fluent methods) and `jaxfne-neural-tensor` (HDP module, tau-law, `RuntimeConfig`).
Require finite arrays, explicit randomness, declared shapes, and strict JSON.

## Stop conditions

Stop for invented APIs, silent zero-node selection, hidden local engines,
NaN/Inf export, unencoded arrays, proxy-as-solver language, or a schema change
without an explicit migration plan and compatibility test.
