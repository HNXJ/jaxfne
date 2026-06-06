---
summary: Use for jaxfne Config/Net/Paradigm/Objective/Trainer/Signals design, connectivity rules, selectors, welding, AGSDR, trainables, and objective outputs.
trigger: Use whenever the task mentions cfg, Config, Configuration, Net, Model, Paradigm, Objective, Trainer, Signals, signal layout, get_signal, AGSDR, tune, trainable, objective output, connectivity, selector, area-id-layer-type, quartet, mechanism, synapse, plasticity, weld, clone, construct, reconstruct.
---

# jaxfne Modeling and Optimization Schema Skill

## Route first

Map requested changes to the owner module:

```text
Config/sub-specs        -> jaxfne.config
compiled circuit        -> jaxfne.net
stimulus/task schedule  -> jaxfne.paradigm
objective metrics/gates -> jaxfne.objective
trainer loop            -> jaxfne.optim.trainer
signal query/layout     -> jaxfne.signals
selectors/connectivity  -> jaxfne.connectivity
readout operators       -> jaxfne.fields
```

Keep `jaxfne.core` as a facade, not a dumping ground.

## Canonical object flow

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

Compatibility aliases:

```text
Configuration -> Config
Model -> Net
FlatModel -> FlatNet
```

## Config rules

- Config uses typed sub-specs: runtime, geometry, circuit, probes, paradigm, objective, optimizer, metadata.
- Config serialized form includes `schema_version`.
- Config JSON cannot contain raw ndarray unless encoded or artifact-referenced.
- Keep transitional readers for old `metadata["circuit"]` paths if present.

## Selector/quartet rules

Canonical node identity:

```text
area_id:local_id_6digits:layer:cell_type
```

Example:

```text
V1:000042:L4:PV
```

Selectors must be explicit:

```python
{"area": "V1", "layer": "L4", "cell_type": "PV"}
```

Empty selector behavior: fail unless `allow_empty=True`.

## Connectivity rules

Connection rules compile selectors to edge arrays.

Required fields:

```text
name
pre selector
post selector
mechanism
pattern: all_to_all | bernoulli | fixed_indegree | matrix | artifact_ref
weight spec
plasticity metadata
control_key optional
```

For many-to-many selectors, default pattern must be explicit. Do not infer all-to-all silently.

## Source/readout guards

- Use one source mode per run.
- Avoid double-counting synaptic current.
- Proxy LFP/CSD/EEG/MEG are proxy readouts, not calibrated physical measurements.

## Trainer rules

Trainables are path-addressed:

```text
cell.E.drive
cell.PV.noise
conn.feedforward_gain
mechanism.AMPA.g
```

Objective outputs declare source, metric, selector, target/gate, and weight.

TrainingResult must save/load strict JSON and include best_config, best_parameters, best_score, history, metrics, validation, and status metadata.

## Stop conditions

```text
invented API
unsupported trainable silently accepted
connection rule selects zero neurons without allow_empty=True
raw ndarray in JSON Config
proxy readout described as solved field
```
