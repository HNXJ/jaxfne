---
name: jaxfne-modeling-optimization-schema
summary: Design or modify jaxfne Config, Net, Paradigm, Objective, Trainer, Signals, selectors, connectivity, welding, trainables, and AGSDR objectives.
trigger: Use whenever the task mentions config, cfg, Configuration, Config, Net, Model, circuit, cell params, mechanisms, connections, connectivity, quartet, selector, area, layer, cell_type, weld, join, clone, construct, reconstruct, Signals, get_signal, layout, paradigm, stimulus, task, objective, trainable, AGSDR, trainer, optimizer, tune, metric, source, field, probe, LFP, CSD, EEG, MEG, EMM, rate, kappa, truth, or manifest.
---

# jaxfne Modeling and Optimization Schema

## Purpose

Use this for almost every modeling, simulation, configuration, objective, and training change.

## Canonical objects

```text
Config: bio-circuit PCB sketch
Net: compiled biophysical circuit
Paradigm: task/trial/stimulus software
Objective: programmer's measure
Trainer: programmer/tuner
Signals: typed/queryable simulation output
```

## Canonical Config structure

Prefer typed sub-specs, not a metadata God object:

```text
Config(
  schema_version,
  runtime,
  geometry,
  circuit,
  probes,
  paradigm,
  objective,
  optimizer,
  metadata,
)
```

Compatibility:

```text
Configuration is an alias/wrapper for Config.
cfg.metadata["circuit"] is compatibility export only.
New implementation should use cfg.circuit, cfg.paradigm, cfg.objective, cfg.optimizer.
```

## Schema rules

- Include `schema_version` in serialized configs.
- Add migration helpers before changing schema shape.
- All config/manifest/training result data must be strict JSON-safe.
- Raw arrays inside config must be encoded or referenced through `artifact_ref` with path, array_name, and sha256.
- Negative noise and nonfinite drive/a/b/c/d must fail loudly.

## Identity and selector rules

Canonical node identity:

```text
area_id:local_id:layer:cell_type
```

Use six-digit local IDs:

```text
V1:000042:L4:PV
```

Every node must carry:

```text
global_id, area, area_id, local_id, layer, cell_type, quartet
```

Empty selectors fail unless `allow_empty=True` is explicit.

## Connectivity rules

Connection rules are a list of named rules:

```python
{
  "name": "V1_L4_E_to_V4_L2_E",
  "pre": {"area": "V1", "layer": "L4", "cell_type": "E"},
  "post": {"area": "V4", "layer": "L2", "cell_type": "E"},
  "mechanism": "AMPA",
  "pattern": {"mode": "bernoulli", "probability": 0.10, "seed": 101},
  "weight": {"mode": "random_uniform", "low": 0.0, "high": 1.0, "scale": 0.25, "seed": 102},
  "plasticity": {"enabled": False, "rule": "none", "rate": 0.0},
  "control_key": "feedforward_gain"
}
```

Required pattern modes:

```text
all_to_all
bernoulli
fixed_indegree
fixed_outdegree
matrix
artifact_ref
```

Weight artifact paths resolve relative to the config file or a declared `artifact_root`. Missing artifact files fail at compile time unless `lazy=True` is explicit.

## Welding semantics

Config welding is primary:

```python
cfg2 = jtfne.weld(cfg_a, cfg_b, duplicate_policy="suffix")
```

Self-weld behavior:

```text
V1 + V1 -> V1, V1_2
```

Welding preserves each component's internal connections only. It does not create cross-connections between welded components unless the user adds explicit connection rules after welding.

## Paradigm minimum

A minimal Paradigm must exist before construct/simulate gates:

```text
ConstantDCParadigm
EventSpec
ConditionSpec
TrialScheduleSpec
StimulusMappingSpec
```

A direct drive array is allowed as a compatibility input, but the future API is `net.simulate(paradigm=...)`.

## Objective and Trainer rules

Trainables use paths:

```text
cell.E.drive
cell.PV.noise
conn.feedforward_gain
mechanism.AMPA.g
```

Objective outputs declare source, metric, target/gate, weight, and selector.

Training results must save:

```text
best_config
best_parameters
best_score
history
metrics
validation
truth_gates
seed/search budget
```

## Simulation truth gate

Default status:

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Use one source mode per run. Do not double-count synaptic current:

```text
Forbidden: q = chi*(I_cap + I_ion + I_syn) + q_syn + q_ext
```

Required checks:

```text
finite arrays
explicit PRNG
runtime dtype respected
readout shapes declared
source calibration status exported
field solver status exported
mean firing rate in declared range unless null/instability lesson
```

## Stop conditions

Stop and report when any appear:

```text
invented public API
hidden local scientific engine
NaN/Inf export
proxy path described as solved field
uncalibrated source described as physical amplitude
silent placeholder success
connection rule silently selects zero neurons
raw ndarray stored in JSON config without encoding/artifact_ref
```
