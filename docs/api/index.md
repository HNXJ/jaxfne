# API Reference

Canonical import:

```python
import jaxfne as jtfne
```

The root public contract is **191 symbols** (178 CANONICAL + 13 COMPATIBILITY) in
`jaxfne.__all__`. The 5-symbol JDNA additive surface (`PseudoGenome`,
`develop`, `load_pseudogenome`, `load_canonical_pseudogenome`,
`list_canonical_pseudogenomes`) is added in 0.4.17 on top of the frozen
0.4.13 contract. Tier membership is authoritative in
[`jaxfne/public_surface.py`](https://github.com/HNXJ/jaxfne/blob/main/jaxfne/public_surface.py)
and [`artifacts/public_surface_contract_v0413.json`](https://github.com/HNXJ/jaxfne/blob/main/artifacts/public_surface_contract_v0413.json).
See [Public surface contract](../public_surface_contract.md).

**Advanced** symbols (58) and **experimental/internal** symbols (16) are
importable from submodules but are not root exports.

!!! note "Proxy readouts"
    Field, EEG, MEG, and EMM outputs are computational proxies unless an
    explicit calibration receipt is supplied. See
    [Scope & status](../scope_and_status.md).

## Grammars

**Scientific:** Emitter → Source → Field → Probe → Objective → Optimizer → Manifest

**Execution:** CircuitSpec → `construct` → `Model` → `simulate` → `Signals`

`CircuitSpec` is a **conceptual category** (`Configuration | NeuronalTensor`),
not a concrete public production class — `construct` accepts either a
`Configuration` or a `NeuronalTensor` (+ `RuntimeConfiguration`). It is
unrelated to the experimental `jaxfne.experimental_hpc.CircuitSpec` type, which production `construct` does not accept.

## Structural map

### Core execution

Configuration builders, `RuntimeConfiguration`, `construct`, `simulate`,
`Signals`, paradigms, trials, receipts, and runtime validation.

| Resource | Page |
|----------|------|
| Configuration & simulation | [Core](core.md) |
| Runtime & `enable_x64` | [Runtime](runtime.md) |
| Validation helpers | [Validation](validation.md) |

Representative symbols: `Configuration`, `RuntimeConfiguration`, `Model`,
`Signals`, `construct`, `simulate`, `Paradigm`, `RunReceipt`, `validate_model`,
`validate_runtime_config`.

### NeuronalTensor

Declarative `Areas × Layers × NeuronTypes` CircuitSpec, canonical tensor
loaders, and bridges into `Configuration`.

| Resource | Page |
|----------|------|
| Tensor circuit model | [NeuronalTensor](neuronal_tensor.md) |

Representative symbols: `NeuronalTensor`, `Area`, `Layer`, `NeuronType`,
`load`, `load_canonical_neuronal_tensor`, `neuronal_tensor_to_configuration`.

### Fields and probes

Source tensors, laminar projection, field operators, and proxy readouts.

| Resource | Page |
|----------|------|
| Source schema | [Source schema](source_schema.md) |
| Field schema | [Field schema](field_schema.md) |
| Field operators | [Fields](fields.md) |
| Probe operators | [Probes](probes.md) |

Representative symbols: `construct_source_tensor`, `project_laminar_sources`,
`compute_fields`, `FieldOutput`, `eeg_proxy_transform`, `meg_proxy_transform`,
`emm_proxy_transform`.

### Objectives and optimization

Objective grammar, parameter specs, and optimizer entry points.

| Resource | Page |
|----------|------|
| Objectives | [Objectives](objectives.md) |
| Objective grammar guide | [Objective grammar](../guides/objective_grammar.md) |

Representative symbols: `Objective`, `ObjectiveReport`, `rate_targets`,
`agsdr`, `gsdr`, `gsgd`, `edge_parameter`, `matrix_parameter`, `random_search`.

### H-state / HDP adaptation

Finite-dimensional hidden biophysical state \(H\) and adaptive parameter
coordinates \(\Theta\) (synaptic and intrinsic). **H-state** is the latent
representation; **HDP** is the adaptive dynamical formulation mediated through
\(H\). \(\Theta\) denotes adaptive coordinates; synaptic storage \(W\) is
distinct.

| Resource | Page |
|----------|------|
| RBS / RBD / HDP guide | [RBS, RBD, and HDP](../guides/hdp.md) |
| Controllability étude | [HDP controllability / reachability](../etudes/hdp_controllability_reachability.md) |
| Runtime `hdp_params` groups | [Runtime](runtime.md) |

Representative symbols: `DEFAULT_HDP`, `DynamicState`, `ContinuationState`,
`restore_state`, `dynamic_state_from_model`.

Public population semantics: `h_state_locality="population"` with
theta-adaptation coefficients. Internal dispatch rule identifiers are not public
vocabulary.

### Evidence and validation

Manifests, hashes, export helpers, and validation reports.

Representative symbols: `manifest`, `asset_hashes`, `config_hash`,
`export_report`, `validation_report`, `save_receipt`, `sha256_file`,
`json_safe`.

### Compatibility

Deprecated aliases retained through 0.4.14. Prefer canonical names in new code.

| Symbol | Use instead |
|--------|-------------|
| `Net` | `Model` |
| `Config` | `Configuration` |
| `AGSDR` | `agsdr()` |
| `construct_neuronal_tensor` | `construct(tensor, RuntimeConfiguration(...))` |
| `load_neuronal_tensor` | `load(path)` |

Full deprecation notes: [Public surface contract](../public_surface_contract.md).

### Advanced namespaces

Low-level kernels, bridges, solvers, STDP, spectral analysis, and dev utilities.
Import from the defining submodule (see `ADVANCED_NAMESPACE` in
`public_surface.py`).

| Area | Submodule examples |
|------|-------------------|
| Emitter kernels | `jaxfne.emitters` — `simulate_edge_recurrent_izhikevich`, `EdgeList` |
| Bridges | `jaxfne.bridges` — `JaxleyBridge`, `JaxleyEmitterBridge` |
| Plasticity | `jaxfne.plasticity` — `STDPPlasticityConfig`, `update_stdp_weights_jax` |
| Solvers | `jaxfne.solvers` — `EulerSolver`, `DiffraxSolver` |
| Spectral | `jaxfne.analysis.spectral` — `spectrolaminar_psd_jax` |

## Minimal workflow

```python
import jaxfne as jtfne

jtfne.enable_x64()
tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
vm = signals.get("vm")
spk = signals.get("spk")
```

Chainable `Configuration` builders (`build_laminar_column`, suite presets) follow
the same `construct` → `simulate` path. See [Quickstart](../quickstart.md).
