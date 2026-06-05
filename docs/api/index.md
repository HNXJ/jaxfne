# API Reference

Canonical import:

```python
import jaxfne as jtfne
```

This page is the complete index of the public API (`jaxfne.__all__`, 146 names),
grouped by module. Per-module pages carry detailed signatures and examples.

!!! note "Scope & truth gates"
    All field/EEG/MEG/EMM outputs are **computational proxies**, not solved PDE
    or sensor-level signals (`field_solver_status = "laminar_proxy_no_pde"`,
    `physical_amplitude_claim_allowed = False`). The package is a
    `computational_scaffold` for teaching, prototyping, and validation — not a
    calibrated biological simulator. See
    [Scope and limitations](../scope_and_limitations.md).

## Module pages

| Page | Covers | Public names |
|---|---|---|
| [Core](core.md) | Configuration, Model, Simulation, Signals, readouts, receipts, suites | 66 |
| [Emitters](emitters.md) | Izhikevich emitter, receptors, synapses, EIG networks, edge lists | 19 |
| [Fields](fields.md) | Source→laminar projection, `FieldOutput`, proxy diagnostics | 9 |
| [Probes](probes.md) | EEG/MEG/EMM proxy transforms (+ probe operators guide) | 3 |
| [Objectives](objectives.md) | `Objective`, `ObjectiveReport`, rate targets | (see Core) |
| [Runtime](runtime.md) | `RuntimeConfig`, `enable_x64`, `runtime_report` | (see Core) |
| [Validation](validation.md) | config/field validators, `operator_status` | (see Core) |
| _(no page yet)_ | **Optimizers** (`optim`), **IO/receipts**, **Bridges**, **Paradigms**, **Sharding**, **Tutorial utils**, **Experimental HPC** | 39 |

> Several public names (optimizers, IO, bridges, paradigms, sharding) do not yet
> have a dedicated module page — they are listed in the index below. See the
> docs audit (`internal_docs/docs_audit_v0330.md`) for the gap plan.

## Minimal workflow (verified)

```python
import jaxfne as jtfne

cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=1000.0, dt_ms=0.1)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)

vm = signals.get("vm")            # membrane voltage [T, N]
spk = signals.get("spk")          # spikes [T, N]
e_idx = model.select(cell_type="E")
vm_e = signals.get("vm", cell_type="E")
```

---

## Complete public symbol index

`func`/`class`/`const`/`module` as resolved from `jaxfne.__all__`. Summaries are
the first docstring line; `_(undocumented)_` marks public callables with no
docstring (tracked in the docs audit (`internal_docs/docs_audit_v0330.md`)).

### Core (66)

| Symbol | Kind | Summary |
|---|---|---|
| `AxisSpec` | class | Typed descriptor for one tensor axis in the TFNE scaffold. |
| `BasisSpec` | class | Typed descriptor for the computation basis of a TFNE run. |
| `config_to_configuration` | func | Map the `network`/`emitter`/`field`/`probes` sections to a `Configuration`. |
| `config_to_geometry` | func | Map the `geometry` section to a `LaminarSourceGeometry`, or `None`. |
| `config_to_simulation` | func | Map the `run` section of a `JaxFNEConfig` to a `Simulation`. |
| `config_to_trial_batch` | func | Map the `trials` section and conditions to a `TrialBatch`. |
| `config_truth_boundary` | func | Return a JSON-safe copy of the truth boundary section. |
| `Configuration` | class | Declarative TFNE model configuration. |
| `configuration` | func | —  _(undocumented)_ |
| `ConfigValidationResult` | class | Report container for configuration validation. |
| `construct` | func | —  _(undocumented)_ |
| `dataset_spec` | func | Return a DatasetSpec schema declaration. |
| `DatasetSpec` | class | Manifest-safe dataset/alignment declaration for observed data. |
| `default_basis_spec` | func | Return the default BasisSpec matching the current laminar-proxy scaffold. |
| `enable_x64` | func | Enable JAX float64 mode before constructing arrays and report status. |
| `get_signal` | func | Thin free-function accessor that delegates to `Signals.get`. |
| `JaxFNEConfig` | class | JSON-safe container for a complete `.jcfg.json` TFNE specification. |
| `laminar_source_geometry` | func | Build a `LaminarSourceGeometry` from an ordered population sequence. |
| `LaminarPopulation` | class | Metadata descriptor for one named laminar cell population. |
| `LaminarSourceGeometry` | class | Metadata descriptor for the full laminar source geometry. |
| `load_config` | func | Load a `.jcfg.json` file and return a `JaxFNEConfig`. |
| `matrix_parameter` | func | Create a matrix parameter specification for tuning weight matrices. |
| `MatrixParameterSpec` | class | Declarative specification for a tunable weight matrix parameter. |
| `Model` | class | —  _(no docstring; dataclass)_ |
| `Objective` | class | Declarative objective specification: losses, regularizers, and diagnostic gates. |
| `objective` | func | —  _(undocumented)_ |
| `ObjectiveReport` | class | Structured, immutable result of evaluating an Objective against Signals. |
| `operator_status` | func | Return the current operator status registry for all declared operators. |
| `Probe` | class | —  _(no docstring; dataclass)_ |
| `provenance_receipt` | func | Capture release provenance atomically. |
| `rate_targets` | func | Create a multi-group firing-rate objective. |
| `readout_spec` | func | Build a ReadoutSpec for declarative feature extraction. |
| `ReadoutResult` | class | Result of applying a ReadoutSpec to Signals. |
| `ReadoutSpec` | class | Declarative specification for extracting a scalar feature from Signals. |
| `run_receipt` | func | Build a RunReceipt for a completed simulation run. |
| `run_trials` | func | Execute a batch of trials using the model. |
| `RunReceipt` | class | Complete, JSON-safe record of a single simulation run. |
| `runtime` | func | —  _(undocumented)_ |
| `runtime_report` | func | —  _(undocumented)_ |
| `RuntimeConfig` | class | JAX runtime and dtype policy. |
| `Signal` | class | Simulation output container holding multiple arrays. |
| `Signals` | class | Simulation output container holding multiple arrays. |
| `simulate` | func | Run a simulation with the given model. |
| `Simulation` | class | —  _(no docstring; dataclass)_ |
| `simulation` | func | —  _(undocumented)_ |
| `standard_visual_omission` | func | Construct a Paradigm with standard visual oddball/omission task conditions. |
| `stimulus_schedule` | func | Build a `StimulusSchedule` from a sequence of events. |
| `StimulusSchedule` | class | Explicit native-drive schedule for event-aligned stimulus injection. |
| `suite2_celltype_presets` | func | Return compact E/PV/SST/VIP reduced-emitter preset metadata. |
| `suite2_four_celltype_config` | func | Build the Suite No. 2 four-emitter E/PV/SST/VIP configuration. |
| `suite2_net1_config` | func | Build net1: a uniformly sampled 3D E/PV/SST/VIP column. |
| `suite2_run_bundle` | func | Run simulation, readouts, manifest, and receipt for Suite No. 2 notebooks. |
| `suite2_simulation` | func | Create a Suite No. 2 simulation with deterministic runtime metadata. |
| `suite2_single_neuron_config` | func | Build the Suite No. 2 one-emitter configuration. |
| `suite2_tune_noise_agsdr_adam` | func | Tune Poisson-drive amplitude toward a target mean firing-rate range. |
| `suite2_v1_v4_config` | func | Build the Suite No. 2 V1-V4 laminar scaffold with six layers per area. |
| `surrogate_config` | func | Return a SurrogateConfig declaration for an Optax gradient path. |
| `SurrogateConfig` | class | Declared surrogate-gradient metadata for discontinuous emitter paths. |
| `trial_batch` | func | Create a TrialBatch by repeating conditions. |
| `TrialBatch` | class | A collection of trial specifications to be run. |
| `TrialBatchResult` | class | Results from a batch of trials. |
| `TrialResult` | class | Result of a single simulation trial. |
| `TrialSpec` | class | Specification for a single simulation trial. |
| `TuneResult` | class | Result object returned by Model.tune() with multi-parameter optimization. |
| `validate_config` | func | Validate a `JaxFNEConfig` and return a `ConfigValidationResult`. |
| `with_emitter_parameters` | func | Functional wrapper for `Model.with_emitter_parameters`. |

### Emitters (19)

| Symbol | Kind | Summary |
|---|---|---|
| `EdgeList` | class | Sparse recurrent connectivity as a JAX pytree. |
| `EIGNetwork` | class | Lightweight description of an E/PV/SST/VIP-like reduced network. |
| `Emitter` | class | Base class for package-native emitter facades. |
| `GLIFEmitter` | class | Loud stub — raises `NotImplementedError` (not yet implemented). |
| `izhikevich_params_from_labels` | func | Create reduced Izhikevich parameters from explicit cell labels. |
| `IzhikevichEmitter` | class | Reduced Izhikevich emitter facade with JAX-native step. |
| `IzhikevichParams` | class | Parameter container for a reduced Izhikevich population. |
| `LIFEmitter` | class | Loud stub — raises `NotImplementedError` (not yet implemented). |
| `make_edge_list_from_dense` | func | Convert a dense recurrent weight matrix into a sparse EdgeList. |
| `make_eig_network` | func | Build a minimal EIG network with laminar depth positions. |
| `ReceptorSpec` | class | Metadata declaration for a synaptic receptor. Not a biological kernel. |
| `simulate_edge_recurrent_izhikevich` | func | Simulate reduced Izhikevich emitters with sparse recurrent synapses. |
| `simulate_eig_izhikevich` | func | Simulate a reduced EIG Izhikevich scaffold using `jax.lax.scan`. |
| `simulate_receptor_exponential_izhikevich` | func | v0.0.11 receptor-indexed exponential recurrent kernel. |
| `standard_receptor_specs` | func | Provide standard declarative receptor metadata. No biological claim. |
| `standard_receptor_tau_table` | func | Return the receptor_index → tau_ms lookup table used by v0.0.11. |
| `SynapseLayer` | class | Exponential synapse layer returning recurrent input currents. |
| `SynapseSpec` | class | Metadata declaration for a synapse. Not a biological kernel. |
| `SynapseState` | class | —  _(no docstring; dataclass)_ |

### Fields (9)

| Symbol | Kind | Summary |
|---|---|---|
| `compute_conservation_proxy_diagnostics` | func | Compute conservation-inspired proxy diagnostics over existing source/field arrays. |
| `construct_source_tensor` | func | —  _(undocumented)_ |
| `FieldOutput` | class | Container for laminar proxy field/readout arrays. |
| `LinearReadout` | class | —  _(no docstring; dataclass)_ |
| `probe_laminar_modes` | func | —  _(undocumented)_ |
| `project_laminar_sources` | func | Project source traces to laminar proxy contacts. |
| `project_sources_to_laminar_field` | func | —  _(undocumented)_ |
| `validate_projection_invariants` | func | —  _(undocumented)_ |
| `validate_source_field_status` | func | Return truth-preserving status for source-field readouts. |

### Probes (3)

| Symbol | Kind | Summary |
|---|---|---|
| `eeg_proxy_transform` | func | Compute EEG-proxy readout via linear leadfield projection. |
| `emm_proxy_transform` | func | Compute EMM-proxy (normalized activity/source/field cost) readout. |
| `meg_proxy_transform` | func | Compute MEG-proxy readout via linear leadfield projection. |

### Optimizers — `optim` (15)

| Symbol | Kind | Summary |
|---|---|---|
| `AGSDR` | class | Legacy AGSDR adapter retained for old notebooks and tests. |
| `agsdr` | func | Return an optimizer spec for AGSDR. |
| `agsdr_transform` | func | Return an Optax-compatible GradientTransformation for Adaptive GSDR. |
| `AGSDROptimizerSpec` | class | Multi-parameter AGSDR optimizer specification with execution parameters. |
| `AGSDRState` | class | Adaptive Genetic Stochastic Delta Rule optimizer state. |
| `gsdr` | func | Return an OptimizerSpec for the GSDR (Genetic Stochastic Delta Rule) optimizer. |
| `gsdr_transform` | func | Return an Optax-compatible GradientTransformation for Genetic SDR. |
| `GSDRState` | class | Genetic Stochastic Delta Rule optimizer state. |
| `optax_adam` | func | Return an OptimizerSpec for Optax Adam. |
| `optax_sgd` | func | Return an OptimizerSpec for Optax SGD. |
| `OptimizerSpec` | class | Declarative optimizer specification with differentiability metadata. |
| `random_search` | func | Return an OptimizerSpec for random search. |
| `require_optax` | func | Import Optax lazily with an informative error. |
| `sdr_transform` | func | Return an Optax-compatible GradientTransformation for Stochastic Delta Rule. |
| `SDRState` | class | Stochastic Delta Rule optimizer state. |

### Runtime / Validation registry

| Symbol | Kind | Summary |
|---|---|---|
| `compilation_registry` | const | Automated JAX tracing and compilation tracking registry for v0.3.20. |

> Runtime config (`RuntimeConfig`, `enable_x64`, `runtime_report`, `runtime`) and
> validators (`validate_config`, `validate_source_field_status`,
> `validate_projection_invariants`, `operator_status`, `config_truth_boundary`)
> are defined in **Core** and listed there.

### IO & receipts (7)

| Symbol | Kind | Summary |
|---|---|---|
| `config_hash` | func | Return a compact SHA256 hash for a configuration-like object. |
| `json_safe` | func | Convert common scientific Python/JAX objects into strict JSON values. |
| `manifest` | func | Build a strict JSON-safe run manifest. |
| `save_json` | func | Save strict JSON with `allow_nan=False`. |
| `save_receipt` | func | Save a RunReceipt as strict JSON. |
| `sha256_file` | func | Return SHA256 for a file. |
| `sha256_text` | func | Return SHA256 for a text payload. |

### Bridges — Jaxley (7)

| Symbol | Kind | Summary |
|---|---|---|
| `BridgeSpec` | class | JSON-safe optional-backend bridge declaration. |
| `hh_numpy_reference_trace` | func | Standalone tutorial/reference Hodgkin-Huxley single-compartment trace. |
| `jaxley_trace_to_signals` | func | Convert Jaxley-style voltage trace array to jaxfne Signals. |
| `JaxleyBridge` | class | Jaxley-focused biophysical emitter bridge. |
| `JaxleyEmitterBridge` | class | Jaxley bridge contract for future compartment emitters. |
| `JaxleyTraceSpec` | class | Metadata specification for Jaxley-style voltage trace arrays. |
| `require_jaxley` | func | Import Jaxley lazily with an informative error. |

### Paradigms (4)

| Symbol | Kind | Summary |
|---|---|---|
| `Paradigm` | class | —  _(no docstring; dataclass)_ |
| `paradigm` | func | —  _(undocumented)_ |
| `ParadigmCondition` | class | A specific trial condition: sequence of stimuli and associated events. |
| `ParadigmEvent` | class | Discrete event within a task trial: stimulus, behavioral code, or omission marker. |

### Sharding (4)

| Symbol | Kind | Summary |
|---|---|---|
| `get_sharding_context` | func | Return a dict with `mesh`, `candidate`, and `replicated` sharding specs. |
| `make_candidate_sharding` | func | Return a `jax.sharding.NamedSharding` that slices the first axis. |
| `make_population_mesh` | func | Return a 1-D named `jax.sharding.Mesh` across all visible JAX devices. |
| `make_replicated_sharding` | func | Return a `jax.sharding.NamedSharding` that fully replicates an array. |

### Tutorial utils (4)

| Symbol | Kind | Summary |
|---|---|---|
| `build_tutorial_laminar_column` | func | Build a laminar column scaffold model. |
| `kappa_synchrony` | func | Compute spike synchrony measure (kappa statistic) across neurons. |
| `rate_synchrony_targets` | func | Create an objective specification for AGSDR tuning toward rate and synchrony targets. |
| `select_neurons` | func | Select neuron indices matching given criteria (area, layer, cell_type). |

### Experimental HPC (2)

| Symbol | Kind | Summary |
|---|---|---|
| `NodeIdentity` | class | Stable node identity for selector-addressable circuits. |
| `SelectorSpec` | class | Selector over area/layer/cell-type/id fields. |

### Submodules & constants (5)

| Symbol | Kind | Summary |
|---|---|---|
| `CELL_TYPE_PRESETS` | const | Mapping of cell-type label → preset Izhikevich parameters. |
| `DEFAULT_SPIKE_IMPULSE_GAIN` | const | Default spike-impulse gain for the source proxy. |
| `RECEPTOR_KINETICS` | const | Mapping of receptor name → kinetic time constants. |
| `vis` | module | Visualization package for jaxfne. |
| `_KNOWN_METRICS` | const | ⚠ private name leaking into `__all__` — see docs audit (remove). |

---

See the docs audit & restructure plan (`internal_docs/docs_audit_v0330.md`)
for orphaned pages, duplicate cleanup, and the per-module table migration.
