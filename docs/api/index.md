# API Reference

Canonical import:

```python
import jaxfne as jtfne
```

This page is the complete index of the public API (`jaxfne.__all__`, 179 names),
grouped by module. Per-module pages carry detailed signatures and examples.
Released in `jaxfne==0.3.37` (tag `v0.3.37`, commit `49aa025`).

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
| [Core](core.md) | Configuration, Model, Simulation, Signals, readouts, receipts, suites | 68 |
| [Emitters](emitters.md) | Izhikevich emitter, receptors, synapses, EIG networks, edge lists | 19 |
| [Fields](fields.md) | Source→laminar projection, `FieldOutput`, proxy diagnostics | 12 |
| [Probes](probes.md) | EEG/MEG/EMM proxy transforms (within Fields) | (in Fields) |
| [Objectives](objectives.md) | `Objective`, `ObjectiveReport`, rate targets | (in Core) |
| [Runtime](runtime.md) | `RuntimeConfig`, `enable_x64`, `runtime_report` | (in Core) |
| [Validation](validation.md) | config/field validators, `operator_status`, `is_valid_signal` | 2 |
| _(no page yet)_ | **Optimizers** (`optim`, 15) · **IO/receipts** (10) · **Bridges** (7) · **Paradigms** (6) · **Solvers** (6) · **Sanity-delta** (7) · **Plasticity** (4) · **Tutorial utils** (4) · **Sharding** (4) · **Connectivity** (2) · **PyNWB** (2) · **Experimental HPC** (2) · geometry/builders/streaming/stimulus (4) | 78 |

> Several public names (optimizers, IO, bridges, paradigms, sharding, solvers)
> do not yet have a dedicated module page — they are listed with full signatures
> in the complete symbol index below. Counts sum to **179** (`len(jaxfne.__all__)`).
> See the docs audit (`internal_docs/docs_audit_v0330.md`) for the page-migration plan.

## Minimal workflow (verified)

```python
import jaxfne as jtfne

cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=1000.0, dt_ms=0.1)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)

vm = signals.get("vm")            # membrane voltage [T, N]
spk = signals.get("spk")          # spikes [T, N]
e_idx = model.select(cell_type="E")     # excitatory neuron indices
vm_e = signals.get("vm", cell_type="E") # equivalent to vm[:, e_idx]
assert vm_e.shape[-1] == len(e_idx)
```

---

## Export & figure APIs (root-level, v0.3.37)

The strict-notebook export grammar introduced in `jaxfne==0.3.37` is exposed as
root-level callables (`jaxfne.<name>`). These are importable from the package
root but are not listed in `jaxfne.__all__`; they are the canonical replacement
for direct `matplotlib`/`json` calls in release-facing notebooks.

| Symbol | Kind | Summary |
|---|---|---|
| `save_figure` | func | Save a matplotlib figure to disk. |
| `save_figures` | func | Save multiple figures to an output directory. |
| `export_report` | func | Export a complete report with JSON artifacts and figures. |
| `export_tutorial_artifacts` | func | Export tutorial artifacts (JSON only, no figures). |
| `plot_raster` | func | Plot a spike raster. |
| `plot_spectrolaminar_suite` | func | Plot spectrolaminar suite from a signals object. |

All export helpers honor the truth gates: JSON is written with `allow_nan=False`,
and figure/readout outputs remain proxy diagnostics
(`physical_amplitude_claim_allowed = False`).

---

## Complete public symbol index

`func`/`class`/`const`/`module` as resolved from `jaxfne.__all__` (**179 names**, grouped by defining module). Summaries are the first docstring line; `_(undocumented)_` marks public callables with no docstring in the released `jaxfne==0.3.37` wheel.

### Core (68)
| Symbol | Kind | Summary |
|---|---|---|
| `AxisSpec` | class | Typed descriptor for one tensor axis in the TFNE scaffold. |
| `BasisSpec` | class | Typed descriptor for the computation basis of a TFNE run. |
| `Config` | class | Declarative TFNE model configuration. |
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
| `Model` | class | —  _(dataclass; fields in signature)_ |
| `Net` | class | —  _(dataclass; fields in signature)_ |
| `Objective` | class | Declarative objective specification: losses, regularizers, and diagnostic gates. |
| `objective` | func | —  _(undocumented)_ |
| `ObjectiveReport` | class | Structured, immutable result of evaluating an Objective against Signals. |
| `operator_status` | func | Return the current operator status registry for all declared operators. |
| `Probe` | class | —  _(dataclass; fields in signature)_ |
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
| `Simulation` | class | —  _(dataclass; fields in signature)_ |
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
| `GLIFEmitter` | class | Base class for package-native emitter facades. |
| `izhikevich_params_from_labels` | func | Create reduced Izhikevich parameters from explicit cell labels. |
| `IzhikevichEmitter` | class | Reduced Izhikevich emitter facade with JAX-native step. |
| `IzhikevichParams` | class | Parameter container for a reduced Izhikevich population. |
| `LIFEmitter` | class | Base class for package-native emitter facades. |
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
| `SynapseState` | class | —  _(namedtuple; fields in signature)_ |

### Fields (12)
| Symbol | Kind | Summary |
|---|---|---|
| `compute_conservation_proxy_diagnostics` | func | Compute conservation-inspired proxy diagnostics over existing source/field arrays. |
| `construct_source_tensor` | func | —  _(undocumented)_ |
| `eeg_proxy_transform` | func | Compute EEG-proxy readout via linear leadfield projection. |
| `emm_proxy_transform` | func | Compute EMM-proxy (normalized activity/source/field cost) readout. |
| `FieldOutput` | class | Container for laminar proxy field/readout arrays. |
| `LinearReadout` | class | —  _(dataclass; fields in signature)_ |
| `meg_proxy_transform` | func | Compute MEG-proxy readout via linear leadfield projection. |
| `probe_laminar_modes` | func | —  _(undocumented)_ |
| `project_laminar_sources` | func | Project source traces to laminar proxy contacts. |
| `project_sources_to_laminar_field` | func | —  _(undocumented)_ |
| `validate_projection_invariants` | func | —  _(undocumented)_ |
| `validate_source_field_status` | func | Return truth-preserving status for source-field readouts. |

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

### IO & receipts (10)
| Symbol | Kind | Summary |
|---|---|---|
| `asset_hashes` | func | Create a SHA256 hash manifest for assets. |
| `config_hash` | func | Return a compact SHA256 hash for a configuration-like object. |
| `json_safe` | func | Convert common scientific Python/JAX objects into strict JSON values. |
| `manifest` | func | Build a strict JSON-safe run manifest. |
| `probe_report` | func | Create a probe operator report JSON bundle. |
| `save_json` | func | Save strict JSON with `allow_nan=False`. |
| `save_receipt` | func | Save a RunReceipt as strict JSON. |
| `sha256_file` | func | Return SHA256 for a file. |
| `sha256_text` | func | Return SHA256 for a text payload. |
| `validation_report` | func | Create a validation report JSON bundle. |

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

### Paradigms (6)
| Symbol | Kind | Summary |
|---|---|---|
| `coop_omission_oddball_paradigm` | func | Create a Continuous Omission Oddball Paradigm (COOP) stimulus sequence. |
| `omission_oddball_paradigm` | func | Create an omission/oddball detection paradigm. |
| `Paradigm` | class | —  _(dataclass; fields in signature)_ |
| `paradigm` | module | _(constant; see source)_ |
| `ParadigmCondition` | class | A specific trial condition: sequence of stimuli and associated events. |
| `ParadigmEvent` | class | Discrete event within a task trial: stimulus, behavioral code, or omission marker. |

### Solvers (6)
| Symbol | Kind | Summary |
|---|---|---|
| `DiffraxSolver` | class | Optional Runge-Kutta solver using diffrax (lazily imported). |
| `euler_scan` | func | Forward Euler integration scan (backward compatibility). |
| `euler_step` | func | Single forward Euler step (backward compatibility). |
| `EulerSolver` | class | Forward Euler integrator using JAX and lax.scan. |
| `solve_ode` | func | Public ODE solver entrypoint routing to appropriate solver backend. |
| `SolverConfig` | class | Configuration class for ODE solvers. |

### Sanity-delta runtime (7)
| Symbol | Kind | Summary |
|---|---|---|
| `BackupState` | class | Resumable task state with ring buffer history. |
| `BehaviorGate` | class | Fixation gate: monitors PFC superficial activity. |
| `HierarchicalOddballParadigm` | class | Task paradigm: AAAB oddball sequence with timing and gating. |
| `Manifest` | class | Output manifest: configuration, paradigm, backup, validation. |
| `SanityDeltaConfig` | class | Hierarchical oddball configuration factory and validation. |
| `SanityDeltaModel` | class | Wrapper around constructed hierarchical oddball model. |
| `TaskEpisode` | class | Result of a task episode with probing, export, validation. |

### Plasticity (4)
| Symbol | Kind | Summary |
|---|---|---|
| `plot_stdp_adaptation_suite` | func | Generates and saves the standard STDP adaptation visualization figures. |
| `STDPPlasticityConfig` | class | Configuration class for STDP activity-dependent plasticity. |
| `STDPState` | class | Container for the state variables of the STDP synapse model. |
| `summarize_stdp_adaptation` | func | Computes synapse-by-synapse adaptation statistics. |

### Tutorial utils (4)
| Symbol | Kind | Summary |
|---|---|---|
| `build_tutorial_laminar_column` | func | Build a laminar column scaffold model. |
| `kappa_synchrony` | func | Compute spike synchrony measure (kappa statistic) across neurons. |
| `rate_synchrony_targets` | func | Create an objective specification for AGSDR tuning toward rate and synchrony targets. |
| `select_neurons` | func | Select neuron indices matching given criteria (area, layer, cell_type). |

### Sharding (4)
| Symbol | Kind | Summary |
|---|---|---|
| `get_sharding_context` | func | Return a dict with `mesh`, `candidate`, and `replicated` sharding specs. |
| `make_candidate_sharding` | func | Return a `jax.sharding.NamedSharding` that slices the first |
| `make_population_mesh` | func | Return a 1-D named `jax.sharding.Mesh` across all visible JAX devices. |
| `make_replicated_sharding` | func | Return a `jax.sharding.NamedSharding` that fully replicates an array |

### Connectivity (2)
| Symbol | Kind | Summary |
|---|---|---|
| `compile_connection_rules` | func | Compile declared connection rules into sparse finite edge arrays. |
| `ConnectionCompileResult` | class | Compiled sparse connectivity. |

### Geometry (1)
| Symbol | Kind | Summary |
|---|---|---|
| `make_ei_cloud_network` | func | Generates geometry and initial weights for a 100-neuron E-I cloud network. |

### Builders (1)
| Symbol | Kind | Summary |
|---|---|---|
| `laminar_cortex_config` | func | Generalized laminar cortical configuration builder. |

### Streaming (1)
| Symbol | Kind | Summary |
|---|---|---|
| `run_stdp_stream` | func | Runs simulation in a chunked, streaming fashion to avoid memory explosion. |

### Stimulus (1)
| Symbol | Kind | Summary |
|---|---|---|
| `triangular_drive` | func | Generates a triangular drive trace. |

### Validation registry (2)
| Symbol | Kind | Summary |
|---|---|---|
| `compilation_registry` | const | Automated JAX tracing and compilation tracking registry. |
| `is_valid_signal` | func | Check if signal arrays contain only finite values (no NaN/Inf). |

### PyNWB compatibility (2)
| Symbol | Kind | Summary |
|---|---|---|
| `read_nwb` | func | Placeholder for NWB read (not implemented). |
| `write_nwb` | func | Placeholder for NWB write (not implemented). |

### Experimental HPC (2)
| Symbol | Kind | Summary |
|---|---|---|
| `NodeIdentity` | class | Stable node identity for selector-addressable circuits. |
| `SelectorSpec` | class | Selector over area/layer/cell-type/id fields. |

### Submodules (1)
| Symbol | Kind | Summary |
|---|---|---|
| `vis` | module | Visualization package for jaxfne. |

### Constants (4)
| Symbol | Kind | Summary |
|---|---|---|
| `_KNOWN_METRICS` | const | ⚠ private name leaking into `__all__` — see docs audit (remove). |
| `CELL_TYPE_PRESETS` | const | Mapping of cell-type label → preset Izhikevich parameters. |
| `DEFAULT_SPIKE_IMPULSE_GAIN` | const | Default spike-impulse gain for the source proxy. |
| `RECEPTOR_KINETICS` | const | Mapping of receptor name → kinetic time constants. |

---

See the docs audit & restructure plan (`internal_docs/docs_audit_v0330.md`)
for orphaned pages, duplicate cleanup, and the per-module table migration.
