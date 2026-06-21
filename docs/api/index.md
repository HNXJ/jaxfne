# API Reference

Canonical import:

```python
import jaxfne as jtfne
```

This page is the complete index of the public API (`jaxfne.__all__`, 204 names),
grouped by module. Per-module pages carry detailed signatures and examples.
Current release `jaxfne==0.4.2` (tag `v0.4.2`). The root-level export
helpers introduced in the v0.3.37/v0.3.38 line remain formal `__all__` members.

!!! note "Scope & truth gates"
    All field/EEG/MEG/EMM outputs are computational proxies
    (`claim_level = "computational_scaffold"`, `field_solver_status = "linear_solver"`,
    `field_claim_level = "proxy_readout"`, `physical_amplitude_calibrated = False`).
    See [Limitations and future plans](../limitations_and_future_plans.md) for the
    scope statement.

## Module pages

| Page | Covers | Public names |
|---|---|---|
| [Core](core.md) | Configuration, Model, Simulation, Signals, readouts, receipts, suites | 69 |
| [Emitters](emitters.md) | Izhikevich emitter, receptors, synapses, EIG networks, edge lists | 19 |
| [Fields](fields.md) | Source→laminar projection, `FieldOutput`, proxy diagnostics | 12 |
| [Probes](probes.md) | EEG/MEG/EMM proxy transforms (within Fields) | (in Fields) |
| [Objectives](objectives.md) | `Objective`, `ObjectiveReport`, rate targets | (in Core) |
| [Runtime](runtime.md) | `RuntimeConfig`, `enable_x64`, `runtime_report` | (in Core) |
| [Validation](validation.md) | config/field validators, `operator_status`, `is_valid_signal` | 2 |
| _(no page yet)_ | **Optimizers** (`optim`, 15) · **IO/receipts** (10) · **Export & figures** (6) · **Bridges** (7) · **Paradigms** (6) · **Solvers** (7) · **Sanity-delta** (7) · **Plasticity** (5) · **Tutorial utils** (4) · **Sharding** (4) · **Connectivity** (3) · **PyNWB** (2) · **Experimental HPC** (2) · **JAX Spectral Analysis** (6) · geometry/builders/streaming/stimulus (13) | 102 |

> Several public names (optimizers, IO, export, bridges, paradigms, sharding,
> solvers) do not yet have a dedicated module page — they are listed with full
> signatures in the complete symbol index below. Counts sum to **204**
> (`len(jaxfne.__all__)`). See the docs audit
> (`internal_docs/docs_audit_v0330.md`) for the page-migration plan.

## Minimal workflow (verified)

The pipeline is one linear chain: `setup → config → construct → simulate →
visualize → tune/objective → optimize → export`.

```python
import jaxfne as jtfne
jtfne.enable_x64()                                       # setup

cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")  # config (canonical prior)
cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))

model   = jtfne.construct(cfg)                           # construct -> Model
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)  # simulate -> Signals

vm = signals.get("vm")                  # membrane voltage [T, N]
spk = signals.get("spk")                # spikes [T, N]
e_idx = model.select(cell_type="E")     # excitatory neuron indices
vm_e = signals.get("vm", cell_type="E") # equivalent to vm[:, e_idx]
assert vm_e.shape[-1] == len(e_idx)
```

---

## Column builders & the canonical prior

`build_laminar_column` and `build_multi_area_columns` build a `Configuration`
with every knob defaulted, so the canonical cortex is reproducible from the call
site. Both are partial builders — chain `.set_emitter().probes().field()` before
`construct`.

### `build_laminar_column(...) -> Configuration`

| Parameter | Default | Meaning |
|---|---|---|
| `name` | `"V1"` | Column name. |
| `n` | `1000` | Total neurons; per-layer count ∝ depth-band width. |
| `layers` | `DEFAULT_LAYERS` | 5-layer set (`"L1","L2/3","L4","L5","L6"`). Use `CANONICAL_LAYERS_6L` for the split-L2/L3 form. |
| `layer_fractions` | canonical/even bands | `(z0, z1)` depth band per layer; width sets the count. |
| `cell_type_fractions` | `FLAT_CELL_TYPE_FRACTIONS` | Global E/PV/SST/VIP used by `ei_profile="flat"`. |
| `layer_cell_type_fractions` | `None` | Explicit per-layer composition; overrides `ei_profile`. |
| `ei_profile` *(kw)* | `"flat"` | `"flat"` = legacy depth-invariant; `"canonical"` = verified E:I gradient (E deep ≈90%, I superficial 50%, PV at L4, ≈77E:23I). |
| `geometry` *(kw)* | `"auto"` | `"auto"` → `laminar` when a non-flat composition is requested, else `uniform3d`. `uniform3d` collapses layer identity. |
| `within_connectivity` *(kw)* | `"all_to_all_uniform_random"` | Within-area rule. |
| `within_gain` *(kw)* | `0.45` | Within-area weight gain. |
| `radius_mm`, `height_mm` *(kw)* | `0.25`, `1.6` | Column cylinder geometry. |
| `edge_seed` *(kw)* | `None` | Connectivity edge seed. |

**Returns:** a `Configuration` (single column). **Examples:**

```python
jtfne.build_laminar_column()                        # V1, n=1000, flat E:I, uniform3d (legacy)
jtfne.build_laminar_column(ei_profile="canonical")  # ground-truth gradient, laminar placement
jtfne.build_laminar_column("M1", 500, layers=["L2/3", "L5"], within_gain=0.6)
```

### `build_multi_area_columns(...) -> Configuration`

| Parameter | Default | Meaning |
|---|---|---|
| `areas` | `("V1","V4","PFC")` | Area names, low→high in the hierarchy. |
| `n_per_area` | `200` | Neurons per area. |
| `layers` | `DEFAULT_LAYERS` | Shared layer sequence. |
| `connectivity_mode` | `"sparse"` | Inter-area mode (`"sparse"`/`"all_to_all"`). |
| `ei_profile` *(kw)* | `"flat"` | Per-layer composition applied to every area (see above). |
| `cell_type_fractions` *(kw)* | `FLAT_CELL_TYPE_FRACTIONS` | Global fractions for `ei_profile="flat"`. |
| `within_connectivity`, `within_gain` *(kw)* | `"all_to_all_uniform_random"`, `0.35` | Within-area rule/gain. |
| `p_feedforward`, `p_feedback` *(kw)* | `0.3`, `0.2` | Inter-area connection probabilities. |

**Returns:** a multi-area `Configuration` with declared feedforward/feedback edges.

Legacy compatibility: `ei_profile="flat"` reproduces the pre-sweep behavior
byte-for-byte (depth-invariant composition, `uniform3d` placement). Only the new
opt-in paths (`ei_profile="canonical"` or an explicit `layer_cell_type_fractions`)
change placement to laminar.

---

## Export & figure APIs (root-level)

The strict-notebook export grammar introduced in `jaxfne==0.3.37` is exposed as
root-level callables (`jaxfne.<name>`) and, as of v0.3.38, is registered in
`jaxfne.__all__` as formal public API (see the **Export & figures** group in the
symbol index below). These are the canonical replacement for direct
`matplotlib`/`json` calls in release-facing notebooks. `matplotlib` is imported
lazily inside the plotting/save functions, so importing `jaxfne` does not pull
in a plotting backend.

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
(`physical_amplitude_calibrated = False`).

---

## Complete public symbol index

`func`/`class`/`const`/`module` as resolved from `jaxfne.__all__` (**204 names**, grouped by defining module). Summaries are the first docstring line; `_(undocumented)_` marks public callables with no docstring in the released wheel.

### Core (69)

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
| `DatasetSpec` | class | Manifest-safe dataset/comparison declaration for observed data. |
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
| `migrate_schema` | func | Upgrade a legacy truth/metadata dict to the canonical truth-gate schema. |
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
| `StimulusSchedule` | class | Explicit drive schedule for event-aligned stimulus injection. |
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
| `Emitter` | class | Base class for package-level emitter facades. |
| `GLIFEmitter` | class | Base class for package-level emitter facades. |
| `izhikevich_params_from_labels` | func | Create reduced Izhikevich parameters from explicit cell labels. |
| `IzhikevichEmitter` | class | Reduced Izhikevich emitter facade with a JAX step function. |
| `IzhikevichParams` | class | Parameter container for a reduced Izhikevich population. |
| `LIFEmitter` | class | Base class for package-level emitter facades. |
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
| `SynapseState` | class | —  _(no docstring)_ |

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

### Export & figures (6)

| Symbol | Kind | Summary |
|---|---|---|
| `export_report` | func | Export a complete report with JSON artifacts and figures. |
| `export_tutorial_artifacts` | func | Export tutorial artifacts (JSON only, no figures). |
| `plot_raster` | func | Plot a spike raster. |
| `plot_spectrolaminar_suite` | func | Plot spectrolaminar suite from signals object. |
| `save_figure` | func | Save a matplotlib figure to disk. |
| `save_figures` | func | Save multiple figures to an output directory. |

### Bridges — Jaxley (7)

| Symbol | Kind | Summary |
|---|---|---|
| `BridgeSpec` | class | JSON-safe optional-backend bridge declaration. |
| `hh_numpy_reference_trace` | func | Standalone tutorial/reference Hodgkin-Huxley single-compartment trace. |
| `jaxley_trace_to_signals` | func | Convert Jaxley-style voltage trace array to jaxfne Signals. |
| `JaxleyBridge` | class | Jaxley-focused biophysical emitter bridge. |
| `JaxleyEmitterBridge` | class | Jaxley bridge contract for reserved compartment emitters. |
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

### Solvers (7)

| Symbol | Kind | Summary |
|---|---|---|
| `DiffraxSolver` | class | Optional Runge-Kutta solver using diffrax (lazily imported). |
| `euler_scan` | func | Forward Euler integration scan (backward compatibility). |
| `euler_step` | func | Single forward Euler step (backward compatibility). |
| `EulerSolver` | class | Forward Euler integrator using JAX and lax.scan. |
| `solve_ode` | func | Public ODE solver entrypoint routing to appropriate solver backend. |
| `solve_volume_conductor_experimental` | func | Experimental volume conductor solver skeleton. |
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

### Plasticity (5)

| Symbol | Kind | Summary |
|---|---|---|
| `plot_stdp_adaptation_suite` | func | Generates and saves the standard STDP adaptation visualization figures. |
| `STDPPlasticityConfig` | class | Configuration class for STDP activity-dependent plasticity. |
| `STDPState` | class | Container for the state variables of the STDP synapse model. |
| `summarize_stdp_adaptation` | func | Computes synapse-by-synapse adaptation statistics. |
| `update_stdp_weights_jax` | func | JAX-optimized plasticity weight update kernel (STDP). |

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

### Connectivity (3)

| Symbol | Kind | Summary |
|---|---|---|
| `compile_connection_rules` | func | Compile declared connection rules into sparse finite edge arrays. |
| `compile_connection_rules_jax` | func | Tensorized JAX connectivity compiler producing static-shape edge outputs. |
| `ConnectionCompileResult` | class | Compiled sparse connectivity. |

### Geometry (1)

| Symbol | Kind | Summary |
|---|---|---|
| `make_ei_cloud_network` | func | Generates geometry and initial weights for a 100-neuron E-I cloud network. |

### Builders (10)

| Symbol | Kind | Summary |
|---|---|---|
| `laminar_cortex_config` | func | Generalized multi-area laminar cortical configuration builder. |
| `build_laminar_column` | func | Single-column builder; defaults `name="V1"`, `n=1000`; `ei_profile`/`geometry` select flat-legacy vs canonical laminar prior. |
| `build_multi_area_columns` | func | Multi-area builder; defaults to the `V1→V4→PFC` hierarchy, 200/area, with inter-area feedforward/feedback. |
| `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` | const | Ground-truth per-layer E:I composition (6-layer); E peaks deep, I peaks superficial. |
| `CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L` | const | 5-layer (L2/3 merged) variant of the canonical composition. |
| `CANONICAL_Z_BANDS` | const | Count-proportional depth bands for the canonical 6-layer column. |
| `CANONICAL_Z_BANDS_5L` | const | Count-proportional depth bands for the 5-layer column. |
| `CANONICAL_LAYERS_6L` | const | Canonical 6-layer name tuple `("L1".."L6")`. |
| `FLAT_CELL_TYPE_FRACTIONS` | const | Legacy depth-invariant E/PV/SST/VIP fractions. |
| `DEFAULT_LAYERS` | const | Historical 5-layer default `("L1","L2/3","L4","L5","L6")`. |

### Streaming (1)

| Symbol | Kind | Summary |
|---|---|---|
| `run_stdp_stream` | func | Runs simulation in a chunked, streaming fashion to avoid memory explosion. |

### Stimulus (1)

| Symbol | Kind | Summary |
|---|---|---|
| `triangular_drive` | func | Generates a triangular drive trace. |

### JAX Spectral Analysis (6)

| Symbol | Kind | Summary |
|---|---|---|
| `spectrolaminar_psd_jax` | func | Compute spectrolaminar PSD averaged across trials using JAX. |
| `bandpower_jax` | func | Compute average power within a frequency band normalized by channel max. |
| `spectrolaminar_readout_kernel_jax` | func | Batchable readout kernel computing relative power and normalized band profiles. |
| `spectrolaminar_similarity_kernel_jax` | func | Compute the profile similarity score in JAX. |
| `spectrolaminar_similarity_candidates_jax` | func | Batched vectorization path for similarity scoring. |
| `spectrolaminar_similarity_candidates_seeds_jax` | func | Nested batched vectorization path for seeds and candidates. |

### Validation registry (2)

| Symbol | Kind | Summary |
|---|---|---|
| `compilation_registry` | const | Automated JAX tracing and compilation tracking registry. |
| `is_valid_signal` | func | Check if signal arrays contain only finite values (no NaN/Inf). |

### PyNWB compatibility (2)

| Symbol | Kind | Summary |
|---|---|---|
| `read_nwb` | func | Placeholder for NWB read (reserved status). |
| `write_nwb` | func | Placeholder for NWB write (reserved status). |

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

See the docs audit & restructure notes (`internal_docs/docs_audit_v0330.md`)
for orphaned pages, duplicate cleanup, and the per-module table migration.
