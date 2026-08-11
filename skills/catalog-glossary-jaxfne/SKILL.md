---
name: catalog-glossary-jaxfne
description: >-
  Authoritative flat catalog of the jaxfne public API — every config builder,
  simulator, laminar/spectrolaminar trial pipeline, optimizer, paradigm,
  readout/projection, metric, visualization, and manifest/receipt helper that
  already exists. USE THIS FIRST, before writing any new helper, before
  grepping the source to "find out if a function exists", and before
  reimplementing PSD/raster/LFP/CSD/EEG/MEG/spectrolaminar/AGSDR/manifest logic.
  Trigger whenever a task touches jaxfne (jtfne) — building a model, running a
  simulation, tuning with AGSDR/GSDR/Optax, plotting, computing spectrolaminar
  profiles, writing manifests/receipts, or hashing artifacts. If you are about
  to say "let me write a function to ..." or "I'll compute X manually",
  STOP and check here — it is almost certainly already implemented.
---

# jaxfne API Catalog & Glossary

The public surface includes functions, classes, the laminar-column tutorial
pipeline (`jaxfne.tutorial_utils`), and visualization (`jaxfne.vis`). Counts
are volatile; run the inventory snippet at the bottom before reporting them.
Do not reinvent package behavior.

Canonical import: `import jaxfne as jtfne`. Everything in §1–§11 below is
`jtfne.<name>` unless a submodule is shown (`jtfne.tutorial_utils.*`,
`jtfne.vis.*`). Verify a signature with `inspect.signature(jtfne.<name>)` only
if you need exact kwargs — the name itself is confirmed present on disk.

> **Two `build_laminar_column` exist.** Top-level `jtfne.build_laminar_column(name, n, ...)`
> builds a `Configuration`. The tutorial pipeline one
> `jtfne.tutorial_utils.build_laminar_column(cfg)` takes a `LaminarColumnConfig`
> and returns a model `dict`. For the spectrolaminar trial pipeline use the
> **tutorial_utils** version (see §2).

---

## 1. Build & configure a model (Configuration → Model)

- `laminar_cortex_config(*, seed, duration_ms, dt_ms, areas, layers, cell_types, n, emitter, baseline_drive_by_cell_type)` → `Configuration` — **the** multi-area laminar builder; `baseline_drive_by_cell_type` injects native Izhikevich drive (eliminates silent neurons at the ODE source).
- `default_cortical_column_config(...)`, `build_multi_area_columns(areas, n_per_area, layers, connectivity_mode)` — ready-made `Configuration`s. Verify any other builder name against the live export surface.
- `construct(cfg, *, geometry=None)` → `Model` — turn a Configuration into a runnable Model.
- `build_laminar_column(name, n, ...)` → `Configuration` (single column; top-level).
- Fresh/complete builders: `configuration()` → empty `Configuration` builder; `default_complete_configuration(column_name="V1", nucleus_name="thalamus", n_column=100, n_nucleus=60, layers=None, seed=None, duration_ms=1000.0, dt_ms=0.1)` → broadest default: laminar cortical column wired to a non-laminar nucleus (cortex + subcortex in one config).
- Model diff/validation helpers (`jaxfne/util.py`): `configuration_diff(a, b)` → `{field: (a_val, b_val)}` for differing declarative fields; `model_diff(a, b, *, atol=1e-6)` → sweep-comparison summary between two built Models; `validate_model(model, *, strict=False)` → structural/numerical consistency warnings as `list[str]`.
- Suite No. 2 configs: `suite2_single_neuron_config`, `suite2_four_celltype_config`, `suite2_net1_config`, `suite2_v1_v4_config`, `suite2_simulation`, `suite2_run_bundle`, `suite2_celltype_presets`.
- Connectivity: `connect_columns(cfg, src, tgt, mode, ...)`, `all_to_all_intercolumn_connectivity(...)`, `sparse_intercolumn_connectivity(...)`, `build_laminar_connections(model, cfg)`, `compile_connection_rules(...)`, `make_edge_list_from_dense(weights, ...)`. Model-level fusion: `connect(*models, edges=None, namespace=None, layout="offset_x", strict=True, name=None)` → `Model` — fuse two or more constructed Models into one ensemble (distinct from the config-level `connect_columns`).
- Construct extras (`jaxfne/_construct_extras.py`): `laminar_source_geometry(populations)` → `LaminarSourceGeometry` (source geometry from an ordered population sequence, for `construct(..., geometry=...)`); `dataset_spec(**kwargs)` → `DatasetSpec`; `operator_status()` → `dict[str, str]` (operator status registry).
- Cells/emitters: `make_cell_dist`, `make_cell_type_catalog`, `make_eig_network`, `izhikevich_params_from_labels(labels, *, drive_overrides=...)`, `with_emitter_parameters(model, ...)`, `standard_receptor_specs`, `standard_receptor_tau_table`.
- Legacy `.jcfg.json`/`JaxFNEConfig` names are not part of the current public
  API. Verify the current `Configuration`-native validator and loaders before use.

### 1b. NeuronalTensor — tensor-first build path (`jtfne.NeuronalTensor`,
`jtfne.NeuronType`, etc. are top-level, no submodule import needed)

`NeuronalTensor = [Areas, AreaConnections]`, `Area = [Layers x NeuronTypes, InterConnections]`.
This is a SEPARATE build path from `Configuration` — both converge on the same
`Model` type via `construct()`:

- `NeuronalTensor(areas=[...], name=...)`, `Area(name, layers=[...])`, `Layer(name, n_neurons, neuron_types=[...])`, `InterConnection(...)`, `AreaConnection(...)`.
- Tensor geometry/placement classes include `Geometry3D`, `Pose3D`, `PlasticParams`,
  and `StaticParams`; verify their live signatures and relative-value semantics
  before use.
- `NEURONAL_TENSOR_SCHEMA_VERSION` is the schema tag for tensor loading; verify
  the active value and migrate rather than patch around mismatches.
- `NeuronType.make(name, *, relative_size=None, fraction=None, value_tag=...)` —
  `fraction` declares an explicit population fraction; if every type in a
  `Layer` declares one, normalized fractions populate
  `Configuration.metadata["area_layer_cell_types"][area][layer]`; otherwise the
  layer uses its documented fallback.
- `neuronal_tensor_to_configuration(tensor, *, seed, duration_ms, dt_ms)` → `Configuration` — the internal bridge `construct()` uses.
- `construct(tensor, runtime_configuration)` → `Model` — same top-level `construct` as the `Configuration` path, dispatches on input type.
- `RuntimeConfiguration` (`neuronal_tensor.py`, frozen, execution-only: seed/duration_ms/dt_ms/etc.) — **distinct from** `RuntimeConfig` (now in `jaxfne/_runtime_config.py`, re-exported unchanged from `jaxfne.core`/top-level `jaxfne`; has `enable_hdp`/`hdp_params`). `RuntimeConfiguration` has NO HDP field.
- `load`, `load_neuronal_tensor`, `load_canonical_neuronal_tensor`, `list_canonical_neuronal_tensors`, `merge_neuronal_tensors`, `construct_neuronal_tensor`.
- `configs_dir()` → `Path` — package-data location of the canonical NeuronalTensor JSON library (`jaxfne/configs/`).
- `make_minimal_ei_tensor(n=8, e_fraction=0.75, *, layer_name="L1", area_name="minimal", h=1.0)` → `NeuronalTensor` — one flat Layer split E/PV by `e_fraction`, all four pairwise E/PV InterConnections (AMPA from E, GABA from PV), `plastic.H=h` on every edge — the canonical small-HDP test tensor.
- Tensor introspection (`jaxfne/util.py`): `tensor_summary(nt)` → JSON-safe `dict` (counts + cell-type inventory); `validate_neuronal_tensor(nt, *, strict=False)` → structural-consistency warnings `list[str]`.
- `default_relative_size(neuron_type: str) -> float` — default relative soma size
  by cell type (E=5.0, PV=1.0, SST/VIP=1.5, … from
  `emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`); single source of truth for NeuronType sizes and
  HDP `tau_i = tau_0_ms * size_i**3` scaling.

**Enabling HDP on a tensor-built Model (no new public API needed):** build via
`construct(tensor, RuntimeConfiguration(...))`, then pass an explicit
`runtime=RuntimeConfig(enable_hdp=True, hdp_params={...})` to `simulate()` —
the explicit `runtime=` kwarg overrides any `Configuration`-derived config.
See `jaxfne-modeling-optimization-schema` for the full pattern + cube-law tau formula.

### 1c. HDP homeostatic plasticity module (`jaxfne/hdp_network.py`) — generic config-driven builder, no per-N functions

- HDP presets and parameters are defined in `jaxfne/hdp_network.py`; query the
  live dictionaries rather than copying values into a skill.
- `K_w_ctrl` is a current parameter in the HDP dispatch. Verify its effect,
  defaults, and stability for the specific topology and run.
- `BASE_HDP_KWARGS_DEFAULT` (H_min=0.1, H_max=10.0, alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0, ...), `BASE_DRIVE_BY_CELL_TYPE_DEFAULT = {"E":4.0,"PV":4.0,"SST":4.0,"VIP":4.0}`, `DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT`.
- Kernel: `simulate_edge_recurrent_izhikevich_hdp` (`emitters.py`). Verify the
  active parameter-forwarding path, scaling law, and supported `hdp_params` keys
  in the current model implementation before relying on them.
- True turn-to-turn HDP state uses the six-field `DynamicState(v, u,
  prev_spikes, syn_state, H, w)` carrier through the continuation pipeline;
  `Model.with_hdp_initial_state` remains partial H/W initialization.
- `model.last_hdp_diagnostics()` → dict with `H_trace`, weight trace, per-edge `receptor_index`.

### 1d. `homeostatic_ei` — second canonical HDP sanity emitter (`jaxfne/emitters_homeostatic_ei.py`)

- **Purpose:** minimal 2-neuron E/I circuit with an explicit conductance
  matrix `G` and HDP state `H`, updated as three separate staged steps (fast
  `x`, intermediate `G`, slow `H`) instead of one fused rule. A separate
  emitter family from Izhikevich, not built on `HDPColumnConfig`.
- **Input:** `Configuration().network(...).set_emitter("homeostatic_ei",
  activation_rule=..., conductance_rule=..., homeostasis_rule=...,
  bound_mode="minimal")` — rule names from `ACTIVATION_RULES`
  (linear/cubic/logistic), `CONDUCTANCE_RULES` (hebbian/bcm/linear/
  **hebbian_pairwise**), `HOMEOSTASIS_RULES` (linear/logistic/cubic_penalty/
  **cubic_penalty_coupled**), or a custom callable passed directly to
  `simulate_homeostatic_ei()` (only registry *names* pass through
  `Configuration`, which must stay JSON-safe).
- **Output:** standard `Signals` (`V_m`, `spikes`, `sources`) plus
  `metadata["hdp"]["G_trace"]`/`["H_trace"]`/`["rules"]["bound_mode"]`.
- **How to use:** `construct(cfg)` then `.simulate(sim)` — dispatches
  automatically, no special-case call needed. `G0` is tunable through the
  existing AGSDR path via `matrix_parameter(mask=..., bounds=..., target="G0")`.
  `make_minimal_ei_params(n, ...)` builds a `HomeostaticEIParams` directly
  (any `n>=2`) for ad hoc scripts without going through `Configuration`.
- **`bound_mode`** (default `"minimal"`): `"stable"` applies a smooth tanh
  soft-bound to `x`/`G`/`H`
  instead of `jnp.clip` -- a bounded codomain that can't be numerically
  outrun. Verify the selected bounds and finite behavior for the requested
  configuration.
- **`hebbian_pairwise`** conductance rule: independent gains
  per population pair (E-E/E-I/I-E/I-I) via
  `make_hebbian_pairwise_rule(k_ee, k_ei, k_ie, k_ii)`; default gains all
  `1.0` == plain `hebbian`. Custom gains are a callable, so only reachable
  by calling `simulate_homeostatic_ei` directly (same `Configuration`
  JSON-safety limitation as any custom rule).
- **`cubic_penalty_coupled`** homeostasis rule: adds E<->I
  cross-population coupling to `cubic_penalty` -- every other rule's `dH`
  depends only on that neuron's own `x`; this one lets one population's H
  respond to the other's activity.
- **Notes:** Verify supported `Model` lifecycle methods, rule behavior, and
  compilation through the current tests before relying on them.

## 2. Laminar-column TRIAL pipeline — `jtfne.tutorial_utils` (the one most often rediscovered)

This is the canonical multi-trial laminar/spectrolaminar path. **Do not hand-roll PSDs.**

```python
from jaxfne.tutorial_utils import (
    make_laminar_column_config, build_laminar_column,
    simulate_laminar_trials, summarize_spectrolaminar_similarity,
    spectrolaminar_from_trials,
)
from jaxfne.vis.tutorial_panels import spectrolaminar_suite_3panel

cfg    = make_laminar_column_config(areas=("V1","V4","PFC"), cell_types=("E","PV","SST","VIP"),
                                    n_neuron_per_column=200, duration_ms=1000, dt_ms=0.5,
                                    n_trials=8, n_contacts=24, freq_count=96, seed=0)  # -> LaminarColumnConfig
model  = build_laminar_column(cfg)                       # -> model dict (CSD/LFP/EEG/MEG contacts)
trials = simulate_laminar_trials(model, cfg, n_trials=8) # -> dict: spikes, voltage_mV, lfp_contacts, csd_contacts, contact_depths_m, area_names, ...
scores, specs = summarize_spectrolaminar_similarity(trials, cfg)         # -> (DataFrame, specs dict)
figs = spectrolaminar_suite_3panel(specs, model, cfg, areas=[...])       # -> {area: Figure} 3-panel depth×freq suite
prof, info = spectrolaminar_from_trials(trials, cfg, signal_key="csd_contacts", area_name="V1")
#   info keys: freq_hz, pos_from_l4 (depths), relative_power, alpha_beta, gamma
```

- `tune_laminar_agsdr(model, cfg, *, target_rate_hz, ...)` — AGSDR fine-tune laminar control to a target rate.
- `single_cell_waveforms(...)`, `make_izhikevich_control_panel(...)` (ipywidgets).

**Spectrolaminar = depth × FREQUENCY relative power** (alpha-beta deep vs gamma superficial). **Field-laminar = depth × TIME.** They are different; don't conflate.

## 3. Simulate (single-run)

- `simulate(model, sim=None, paradigm=None, **kwargs)` → `Signals` — main entry. kwargs include `duration_ms`, `dt_ms`, `seed`.
- Kernels (advanced): `simulate_eig_izhikevich`, `simulate_edge_recurrent_izhikevich`, `simulate_receptor_exponential_izhikevich`.
- `run_trials(model, batch, sim, ...)` → `TrialBatchResult`.
- ODE: `euler_scan(y_init, t_start, dt, n_steps, dydt_fn)` and `euler_step(...)`;
  use the `lax.scan`-based path for grad-through-time and verify any solver
  class signature against the active checkout.
- Stimulus: `make_stimulus(...)`, `stimulus_schedule(...)`. Per-neuron
  targeting uses `target_indices` on the event dictionary, not a schedule
  constructor argument; derive indices from `model.neuron_table()`.
- Noise controls are kernel-dependent. Verify supported kwargs and defaults
  against the current emitter signature.
- Read a signal: `Signals.get(key)` or free fn `get_signal(obj, key)`. Keys accept aliases: `"V_m"`/`"vm"`, `"spikes"`/`"spk"`, `"lfp_contacts"`, `"csd_contacts"`, `"source_native"`.

## 4. Readouts, projections, fields (default path is PROXY; one real solver is experimental)

- `project_laminar_sources(...)` → `FieldOutput`. Verify the projection mode
  and normalization choice in the current implementation and state it in the
  readout metadata.
- `project_sources_to_laminar_field(...)`, `probe_laminar_modes(field_output, modes)`.
- Lead-field proxies: `eeg_proxy_transform(source, leadfield)`, `meg_proxy_transform(source_oriented, leadfield)`, `emm_proxy_transform(...)`.
- `construct_source_tensor(*, mode, ...)`, `compute_conservation_proxy_diagnostics(...)`, `validate_projection_invariants(...)`, `validate_source_field_status(...)`.
- **Experimental 1D Poisson solve, separate from the proxy path above**:
  `experimental_poisson_1d(...)` is an opt-in solver path. Verify convergence
  and admissibility for the requested grid and dtype; do not copy a
  performance ceiling into persistent context.

## 5. Optimizers & tuning (AGSDR/GSDR/SDR/Optax)

- Specs: `agsdr(...)`, `gsdr(...)`, `random_search(...)`, `optax_adam(...)`, `optax_sgd(...)`. GSGD: `GSGDState(count=jnp.ndarray, step_size=jnp.ndarray)` — per-parameter iteration state for the GSGD optimizer step (`jaxfne/optim/gsgd.py`).
- Optax GradientTransformations: `agsdr_transform(...)`, `gsdr_transform(...)`, `sdr_transform(...)`.
- Objectives: `objective()`, `rate_targets(groups, targets_hz, weights)`, `rate_synchrony_targets(target_rate_hz, target_kappa_synchrony, ...)`, `readout_spec(...)`, `matrix_parameter(*, mask, bounds, ...)`, `surrogate_config(...)`.
- High-level: `Model.tune(obj, optimizer=..., ...)` → `TuneResult`; `suite2_tune_noise_agsdr_adam(model, ...)`; `tutorial_utils.tune_laminar_agsdr(...)`.

## 6. Paradigms & trials

- `paradigm(name)`, `evoked_l4_drive_paradigm(...)`, `omission_oddball_paradigm(...)`, `standard_visual_omission()`.
- `coop_omission_oddball_paradigm(duration_ms=..., freq_hz=..., omission_prob=...)` — Continuous Omission Oddball Paradigm (the original, narrower COOP).
- **`general_sequential_oddball_paradigm(...)`** — the generic backbone (omission/global/local/sync/async/active/passive families via event windows + token sequences or explicit event lists). Prefer this over hand-rolling a new fixed-shape builder; see `jaxfne-paradigm-design` for the full grammar.
- `general_delayed_match_to_sample_paradigm(...)` — thin DMS-flavored wrapper over the backbone.
- `trial_batch(conditions, n_reps, seed, ...)` → `TrialBatch`.

### 6b. Hierarchical global-local oddball task stack (`jaxfne/sanity_delta.py`) — AAAB task family

A distinct gated oddball-task pipeline (fixation gate, presentation/delay/review windows) for the hierarchical global-local AAAB task; classes below are the full task stack:

- `SanityDeltaConfig(seed, duration_ms, dt_ms, neurons_per_area, areas, hierarchy, cell_counts, stimulus_frequency_hz, stimulus_map, claim_level="computational_scaffold", ...)` — configuration factory + validation for the hierarchical oddball task.
- `SanityDeltaModel(config, model_state, n_neurons, n_steps, plasticity_enabled=False, plasticity_config=None)` — wrapper around a constructed hierarchical oddball network.
- `HierarchicalOddballParadigm(config, name, sequence, prefix_ms, fixation_required_ms, presentation_ms, delay_ms, review_ms)` — AAAB oddball task timing/gating; `BackupState(paradigm, time_ms, vm, ...)` — resumable task state with ring-buffer history; `BehaviorGate(paradigm, area, layer_group, target_rate_hz, tolerance_hz, window_ms)` — fixation gate monitoring PFC superficial activity.
- `TaskEpisode(config, paradigm, model, backup, spikes, vm, ...)` → task-episode result with probing/export/validation; `Manifest(config, paradigm, backup, episode_metadata, generated_at_utc, strict_json=True)` — output manifest for one task episode (task-level `Manifest`, distinct from the §9 `manifest()`/build_manifest receipt helpers).

## 7. Metrics & summaries

- `kappa_synchrony(spikes, dt_ms)`, `tutorial_utils.population_rate_hz(spikes, dt_ms)`.
- `layer_celltype_count_table(cfg_or_model)`, `column_density_table(cfg_or_model)` — accept `Configuration` (via `construct`) or `Model`; counts/densities from `neuron_table()`.
- `configuration_table(cfg)`, `config_summary_frame(cfg)`, `cell_catalog_frame(catalog)`.
- `spectrolaminar_motif_score(alpha_beta, gamma)` — anti-correlation score (0-100) between a deep alpha/beta and superficial gamma depth profile; `summarize_spectrolaminar_similarity` calls this internally per area. Distinct from `spectrolaminar_similarity_kernel_jax`, which scores against an explicit external target — see `jaxfne-spectrolaminar-suite` for the full distinction.

## 8. Visualization — `jtfne.vis.*` (signal-driven, proxy-safe)

Pass a `Signals` object; each returns a matplotlib fig (and a `*_with_meta` variant returns a JSON-safe metadata container):
`raster`, `vm`, `rate`, `source`, `lfp`, `lfp_traces`, `csd`, `csd_traces`, `eeg`, `meg`, `emm`, `psd`, `spectrogram`, `bandpower`, `connectivity`/`connectivity_matrix`, `laminar_profile`/`layer_celltype_counts`, `geometry3d`/`circuit3d`/`column_geometry`, `multi_area_layout`, `summary`, `objective_report`.
- `vis.spectrolaminar(signals)` — 3-panel spectrolaminar from a Signals object; `vis.spectrolaminar_suite(signals)` — Suite No. 2 readout panel.
- `vis.visualize_network_3d(data, *, output_html=..., show_edges=...)` — **interactive Plotly** 3D network (use for scaffold cells; supports HTML export, pan/zoom).
- `vis.visualize_laminar_column_3d(model, cfg, ...)`.

### `jtfne.vis.tutorial_panels.*` (trial/specs-driven suites)
- `spectrolaminar_suite_3panel(specs, model, cfg, areas=..., output_dir=..., theme="dark")` → `{area: Figure}` — also re-exported as `jtfne.vis.spectrolaminar_suite_3panel`.
- `activity_trace_suite(trials, cfg, ...)` — raster + LFP + CSD + PSD.
- `visualize_laminar_column_3d(model, cfg, ...)`.

### `jtfne.tutorial_utils.plot_*` (array-driven quick plots)
`plot_raster`, `plot_laminar_readout`, `plot_population_rate`, `plot_voltage_samples`, `plot_connectivity_matrix`, `save_png(fig, name, fig_dir)`. (No `tutorial_utils` short-name wrapper exists for spectrolaminar-power plotting — use `jtfne.vis.plot_spectrolaminar_power_array(t, signal, freq_min=1.0, freq_max=120.0, n_freqs=96, ...)` directly, defined in `jaxfne/vis/tutorial_array_plots.py`.)

## 9. Manifests, receipts, JSON, hashing (Truth plane)

- `manifest(cfg, signals=None, readout=None, paradigm=None, objective=None, ...)` → strict JSON-safe run manifest.
- `validation_report(config_valid, issues, metadata)`, `probe_report(n_probes, probe_types, metadata)`.
- `run_receipt(model, signals, *, tags=...)` → `RunReceipt`; `save_receipt(receipt, path, *, overwrite=False)` (write-once); `provenance_receipt(branch, sha, dirty)`.
- `save_json(obj, path)` (allow_nan=False), `json_safe(obj)`.
- Hashing: `asset_hashes(assets)`, `sha256_file(path)`, `sha256_text(text)`, `config_hash(cfg)`.
- `export_tutorial_artifacts(cfg, manifest_dict, metrics_dict, validation_dict, output_dir)`.
- Declarative descriptors (`jaxfne/_signals.py`, manifest-safe): `AxisSpec(name, status="active", size=None, units_or_status="declared")` — typed descriptor for one tensor axis; `BasisSpec(space_basis="laminar_depth", time_basis="continuous_ms", field_regime="laminar_proxy", source_mode="proxy_no_field_solve", probe_basis="multimodal_proxy", axes=...)` — computation basis of a run; `DatasetSpec(name="unnamed_dataset", modality="unspecified", source_format="unspecified", comparison_label="p1", comparison_code=101, ...)` — manifest-safe dataset/comparison declaration for observed data. Factories: `default_basis_spec()` → `BasisSpec` (default matching the laminar-proxy scaffold).

## 10. Runtime / JAX / x64 / sharding

- `enable_x64()` — call **before** building arrays; verify with `runtime_report()["actual_dtype"]`.
- `runtime_report(runtime_config=None)`, `RuntimeConfig`.
- RuntimeConfig helpers (`jaxfne/util.py`): `merge_runtime_configs(*cfgs, **overrides)` → `RuntimeConfig` (layers configs left-to-right, later wins per field, then `overrides` on top); `runtime_config_diff(a, b)` → `{field: (a_val, b_val)}` for every differing field; `validate_runtime_config(cfg, *, strict=False)` → consistency warnings `list[str]` beyond `__post_init__`.
- `compilation_registry` — module-level `CompilationRegistry` instance (`jaxfne/validation.py`) tracking trace-shape compilation guarding (set mode with `set_mode(recompilation_guard)`); read `is_valid_signal`/registry state before trusting compiled-program reuse.
- Sharding: `make_population_mesh()`, `make_candidate_sharding(mesh)`, `make_replicated_sharding(mesh)`, `get_sharding_context()`.
- Lazy optional deps: `require_jaxley()`, `require_optax()` (and `vis.require_matplotlib()`).
- Bridge declarations (`jaxfne/bridges.py`): `BridgeSpec(name, backend, status="schema_only_no_backend_constructed", source_calibration_status="uncalibrated_bridge_output", metadata={})` — JSON-safe optional-backend bridge declaration; `JaxFemFieldBridge(geometry="laminar_column", n_layers=None, source_calibration_status="uncalibrated_jax_fem_bridge", metadata={})` — bridge contract for a future differentiable volumetric field solve (declaration only).
- HH reference traces (tutorial/comparison, not Jaxley-bridge validation):
  `hh_numpy_reference_trace(duration_ms, dt_ms, current_amplitude)` (standalone,
  no optional deps) and `hh_jaxley_reference_trace(duration_ms, dt_ms,
  current_amplitude)` (real Jaxley HH channel, raises via `require_jaxley()`
  if not installed — its `jaxley` import is deferred inside the function body,
  so `import jaxfne` remains independent of the optional dependency). Verify
  current export paths before use.

## 11. Selection helpers

- `select_cells(model, area, layers, cell_types, fraction, max_cells, seed)`, `select_neurons(model, area, layer, cell_type)`, `Model.select(area=..., ...)`.
- `SelectorSpec(area=None, area_id=None, layer=None, cell_type=None, ids=None)` — selector over area/layer/cell-type/id fields (`jaxfne/experimental_hpc/contracts.py`).

## Key classes (recognize, don't redefine)

Config/model: `Configuration`, `Config`, `LaminarColumnConfig`, `RuntimeConfig`, `RuntimeConfiguration`, `Simulation`, `Model`, `Net`, `EIGNetwork`, `LaminarPopulation`, `LaminarSourceGeometry`.
Emitters: `Emitter`, `IzhikevichEmitter`, `IzhikevichParams`, `LIFEmitter`, `GLIFEmitter`.
Edges/synapses: `EdgeList`, `SynapseLayer`, `SynapseSpec`, `SynapseState`, `ReceptorSpec`, `ConnectionCompileResult`.
Signals/probes/fields: `Signals`, `Signal`, `Probe`, `ReadoutSpec`, `ReadoutResult`, `LinearReadout`, `FieldOutput`.
Optimize: `Objective`, `ObjectiveReport`, `OptimizerSpec`, `AGSDR`/`AGSDROptimizerSpec`/`AGSDRState`, `GSDRState`, `SDRState`, `SurrogateConfig`, `MatrixParameterSpec`, `TuneResult`.
Paradigm/trials: `Paradigm`, `ParadigmCondition`, `ParadigmEvent`, `TrialBatch`/`TrialBatchResult`/`TrialResult`/`TrialSpec`, `StimulusSchedule`, `Simulation`.
Receipts: `RunReceipt`, `RuntimeConfig`, `CellTypePreset`, `NodeIdentity`.
Bridges: `JaxleyBridge`, `JaxleyEmitterBridge`, `JaxleyTraceSpec`.
Tensor: `NeuronalTensor`, `Area`, `Layer`, `NeuronType`, `InterConnection`, `AreaConnection`.

## Plasticity word overload (disambiguate before editing)

Three mechanisms share "plasticity" — do not conflate:

1. `Configuration.plasticity()` — declarative metadata only (`declared_not_wired_to_simulate`).
2. `Configuration.homeostasis(eta=...)` — wired synaptic homeostasis in `simulate_edge_recurrent_izhikevich_homeostatic`.
3. `run_stdp_stream` / `make_ei_cloud_network` — separate STDP path, not connected to `Model.simulate()`.

STDP config/state classes (`jaxfne/plasticity.py`): `STDPPlasticityConfig(A_plus=0.01, A_minus=0.012, tau_plus=20.0, tau_minus=20.0, w_min=0.0, w_max=1.5)` — activity-dependent STDP parameter configuration; `STDPState(W, trace_pre, trace_post)` — state container for the STDP synapse model.

`homeostasis(k_gain=...)` is a one-sided excitability damper, not a bidirectional rate setpoint (see repo `AGENTS.md`).

## Truth-plane reminders (always)

- Wording: **proxy only** — EEG-proxy, MEG-proxy, LFP-proxy, CSD-proxy, spectrolaminar-proxy, field-laminar-proxy. Never "real EEG/MEG", "calibrated amplitude", "solved field", "physical CSD", "mechanism proof", "biological validation/learning".
- Read the current status fields from `docs/scope_and_status.md` and live
  manifests. Never escalate them. Legacy status keys are migration inputs, not
  current output keys.

## Keeping this catalog honest

This is a snapshot. If a name here is missing at runtime (or you suspect drift), regenerate the inventory and reconcile — do not guess:

```bash
python3 - <<'PY'
import jaxfne as j, inspect
fns = sorted(n for n in dir(j) if not n.startswith("_") and not inspect.isclass(getattr(j,n)))
cls = sorted(n for n in dir(j) if not n.startswith("_") and inspect.isclass(getattr(j,n)))
print(f"{len(fns)} functions, {len(cls)} classes")
print("functions:", ", ".join(fns))
print("classes:", ", ".join(cls))
PY
```

If you find a function that belongs in a section above, add it. The cost of a
stale catalog is the exact failure this skill prevents: re-deriving an API that
already exists.
