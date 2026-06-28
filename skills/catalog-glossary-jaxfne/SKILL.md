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

**jaxfne has ~210 public functions + ~85 public classes at the top level (verified 2026-06-25; was ~120/~60, grown substantially with NeuronalTensor/HDP — re-run the inventory snippet at the bottom of this file before trusting either number), plus a
laminar-column tutorial pipeline (`jaxfne.tutorial_utils`) and a visualization
package (`jaxfne.vis`). Do not reinvent any of it.** This catalog exists so you
recognize what is already built instead of "rediscovering" it.

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
- `default_cortical_column_config(...)`, `default_spectrolaminar_config(areas, n_per_area, ...)`, `build_multi_area_columns(areas, n_per_area, layers, connectivity_mode)` — ready-made `Configuration`s.
- `construct(cfg, *, geometry=None)` → `Model` — turn a Configuration into a runnable Model.
- `build_laminar_column(name, n, ...)` → `Configuration` (single column; top-level).
- Suite No. 2 configs: `suite2_single_neuron_config`, `suite2_four_celltype_config`, `suite2_net1_config`, `suite2_v1_v4_config`, `suite2_simulation`, `suite2_run_bundle`, `suite2_celltype_presets`.
- Connectivity: `connect_columns(cfg, src, tgt, mode, ...)`, `all_to_all_intercolumn_connectivity(...)`, `sparse_intercolumn_connectivity(...)`, `build_laminar_connections(model, cfg)`, `compile_connection_rules(...)`, `make_edge_list_from_dense(weights, ...)`.
- Cells/emitters: `make_cell_dist`, `make_cell_type_catalog`, `make_eig_network`, `izhikevich_params_from_labels(labels, *, drive_overrides=...)`, `with_emitter_parameters(model, ...)`, `standard_receptor_specs`, `standard_receptor_tau_table`.
- `.jcfg.json` files: `load_config(path)`, `config_to_configuration`, `config_to_simulation`, `config_to_geometry`, `config_to_trial_batch`, `validate_config`, `validate_configuration`.

### 1b. NeuronalTensor (0.4.7) — tensor-first build path (`jtfne.NeuronalTensor`, `jtfne.NeuronType`, etc. are top-level, no submodule import needed)

`NeuronalTensor = [Areas, AreaConnections]`, `Area = [Layers x NeuronTypes, InterConnections]`.
This is a SEPARATE build path from `Configuration` — both converge on the same
`Model` type via `construct()`:

- `NeuronalTensor(areas=[...], name=...)`, `Area(name, layers=[...])`, `Layer(name, n_neurons, neuron_types=[...])`, `InterConnection(...)`, `AreaConnection(...)`.
- `NeuronType.make(name, *, relative_size=None, fraction=None, value_tag=...)` — `fraction` (0.4.7 addition, default `None`) declares an explicit population fraction; if **every** type in a `Layer` declares one, those normalized fractions populate `Configuration.metadata["area_layer_cell_types"][area][layer]`; if any type omits it, the whole layer falls back to an even split (backward-compatible).
- `neuronal_tensor_to_configuration(tensor, *, seed, duration_ms, dt_ms)` → `Configuration` — the internal bridge `construct()` uses.
- `construct(tensor, runtime_configuration)` → `Model` — same top-level `construct` as the `Configuration` path, dispatches on input type.
- `RuntimeConfiguration` (`neuronal_tensor.py`, frozen, execution-only: seed/duration_ms/dt_ms/etc.) — **distinct from** `RuntimeConfig` (`core.py`, has `enable_hdp`/`hdp_params`). `RuntimeConfiguration` has NO HDP field.
- `load`, `load_neuronal_tensor`, `load_canonical_neuronal_tensor`, `list_canonical_neuronal_tensors`, `merge_neuronal_tensors`, `construct_neuronal_tensor`.
- `DEFAULT_RELATIVE_SIZE` re-exports `emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`
  (E=5.0, PV=1.0, SST/VIP=1.5, …) — single source of truth for NeuronType sizes and
  HDP `tau_i = tau_0_ms * size_i**3` scaling.

**Enabling HDP on a tensor-built Model (no new public API needed):** build via
`construct(tensor, RuntimeConfiguration(...))`, then pass an explicit
`runtime=RuntimeConfig(enable_hdp=True, hdp_params={...})` to `simulate()` —
the explicit `runtime=` kwarg overrides any `Configuration`-derived config.
See `jaxfne-modeling-optimization-schema` for the full pattern + cube-law tau formula.

### 1c. HDP homeostatic plasticity module (`jaxfne/hdp_network.py`) — generic config-driven builder, no per-N functions

- `DEFAULT_HDP = dict(K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0, barrier_c=0.01, barrier_d=0.01)`.
- `BASE_HDP_KWARGS_DEFAULT` (H_min=0.1, H_max=10.0, alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0, ...), `BASE_DRIVE_BY_CELL_TYPE_DEFAULT = {"E":4.0,"PV":4.0,"SST":4.0,"VIP":4.0}`, `DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT`.
- Kernel: `simulate_edge_recurrent_izhikevich_hdp` (`emitters.py`) — `tau_i = tau_0_ms * size_i**3` (cube law, verified 0.4.7; NOT `size_i**2`). `hdp_params` is a free-form dict forwarded through `core.py`'s `_hdp_packed`; any new key (e.g. `size_scale_by_cell_type`, `size_scale_override`) must be explicitly added there or it is silently dropped — verify with `grep -n size_scale_by_cell_type jaxfne/core.py` before trusting a new `hdp_params` key reaches the kernel.
- `model.last_hdp_diagnostics()` → dict with `H_trace`, weight trace, per-edge `receptor_index`.

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
- ODE: `euler_scan(y_init, t_start, dt, n_steps, dydt_fn)`, `euler_step(...)` (use `lax.scan`-based `euler_scan` for grad-through-time).
- Stimulus: `make_stimulus(*, kind, duration_ms, dt_ms, amplitude, frequency_hz, ...)`, `stimulus_schedule(events, n_neurons, ...)`. **Per-neuron-subset targeting:** `target_indices` is a per-event dict key (not a `StimulusSchedule` constructor kwarg), read by `StimulusSchedule.to_array`/`to_array_jax` (`StimulusSchedule` class at `jaxfne/core.py:3185`) to restrict that event to a specific neuron subset (e.g. only L4 E cells) instead of the whole column — build the index list from `model.neuron_table()` filtered by `layer`/`cell_type` and put it on each event dict, e.g. `StimulusSchedule(events=({"onset_ms":0.0,"duration_ms":50.0,"amplitude":5.0,"target_indices":l4e_idx},), n_neurons=model.n_neurons)`. Don't hand-roll a per-neuron drive mask for this; see `jaxfne-paradigm-design` for the full pattern.
- Noise control on the Config-path is **kernel-dependent**, not uniform: `simulate_eig_izhikevich`, `simulate_edge_recurrent_izhikevich`, and the homeostatic variant accept `noise_scale=` (`None` = historical `0.5`); `simulate_receptor_exponential_izhikevich` hardcodes `0.5` inline with no override kwarg at all.
- Read a signal: `Signals.get(key)` or free fn `get_signal(obj, key)`. Keys accept aliases: `"V_m"`/`"vm"`, `"spikes"`/`"spk"`, `"lfp_contacts"`, `"csd_contacts"`, `"source_native"`.

## 4. Readouts, projections, fields (all PROXY — no PDE solve)

- `project_laminar_sources(sources, positions, *, n_contacts, width, mode="density_preserving")` → `FieldOutput`. Default **`mode="density_preserving"`** (SUM-like, preserves density). Use **`mode="row_normalize"`** only for explicit opt-in / backward compatibility — it flattens depth structure when contacts fall outside the population (see `skills/FRICTIONS_STACK.md` F-003).
- `project_sources_to_laminar_field(...)`, `probe_laminar_modes(field_output, modes)`.
- Lead-field proxies: `eeg_proxy_transform(source, leadfield)`, `meg_proxy_transform(source_oriented, leadfield)`, `emm_proxy_transform(...)`.
- `construct_source_tensor(*, mode, ...)`, `compute_conservation_proxy_diagnostics(...)`, `validate_projection_invariants(...)`, `validate_source_field_status(...)`.

## 5. Optimizers & tuning (AGSDR/GSDR/SDR/Optax)

- Specs: `agsdr(...)`, `gsdr(...)`, `random_search(...)`, `optax_adam(...)`, `optax_sgd(...)`.
- Optax GradientTransformations: `agsdr_transform(...)`, `gsdr_transform(...)`, `sdr_transform(...)`.
- Objectives: `objective()`, `rate_targets(groups, targets_hz, weights)`, `rate_synchrony_targets(target_rate_hz, target_kappa_synchrony, ...)`, `readout_spec(...)`, `matrix_parameter(*, mask, bounds, ...)`, `surrogate_config(...)`.
- High-level: `Model.tune(obj, optimizer=..., ...)` → `TuneResult`; `suite2_tune_noise_agsdr_adam(model, ...)`; `tutorial_utils.tune_laminar_agsdr(...)`.

## 6. Paradigms & trials

- `paradigm(name)`, `evoked_l4_drive_paradigm(...)`, `omission_oddball_paradigm(...)`, `standard_visual_omission()`.
- `trial_batch(conditions, n_reps, seed, ...)` → `TrialBatch`.

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
`plot_raster`, `plot_spectrolaminar_power(t, signal, freq_min, freq_max, n_freqs)`, `plot_laminar_readout`, `plot_population_rate`, `plot_voltage_samples`, `plot_connectivity_matrix`, `save_png(fig, name, fig_dir)`.

## 9. Manifests, receipts, JSON, hashing (Truth plane)

- `manifest(cfg, signals=None, readout=None, paradigm=None, objective=None, ...)` → strict JSON-safe run manifest.
- `validation_report(config_valid, issues, metadata)`, `probe_report(n_probes, probe_types, metadata)`.
- `run_receipt(model, signals, *, tags=...)` → `RunReceipt`; `save_receipt(receipt, path, *, overwrite=False)` (write-once); `provenance_receipt(branch, sha, dirty)`.
- `save_json(obj, path)` (allow_nan=False), `json_safe(obj)`.
- Hashing: `asset_hashes(assets)`, `sha256_file(path)`, `sha256_text(text)`, `config_hash(cfg)`.
- `export_tutorial_artifacts(cfg, manifest_dict, metrics_dict, validation_dict, output_dir)`.

## 10. Runtime / JAX / x64 / sharding

- `enable_x64()` — call **before** building arrays; verify with `runtime_report()["actual_dtype"]`.
- `runtime_report(runtime_config=None)`, `RuntimeConfig`.
- Sharding: `make_population_mesh()`, `make_candidate_sharding(mesh)`, `make_replicated_sharding(mesh)`, `get_sharding_context()`.
- Lazy optional deps: `require_jaxley()`, `require_optax()` (and `vis.require_matplotlib()`).

## 11. Selection helpers

- `select_cells(model, area, layers, cell_types, fraction, max_cells, seed)`, `select_neurons(model, area, layer, cell_type)`, `Model.select(area=..., ...)`.

## Key classes (recognize, don't redefine)

Config/model: `Configuration`, `Config`, `JaxFNEConfig`, `LaminarColumnConfig`, `RuntimeConfig`, `RuntimeConfiguration`, `Simulation`, `Model`, `Net`, `EIGNetwork`, `LaminarPopulation`, `LaminarSourceGeometry`.
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

`homeostasis(k_gain=...)` is a one-sided excitability damper, not a bidirectional rate setpoint (see repo `AGENTS.md`).

## Truth-plane reminders (always)

- Wording: **proxy only** — EEG-proxy, MEG-proxy, LFP-proxy, CSD-proxy, spectrolaminar-proxy, field-laminar-proxy. Never "real EEG/MEG", "calibrated amplitude", "solved field", "physical CSD", "mechanism proof", "biological validation/learning".
- Gates that ride along on dataclasses (v0.4.0 canonical schema): `claim_level=computational_scaffold`, `field_solver_status=linear_solver`, `field_claim_level=proxy_readout`, `physical_amplitude_calibrated=False`, `biological_learning_claim=False`, `mechanism_claim_status=not_claimed`. Read them; never escalate. (RETIRED, do not emit: `truth_mode`/`truth_safe_unverified`, `laminar_proxy_no_pde`, `proxy_readout_only`, `physical_amplitude_claim_allowed`. Migrate legacy JSON via `jtfne.migrate_schema`.)

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
