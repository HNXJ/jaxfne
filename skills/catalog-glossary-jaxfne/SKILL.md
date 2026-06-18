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

**jaxfne has ~120 public functions + ~60 public classes at the top level, plus a
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
- Stimulus: `make_stimulus(*, kind, duration_ms, dt_ms, amplitude, frequency_hz, ...)`, `stimulus_schedule(events, n_neurons, ...)`.
- Read a signal: `Signals.get(key)` or free fn `get_signal(obj, key)`. Keys accept aliases: `"V_m"`/`"vm"`, `"spikes"`/`"spk"`, `"lfp_contacts"`, `"csd_contacts"`, `"source_native"`.

## 4. Readouts, projections, fields (all PROXY — no PDE solve)

- `project_laminar_sources(sources, positions, *, n_contacts, width)` → `FieldOutput`; `project_sources_to_laminar_field(...)`; `probe_laminar_modes(field_output, modes)`.
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
- `column_density_table(cfg)`, `layer_celltype_count_table(cfg)`, `configuration_table(cfg)`, `config_summary_frame(cfg)`, `cell_catalog_frame(catalog)`.

## 8. Visualization — `jtfne.vis.*` (signal-driven, proxy-safe)

Pass a `Signals` object; each returns a matplotlib fig (and a `*_with_meta` variant returns a JSON-safe metadata container):
`raster`, `vm`, `rate`, `source`, `lfp`, `lfp_traces`, `csd`, `csd_traces`, `eeg`, `meg`, `emm`, `psd`, `spectrogram`, `bandpower`, `connectivity`/`connectivity_matrix`, `laminar_profile`/`layer_celltype_counts`, `geometry3d`/`circuit3d`/`column_geometry`, `multi_area_layout`, `summary`, `objective_report`.
- `vis.spectrolaminar(signals)` — 3-panel spectrolaminar from a Signals object; `vis.spectrolaminar_suite(signals)` — Suite No. 2 readout panel.
- `vis.visualize_network_3d(data, *, output_html=..., show_edges=...)` — **interactive Plotly** 3D network (use for scaffold cells; supports HTML export, pan/zoom).
- `vis.visualize_laminar_column_3d(model, cfg, ...)`.

### `jtfne.vis.tutorial_panels.*` (trial/specs-driven suites)
- `spectrolaminar_suite_3panel(specs, model, cfg, areas=..., output_dir=..., theme="dark")` → `{area: Figure}` — the genuine depth×freq + band-crossover template.
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

Config/model: `Configuration`, `Config`, `JaxFNEConfig`, `LaminarColumnConfig`, `RuntimeConfig`, `Simulation`, `Model`, `Net`, `EIGNetwork`, `LaminarPopulation`, `LaminarSourceGeometry`.
Emitters: `Emitter`, `IzhikevichEmitter`, `IzhikevichParams`, `LIFEmitter`, `GLIFEmitter`.
Edges/synapses: `EdgeList`, `SynapseLayer`, `SynapseSpec`, `SynapseState`, `ReceptorSpec`, `ConnectionCompileResult`.
Signals/probes/fields: `Signals`, `Signal`, `Probe`, `ReadoutSpec`, `ReadoutResult`, `LinearReadout`, `FieldOutput`.
Optimize: `Objective`, `ObjectiveReport`, `OptimizerSpec`, `AGSDR`/`AGSDROptimizerSpec`/`AGSDRState`, `GSDRState`, `SDRState`, `SurrogateConfig`, `MatrixParameterSpec`, `TuneResult`.
Paradigm/trials: `Paradigm`, `ParadigmCondition`, `ParadigmEvent`, `TrialBatch`/`TrialBatchResult`/`TrialResult`/`TrialSpec`, `StimulusSchedule`, `Simulation`.
Receipts: `RunReceipt`, `RuntimeConfig`, `CellTypePreset`, `NodeIdentity`.
Bridges: `JaxleyBridge`, `JaxleyEmitterBridge`, `JaxleyTraceSpec`.

## Truth-plane reminders (always)

- Wording: **proxy only** — EEG-proxy, MEG-proxy, LFP-proxy, CSD-proxy, spectrolaminar-proxy, field-laminar-proxy. Never "real EEG/MEG", "calibrated amplitude", "solved field", "physical CSD", "mechanism proof", "biological validation/learning".
- Gates that ride along on dataclasses: `truth_mode=truth_safe_unverified`, `claim_level=computational_scaffold`, `field_solver_status=laminar_proxy_no_pde`, `physical_amplitude_claim_allowed=False`, `biological_learning_claim=False`, `mechanism_claim_status=not_claimed`. Read them; never escalate.

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
