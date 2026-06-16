# jaxfne Agent API Catalog — v0.3.31

**Purpose:** a curated lookup table of package-native jaxfne functions so agents
stop rediscovering / hand-rolling code that already exists. Canonical import:
`import jaxfne as jtfne`. Submodule paths (`jtfne.tutorial_utils`, `jtfne.vis`)
are shown explicitly. **No invented APIs** — every entry is verified present in
v0.3.31. Curated, ≤100 entries; the exhaustive list lives in the
`catalog-glossary-jaxfne` agent skill.

> **Read this before writing a helper.** If you are about to compute a PSD,
> raster, LFP/CSD/EEG/MEG proxy, spectrolaminar profile, AGSDR tune, manifest,
> or hash — it already exists below.

## Forbidden substitutions (do NOT do these)

- **Do not hand-roll a PSD / spectrogram for spectrolaminar.** Use the native
  spectrolaminar pipeline (section "Spectrolaminar-proxy pipeline").
- **Do not call a depth × time field map "spectrolaminar."** Spectrolaminar is
  depth × **frequency** relative power. Depth × time = *field-laminar-proxy*.
- **Do not use `*-like` public labels for modalities.** Always use the `-proxy`
  suffix: LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy — never the `_like` variant.
- **Do not call `jtfne.load_json`** — it does not exist. Use `json.load(open(p))`
  for plain reads, `jtfne.load_config(path)` for `.jcfg.json`, and
  `jtfne.save_json(obj, path)` to write strict JSON.

## Core: configure / construct / simulate

| function | import | purpose | I/O |
|---|---|---|---|
| `laminar_cortex_config` | `jtfne` | multi-area laminar Configuration; `baseline_drive_by_cell_type=` injects native drive | kwargs → `Configuration` |
| `default_cortical_column_config` | `jtfne` | single-column default Configuration | → `Configuration` |
| `default_spectrolaminar_config` | `jtfne` | multi-area spectrolaminar Configuration (CSD/LFP/EEG/MEG readouts) | → `Configuration` |
| `build_multi_area_columns` | `jtfne` | multi-area Configuration with inter-area connectivity | → `Configuration` |
| `construct` | `jtfne` | Configuration → runnable Model (also the "reconstruct" step) | `Configuration` → `Model` |
| `simulate` | `jtfne` | run a single simulation | `Model`, duration_ms, dt_ms, seed → `Signals` |
| `run_trials` | `jtfne` | run a batch of trials | `Model`, `TrialBatch`, `Simulation` → `TrialBatchResult` |
| `euler_scan` | `jtfne` | forward-Euler over n_steps via `lax.scan` (grad-safe) | y_init,dt,n,dydt → (ys,ts) |
| `euler_step` | `jtfne` | single forward-Euler step | y,t,dt,dydt → y |
| `make_stimulus` | `jtfne` | build a stimulus array (constant/sine/pulse/noise) | kwargs → ndarray |
| `stimulus_schedule` | `jtfne` | build a `StimulusSchedule` from events | events,n → `StimulusSchedule` |

## Laminar cortex / multi-area scaffold

| function | import | purpose | I/O |
|---|---|---|---|
| `make_laminar_column_config` | `jtfne.tutorial_utils` | LaminarColumnConfig (areas, layers, contacts, trials, freq grid) | kwargs → `LaminarColumnConfig` |
| `build_laminar_column` | `jtfne.tutorial_utils` | LaminarColumnConfig → model dict (CSD/LFP contacts) | `cfg` → model dict |
| `build_laminar_connections` | `jtfne` | extract connection matrices from a laminar model | model,cfg → matrices |
| `connect_columns` | `jtfne` | add inter-column connectivity to a Configuration | cfg,src,tgt → `Configuration` |
| `all_to_all_intercolumn_connectivity` | `jtfne` | dense inter-column connectivity spec | → dict |
| `sparse_intercolumn_connectivity` | `jtfne` | sparse inter-column connectivity spec | → dict |
| `select_cells` | `jtfne` | select cell indices by area/layer/cell_type | model,... → ndarray |
| `select_neurons` | `jtfne` | select neuron indices by area/layer/cell_type | model,... → ndarray |

## Native drive / Izhikevich emitters

| function | import | purpose | I/O |
|---|---|---|---|
| `izhikevich_params_from_labels` | `jtfne` | reduced Izhikevich params; `drive_overrides=` sets native baseline drive | labels → `IzhikevichParams` |
| `with_emitter_parameters` | `jtfne` | functional override of emitter params on a Model | `Model`,... → `Model` |
| `simulate_eig_izhikevich` | `jtfne` | scan-based reduced Izhikevich kernel | params,n,dt,key → (v,u,spk) |
| `simulate_edge_recurrent_izhikevich` | `jtfne` | sparse recurrent Izhikevich kernel | params,edges,... → arrays |
| `make_eig_network` | `jtfne` | minimal EIG network with laminar depth | n → `EIGNetwork` |
| `make_edge_list_from_dense` | `jtfne` | dense W → sparse `EdgeList` | W → `EdgeList` |
| `make_cell_type_catalog` | `jtfne` | E/PV/SST/VIP Izhikevich preset catalog | → dict |
| `make_cell_dist` | `jtfne` | layer × cell-type distribution matrix | layers,types → ndarray |

## Signals and readouts

| function | import | purpose | I/O |
|---|---|---|---|
| `Signals.get` / `get_signal` | `jtfne` | read a signal; aliases `V_m`/`vm`, `spikes`/`spk`, `lfp_contacts`, `csd_contacts` | key → array |
| `run_receipt` | `jtfne` | build a RunReceipt for a completed run | model,signals → `RunReceipt` |
| `kappa_synchrony` | `jtfne` | spike synchrony (kappa) | spikes,dt → float |
| `population_rate_hz` | `jtfne.tutorial_utils` | mean population firing rate | spikes,dt → float |
| `configuration_table` | `jtfne` | human-readable config summary | cfg → dict |
| `column_density_table` | `jtfne` | neurons/mm³ per layer | cfg → dict |
| `layer_celltype_count_table` | `jtfne` | counts by layer × cell type | cfg → dict |

## EEG-proxy / MEG-proxy / LFP-proxy / CSD-proxy

| function | import | purpose | I/O |
|---|---|---|---|
| `eeg_proxy_transform` | `jtfne` | EEG-proxy via linear leadfield projection | source,leadfield → array |
| `meg_proxy_transform` | `jtfne` | MEG-proxy via oriented leadfield projection | source,leadfield → array |
| `emm_proxy_transform` | `jtfne` | normalized activity/source/field cost (EMM-proxy) | arrays → array |
| `project_laminar_sources` | `jtfne` | project sources to laminar proxy contacts | sources,pos → `FieldOutput` |
| `project_sources_to_laminar_field` | `jtfne` | source → laminar field-proxy contacts | sources,pos → `FieldOutput` |
| `probe_laminar_modes` | `jtfne` | extract declared modes from a FieldOutput | field,modes → dict |
| `vis.lfp` / `vis.csd` | `jtfne.vis` | LFP-proxy / CSD-proxy heatmap from Signals | signals → fig |
| `vis.eeg` / `vis.meg` | `jtfne.vis` | EEG-proxy / MEG-proxy traces from Signals | signals → fig |
| `vis.raster` / `vis.vm` / `vis.rate` | `jtfne.vis` | raster / Vm / population-rate plots | signals → fig |

## Spectrolaminar-proxy pipeline (the exact path — do not rediscover)

| function | import | purpose | I/O |
|---|---|---|---|
| `make_laminar_column_config` | `jtfne.tutorial_utils` | step 1: build LaminarColumnConfig | kwargs → cfg |
| `build_laminar_column` | `jtfne.tutorial_utils` | step 2: cfg → model dict | cfg → model |
| `simulate_laminar_trials` | `jtfne.tutorial_utils` | step 3: multi-trial sim | model,cfg,n_trials → trials dict |
| `summarize_spectrolaminar_similarity` | `jtfne.tutorial_utils` | step 4: scores + specs (all areas) | trials,cfg → (DataFrame, specs) |
| `spectrolaminar_from_trials` | `jtfne.tutorial_utils` | per-area depth×freq profile | trials,cfg → (profile, info) |
| `spectrolaminar_suite_3panel` | `jtfne.vis.tutorial_panels` | **3-panel depth×freq suite** (map + band crossover + spectra) | specs,model,cfg → {area: Figure} |
| `vis.spectrolaminar` | `jtfne.vis` | 3-panel spectrolaminar from a Signals object | signals → fig |
| `plot_spectrolaminar_power` | `jtfne.tutorial_utils` | quick depth×freq PSD heatmap from an array | t,signal → fig |
| `activity_trace_suite` | `jtfne.vis.tutorial_panels` | raster + LFP-proxy + CSD-proxy + PSD suite | trials,cfg → fig |

`info` from `spectrolaminar_from_trials` keys: `freq_hz`, `pos_from_l4` (depths),
`relative_power`, `alpha_beta`, `gamma`. Spectrolaminar = **depth × frequency**.

## Output bundle helpers (Truth plane)

| function | import | purpose | I/O |
|---|---|---|---|
| `manifest` | `jtfne` | strict JSON-safe run manifest | cfg[,signals,...] → dict |
| `validation_report` | `jtfne` | validation report bundle | valid,issues,meta → dict |
| `probe_report` | `jtfne` | probe operator report bundle | n,types → dict |
| `export_tutorial_artifacts` | `jtfne` | export cfg+results as strict JSON | cfg,dicts → dict |
| `save_json` | `jtfne` | write strict JSON (`allow_nan=False`) | obj,path → None |
| `json_safe` | `jtfne` | coerce scientific objects to JSON values | obj → obj |
| `asset_hashes` | `jtfne` | SHA256 manifest for a set of assets | dict → dict |
| `sha256_file` / `sha256_text` | `jtfne` | SHA256 of a file / text | path|text → hex |
| `config_hash` | `jtfne` | compact SHA256 of a config-like object | cfg → hex |
| `save_png` | `jtfne.tutorial_utils` | save a matplotlib fig to a figures dir | fig,name,dir → path |

## Optimizer / AGSDR / objective utilities

| function | import | purpose | I/O |
|---|---|---|---|
| `agsdr` | `jtfne` | AGSDR optimizer spec | kwargs → spec |
| `gsdr` | `jtfne` | GSDR optimizer spec | kwargs → `OptimizerSpec` |
| `random_search` | `jtfne` | random-search optimizer spec | → `OptimizerSpec` |
| `optax_adam` / `optax_sgd` | `jtfne` | Optax optimizer specs | lr → `OptimizerSpec` |
| `agsdr_transform` / `gsdr_transform` / `sdr_transform` | `jtfne` | Optax GradientTransformations | kwargs → transform |
| `objective` | `jtfne` | empty `Objective` builder | → `Objective` |
| `rate_targets` | `jtfne` | multi-group firing-rate objective | groups,targets → `Objective` |
| `rate_synchrony_targets` | `jtfne` | rate + kappa-synchrony objective | targets → spec |
| `tune_laminar_agsdr` | `jtfne.tutorial_utils` | AGSDR-tune laminar control to a target rate | model,cfg → tuple |
| `suite2_tune_noise_agsdr_adam` | `jtfne` | tune Poisson drive to a target rate range | model → `TuneResult` |
| `Model.tune` | `jtfne` (method) | tune a Model with an Objective+optimizer | obj,optimizer → `TuneResult` |

## Save / load / reconstruct helpers

| function | import | purpose | I/O |
|---|---|---|---|
| `save_json` | `jtfne` | write strict JSON | obj,path → None |
| `load_config` | `jtfne` | load a `.jcfg.json` → `JaxFNEConfig` | path → `JaxFNEConfig` |
| `save_receipt` | `jtfne` | write a RunReceipt (write-once) | receipt,path → None |
| `provenance_receipt` | `jtfne` | capture branch/sha/dirty provenance | branch,sha → dict |
| `validate_configuration` | `jtfne` | validate a Configuration against gates | cfg → dict |
| `validate_config` | `jtfne` | validate a `JaxFNEConfig` | cfg → `ConfigValidationResult` |
| (reconstruct) `construct` | `jtfne` | rebuild a Model from a (re)loaded config | cfg → `Model` |
| (plain read) `json.load` | stdlib | read back a saved JSON dict (no `jtfne.load_json`) | path → dict |

## Optional Jaxley / PyNWB placeholders

| function | import | purpose | I/O |
|---|---|---|---|
| `require_jaxley` | `jtfne` | lazy import Jaxley with an install hint | → module |
| `require_optax` | `jtfne` | lazy import Optax with an install hint | → module |
| `jaxley_trace_to_signals` | `jtfne` | convert a Jaxley voltage trace → `Signals` | trace → `Signals` |
| `read_nwb` / `write_nwb` | `jtfne` | NWB placeholders (not implemented — do not assume I/O) | placeholder |

---

**Truth gates ride on every crossing dataclass and must never be escalated:**
`field_solver_status=linear_solver`, `physical_amplitude_calibrated=False`,
`biological_learning_claim=False`, `mechanism_claim_status=not_claimed`.
