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

**jaxfne has 162 public functions + 82 public classes + 12 public constants at the top level (checked 2026-08-07 by inspecting the runtime package: 256 `__all__` entries, all resolvable; the earlier "~210/~85" (2026-06-25) and "~120/~60" estimates were stale — re-run the inventory snippet at the bottom of this file before trusting either number again), plus a
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
- `default_cortical_column_config(...)`, `build_multi_area_columns(areas, n_per_area, layers, connectivity_mode)` — ready-made `Configuration`s. **`default_spectrolaminar_config`/`default_nuclei_config` were REMOVED** (2026-06-30) — their JSON output is archived read-only at `jaxfne/configs/legacy/{spectrolaminar_default,nuclei_default}.json` (reproduce inline via the fluent builder if you need the shape, don't try to call the removed functions).
- Canonical layer/geometry constants (`jaxfne/builders.py`): `CANONICAL_LAYERS_6L = ("L1","L2","L3","L4","L5","L6")`; `DEFAULT_LAYERS = ("L1","L2/3","L4","L5","L6")` (5-layer, L2/3 merged); `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` / `CANONICAL_LAYER_CELL_TYPE_FRACTIONS_5L` (per-layer E/PV/SST/VIP fractions); `CANONICAL_Z_BANDS` / `CANONICAL_Z_BANDS_5L` (normalized depth bands per layer); `FLAT_CELL_TYPE_FRACTIONS = {"E":0.75,"PV":0.10,"SST":0.08,"VIP":0.07}`. These drive `ei_profile="canonical"` layout checks — feed them to `laminar_cortex_config`/`NeuronalTensor`, don't hand-roll competing tables.
- Cell/receptor presets (`jaxfne/presets.py`): `CELL_TYPE_PRESETS` (Izhikevich presets by label: `E_RS`, `PV_FS`, `SST_LTS`, `VIP_IS`); `RECEPTOR_KINETICS` (AMPA/NMDA/GABA_A/GABA_B kinetics: `receptor_index`, `tau_ms`, `reversal_mV`, `sign`); `DEFAULT_SPIKE_IMPULSE_GAIN = 20.0` (hardcoded spike gain; **keep `emitters` dense/edge kernels in sync with it** — see AGENTS.md fragility).
- `construct(cfg, *, geometry=None)` → `Model` — turn a Configuration into a runnable Model.
- `build_laminar_column(name, n, ...)` → `Configuration` (single column; top-level).
- Suite No. 2 configs: `suite2_single_neuron_config`, `suite2_four_celltype_config`, `suite2_net1_config`, `suite2_v1_v4_config`, `suite2_simulation`, `suite2_run_bundle`, `suite2_celltype_presets`.
- Connectivity: `connect_columns(cfg, src, tgt, mode, ...)`, `all_to_all_intercolumn_connectivity(...)`, `sparse_intercolumn_connectivity(...)`, `build_laminar_connections(model, cfg)`, `compile_connection_rules(...)`, `make_edge_list_from_dense(weights, ...)`.
- Cells/emitters: `make_cell_dist`, `make_cell_type_catalog`, `make_eig_network`, `izhikevich_params_from_labels(labels, *, drive_overrides=...)`, `with_emitter_parameters(model, ...)`, `standard_receptor_specs`, `standard_receptor_tau_table`.
- **`.jcfg.json`/`JaxFNEConfig` format DELETED (2026-06-30)**: `load_config`, `validate_config`, `config_to_configuration`, `config_to_simulation`, `config_to_geometry`, `config_to_trial_batch`, `config_truth_boundary`, `ConfigValidationResult`, `JaxFNEConfig` no longer exist — legacy format lived only in tests, never a real asset. `validate_configuration` (the `Configuration`-native validator, distinct name) still exists.

### 1b. NeuronalTensor (0.4.7) — tensor-first build path (`jtfne.NeuronalTensor`, `jtfne.NeuronType`, etc. are top-level, no submodule import needed)

`NeuronalTensor = [Areas, AreaConnections]`, `Area = [Layers x NeuronTypes, InterConnections]`.
This is a SEPARATE build path from `Configuration` — both converge on the same
`Model` type via `construct()`:

- `NeuronalTensor(areas=[...], name=...)`, `Area(name, layers=[...])`, `Layer(name, n_neurons, neuron_types=[...])`, `InterConnection(...)`, `AreaConnection(...)`.
- Tensor geometry/placement classes: `Geometry3D(distribution="uniform_random", x_range=(0,1), y_range=(0,1), z_range=(0,1), value_tag="relative")` (always 3D; fix an axis at 0.0 to collapse it); `Pose3D(plane="xy", rotation_deg=0.0, translation=(0,0,0), value_tag="relative")` (where an Area's layer stack sits in global 3D space); `PlasticParams(w_mech=1.0, H=0.0, value_tag="relative")` (trainable per-connection gain + homeostatic H factor); `StaticParams(g_mech=..., reversal_potentials_mV=..., dT_ms=0.1, value_tag="relative")` (never plastic: conductances, reversal potentials, dT). All dataclasses from `jaxfne/neuronal_tensor.py`.
- `NEURONAL_TENSOR_SCHEMA_VERSION = "neuronal_tensor_v1"` — schema tag checked by `load_neuronal_tensor`/canonical-tensor loading; migrate, don't patch around mismatches.
- `NeuronType.make(name, *, relative_size=None, fraction=None, value_tag=...)` — `fraction` (0.4.7 addition, default `None`) declares an explicit population fraction; if **every** type in a `Layer` declares one, those normalized fractions populate `Configuration.metadata["area_layer_cell_types"][area][layer]`; if any type omits it, the whole layer falls back to an even split (backward-compatible).
- `neuronal_tensor_to_configuration(tensor, *, seed, duration_ms, dt_ms)` → `Configuration` — the internal bridge `construct()` uses.
- `construct(tensor, runtime_configuration)` → `Model` — same top-level `construct` as the `Configuration` path, dispatches on input type.
- `RuntimeConfiguration` (`neuronal_tensor.py`, frozen, execution-only: seed/duration_ms/dt_ms/etc.) — **distinct from** `RuntimeConfig` (now in `jaxfne/_runtime_config.py`, re-exported unchanged from `jaxfne.core`/top-level `jaxfne`; has `enable_hdp`/`hdp_params`). `RuntimeConfiguration` has NO HDP field.
- `load`, `load_neuronal_tensor`, `load_canonical_neuronal_tensor`, `list_canonical_neuronal_tensors`, `merge_neuronal_tensors`, `construct_neuronal_tensor`.
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

- `DEFAULT_HDP = dict(K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0, barrier_c=0.01, barrier_d=0.01)`.
- `DEFAULT_HDP_DESYNC = dict(K_HDP=0.01, tau_0_ms=5.0, K_ctrl=0.15, rho_passive=0.0, barrier_c=0.01, barrier_d=0.01, alpha=0.05, gamma=0.5, C_spike=0.0)` — "responsive H" family (faster `tau_0_ms`, rate-drain `gamma`), vs. `DEFAULT_HDP`'s near-static profile.
- `DEFAULT_HDP_V1_PFC_AAAB = dict(DEFAULT_HDP_DESYNC, K_HDP=0.003, K_w_ctrl=0.001)` + `v1_pfc_aaab_hdp_params()` (added 2026-07-05) — the named preset for the V1-PFC continuous AAAB paradigm; `v1_pfc_aaab_hdp_params()` returns the full `BASE_HDP_KWARGS_DEFAULT` + preset + `size_scale_by_cell_type` assembly, single source of truth (don't hand-roll this dict in a script).
- `K_w_ctrl` (added 2026-07-04) — weight-magnitude two-sided restoring force, `dwmag/dt += K_w_ctrl*(wmag_baseline - wmag)`, mirroring `K_ctrl`'s form for `H`; default `0.0` (backward compatible). Fixes the previously-real weight-carryover runaway when chaining `Model.with_hdp_initial_state(H0=..., w0=...)` across trials. Verified stable at 100 chained trials with `K_w_ctrl=0.001`.
- `BASE_HDP_KWARGS_DEFAULT` (H_min=0.1, H_max=10.0, alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0, ...), `BASE_DRIVE_BY_CELL_TYPE_DEFAULT = {"E":4.0,"PV":4.0,"SST":4.0,"VIP":4.0}`, `DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT`.
- Kernel: `simulate_edge_recurrent_izhikevich_hdp` (`emitters.py`) — `tau_i = tau_0_ms * size_i**3` (cube law, verified 0.4.7; NOT `size_i**2`). `hdp_params` is a free-form dict forwarded through `_model.py`'s `_hdp_packed` (moved from `core.py` during the 2026-07-04/05 monolith split — core.py is now a 233-line pure re-export aggregator); any new key (e.g. `size_scale_by_cell_type`, `size_scale_override`, `K_w_ctrl`) must be explicitly added there or it is silently dropped — verify with `grep -n size_scale_by_cell_type jaxfne/_model.py` before trusting a new `hdp_params` key reaches the kernel.
- True turn-to-turn HDP state (low-level, `_pipeline.py`): `DynamicState(v, u, prev_spikes, syn_state, H, w)` — full six-field carry tuple for continuous multi-turn runs via `compile_step_fn`/`scan_network` (the canonical low-level HDP call pattern; `Model.with_hdp_initial_state` only carries H/w partially).
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
- **`bound_mode`** (2026-07-16, default `"minimal"` = unchanged prior
  behavior): `"stable"` applies a smooth tanh soft-bound to `x`/`G`/`H`
  instead of `jnp.clip` -- a bounded codomain that can't be numerically
  outrun (verified: fixes a real N=16 divergence to NaN under the flat
  canonical `G_max=5.0`, reproducible across seeds under `"minimal"`).
  `x` was previously the only *unbounded* state (`G`/`H` were already
  clipped) -- new `HomeostaticEIParams.x_min`/`.x_max` fields back this,
  default a wide `+-1e6` (safe/inert under `"minimal"`).
- **`hebbian_pairwise`** conductance rule (2026-07-16): independent gains
  per population pair (E-E/E-I/I-E/I-I) via
  `make_hebbian_pairwise_rule(k_ee, k_ei, k_ie, k_ii)`; default gains all
  `1.0` == plain `hebbian`. Custom gains are a callable, so only reachable
  by calling `simulate_homeostatic_ei` directly (same `Configuration`
  JSON-safety limitation as any custom rule).
- **`cubic_penalty_coupled`** homeostasis rule (2026-07-16): adds E<->I
  cross-population coupling to `cubic_penalty` -- every other rule's `dH`
  depends only on that neuron's own `x`; this one lets one population's H
  respond to the other's activity.
- **Notes:** `activation_rule="linear"` diverges once `G`-adaptation is on —
  the default is `"cubic"`. `Model.summary()`/`.neuron_table()`/
  `.checkpoint()`/`.with_emitter_parameters()`/`.simulate_batch()` raise
  `NotImplementedError` for this family. `simulate_homeostatic_ei` is
  `jax.jit`-compiled (repeated calls with the same static rule/mode config
  reuse one compiled program). Milestones 1-3 covered by
  `tests/test_homeostatic_ei_*.py`; Milestones 4-6 tracked in
  `artifacts/developer/plans.json`.

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
- ODE: `euler_scan(y_init, t_start, dt, n_steps, dydt_fn)`, `euler_step(...)` (use `lax.scan`-based `euler_scan` for grad-through-time). Solver classes (`jaxfne/solvers.py`): `EulerSolver(dt)` (forward Euler via JAX `lax.scan`), `DiffraxSolver(dt, rtol=1e-3, atol=1e-6, solver_type=None)` (optional diffrax Runge-Kutta, lazily imported), `SolverConfig(method="euler", dt=0.1, rtol=1e-3, atol=1e-6, solver_type=None)` (dataclass for ODE solver configuration).
- Stimulus: `make_stimulus(*, kind, duration_ms, dt_ms, amplitude, frequency_hz, ...)`, `stimulus_schedule(events, n_neurons, ...)` (builder function, `jaxfne/core.py`). **Per-neuron-subset targeting:** `target_indices` is a per-event dict key (not a `StimulusSchedule` constructor kwarg), read by `StimulusSchedule.to_array`/`to_array_jax` (`StimulusSchedule` class now lives in `jaxfne/_signals.py`, re-exported from `jaxfne.core`/top-level `jaxfne` unchanged) to restrict that event to a specific neuron subset (e.g. only L4 E cells) instead of the whole column — build the index list from `model.neuron_table()` filtered by `layer`/`cell_type` and put it on each event dict, e.g. `StimulusSchedule(events=({"onset_ms":0.0,"duration_ms":50.0,"amplitude":5.0,"target_indices":l4e_idx},), n_neurons=model.n_neurons)`. Don't hand-roll a per-neuron drive mask for this; see `jaxfne-paradigm-design` for the full pattern. An event dict may also carry `frequency_hz` (added since): when present, the flat `amplitude` plateau is replaced by a true sinusoidal drive `amplitude * sin(2*pi*frequency_hz*t)` instead of a flat plateau; absent `frequency_hz` reproduces the old flat-plateau behavior unchanged.
- Noise control on the Config-path is **kernel-dependent**, not uniform: `simulate_eig_izhikevich`, `simulate_edge_recurrent_izhikevich`, and the homeostatic variant accept `noise_scale=` (`None` = historical `0.5`); `simulate_receptor_exponential_izhikevich` hardcodes `0.5` inline with no override kwarg at all.
- Read a signal: `Signals.get(key)` or free fn `get_signal(obj, key)`. Keys accept aliases: `"V_m"`/`"vm"`, `"spikes"`/`"spk"`, `"lfp_contacts"`, `"csd_contacts"`, `"source_native"`.

## 4. Readouts, projections, fields (default path is PROXY; one real solver is experimental)

- `project_laminar_sources(sources, positions, *, n_contacts, width, mode="density_preserving")` → `FieldOutput`. Default **`mode="density_preserving"`** (SUM-like, preserves density). Use **`mode="row_normalize"`** only for explicit opt-in / backward compatibility — it flattens depth structure when contacts fall outside the population (see `skills/FRICTIONS_STACK.md` F-003). This remains the default `simulate()` field dispatch, unaffected by the solver below.
- `project_sources_to_laminar_field(...)`, `probe_laminar_modes(field_output, modes)`.
- Lead-field proxies: `eeg_proxy_transform(source, leadfield)`, `meg_proxy_transform(source_oriented, leadfield)`, `emm_proxy_transform(...)`.
- `construct_source_tensor(*, mode, ...)`, `compute_conservation_proxy_diagnostics(...)`, `validate_projection_invariants(...)`, `validate_source_field_status(...)`.
- **Real (experimental) 1D Poisson solve, separate from the proxy path above**: `experimental_poisson_1d(sources, conductivity, dx)` (`jaxfne/fields/solvers.py`) actually assembles and solves a linear system (`field_solver_status="experimental_pde_solver"`), supporting uniform or layered (per-face array) conductivity. Confirmed convergence ceiling: reliable to roughly N~150 grid points in float32, degrades sharply above that (`convergence_status` self-reports `"failed"`, checked). `experimental_poisson_1d_from_neuron_table(neuron_table, sources, conductivity, n_bins)` bridges it to a real `Model.neuron_table()`/`Signals.sources` — an explicitly opt-in accessor called after `construct()`/`simulate()`, kept fully separate from the `project_laminar_sources` dispatch above. Toward `plans.json:novelty::tfne-differentiable-field-solver`.

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
- Declarative descriptors (`jaxfne/_signals.py`, manifest-safe): `AxisSpec(name, status="active", size=None, units_or_status="declared")` — typed descriptor for one tensor axis; `BasisSpec(space_basis="laminar_depth", time_basis="continuous_ms", field_regime="laminar_proxy", source_mode="proxy_no_field_solve", probe_basis="multimodal_proxy", axes=...)` — computation basis of a run; `DatasetSpec(name="unnamed_dataset", modality="unspecified", source_format="unspecified", comparison_label="p1", comparison_code=101, ...)` — manifest-safe dataset/comparison declaration for observed data.

## 10. Runtime / JAX / x64 / sharding

- `enable_x64()` — call **before** building arrays; verify with `runtime_report()["actual_dtype"]`.
- `runtime_report(runtime_config=None)`, `RuntimeConfig`.
- `compilation_registry` — module-level `CompilationRegistry` instance (`jaxfne/validation.py`) tracking trace-shape compilation guarding (set mode with `set_mode(recompilation_guard)`); read `is_valid_signal`/registry state before trusting compiled-program reuse.
- Sharding: `make_population_mesh()`, `make_candidate_sharding(mesh)`, `make_replicated_sharding(mesh)`, `get_sharding_context()`.
- Lazy optional deps: `require_jaxley()`, `require_optax()` (and `vis.require_matplotlib()`).
- Bridge declarations (`jaxfne/bridges.py`): `BridgeSpec(name, backend, status="schema_only_no_backend_constructed", source_calibration_status="uncalibrated_bridge_output", metadata={})` — JSON-safe optional-backend bridge declaration; `JaxFemFieldBridge(geometry="laminar_column", n_layers=None, source_calibration_status="uncalibrated_jax_fem_bridge", metadata={})` — bridge contract for a future differentiable volumetric field solve (declaration only).
- HH reference traces (tutorial/comparison, not Jaxley-bridge validation):
  `hh_numpy_reference_trace(duration_ms, dt_ms, current_amplitude)` (standalone,
  no optional deps) and `hh_jaxley_reference_trace(duration_ms, dt_ms,
  current_amplitude)` (real Jaxley HH channel, raises via `require_jaxley()`
  if not installed — its `jaxley` import is deferred inside the function body,
  so `import jaxfne` never fails regardless). Both exported at root as of
  2026-07-05 (the jaxley-backed one was previously only reachable via
  `jaxfne.bridges.hh_jaxley_reference_trace` — fixed as an inconsistency, not
  a design choice).

## 11. Selection helpers

- `select_cells(model, area, layers, cell_types, fraction, max_cells, seed)`, `select_neurons(model, area, layer, cell_type)`, `Model.select(area=..., ...)`.
- `SelectorSpec(area=None, area_id=None, layer=None, cell_type=None, ids=None)` — selector over area/layer/cell-type/id fields (`jaxfne/experimental_hpc/contracts.py`).

## Key classes (recognize, don't redefine)

Config/model: `Configuration`, `Config`, `LaminarColumnConfig`, `RuntimeConfig`, `RuntimeConfiguration`, `Simulation`, `Model`, `Net`, `EIGNetwork`, `LaminarPopulation`, `LaminarSourceGeometry`. (`JaxFNEConfig` DELETED 2026-06-30.)
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
