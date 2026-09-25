# `artifacts/memory.md` — jaxfne agent orientation (durable facts only)

<!-- Every backticked repo path and `jaxfne.*` symbol here is resolved by
tests/test_memory_brief.py; keep SHAs, versions, counts and status out. -->

> Read this first. Mutable state lives elsewhere: remaining work in
> `artifacts/todo_stack.md`, stable human-authorized facts in
> `artifacts/fact_stack.md`, procedure in `artifacts/skills/`,
> reusable failure lessons in `MEMORY.md`, rules in `artifacts/AGENTS.md`.
> This file names those files instead of restating them. Import convention:
> `import jaxfne as jtfne` (`README.md`); Atlas scripts alias `J`.

## 1. What jaxfne is

- jaxfne = JAX package for Tensor-Field Neural Equations (TFNE): a **containment
  and composition model** for neural models at different resolutions, not one
  prescribed biophysical equation (`artifacts/AGENTS.md`, `README.md`).
- Biology, dynamics, connectivity, geometry, observations change inside one model
  (`README.md`); workflow is change biology → change dynamics → simulate → measure.
- Scientific grammar: `Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest`.
- Execution grammar: `CircuitSpec -> construct -> Model -> simulate -> Signals`.
  `CircuitSpec` = `Configuration` or `NeuronalTensor` forms.
- Nesting rule: factor at specification time, flatten at execution time; biological
  identity, topology, signs, mechanism identity, geometry, parameter ownership
  survive compilation and optimization.
- Internal quantities stay relative; absolute units arise only via explicit
  calibration transforms at semantic boundaries (doctrine:
  `docs/doctrine/relative_quantity_grammar.md`).
- Numerical kernels are JAX-first; public API is object-oriented (`jaxfne/__init__.py`).
- `NeuronalTensor` is the preferred `construct` input; `Configuration` is the
  supported compatibility path (`docs/quickstart.md`).
- Paradigm, Objective, tuning, visualization, export are optional downstream
  components, not stages of either grammar.
- Evidence vocabulary is SPECIFIED / IMPLEMENTED / TESTED / OBSERVED; failed
  prospective receipts are preserved, never tuned post-hoc.

## 2. Repository map

### Top level

| Path | Purpose |
|---|---|
| `jaxfne/` | The package (see §3) |
| `tests/` | Executable verification, incl. `conftest.py`, `_numeric_gates.py` |
| `scripts/` | Gates, audits, benchmarks, generators, etude runners (see §6) |
| `scripts/harness/` | Harness integrity scripts + `HARNESS_MANIFEST.json` |
| `scripts/release/` | Release tooling (manifest, reconcile) |
| `artifacts/` | Project state, evidence, programme records (see §7) |
| `docs/` | Published documentation source (mkdocs strict) |
| `docs/doctrine/` | Canonical math: TFNE, RBS/RBD/HDP, quantity grammar |
| `docs/guides/` | How-to guides (`configuration_grammar.md`, `hdp.md`, `jdna.md`, `calibration.md`, `atlas_suite.md`, `jaxley_interop.md`, …) |
| `docs/api/` | Per-area API pages (`core.md`, `neuronal_tensor.md`, `fields.md`, `emitters.md`, …) |
| `docs/tutorials/`, `docs/tutorials_v030/` | Executed tutorial notebooks/figures |
| `docs/etudes/`, `docs/protocols/`, `docs/appendix/`, `docs/reference/` | Etudes, protocol specs, supplements |
| `examples/` | Release-smoked scripts (`00_minimal_column.py` … `08_neuronal_tensor_first.py`) |
| `.github/workflows/` | `ci.yml` (Fast), `release_ci.yml`, `notebook_execution.yml`, `publish.yml`, `macroscope-trigger.yml` |
| `scratch/` | Untracked work area; `scratch/CURRENT_TASK.md` carries gate mode |
| `pyproject.toml` | Markers (`slow`, `notebook`, `release`), ruff config, extras (`dev`, `viz`, `jaxley`) |
| `Makefile` | Thin gate aliases (`test-dev`, `test-broad`, `test-release`, `test-rc`, …) |
| `mkdocs.yml` | Docs nav (orphan pages fail the gate even though mkdocs only warns) |
| `MEMORY.md` | Verified reusable failure lessons (root) |
| `CITATION.cff` | Citation metadata |

### `jaxfne/` key files (flat core) and subpackages

| Path | Purpose |
|---|---|
| `jaxfne/__init__.py` | Root re-exports; `__all__` comes from `public_surface.PUBLIC_EXPORTS` |
| `jaxfne/public_surface.py` | Tier contract: CANONICAL / ADVANCED / COMPATIBILITY / EXPERIMENTAL_INTERNAL; `hdp_params` semantic groups |
| `jaxfne/core.py` | Re-export hub for the construct/simulate surface |
| `jaxfne/_construct*.py` | Implementation split: `_construct_core.py` (`simulate`, `compute_fields`, `construct`), `_construct_connectivity.py` (`connect`, `ensemble_member_seed`, `ensemble_edge_ownership`), `_construct_presets.py` (`configuration`, `simulation`, `runtime`), `_construct_population.py`, `_construct_extras.py` (`get_signal`) |
| `jaxfne/_config.py` | `Configuration` fluent builder |
| `jaxfne/_model*.py` | `Model` incl. `_model_simulate.py` (`simulate`), `_model_manifest.py` (`manifest`), `_model_evaluate.py`, `_model_tune.py`, `_model_readout.py` |
| `jaxfne/_signals.py` | `Signals`, `Signal`, `Simulation`, `Objective` |
| `jaxfne/_runtime_config.py` | `RuntimeConfig` execution policy (backend/dtype/jit/vmap/kernels) |
| `jaxfne/_pipeline.py` | `DynamicState`, `checkpoint_state`, `restore_state`, `compile_step_fn`, `scan_network`, `run_continuation` |
| `jaxfne/neuronal_tensor.py` | `NeuronalTensor`, `Area`, `Layer`, `NeuronType`, `InterConnection`, `AreaConnection`, `Geometry3D`, `Pose3D`, `StaticParams`, `PlasticParams`, `RuntimeConfiguration`, canonical tensor loaders |
| `jaxfne/tfne.py` | TFNE specification-language compiler (advanced tier, reached as `jaxfne.tfne`) |
| `jaxfne/jdna/` | `PseudoGenome`, `develop`, canonical pseudogenome loaders, `completion.py` (TFNE→tensor completion) |
| `jaxfne/emitters.py` | `Emitter`, `IzhikevichEmitter`, EIG/edge-list kernels, receptor specs/kinetics |
| `jaxfne/emitters_homeostatic_ei.py` | Homeostatic E/I emitter variants |
| `jaxfne/fields/` | `FieldOutput`, `LinearReadout`, source construction/projection, laminar probes, EEG/MEG/EMM proxies, CSD, cable filter, `population_rate`, `cross_area_coherence` |
| `jaxfne/connectivity.py` | `compile_connection_rules`, JAX variant, weight/mechanism resolution |
| `jaxfne/builders.py` | `build_laminar_column`, `build_multi_area_columns`, `connect_columns`, column presets/tables, `validate_configuration` |
| `jaxfne/presets.py` | `CELL_TYPE_PRESETS`, `RECEPTOR_KINETICS`, `DEFAULT_SPIKE_IMPULSE_GAIN` |
| `jaxfne/hdp_network.py` | HDP kernels, `DEFAULT_HDP`, `plasticity_mask` validation |
| `jaxfne/hdp_rule.py` | Registrable rules: `register_hdp_rule`, `HDPRuleContext/Descriptor/Update` |
| `jaxfne/_hdp_adaptive.py`, `jaxfne/_hdp_registrable_kernel.py` | Adaptive/registrable HDP kernel internals |
| `jaxfne/plasticity.py` | STDP config/state/update (`STDPPlasticityConfig`, `update_stdp_weights_jax`) |
| `jaxfne/w1a_omega_plasticity.py`, `jaxfne/w1b_shadow_plasticity.py` | Protocol-W plasticity lanes |
| `jaxfne/w2_parameter_expression.py`, `jaxfne/w3_stability_analysis.py`, `jaxfne/w3a_stability_analysis.py`, `jaxfne/w3b_parameter_domain.py` | Protocol-W analysis lanes |
| `jaxfne/intervene.py` | Causal intervention objects with manifest roundtrip |
| `jaxfne/optim/` | `agsdr`, `gsdr`, `gsgd`, `random_search`, `optax_adam/sgd`, optimizer specs/states |
| `jaxfne/objectives.py`, `jaxfne/paradigm.py` | Objectives; `Paradigm`, oddball/DMS/evoked paradigms |
| `jaxfne/solvers.py` | `euler_step`, `euler_scan`, `EulerSolver`, `DiffraxSolver`, `SolverConfig`, `solve_ode` |
| `jaxfne/bridges.py` | `JaxleyBridge`, `JaxleyEmitterBridge`, `JaxFemFieldBridge`, trace↔`Signals` converters, `require_jaxley/jax_fem` |
| `jaxfne/stimulus.py` | `triangular_drive` and drive helpers |
| `jaxfne/streaming.py` | `run_stdp_stream` (STDP runs here, not in `simulate`) |
| `jaxfne/geometry.py` | `make_ei_cloud_network` |
| `jaxfne/analysis/`, `jaxfne/analysis/spectral.py` | Metrics; `spectrolaminar_psd_jax`, `bandpower_jax`, similarity kernels |
| `jaxfne/vis/` | View layer only: `atlas_suite.py` (7-panel Atlas), `visualize.py`, `network3d.py`, `fields.py`, plotly/matplotlib panels; lazy-imported (zero graphics overhead on `import jaxfne`) |
| `jaxfne/io.py` | `manifest`, `save_receipt`, `save_json`, `config_hash`, `sha256_*`, `validation_report`, `probe_report`, `asset_hashes` |
| `jaxfne/export.py` | `save_figure(s)`, `export_report`, `export_tutorial_artifacts` |
| `jaxfne/validation.py` | `compilation_registry`, `is_valid_signal` |
| `jaxfne/util.py` | `validate_*`, `*_diff`, `tensor_summary`, `canonical_compact_summary` |
| `jaxfne/units.py` | Dtype-keyed epsilon/dither defaults + `warn_dt_dtype_mismatch` (warn-only) |
| `jaxfne/sharding_utils.py` | Sharding stubs (experimental-internal) |
| `jaxfne/sanity_delta.py`, `jaxfne/sanity_runtime.py` | Hierarchical-oddball sanity API (experimental-internal) |
| `jaxfne/pynwb_compat.py` | NWB compat placeholder (see the todo stack before building on it) |
| `jaxfne/tutorial_utils.py` | Tutorial scaffolds incl. `build_laminar_column` alias root as `build_tutorial_laminar_column` |
| `jaxfne/configs/` | Canonical `NeuronalTensor` JSON genomes (e.g. `canonical-v1-column-1000n`) |
| `jaxfne/experiment_a/`, `jaxfne/experimental_hpc/`, `jaxfne/protocol_c/`, `jaxfne/protocol_d_biological_rbs/`, `jaxfne/protocol_e_integration/`, `jaxfne/publication/` | Protocol/experiment subpackages (advanced) |

### `artifacts/` (project state and evidence)

| Path | Purpose |
|---|---|
| `artifacts/AGENTS.md` | Binding agent rules ( grammars, evidence, work loop, H-series) |
| `artifacts/todo_stack.md` | Remaining work only; done items are removed, not ticked |
| `artifacts/fact_stack.md` | Stable human-authorized facts (challenge, don't edit) |
| `artifacts/programme/` | Per-item receipts (`*_receipt.md`), `atlas_coverage.json`, `atlas_gap_05*.md`, `capability_*.md`, TFNE conformance receipts |
| `artifacts/project_sources/` | Authoritative spec sources (`8_atlas.md` = Atlas source; `6_other_important_notes.md` = long-term plan; `4_tfne_theory_and_neural_tensor.md`) |
| `artifacts/atlas/` | Atlas assays: `at01_at10_toy.py`, `at01_at06_052.py`, `at07_at04_053.py`, `at08_at09_054.py` |
| `artifacts/release/` | Per-version release receipts + `current_release_authorities.json` |
| `artifacts/release_candidate/` | RC working area |
| `artifacts/publication/` | Frozen Figure 1–7 snapshot; `frozen_manifest.json` enumerates immutable files |
| `artifacts/attestations/` | Gitignored RC JUnit + gate attestation (untracked by design) |
| `artifacts/skills/` | Canonical skills: `jaxfne-core`, `jaxfne-science`, `jaxfne-workflow`, `jaxfne-repo`, `jaxfne-audit`, `jaxfne-release`, `jaxfne-seal`, `vocabulary-audit` |
| `artifacts/etudes/`, `artifacts/science/`, `artifacts/figures/` | Etude bundles, scientific notes, generated figures |
| `artifacts/tutorials/`, `artifacts/vocabulary/` | Tutorial evidence, vocabulary audits |
| `artifacts/protocol_c/`, `artifacts/protocol_d_biological_rbs/`, `artifacts/protocol_e_integration/`, `artifacts/protocol_h_rbd/`, `artifacts/protocol_w/` | Protocol evidence homes |
| `artifacts/harness/`, `artifacts/audit/`, `artifacts/issue_log/`, `artifacts/roadmap/` | Harness records, audits, issues, roadmap |
| `artifacts/public_surface_contract_v0413.json`, `artifacts/public_api_before.json` | Surface snapshots (see §8) |

### `scripts/` (named gates and tools)

| Script | Checks / does |
|---|---|
| `scripts/run_test_gate.py` | Single source of truth for gates `dev/broad/release/rc/publication`; owns target lists, marker exprs, check families |
| `scripts/check_environment_parity.py` | RC precondition: optional deps present so collection doesn't silently under-cover |
| `scripts/check_junit_parity.py`, `scripts/compare_pytest_junit.py` | RC-vs-CI per-node JUnit comparison |
| `scripts/audit_public_docs_language.py --check` | No leaked doctrine / obfuscated identifiers in public docs |
| `scripts/audit_notebook_grammar.py --check` | Notebook grammar conformance |
| `scripts/audit_vocabulary.py --check` | Controlled vocabulary conformance |
| `scripts/check_docs_orphans.py` | Every built page reachable from nav |
| `scripts/audit_doc_code_integrity.py --check` | Doc fences resolve against live symbols; allowlist in `scripts/doc_code_integrity_allowlist.json` |
| `scripts/check_atlas_coverage.py --check` | `atlas_coverage.json` seal rule mechanically enforced |
| `scripts/generate_public_surface_contract.py` | Regenerates tracked surface contract from live module |
| `scripts/snapshot_public_api.py` | API snapshot helper |
| `scripts/generate_surface_contract.py` | Dev-local surface contract (gitignored `artifacts/developer/`) |
| `scripts/build_release_manifest.py` | Binds run→commit→tree→SHA256 for shipped bytes |
| `scripts/repo_state_snapshot.py` | Publication-gate repo snapshot |
| `scripts/generate_readme_atlas.py --html-only` | Regenerates canonical Atlas panels |
| `scripts/benchmark_050_baseline.py`, `scripts/benchmark_051_matrix.py`, `scripts/benchmark_054_scaling.py`, `scripts/profile_050_phases.py` | Perf matrix + profiling harness |
| `scripts/harness/sync_skills.py --update --manifest` | Canonical-skills→mirrors sync + manifest update (only sanctioned edit→sync flow) |
| `scripts/harness/check_harness_integrity.py`, `check_agent_refs.py`, `check_memory_hygiene.py` | Harness hash / ref / memory hygiene |
| `scripts/gate_etude_no_resim.py` | Frozen-etude no-resimulation gate |
| `scripts/run_all_tutorials.py`, `scripts/run_tutorial_smoke.py`, `scripts/validate_tutorial_outputs.py` | Tutorial execution + output validation |
| `scripts/snt_pipeline.py`, `scripts/run_experiment_a.py`, `scripts/run_protocol_h_h4_matrix.py`, `scripts/mcc3_10s_scientific_checkpoint.py` | Named experiment/protocol runners |

### `tests/` organization

- Markers (`pyproject.toml`): `slow` (long multi-trial/sweep tests), `notebook`
  (interactive notebook execution), `release` (full T2 release-gate incl. docs/examples).
- Curated `dev` gate modules (`scripts/run_test_gate.py:DEV_PYTEST_TARGETS`):
  `test_api_smoke.py`, `test_root_import_lightweight.py`, `test_signals_get_v0329.py`,
  `test_neuronal_tensor_connectivity.py`, `test_neuronal_tensor.py`,
  `test_connection_rule_compile_v0330.py`, `test_continuation_contract.py`, `test_mcc.py`,
  `test_tfne_algebra.py`, `test_tfne_execution.py`, `test_tfne_parameter_transfer.py`.
- Notebook sweep is scoped to 4 modules (`test_neuronal_tensor_notebook_execution.py`,
  `test_notebook_execution_suite.py`, `test_suite_no1_notebook_execution.py`,
  `test_suite_no4_notebook_execution.py`) for Windows kernel/zmq stability.
- Shared numeric helpers: `tests/_numeric_gates.py` (allclose form: abs leg governs
  near-zero only). Atlas firewall: `tests/test_atlas_firewall.py` (AT scenarios use
  public surface only). Gate hierarchy proof: `tests/test_release_gate_hierarchy.py`.
  Surface pins: `tests/test_public_surface_contract_v0413.py`,
  `tests/test_public_api_snapshot_v034.py`.

## 3. Package map (module → responsibility → main entry points)

Tiers per `jaxfne/public_surface.py`: CANONICAL (normal use), ADVANCED (kernels/bridges/
extension surfaces, reachable but not in `__all__`), COMPATIBILITY (`Net`, `Config`,
`AGSDR`, `*_transform`, `construct_neuronal_tensor`, `load_neuronal_tensor` — retained
with declared replacements), EXPERIMENTAL_INTERNAL (do not build on).

| Module | Responsibility | Main entry points |
|---|---|---|
| `jaxfne` (root, CANONICAL) | CircuitSpec→Model→Signals | `configuration`, `construct`, `simulate`, `simulation`, `runtime`, `connect`, `compute_fields`, `manifest`, `get_signal`, `run_trials`, `trial_batch`, `run_continuation`, `run_receipt`, `provenance_receipt` |
| `jaxfne.neuronal_tensor` | Preferred CircuitSpec form | `NeuronalTensor`, `Area`, `Layer`, `NeuronType`, `InterConnection`, `AreaConnection`, `Geometry3D`, `Pose3D`, `StaticParams`, `PlasticParams`, `RuntimeConfiguration`, `load`, `load_canonical_neuronal_tensor`, `list_canonical_neuronal_tensors`, `save_neuronal_tensor`, `neuronal_tensor_to_configuration`, `merge_neuronal_tensors`, `make_minimal_ei_tensor`, `configs_dir`, `NEURONAL_TENSOR_SCHEMA_VERSION` |
| `jaxfne.tfne` (ADVANCED) | Spec-language compiler | module `tfne` (parse/resolve/realize inside; names stay in-module) |
| `jaxfne.jdna` | Development genome→tensor | `PseudoGenome`, `develop`, `load_pseudogenome`, `load_canonical_pseudogenome`, `list_canonical_pseudogenomes` |
| `jaxfne.emitters` (family types CANONICAL; kernels ADVANCED) | Neural dynamics | `Emitter`, `IzhikevichEmitter`; ADVANCED: `EdgeList`, `EIGNetwork`, `IzhikevichParams`, `ReceptorSpec`, `SynapseSpec/Layer/State`, `make_edge_list_from_dense`, `make_eig_network`, `izhikevich_params_from_labels`, `simulate_edge_recurrent_izhikevich`, `simulate_eig_izhikevich`, `simulate_receptor_exponential_izhikevich`, `standard_receptor_specs`, `standard_receptor_tau_table`, `synaptic_tau_from_mechanism`, `synaptic_current_tensor`, `synaptic_tensor_report` |
| `jaxfne.fields` | Sources→fields→probes | `FieldOutput`, `LinearReadout`, `construct_source_tensor`, `project_laminar_sources`, `project_sources_to_laminar_field`, `probe_laminar_modes`, `csd_tensor`, `cable_filter_sources/tau/report`, `eeg/meg/emm_proxy_transform`, `validate_projection_invariants`, `validate_source_field_status`, `population_rate`, `cross_area_coherence` |
| `jaxfne.connectivity` | Connection compilation | `compile_connection_rules`, `ConnectionCompileResult`; ADVANCED: `compile_connection_rules_jax` |
| `jaxfne.builders` + `jaxfne.presets` | Column/area construction | `build_laminar_column`, `build_multi_area_columns`, `build_tutorial_laminar_column`, `default_cortical_column_config`, `default_complete_configuration`, `laminar_cortex_config`, `connect_columns`, `sparse/all_to_all_intercolumn_connectivity`, `layer_celltype_count_table`, `column_density_table`, `configuration_table`, `validate_configuration`; `CELL_TYPE_PRESETS`, `RECEPTOR_KINETICS`, `DEFAULT_SPIKE_IMPULSE_GAIN`, `CANONICAL_LAYERS_6L`, `CANONICAL_LAYER_CELL_TYPE_FRACTIONS(_5L)`, `CANONICAL_Z_BANDS(_5L)`, `FLAT_CELL_TYPE_FRACTIONS`, `DEFAULT_LAYERS` |
| `jaxfne.hdp_rule` + `jaxfne.hdp_network` (ADVANCED surface) | Plasticity extension | `register_hdp_rule`, `HDPRuleContext/Descriptor/Update`, `DEFAULT_HDP` |
| `jaxfne.plasticity` (ADVANCED) | STDP lane | `STDPPlasticityConfig`, `STDPState`, `summarize_stdp_adaptation`, `update_stdp_weights_jax` |
| `jaxfne.streaming` (ADVANCED) | Streaming plasticity | `run_stdp_stream` |
| `jaxfne.optim` | Tuning optimizers | `agsdr`, `gsdr`, `gsgd`, `random_search`, `optax_adam`, `optax_sgd`, `require_optax`, `OptimizerSpec`, `AGSDROptimizerSpec`, `edge_parameter`, `matrix_parameter`, `EdgeParameterSpec`, `MatrixParameterSpec` |
| `jaxfne.paradigm` | Stimulus paradigms | `Paradigm`, `ParadigmCondition/Event`, `paradigm`, `omission_oddball_paradigm`, `general_sequential_oddball_paradigm`, `general_delayed_match_to_sample_paradigm`, `evoked_l4_drive_paradigm`, `standard_visual_omission`, `coop_omission_oddball_paradigm/for_model/for_neuronal_tensor`, `paradigm_target_indices_from_model`, `rate_targets`, `rate_synchrony_targets`, `objective`, `Objective`, `ObjectiveReport`, `StimulusSchedule`, `stimulus_schedule`, `DatasetSpec`, `dataset_spec` |
| `jaxfne.solvers` (ADVANCED) | Integrators | `euler_step`, `euler_scan`, `SolverConfig`, `EulerSolver`, `DiffraxSolver`, `solve_ode` |
| `jaxfne.bridges` (ADVANCED) | External interop | `JaxleyBridge`, `JaxleyEmitterBridge`, `JaxleyTraceSpec`, `BridgeSpec`, `jaxley_to_signals`, `jaxley_trace_to_signals`, `require_jaxley`, `hh_numpy/jaxley_reference_trace`, `JaxFemFieldBridge`, `require_jax_fem` |
| `jaxfne.analysis.spectral` (ADVANCED) | Spectral kernels | `spectrolaminar_psd_jax`, `bandpower_jax`, `spectrolaminar_readout_kernel_jax`, `spectrolaminar_similarity_kernel_jax/candidates_jax/candidates_seeds_jax` |
| `jaxfne.io` | Evidence I/O | `manifest`, `save_receipt`, `save_json`, `config_hash`, `sha256_file/text`, `asset_hashes`, `validation_report`, `probe_report` |
| `jaxfne.export` | Reports/figures export | `export_report`, `export_tutorial_artifacts`, `save_figure(s)` |
| `jaxfne._pipeline` | State/continuation | `DynamicState`, `dynamic_state_from_model`, `checkpoint_state`, `restore_state`, `compile_step_fn`, `scan_network`, `run_continuation`, `ContinuationState` |
| `jaxfne.validation` (ADVANCED) | Contract checks | `compilation_registry`, `is_valid_signal` |
| `jaxfne.util` | Inspection/diffs | `validate_runtime_config/model/neuronal_tensor`, `runtime_config_diff`, `merge_runtime_configs`, `model_diff`, `configuration_diff`, `tensor_summary`, `canonical_compact_summary`, `format_canonical_text_bundle` |
| `jaxfne.geometry` (ADVANCED) | Spatial networks | `make_ei_cloud_network` |
| `jaxfne.stimulus` (ADVANCED) | Drives | `triangular_drive` |
| `jaxfne.tutorial_utils` | Tutorial scaffolds/metrics | `select_neurons`, `kappa_synchrony`, `LaminarColumnConfig`, `CellTypePreset`, `make_laminar_column_config`, `simulate_laminar_trials`, `spectrolaminar_from_trials/motif_score` |
| `jaxfne.vis` (lazy) | View-only panels | `vis`, `visualize`, `plot_raster`, `plot_spectrolaminar_suite`, `plot_stdp_adaptation_suite`; `jaxfne.vis.atlas_suite` = 7-panel Atlas contract |
| `jaxfne.intervene` | Causal experiments | intervention objects with manifest roundtrip |
| `jaxfne.units` | Numeric defaults | `epsilon`, `dither_scale`, `warn_dt_dtype_mismatch` |
| `jaxfne.sanity_delta` etc. (EXPERIMENTAL_INTERNAL) | Oddball sanity lane | `SanityDeltaConfig/Model`, `HierarchicalOddballParadigm`, `BehaviorGate`, `BackupState`, `TaskEpisode`, `Manifest`, `surrogate_config` |
| `jaxfne.sharding_utils` (EXPERIMENTAL_INTERNAL) | Sharding stubs | `get_sharding_context`, `make_candidate_sharding`, `make_population_mesh`, `make_replicated_sharding` |

`Net = Model`, `Config = Configuration` are compatibility aliases only.

## 4. Core workflow (shortest correct paths)

Preferred tensor path (`docs/quickstart.md`):

```python
import jaxfne as jtfne
jtfne.enable_x64()  # before array construction, for float64
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5)
model   = jtfne.construct(tensor, runtime)
signals = jtfne.simulate(model)  # inherits duration/dt/seed from runtime
spk     = signals.get("spk")
```

`Configuration` path (`docs/quickstart.md`):

```python
cfg = (jtfne.build_laminar_column(n=1000, ei_profile="canonical")
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy",
                 boundary="mean_zero_neumann"))
model   = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

Fields → readouts → manifest (CI wheel-smoke contract, `.github/workflows/ci.yml`):

```python
fields   = jtfne.compute_fields(model, signals)
readouts = model.compute_readout(signals, [jtfne.readout_spec("r", "spike_rate_hz")])
manifest = model.manifest(signals, readouts)  # json-safe; physical_amplitude_calibrated is False
```

HDP plasticity: pass `RuntimeConfig`, not `RuntimeConfiguration`, to `simulate`
(`docs/quickstart.md`): `jtfne.RuntimeConfig(enable_hdp=True, hdp_params={...})`.
STDP is separate: `plasticity()` records manifest intent only; execution is
`run_stdp_stream`, not `simulate` (`docs/quickstart.md`).
Tune: `obj = jtfne.rate_synchrony_targets(); result = model.tune(obj, optimizer="AGSDR", steps=50)`.
Continuation/replay: `checkpoint_state` / `restore_state` / `run_continuation`;
chunked ≡ continuous bit-exact incl. delays in flight.
One simulation → many views: `jaxfne.vis.atlas_suite` consumes the declared data
contract; never simulate inside visualization.

## 5. Key concepts glossary

| Term | Meaning |
|---|---|
| TFNE | Tensor-Field Neural Equations: specification language + containment/composition model; `jaxfne.tfne` compiles it; doctrine `docs/doctrine/tfne_algebra.md`, `tfne_containment_architecture.md`, `tfne_jdna_boundary.md` |
| Configuration | Fluent `CircuitSpec` builder (`configuration().network(n=…).emitter(…).field(…).probe(…)`); compatibility path, still supported |
| NeuronalTensor | Preferred `CircuitSpec`: `Area → Layer → NeuronType` + `InterConnection/AreaConnection`, `Geometry3D/Pose3D`, `StaticParams/PlasticParams`; JSON-serializable, canonical genomes in `jaxfne/configs/` |
| emitter | Neural dynamics operator (e.g. `IzhikevichEmitter`); maps state+input to spikes/traces; receptor kinetics via `standard_receptor_specs()` |
| source | Transmembrane-current tensor `Q` derived from executed activity `X`; `construct_source_tensor` |
| field | Potential field `Φ(r,t)` from projected sources; proxy solve only (`conductivity="proxy"`); never a calibrated instrument output unless validated |
| probe | Observer: contacts/modes/readouts (`Probe`, `ReadoutSpec`, `probe_laminar_modes`); invented contacts refused unless opted in |
| H / RBS | Relative Biophysical State: finite-dimensional dependency-state container, not intrinsically homeostasis, not one scalar driving all operators (`artifacts/fact_stack.md`, `docs/doctrine/rbs_rbd_hdp.md`) |
| RBD | State dynamics involving H (`dH/dt`); fixed-`W` RBD (`Ẇ=0`) is valid |
| HDP | Hidden-state Dependent Plasticity: general plasticity abstraction (`dΘ/dt` per-rule), not one homeostatic mechanism; `DEFAULT_HDP`, `hdp_params` transport, registrable via `register_hdp_rule` |
| W | Weights/efficacy (mutable parameter `Θ`); owned, inspectable, replayable; lazy-`W` + `_SPARSE_DIRECT_N` equivalence fenced by REP-03 |
| plasticity masks | Per-edge gate `plasticity_mask` in `hdp_params` (shape `(n_edges,)`, finite); per-rule/per-projection enable/disable/clamp; stale-mask-vs-`enable` contradiction raises |
| ensembles / connect | `connect()` composes areas (`N_A ⊕_C N_B → (s, h₀, I)`); solo ≡ member-with-zero-cross; `ensemble_member_seed` (RNG domains), `ensemble_edge_ownership` (member + per-rule cross ranges) |
| JDNA / PseudoGenome | Developmental completion: TFNE constrains (possibly underdetermined), JDNA completes under `D + K_D`; `develop(G, seed)`; canonical genomes listed/loaded via `list/load_canonical_pseudogenome` |
| Atlas AT-01…AT-10 | 10 canonical simulations from `artifacts/project_sources/8_atlas.md` (S<n> ≡ AT-0n): AT-01 1 HH neuron (Jaxley anchor); AT-02 E→E pair; AT-03 E↔I oscillator; AT-04 field/state coupling; AT-05 E/I population; AT-06 structured column; AT-07 plastic population; AT-08 2-area adaptation; AT-09 2-area plastic coupling; AT-10 20-area JDNA system |
| measurement vector Y | Common 13-key vector `{X,H,W,Q,Phi_E,Phi_B,SPK,PSD,C,phi,E_reduction,T_compute,M_compute}` plus 10 area/cross cells; frozen schema and extractors in `artifacts/atlas/y_schema.py` (IMPLEMENTED only when the run wrote the value); requirement AT-00-R1 in `artifacts/programme/atlas_coverage.json` |
| OMITTED / REFUSED | Absent quantity = `OMITTED`; refused capability = `REFUSED`; never synthesized — explicit sentinels + omission cards, e.g. `jaxfne/vis/atlas_suite.py` renders omission instead of substitutes |
| calibration levels | Epistemic ladder CALIBRATED ≠ REDUCED_PHYSICAL ≠ RELATIVE_PROXY; outputs labelled, sealed calibration + refusal gate; live fields carry `claim_level="computational_scaffold"` / `"proxy_readout"` with `physical_amplitude_calibrated=False` |

## 6. Verification

- Gates (single source of truth `scripts/run_test_gate.py`, aliases in `Makefile`):
  `dev` (curated arch gate), `broad` (`-m "not slow"`), `release` (broad + slow +
  notebook + strict docs + examples), `rc` (release + parity/build/twine/isolated
  smoke + attestation), `publication` (repo snapshot + frozen experiments).
- Check families per gate: dev = compileall/pytest_dev/docs_language/vocabulary;
  broad adds lint_ruff/notebook_grammar/docs_orphans/doc_code_integrity/pytest_broad;
  release adds docs_build_strict/pytest_slow/pytest_notebook/examples_smoke;
  rc adds environment_parity/junit_parity/package_build/twine_check/isolated_wheel_smoke.
- Marker algebra is exhaustive: broad `not slow`, slow `slow and not notebook`,
  notebook `notebook`; release CI runs unfiltered `pytest tests` (proven by
  `tests/test_release_gate_hierarchy.py`).
- CI Fast (`.github/workflows/ci.yml`, ubuntu, py 3.11+3.14): compileall → ruff →
  docs-language audit → atlas firewall → atlas coverage → notebook grammar →
  docs strict build → orphan check → doc↔code integrity → `run_test_gate.py dev`
  (dev branch) or `broad` (main) → examples loop → wheel build/twine/isolated smoke.
- Release CI (`.github/workflows/release_ci.yml`, main/schedule/dispatch): unfiltered
  pytest with JUnit artifacts → examples → single build → `RELEASE_MANIFEST.json` →
  twine → isolated smoke. RC gate must subsume it (`PRE_RELEASE_GATE >= RELEASE_CI_GATE`).
- Other workflows: `notebook_execution.yml`, `publish.yml`, `macroscope-trigger.yml`.
- Docs policy: `docs/ci_policy.md`. Attestations/JUnit land gitignored in
  `artifacts/attestations/` (a tracked attestation would change the attested SHA).
- Exact commands (from `Makefile`, `.github/workflows/ci.yml`, `docs/quickstart.md`):

```bash
make test-dev            # or: python scripts/run_test_gate.py dev
make test-broad          # or: python scripts/run_test_gate.py broad
make test-release        # or: python scripts/run_test_gate.py release
make test-rc             # run on the candidate SHA itself
python3 -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests -q -m "not slow"
python3 -m ruff check jaxfne/
python3 -m mkdocs build --strict
```

## 7. Project state and control files

| File | Lives there | Who may edit |
|---|---|---|
| `artifacts/todo_stack.md` | Remaining work only (0.5.x ENGINE∥ATLAS stacks, decisions, acceptance); evidence stays in git/tests/receipts | Agents: small in-place amend/reorder/annotate; never wholesale rewrite; re-read before editing (concurrent human + agent editors) |
| `artifacts/fact_stack.md` | Stable authorized facts (HDP, RBS/RBD, continuation ownership, configured≠realized≠executed≠effective, relative≠calibrated, proxy≠measurement) | Human authorization only; agents read/use/test/challenge, flag contradictions |
| `artifacts/programme/` | Item receipts, `atlas_coverage.json` (requirement rows `AT-0n-R<k>`, states PLANNED/SUPPORTED/VALIDATED/CANONICAL/OUT_OF_SCOPE, `seal_rule`, `sealed_releases`), `atlas_gap_05*.md`, `capability_*.md` | Task owners append receipts; coverage checked mechanically by `scripts/check_atlas_coverage.py` (a PLANNED row in a sealed release fails) |
| `artifacts/release/` | Per-version receipts + `current_release_authorities.json` (version-specific release authorities) | Release tasks only |
| `artifacts/atlas/` | Executable Atlas assays (`at01_at10_toy.py`, `at01_at06_052.py`, `at07_at04_053.py`, `at08_at09_054.py`) and the frozen Y schema `y_schema.py`; firewall-pinned to public surface | Atlas lane tasks |
| `artifacts/skills/` | Canonical skill sources (8 skills) | Edit canonical → `scripts/harness/sync_skills.py --update --manifest` → verify; never hand-edit mirrors |
| `scripts/harness/HARNESS_MANIFEST.json` | Hashes of kernel/router/skills/gates/schemas (harness v2.1) | Harness sync flow only |
| `artifacts/publication/frozen_manifest.json` | Enumerates immutable Figure 1–7 artifacts | Immutable |
| `scratch/CURRENT_TASK.md` | Active `mode:` (READ/CODE/SCIENCE/RELEASE/PUBLICATION) for Gate 0 + compact `C_*` identity lines | Runtime-generated, gitignored; referenced via allowlist, never hard-required |

## 8. Conventions and pitfalls

- Dtypes: `float32` default; `float64` honored only with JAX x64 enabled
  (`enable_x64()` before array construction); `bfloat16` allowed
  (`jaxfne/_runtime_config.py:_ALLOWED_DTYPES`); `jaxfne.units` gives
  dtype-keyed epsilon/dither; `warn_dt_dtype_mismatch` warns only, never adjusts.
- Units: time in ms (`duration_ms`, `dt_ms`, `tau_*_ms`); doc atlases use binary-exact
  `dt_ms=0.5` (0.1 drifts the float32 time grid → `INVALID_TIME_GRID`); delay is
  declared in ms, realized as `delay_steps = round(delay_ms / dt_ms)`, positive
  delay rounding to 0 steps is refused; geometry is relative fractions in [0,1],
  outside refused (see `MEMORY.md` one-liners below).
- Seeds/RNG: explicit `seed` on `RuntimeConfiguration`/`Simulation`;
  `ensemble_member_seed` derives per-member streams (solo ≡ member); replay is
  bit-identical with declared RNG domains; chunked ≡ continuous for all mutable
  state incl. delays in flight; Model-level stochastic runs use the chain-consistent
  noise stream.
- Configured ≠ realized ≠ executed ≠ effective: inspect each stage separately;
  equality of counts/names/shapes is too weak; perturbations must be asymmetric.
  Relative ≠ calibrated; proxy ≠ physical measurement.
- Frozen/immutable: `frozen_manifest.json` files; frozen etude bundles
  (no-resim gate `scripts/gate_etude_no_resim.py`); write-once-per-figure Atlas outputs.
- Public surface: tiers + `hdp_params` groups + deprecation map owned by
  `jaxfne/public_surface.py`; `__all__` = canonical + compatibility; pins are
  `artifacts/public_surface_contract_v0413.json`, `tests/test_public_surface_contract_v0413.py`,
  `tests/test_public_api_snapshot_v034.py`, regenerated via
  `scripts/generate_public_surface_contract.py`; check CI on the pushed SHA after
  any export change.
- Docs: prose + fences are checked (`audit_public_docs_language.py`,
  `audit_notebook_grammar.py`, `audit_vocabulary.py`, `check_docs_orphans.py`,
  `audit_doc_code_integrity.py` + `scripts/doc_code_integrity_allowlist.json`);
  rendering green ≠ truth. Public docs stay compact positive math; engineering
  history and agent governance stay out.
- Recurring failure classes (one line each; detail in `MEMORY.md`):
  gitignored skill refs break fresh-clone gates → explicit generated-ref allowlist;
  abs+rel conjunctive freeze unsatisfiable → allclose form, abs near-zero only;
  shared helpers upcasting float32→float64 → preserve input dtype, normalized diff;
  whole-file reflow from edits → diff-stat review, P-ID unattributed churn;
  PowerShell quoting/alias breaks → single-purpose calls, file-ified quoting;
  `dt_ms=0.1` long-run grid drift → binary-exact 0.5 for atlases;
  Plotly UUID byte diffs → manifest-sha + linkcheck, not byte diff;
  frozen-bundle float-hash drift → two-tier check (spike/geometry exact, floats in tolerance);
  local-green/CI-red env split → verify CI on exact pushed SHA, never release from red;
  docs-build-green-but-lying → doc↔code integrity gate;
  subagent overlap/interruption → disjoint file sets, `git status` + gates before trust;
  text-mode CRLF whole-file diffs → binary-safe patches for existing `.py`;
  stash ownership → never bare-pop, pre-existing stashes read-only;
  per-step local-only verification → read each push's CI result before next step.

## 9. Where to look first for common tasks

| Task | Read first |
|---|---|
| Build + run a model | `docs/quickstart.md`, `docs/guides/configuration_grammar.md`, `docs/api/core.md`, `docs/api/neuronal_tensor.md` |
| NeuronalTensor circuits, canonical genomes | `jaxfne/neuronal_tensor.py`, `jaxfne/configs/`, `docs/api/neuronal_tensor.md`, `docs/guides/jdna.md` |
| TFNE program → execution | `jaxfne/tfne.py`, `docs/doctrine/tfne_algebra.md`, `docs/doctrine/tfne_jdna_boundary.md`, `tests/test_tfne_execution.py`, `tests/test_tfne_parameter_transfer.py` |
| JDNA development / genome | `jaxfne/jdna/`, `docs/api/jdna.md`, `artifacts/project_sources/8_atlas.md` |
| Plasticity (HDP/RBD/HDP rules) | `docs/guides/hdp.md`, `docs/doctrine/rbs_rbd_hdp.md`, `jaxfne/hdp_rule.py`, `jaxfne/hdp_network.py`, `jaxfne/_hdp_registrable_kernel.py` |
| STDP lane | `jaxfne/plasticity.py`, `jaxfne/streaming.py` (`run_stdp_stream`) |
| Sources → fields → probes, calibration | `docs/guides/calibration.md`, `docs/guides/probe_operators.md`, `jaxfne/fields/`, `docs/api/fields.md`, `docs/api/field_schema.md`, `docs/api/source_schema.md` |
| Multi-area composition | `jaxfne/_construct_connectivity.py` (`connect`), `jaxfne/builders.py` (`connect_columns`, `build_multi_area_columns`), 0.5.4 receipt `artifacts/programme/composition_054.md` |
| Tune toward targets | `docs/guides/objective_grammar.md`, `jaxfne/optim/`, `jaxfne/objectives.py`, `docs/api/objectives.md` |
| Continuation / replay / interventions | `jaxfne/_pipeline.py`, `jaxfne/intervene.py`, `artifacts/programme/continuation_053.md`, `intervention_053.md`, `replay_domains_053.md` |
| Visualize (never simulate in vis) | `docs/guides/atlas_suite.md`, `docs/guides/plotly_visualization.md`, `jaxfne/vis/atlas_suite.py`, `docs/api/vis.md` |
| Jaxley / JAX-FEM bridges | `docs/guides/jaxley_interop.md`, `docs/guides/jax_fem_interop.md`, `jaxfne/bridges.py`, `docs/api/bridges.md` |
| Solvers / dt acceptance | `docs/guides/solver_acceptance.md`, `jaxfne/solvers.py`, `docs/api/solvers.md` |
| Atlas simulation or gap row | `artifacts/project_sources/8_atlas.md`, `artifacts/programme/atlas_coverage.json`, `artifacts/atlas/at*.py`, `artifacts/programme/atlas_gap_05*.md` |
| Add/verify a public symbol | `jaxfne/public_surface.py`, `jaxfne/__init__.py`, `scripts/generate_public_surface_contract.py`, surface snapshot tests (§2) |
| Change docs | The 5 doc audits in §6 + `docs/ci_policy.md`; `scripts/doc_code_integrity_allowlist.json` for generated refs |
| Release work | `artifacts/skills/jaxfne-release/SKILL.md`, `artifacts/release/current_release_authorities.json`, `scripts/release/`, `scripts/build_release_manifest.py` |
| Numeric mismatch / dtype / determinism | `tests/_numeric_gates.py`, `jaxfne/units.py`, `jaxfne/util.py` (diffs/summaries), `MEMORY.md` |
| Perf question | `artifacts/perf/` baselines, `scripts/benchmark_051_matrix.py`, `scripts/profile_050_phases.py` |
| Inspect a running model | `docs/guides/model_inspection.md`, `jaxfne/util.py`, `jaxfne/validation.py` |
| Claim → evidence for manuscript | `scripts/build_manuscript.py`, `scripts/publication_figures/`, claim-ledger pattern in `artifacts/publication/` |
