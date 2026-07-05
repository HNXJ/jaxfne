<!-- auto-generated from progress.json by scripts/prp_to_markdown.py — do not hand-edit -->

| path | score | status | tbi | tbd | last_verified |
|---|---|---|---|---|---|
| scripts/evidence_figures/ |  | open |  |  | 2026-07-01 |
| outputs/ff_fb_hypothesis_bundle/run_analysis.py | 45 | deferred | All 4 scripts it dispatches to (test_ff_fb_hypothesis_proper.py, spectrolaminar_3panel_suite.py, laminar_raster_6panel_1 |  | 2026-07-01 |
| docs/tutorials/05_v1_pfc_dual_column.md | 45 | needs_followup |  | SCOPE REPLACED 2026-07-03 (user directive): the old vague inter_areal_connectivity/dual_laminar_column/traveling_waves/c | 2026-07-03 |
| scripts/evidence_figures/ed05_manifest_hashes.py | 68 | reviewed | METADATA_ARTIFACTS checklist entry path is stale/wrong -- will cause a spurious RuntimeError, confirmed by static trace | fix METADATA_ARTIFACTS path to use evidence_checklist_path() | 2026-07-04 |
| scripts/hdp_v2_rho_sweep.py | 70 | done | Will need re-running (full duration, multi-seed) against any new candidate formula from plans.json item hdp-stability-fo |  | 2026-07-01 |
| jaxfne/configs/default_macaque_V1.json | 70 | open | Per-layer density fractions are inherited from the existing canonical-v1-column-1000n.json template (cross-checked quali |  | 2026-06-30 |
| scripts/evidence_figures/fig07_reproducibility_artifacts.py | 72 | reviewed | checklist_exists field in the manifest/figure will always be False due to hardcoded stale path, contradicting the succes | replace the hardcoded path with evidence_checklist_path() (already imported) | 2026-07-04 |
| jaxfne/tutorial_utils.py | 75 | reviewed | build_laminar_column's inline comment says 'placeholder: sparse random' for W_local_exc/W_local_inh/W_ff/W_fb, but the a | delete the unused W_parts/build_laminar_connections dead-code path, or wire it in, or fix the comment | 2026-07-04 |
| tests/test_sanity_delta_report_schema_full.py | 75 | reviewed | test name/docstring ('strict JSON compliant') overclaims what the assertions check (mere key presence + json.loads-abili | either rename to reflect it only checks key presence, or add an assertion on checks['strict_json'] tied to a real outcom | 2026-07-04 |
| scripts/hdp_1000_neuronal_tensor_column.py | 78 | reviewed | apply_drive_correction() is a real, working, LOCAL-ONLY duplicate of jaxfne.hdp_network.apply_drive_correction -- but NO |  | 2026-07-01 |
| scripts/repair_notebooks.py | 78 | reviewed |  |  | 2026-07-04 |
| scripts/spectrolaminar_tfne_izhikevich_pipeline.py | 78 | reviewed | build_model(cfg: dict) is a real, working, LOCAL-ONLY duplicate of jaxfne.hdp_network.build_model -- but NOT a safe cand |  | 2026-07-01 |
| tests/test_interactive_tutorial_artifacts_v0221.py | 78 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_plasticity_report_full.py | 78 | reviewed |  | add a companion test calling export() with plasticity left disabled that asserts plasticity_report.json's weights are NO | 2026-07-04 |
| tests/test_tutorial_figure_manifest_v028.py | 78 | reviewed | test_jaxfne_version_current hardcodes expected '0.3.4' against a frozen manifest while pyproject is 0.4.5 -- not current | either drop the hardcoded version literal in favor of asserting the manifest's version is internally self-consistent, or | 2026-07-04 |
| benchmarks/scaling_benchmark.py | 80 | open |  |  | 2026-06-30 |
| jaxfne/optim/base.py | 80 | open |  |  | 2026-06-30 |
| tests/test_canonical_biophysics.py | 80 | reviewed | test_pv_e_strengthened_canonical_only constructs the flat-baseline model mf but never uses it in any assertion -- it onl | either use mf in a real comparison (pv_e_meanabs(mc)>pv_e_meanabs(mf)) or remove the unused var and soften the comment | 2026-07-04 |
| jaxfne/sanity_delta.py | 82 | open | export()'s equivalence-check block hardcodes start_ms=2100.0 and segment 'd4' rather than deriving from config -- works  |  | 2026-07-05 |
| scripts/find_5hz_vmap.py | 82 | reviewed |  |  | 2026-07-04 |
| scripts/run_neuron_sweeps.py | 82 | reviewed |  |  | 2026-07-04 |
| tests/test_docs_equations_plotly_v0214.py | 82 | reviewed | test_no_like_terminology_in_new_docs lists 'docs/skills/skill_visual_outputs.md' as a new_docs path to check, but that f | update the new_docs list to point at the current .legacy/internal_docs/skills/ location (or drop the stale entry) so the | 2026-07-04 |
| docs/CORTEX_CALIBRATION_CHECKLIST.md | 82 | done |  |  | 2026-07-03 |
| docs/STDP_HOMEOSTATIC_REPORT.md | 82 | done | The all-neuron-plasticity result depends on a script-level code path (cortex_100_homeostatic_stdp.py, not the package's  |  | 2026-07-03 |
| docs/faq.md | 82 | done |  |  | 2026-07-03 |
| docs/releases/v0.3.4.md | 82 | done |  | Code sample (cfg.runtime/.column/.cell_types/.connectivity/.set_emitter/.probes) uses a DIFFERENT chaining vocabulary th | 2026-07-03 |
| docs/tutorials/08_jaxfne_suite_no_3_low_frequency_scaling.md | 82 | done |  |  | 2026-07-03 |
| docs/BASELINE_DRIVE_REFERENCE.md | 83 | done |  |  | 2026-07-03 |
| docs/STDP_CLOSED_LOOP_REPORT.md | 84 | done |  |  | 2026-07-03 |
| docs/STDP_GLOBAL_SCALE_REPORT.md | 84 | done |  |  | 2026-07-03 |
| docs/STDP_LOWRATE_REGIME_REPORT.md | 84 | done |  |  | 2026-07-03 |
| docs/STDP_REAL_TEST_REPORT.md | 84 | done |  |  | 2026-07-03 |
| jaxfne/experimental_hpc/__init__.py | 85 | open |  |  | 2026-06-30 |
| jaxfne/streaming.py | 85 | done |  |  | 2026-07-01 |
| jaxfne/vis/plotly/network.py | 85 | reviewed |  | No dedicated test file (grep -rln 'plotly.network|plot_network_3d' tests/ found test_vis_network3d_public_api.py and tes | 2026-06-30 |
| jaxfne/vis/tutorial_panels.py | 85 | reviewed | visualize_laminar_column_3d's return type annotation (line 54) and activity_trace_suite's/spectrolaminar_suite_3panel's  |  | 2026-06-30 |
| scripts/characterize_neuron_io_curves.py | 85 | reviewed |  |  | 2026-07-04 |
| scripts/verify_5hz_traces.py | 85 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_backup_resume.py | 85 | reviewed |  | no file in the sanity_delta suite asserts the strict_json check's *value* is meaningful rather than just present | 2026-07-04 |
| tests/test_v0320_recompilation_guards.py | 85 | reviewed | the underlying mechanism still conflates 'a new Python wrapper handed to jax.jit' with 'an actual XLA recompilation' (wo | if the registry's Python-call-counting design changes to be JIT-hook-based, these tests need re-verification since they  | 2026-07-04 |
| scripts/macaque_v1_n_parametrized_smoke.py | 85 | open |  | Only validated at N=10 per locked user decision (N=10-only scope) -- not yet checked at a larger N (e.g. 1000) to confir | 2026-06-30 |
| jaxfne/_pipeline.py | 85 | done | configuration_to_tensor -- no Configuration->NeuronalTensor converter exists anywhere in the codebase (only the reverse, | Whether compile_step_fn/scan_network ever expand beyond HDP edge-list (homeostasis and dense backends are explicitly out | 2026-07-03 |
| docs/HDP_REPORT.md | 85 | done | The doc's phrase 'Generic builder (jaxfne.hdp_network)' names jaxfne.hdp_network as if it were a callable builder; it is |  | 2026-07-03 |
| docs/api/plasticity.md | 85 | done |  |  | 2026-07-03 |
| docs/colab.md | 85 | done |  | manifest['basis'] key names in the printed example ('jaxfne_version', 'model_status', 'amplitude_status', 'field_solver_ | 2026-07-03 |
| docs/conservation_proxy_diagnostics.md | 85 | done |  |  | 2026-07-03 |
| docs/tutorial_figures.md | 85 | done |  |  | 2026-07-03 |
| docs/tutorials/04_v1_column.md | 85 | done |  |  | 2026-07-03 |
| docs/tutorials/09_v0310_eeg_meg_emm_proxy_bundle.md | 85 | done |  |  | 2026-07-03 |
| scripts/v1_pfc_continuous_aaab_smoke_test.py | 85 | reviewed | Long-term (trial-to-trial) adaptation is demonstrated only with H-only carryover (carry_weights=False). Carrying synapti | A weight-homeostat / synaptic-normalization term would be needed to make weight-carryover stable across trials; not in s | 2026-07-04 |
| scripts/evidence_figures/ed10_release_archive_receipt.py | 87 | reviewed |  |  | 2026-07-04 |
| scripts/run_delta_notebook_01.py | 87 | reviewed |  |  | 2026-07-04 |
| docs/tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md | 87 | done |  |  | 2026-07-03 |
| jaxfne/bridges.py | 88 | reviewed | hh_jaxley_reference_trace defined but not exported anywhere in __init__.py, only reachable via the submodule path |  | 2026-07-04 |
| jaxfne/sanity_runtime.py | 88 | done |  |  | 2026-07-05 |
| scripts/build_v037_source_column_3d.py | 88 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/ed03_notebook_execution_receipts.py | 88 | reviewed |  |  | 2026-07-04 |
| scripts/generate_tutorial_figures.py | 88 | reviewed |  |  | 2026-07-04 |
| scripts/run_all_tutorials.py | 88 | reviewed |  |  | 2026-07-04 |
| scripts/validate_tutorial_outputs.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_agent_api_catalog.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_backend_parity_v020.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_multi_area_emitter_runtime.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_multi_area_source_projector.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_performance_reports_v030.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_public_docs_hygiene.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_hierarchical_oddball_config.py | 88 | reviewed |  | no file asserts on the actual value of the strict_json check result | 2026-07-04 |
| tests/test_tutorial_smoke_runner_v0217.py | 88 | reviewed |  | rename test_version_remains_0_2_10 to something version-neutral to stop the name drifting further behind reality | 2026-07-04 |
| tests/test_v0331_jaxley_lazy.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_v033_two_neuron_ei.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_v0342_evidence_inventory_paths.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_vis_deduplication.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_vis_proxy_safe_titles.py | 88 | reviewed |  |  | 2026-07-04 |
| tests/test_vis_suite.py | 88 | reviewed |  |  | 2026-07-04 |
| docs/changelog.md | 88 | done |  |  | 2026-07-03 |
| docs/guides/jaxley_interop.md | 88 | done |  | examples/03_jaxley_bridge_smoke.py also exists but is not referenced from this page -- could be added alongside the refe | 2026-07-03 |
| docs/interactive_visualizations.md | 88 | done |  |  | 2026-07-03 |
| docs/releases/v0.2.10.md | 88 | done |  |  | 2026-07-03 |
| docs/tutorials/02_two_neuron_ei.md | 88 | done |  |  | 2026-07-03 |
| docs/tutorials/06_jaxfne_suite_no_1_computational_biophysics.md | 88 | done |  |  | 2026-07-03 |
| docs/tutorials/06_v036_100_neuron_ei_population.md | 88 | done |  |  | 2026-07-03 |
| docs/tutorials/11_multi_laminar_cortical_agsdr.md | 88 | done |  |  | 2026-07-03 |
| docs/v047_refactor_audit.md | 88 | done |  | Doc is an explicitly dated snapshot (header states 'Generated: 2026-06-28', SHA a16b0ea) but is read by users/agents as  | 2026-07-03 |
| local/gen_test_data_100n_1000ms.py | 88 | done |  |  | 2026-07-01 |
| jaxfne/optim/core.py | 88 | done |  |  | 2026-07-01 |
| scripts/evidence_figures/ed08_tutorial_atlas_coverage.py | 89 | reviewed |  |  | 2026-07-04 |
| scripts/run_agsdr_gain_optimization.py | 89 | reviewed |  |  | 2026-07-04 |
| scripts/run_tutorial_smoke.py | 89 | reviewed |  |  | 2026-07-04 |
| tests/test_multi_area_spectrolaminar_objective.py | 89 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_optional_imports.py | 89 | reviewed | test_jaxfne_imports_without_notebook_deps has a genuinely vacuous assertion (the for-loop's if-branch always just passes |  | 2026-07-04 |
| tests/test_pipeline_pure_functions.py | 90 | open |  |  | 2026-06-30 |
| examples/02_spectrolaminar_oddball_scaffold.py | 90 | reviewed |  |  | 2026-07-04 |
| examples/03_jaxley_bridge_smoke.py | 90 | reviewed |  |  | 2026-07-04 |
| jaxfne/__init__.py | 90 | reviewed | hh_jaxley_reference_trace (bridges.py) not exported at root, only its numpy fallback is -- may be intentional, worth a d |  | 2026-07-04 |
| jaxfne/connectivity.py | 90 | done |  |  | 2026-07-05 |
| jaxfne/fields/solvers.py | 90 | reviewed |  |  | 2026-07-04 |
| jaxfne/plasticity.py | 90 | reviewed |  |  | 2026-07-04 |
| jaxfne/validation.py | 90 | done |  |  | 2026-07-05 |
| scripts/audit_notebooks_and_assets.py | 90 | reviewed |  |  | 2026-07-04 |
| scripts/benchmark_scan_backends.py | 90 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/ed02_json_schema_validation.py | 90 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/ed06_benchmark_scaling_tables.py | 90 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig05_runtime_scaling.py | 90 | reviewed |  |  | 2026-07-04 |
| scripts/make_delta_test_01_report.py | 90 | reviewed |  |  | 2026-07-04 |
| scripts/tutorial_plotly_utils.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/_notebook_exec_helpers.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_api_smoke.py | 90 | reviewed |  | stale unused constant _MANIFEST_SCHEMA_VERSION='manifest.v0.0.21' exists in _model.py alongside the actually-used '0.0.4 | 2026-07-04 |
| tests/test_artifact_json_safety_v0330.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_compact_facade_v034.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_config_circuit_ownership_v0328_completion.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_config_json_roundtrip.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_connect_ensemble.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_connect_golden_snapshot.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_connection_rule_compile_v0330.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_connection_rule_mechanisms_v0330.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_connectivity_scaling.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_construct_golden_snapshot.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_coop_v035.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_delta_notebook_01_receipt.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_delta_notebook_01_static.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_docs_links_v0330.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_ed10_release_archive.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_ed9_evidence.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_ed9_hdp_evidence.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_etude1_notebook_thinness.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_network_100_ei_colab_v0210.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_notebook_status_doc_consistency.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_notebook_structure_v0330.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_objectives_v020.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_optim_tune.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_probe_report_contract_v0212.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_readout_spec_v017.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_full_runtime_mode.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_plasticity.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_proxy_readout_names.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_semantic_correctness_v020.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_single_neuron_colab_v028.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_single_neuron_multimodal_v023.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_spectrolaminar_public_path_v020.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no1_public_grammar_validation.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_tfne_global_assumptions.py | 90 | reviewed |  | consider having test_locality_contract/test_linear_projection_superposition_and_scaling call the real emitters/fields fu | 2026-07-04 |
| tests/test_tfne_izhikevich_3d.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_trial_runner_v014.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_tune_parameter_propagation.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_tutorial_figure_contract_v0219.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_two_neuron_ei_colab_v029.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v006_v008.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0313_omission_oddball_tutorial.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0314_ablation_controls.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0317_dtype_invariants.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0318_sharding_stubs.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0321_migration_boundaries.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0331_output_bundles.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0331_pynwb_placeholder.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v033_api_extensions.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v0341_kernels.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v036_100_neuron_ei_population_tutorial.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v037_interactive_column_docs.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_v038_lfp_csd_readout_tutorial.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_vectorized_equivalence.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_vis_network3d_public_api.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_vis_psd_spectrogram.py | 90 | reviewed |  |  | 2026-07-04 |
| tests/test_epv_2neuron_pipeline_smoke.py | 90 | done |  | 4 more configs from the matrix still not built: jaxley-HH parity (same connectivity, different intrinsic params), Izhike | 2026-07-01 |
| jaxfne/core.py::Configuration.uniform3d | 90 | done |  |  | 2026-07-01 |
| tests/test_ei_jaxley_izhikevich_parity.py | 90 | done |  | Config #2b (jaxley+HH via simulate_laminar_field, 5-contact LFP/CSD) not yet built -- staged in plans.json smart-test-ma | 2026-07-01 |
| jaxfne/core.py::Configuration.population | 90 | done |  | Confirmed via reproduction 2026-07-01: L4's hardcoded _SUITE2_LAYER_CELL_TYPES_V1 default (E:0.25,PV:0.45,SST:0.15,VIP:0 | 2026-07-01 |
| jaxfne/core.py::RuntimeConfig.dtype + Model.with_hdp_initial_state | 90 | done |  |  | 2026-07-01 |
| jaxfne/units.py | 90 | done | Scope deliberately limited to plans.json bf16-quantized-tfne-izhikevich-mode items 5+7 (epsilon/dither_scale defaults) - |  | 2026-07-01 |
| tests/test_vis_smoke_all.py | 90 | done |  | Smoke-only by design (no exception + non-empty/meaningful return) -- does not catch value/pixel regressions. A future pa | 2026-07-01 |
| docs/NEURON_IO_CHARACTERIZATION.md | 90 | done |  |  | 2026-07-03 |
| docs/ci_policy.md | 90 | done |  |  | 2026-07-03 |
| docs/contributing.md | 90 | done |  |  | 2026-07-03 |
| docs/guides/calibration.md | 90 | done |  |  | 2026-07-03 |
| docs/guides/plotly_visualization.md | 90 | done |  | Page footer states 'Status: v0.2.14 / Last updated: 2026-05-20' and requirements.txt example pins 'jaxfne>=0.2.14' -- cu | 2026-07-03 |
| docs/releases/v0.2.18.md | 90 | done |  |  | 2026-07-03 |
| docs/tensor_network_ancestry.md | 90 | done | BasisSpec usage example (Part 5, lines 158-163: `BasisSpec(name=..., units=..., n_dims=...)`) not verified against the r |  | 2026-07-03 |
| docs/tutorials/01_single_neuron_multimodal.md | 90 | done | readout_spec('mean_voltage', 'voltage_mean') at line 68 uses metric name 'voltage_mean', which is NOT in the verified _K |  | 2026-07-03 |
| docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md | 90 | done |  |  | 2026-07-03 |
| docs/tutorials/08_v038_lfp_csd_readout.md | 90 | done |  |  | 2026-07-03 |
| docs/tutorials/12_izhikevich_single_emitter_explorer.md | 90 | done |  |  | 2026-07-03 |
| scripts/hdp_suite2_visualizations.py | 90 | done |  |  | 2026-07-01 |
| examples/02_omission_scaffold.py | 90 | done |  |  | 2026-07-01 |
| examples/03_objective_and_tune_smoke.py | 90 | done |  |  | 2026-07-01 |
| tests/test_jaxley_emitter_bridge_e2e.py | 90 | done |  |  | 2026-07-01 |
| jaxfne/fields/proxy.py | 90 | done |  |  | 2026-07-01 |
| tutorials/etudes/jaxfne_etude_no_10_global_local_oddball.ipynb | 90 | done | The progress.json TBI note this replaces suggested wiring to HierarchicalOddballParadigm (jaxfne/sanity_delta.py) -- che |  | 2026-07-02 |
| tutorials/etudes/jaxfne_etude_no_11_omission_local.ipynb | 90 | done |  |  | 2026-07-02 |
| tutorials/etudes/jaxfne_etude_no_12_omission_global_coop.ipynb | 90 | done |  |  | 2026-07-02 |
| scripts/evidence_figures/ed04_optional_dependency_laziness.py | 91 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/ed07_probe_operator_contracts.py | 91 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_bifurcation_trace.py | 91 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_small_scale_balance_sweep.py | 91 | reviewed |  |  | 2026-07-04 |
| scripts/spectrolaminar_drive_sweep_l23_l4_l56.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_evoked_l4_drive.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_fields_projection_finite_and_normalization.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_global_superposition.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_multi_area_config.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_multi_area_spectrolaminar_readout.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_optim_report_hardening.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_runtime_dtype_v020.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no1_public_grammar.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no3_low_frequency_scaling_tutorial.py | 91 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no4_notebook_execution.py | 91 | reviewed |  |  | 2026-07-04 |
| docs/guides/showcases.md | 91 | done |  | Side-finding documented inline in the page itself (VIP neurons appearing in neuron_table() despite being omitted from a  | 2026-07-03 |
| jaxfne/experimental_hpc/contracts.py | 92 | open | Documented-as-intentional TBI surface (by design, not a bug): Config.with_runtime/.with_circuit/.with_probes/.validate/. |  | 2026-06-30 |
| jaxfne/export.py | 92 | reviewed |  |  | 2026-07-04 |
| jaxfne/neuronal_tensor.py | 92 | reviewed |  | Layer.geometry (per-layer distribution/x_range/y_range/z_range) is dropped by neuronal_tensor_to_configuration; only con | 2026-07-04 |
| jaxfne/paradigm.py | 92 | reviewed |  |  | 2026-07-04 |
| jaxfne/runtime.py | 92 | done |  |  | 2026-07-05 |
| jaxfne/sharding_utils.py | 92 | reviewed |  | docstring states stubs 'do not yet drive any real multi-device dispatch...planned for v0.3.20+' -- self-labeled incomple | 2026-07-04 |
| jaxfne/solvers.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/benchmark_jaxfne.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/_figure_common.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig04_minimal_install_run.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig06_readout_family_panel.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_1000_laminar_column_boosted.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_dH_component_trace.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_fi_calibration_curves.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_gain_ratio_sweep.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_synaptic_channel_trace.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/visualize_cylinder_cortex_1000.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_agsdr_multilaminar_api.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_analysis_metrics.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_cable_filter_tensor.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_calibration_contracts_v025.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_cell_params_compilation_v0401.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_computation_basis_v026.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_config_plasticity_homeostasis_baseline.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_configuration_domains_complete.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_configuration_geometry_population_interarea.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_connection_weight_modes_v0330.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_connections_compiler.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_conservation_proxy_diagnostics_v027.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_core_class_hygiene.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_csd_tensor.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_docs_version_alignment.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_edge_backend_v009.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_emitters_generalized.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_field_diagnostics_v026.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_field_proxy_admissibility_v024.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_general_sequential_paradigm.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_hdp_dispatch.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_hdp_kernel_standalone.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_homeostasis_dispatch.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_homeostatic_stability_v042.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_jaxley_bridge.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_jaxley_trace_bridge.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_jit_equivalence_v036.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_laminar_geometry_v013.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_linear_readouts.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_network3d_visualize.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_neuronal_tensor_notebook_execution.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_notebook_standard_v027.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_objectives.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_operator_stage_coverage_v04.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_plasticity_v034.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_release_scripts.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_root_import_lightweight.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_run_receipt_v016.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_runtime_backend_device_v0340.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_runtime_module_v0211.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_resume_equivalence_full.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_runtime_architecture.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_sanity_delta_task_schedule.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_scaling_benchmark.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_scan_backends_v012.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_simulate_runtime_propagation.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_solvers.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_source_bookkeeping.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_source_bookkeeping_v020.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_spectrolaminar_readiness_v011.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_stimulus_injection_v012.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no1_agsdr_public_api.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no1_notebook_execution.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no1_runtime_static_guards.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no2_upgrade.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no4_public_grammar.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_suite_no4_runtime_static_guards.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_synapse_metadata_v010.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_synaptic_kernel_v011.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_tcm_v1_6pop.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_tensor_pipeline_custom_cfg.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_tutorial_utils.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_v0310_eeg_meg_emm_proxy_bundle_tutorial.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_v0331_euler_solver.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_v0331_laminar_config.py | 92 | reviewed |  |  | 2026-07-04 |
| tests/test_with_emitter_parameters.py | 92 | reviewed |  |  | 2026-07-04 |
| scripts/cortical_column_localized_workflow.py | 92 | done | HDP deliberately left off (K_HDP=0 null control) -- the custom 'spend more, lose more' dH/dt formula requested for this  |  | 2026-07-01 |
| docs/api/core.md | 92 | done |  | ReadoutSpec metrics table previously listed max_spike_rate_hz/mean_source/mean_LFP/mean_CSD/burst_frequency_hz as valid  | 2026-07-03 |
| docs/api/fields.md | 92 | done |  |  | 2026-07-03 |
| docs/api/probes.md | 92 | done |  |  | 2026-07-03 |
| docs/citation.md | 92 | done |  |  | 2026-07-03 |
| docs/computation_basis.md | 92 | done |  |  | 2026-07-03 |
| docs/guides/hdp.md | 92 | done |  |  | 2026-07-03 |
| docs/guides/output_bundles.md | 92 | done | Uses an older/simplified API style (model.simulate(sim), model.compute_readout(signals, specs), model.manifest(signals,  |  | 2026-07-03 |
| docs/releases/v0.2.3.md | 92 | done |  |  | 2026-07-03 |
| docs/source_field_equations.md | 92 | done | Code example 'cfg = jtfne.configuration().emitter(...).field(...)' (no assignment reuse of `cfg =` per line, missing lea |  | 2026-07-03 |
| docs/tutorials/07_v037_source_bookkeeping.md | 92 | done |  |  | 2026-07-03 |
| docs/tutorials/10_v0313_omission_oddball.md | 92 | done |  |  | 2026-07-03 |
| docs/tutorials/index.md | 92 | done |  | The `docs/api/neuronal_tensor.md` and `docs/guides/hdp.md` links under 'Featured: NeuronalTensor' resolve to real files  | 2026-07-03 |
| tests/test_etude3_v1_spectrolaminar_1k.py | 92 | done |  |  | 2026-07-01 |
| local/verify_plotly_pipeline.py | 92 | done |  |  | 2026-07-01 |
| jaxfne/optim/manifests.py | 92 | done |  |  | 2026-07-01 |
| jaxfne/ (TODO/FIXME/NotImplementedError triage, 11 files) | 92 | done |  | jaxfne/emitters.py GLIFEmitter/LIFEmitter and jaxfne/pynwb_compat.py write_nwb/read_nwb are exported in __all__ but ALWA | 2026-07-01 |
| tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb | 92 | done |  |  | 2026-07-02 |
| tutorials/etudes/jaxfne_etude_no_9_local_oddball.ipynb | 92 | done |  | Uses build_laminar_column's default geometry (no explicit cylinder/radius) -- consistent with other étude notebooks, not | 2026-07-02 |
| tutorials/etudes/jaxfne_etude_no_8_continuous_adaptation.ipynb | 92 | done |  |  | 2026-07-02 |
| scripts/audit_notebook_grammar.py | 92 | done |  | Checks are deliberately text/regex-based (section-marker presence, forbidden-pattern absence), NOT execution-based -- it | 2026-07-02 |
| jaxfne/_model.py | 92 | done |  |  | 2026-07-05 |
| jaxfne/_construct.py | 92 | done |  |  | 2026-07-05 |
| jaxfne/analysis/metrics.py | 93 | reviewed |  |  | 2026-07-04 |
| jaxfne/fields/__init__.py | 93 | reviewed |  |  | 2026-07-04 |
| jaxfne/fields/diagnostics.py | 93 | reviewed |  |  | 2026-07-04 |
| scripts/ed9_hdp_evidence.py | 93 | reviewed |  |  | 2026-07-04 |
| scripts/ed9_homeostasis_evidence.py | 93 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/ed01_api_stability_snapshot.py | 93 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/ed09_failure_modes_and_nulls.py | 93 | reviewed |  |  | 2026-07-04 |
| scripts/hdp_100_stability_sweep.py | 93 | reviewed |  |  | 2026-07-04 |
| scripts/sync_docs_version.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_emitter_equations_v020.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_experimental_hpc_contracts.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_field_admissibility_v020.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_identity_v0329.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_is_valid_signal.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_jaxley_optional_dependency.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_manifest_readout_compat.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_manifest_v005.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_mechanism_aware_connection_compiler.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_numerical_stability.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_objective_report_v018.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_optim_jax_native_audit.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_paradigm.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_probe_operators_eeg_meg_emm.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_probe_operators_v021.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_public_api_compatibility.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_root_export_api_v0338.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_schema_migration_v0342.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_selectors_v0329.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_signals_get_v0329.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_simulate_homeostasis_metadata_helper.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_solver_smoke_v0401.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_sparse_connectivity_v0401.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_spectrolaminar_objectives.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_spectrolaminar_readout.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_spectrolaminar_sources.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_streaming.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_synapse_connectivity.py | 93 | reviewed |  |  | 2026-07-04 |
| tests/test_synaptic_tensor.py | 93 | reviewed |  |  | 2026-07-04 |
| AGENTS.md | 93 | reviewed | Fixed this session: gradient_path_safe/Model.tune() guard code-location claim, StimulusSchedule file:line, Configuration |  | 2026-07-05 |
| docs/api/bridges.md | 93 | done |  |  | 2026-07-03 |
| docs/guides/configuration_grammar.md | 93 | done |  |  | 2026-07-03 |
| docs/guides/homeostasis.md | 93 | done |  | Consider a one-line note that the built-in kernel's default r_star (0.05, from jaxfne/core.py:2363 RuntimeConfig.homeost | 2026-07-03 |
| docs/quickstart.md | 93 | done |  | hdp_params example passes 'size_scale_by_cell_type' inside RuntimeConfig(hdp_params={...}); RuntimeConfig's own docstrin | 2026-07-03 |
| jaxfne/fields/probes.py | 94 | reviewed |  |  | 2026-07-04 |
| jaxfne/presets.py | 94 | reviewed |  |  | 2026-07-04 |
| jaxfne/vis/plotly/lfp.py | 94 | reviewed |  |  | 2026-07-04 |
| jaxfne/vis/plotly/raster.py | 94 | reviewed |  |  | 2026-07-04 |
| scripts/ed10_release_archive_receipt.py | 94 | reviewed |  |  | 2026-07-04 |
| scripts/release/reconcile_release_target.py | 94 | reviewed |  |  | 2026-07-04 |
| tests/test_emitter_family_validation_v0330.py | 94 | reviewed |  |  | 2026-07-04 |
| tests/test_field_solution_metadata_v0213.py | 94 | reviewed |  |  | 2026-07-04 |
| tests/test_fields_helper_dedup_v0330.py | 94 | reviewed |  |  | 2026-07-04 |
| tests/test_kappa_synchrony_vectorized.py | 94 | reviewed |  |  | 2026-07-04 |
| tests/test_objective_null_reproducibility_v0330.py | 94 | reviewed |  |  | 2026-07-04 |
| tests/test_poisson_admissibility_v0215.py | 94 | reviewed |  |  | 2026-07-04 |
| docs/mathematical_glossary_flow.md | 94 | done |  |  | 2026-07-03 |
| docs/migration_guide.md | 94 | done |  |  | 2026-07-03 |
| docs/tensor_electromagnetics_scope.md | 94 | done |  |  | 2026-07-03 |
| jaxfne/configs/{canonical-v1-column-1000n,canonical-v1-v4-pfc-multiarea,default-column,homeostatic-h-override-demo,lamin | 95 | open |  |  | 2026-06-30 |
| examples/00_generalized_izhikevich_3d_smoke.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/00_minimal_column.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/01_generalized_readout_smoke.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/01_source_field_manifest.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/02_generalized_vis_smoke.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/03_single_neuron_multimodal_probe.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/04_two_neuron_ei_multimodal.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/05_dataset_bridge_manifest.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/05_network_100_ei_multimodal.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/06_edge_list_recurrent_backend.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/07_jaxley_trace_bridge.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/08_neuronal_tensor_first.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/v031_single_izhikevich_neuron.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/v032_single_neuron_parameter_sweep.py | 95 | reviewed |  |  | 2026-07-04 |
| examples/v033_two_neuron_ei_multimodal.py | 95 | reviewed |  |  | 2026-07-04 |
| jaxfne/analysis/__init__.py | 95 | reviewed |  |  | 2026-07-04 |
| jaxfne/analysis/spectral.py | 95 | reviewed |  |  | 2026-07-04 |
| jaxfne/io.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/build_canonical_neuronal_tensor_configs.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig01_architecture.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig02_contracts.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig03_backend.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/evidence_figures/fig08_adjacent_tools_comparison.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/release/assert_release_freeze.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/release/validate_release_artifacts.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/report_hygiene_check.py | 95 | reviewed |  |  | 2026-07-04 |
| scripts/sync_release_metadata.py | 95 | reviewed |  |  | 2026-07-04 |
| tests/test_package_version_alignment.py | 95 | reviewed |  |  | 2026-07-04 |
| tests/test_physical_field_solver_v040_placeholder.py | 95 | reviewed |  |  | 2026-07-04 |
| skills/FRICTIONS_STACK.md | 95 | done |  |  | 2026-07-02 |
| docs/_generated/operator_inventory.md | 95 | done |  |  | 2026-07-03 |
| docs/api/emitters.md | 95 | done |  |  | 2026-07-03 |
| docs/api/objectives.md | 95 | done |  |  | 2026-07-03 |
| docs/api/solvers.md | 95 | done |  |  | 2026-07-03 |
| docs/guides/poisson_admissibility.md | 95 | done |  |  | 2026-07-03 |
| docs/guides/probe_operators.md | 95 | done |  | Version banner says 'v0.2.1' / 'Last updated: 2026-05-20' while package is at 0.4.4 -- same stale-version pattern as plo | 2026-07-03 |
| docs/notes/biophysical_model_comparison.md | 95 | done |  |  | 2026-07-03 |
| docs/reference/glossary_of_methods.md | 95 | done |  |  | 2026-07-03 |
| tests/test_notebook_execution_suite.py | 95 | done |  |  | 2026-07-02 |
| jaxfne/_signals.py | 95 | open |  | Verify full test suite passes after the 3 rounds of missed-re-export fixes this session (_KNOWN_METRICS/_KNOWN_LAYERS/_K | 2026-07-04 |
| jaxfne/_config.py | 95 | open |  |  | 2026-07-04 |
| scripts/evidence_inventory.py | 96 | reviewed |  |  | 2026-07-04 |
| scripts/generate_operator_inventory.py | 96 | reviewed |  |  | 2026-07-04 |
| scripts/validate_json_safe.py | 96 | reviewed |  |  | 2026-07-04 |
| README.md | 96 | done |  |  | 2026-07-05 |
| docs/api/neuronal_tensor.md | 96 | done |  |  | 2026-07-03 |
| docs/api/runtime.md | 96 | done |  |  | 2026-07-03 |
| docs/api/sharding.md | 96 | done |  |  | 2026-07-03 |
| docs/guides/tensor_field_workflows.md | 96 | done |  | STYLE QUESTION RESOLVED (2026-07-03): confirmed this doc's older Configuration builder-facade style (jtfne.configuration | 2026-07-03 |
| docs/limitations_and_future_plans.md | 96 | done |  |  | 2026-07-03 |
| docs/operator_doctrine.md | 96 | done |  |  | 2026-07-03 |
| docs/tutorials/03_network_100_ei.md | 96 | done |  |  | 2026-07-03 |
| docs/tutorials/tutorial_outputs.md | 96 | done |  |  | 2026-07-03 |
| docs/guides/index.md | 97 | done |  |  | 2026-07-03 |
| docs/guides/objective_grammar.md | 97 | done |  |  | 2026-07-03 |
| docs/index.md | 97 | done |  |  | 2026-07-03 |
| docs/install.md | 97 | done |  |  | 2026-07-03 |
| docs/performance_baseline.md | 97 | done |  |  | 2026-07-03 |
| docs/tutorials/13_canonical_column_etude.md | 97 | done |  | No numbered notebook maps 1:1 to this doc (unlike docs/tutorials/0X_*.md pattern) -- closest analog is tutorials/etudes/ | 2026-07-03 |
| jaxfne/experimental_hpc/physical_field_solver_v040.py | 98 | open |  |  | 2026-06-30 |
| scripts/evidence_figures_inventory.py | 98 | reviewed |  |  | 2026-07-04 |
| scripts/snapshot_public_api.py | 98 | reviewed |  |  | 2026-07-04 |
| docs/guides/operator_composition.md | 98 | done |  |  | 2026-07-03 |
| jaxfne/hdp_network.py | 100 | done | CORRECTION 2026-07-01: the note recommending DEFAULT_HDP as a 'verified-stable' default (added this session) is now unde | F-019 (not yet opened, deferred per explicit instruction): redesign of the passive-income rho_passive/H^2 formula -- F-0 | 2026-07-01 |
| jaxfne/emitters.py::simulate_edge_recurrent_izhikevich_hdp | 100 | done |  |  | 2026-07-01 |
| jaxfne/objectives.py | 100 | done |  |  | 2026-07-05 |
| jaxfne/vis/hdp_diagnostics.py | 100 | done |  |  | 2026-07-05 |
| jaxfne/vis/network3d.py | 100 | reviewed |  |  | 2026-06-30 |
| jaxfne/vis/plasticity_viz.py | 100 | done |  |  | 2026-07-05 |
| jaxfne/vis/report_plots.py | 100 | done |  |  | 2026-07-05 |
| jaxfne/vis/tutorial_array_plots.py | 100 | done |  |  | 2026-07-05 |
| jaxfne/optim/sdr.py | 100 | done |  |  | 2026-06-30 |
| jaxfne/optim/gsdr.py | 100 | done |  |  | 2026-06-30 |
| jaxfne/optim/agsdr.py | 100 | done |  |  | 2026-06-30 |
| jaxfne/core.py::Model / Configuration / JaxFNEConfig | 100 | done |  | Whether Model._simulate_arrays's five-way Python dispatcher (homeostasis/HDP/edge_list/dense x ablation_mode) is ever un | 2026-07-01 |
| jaxfne/optim/gsgd.py | 100 | done |  |  | 2026-06-30 |
| jaxfne/emitters.py | 100 | done |  | simulate_edge_recurrent_izhikevich_hdp's asymmetric double-barrier term (barrier_c/(H-H_min)^2 - barrier_d/(H_max-H)^2,  | 2026-07-04 |
| docs/api/index.md | 100 | done |  |  | 2026-07-03 |
| docs/api/validation.md | 100 | done |  | Optional follow-up (not blocking): document jaxfne.builders.validate_configuration(cfg: Configuration, strict: bool=True | 2026-07-03 |
