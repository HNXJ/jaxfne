<!-- auto-generated from plans.json by scripts/prp_to_markdown.py — do not hand-edit -->

## items

| id | title | status | target_files |
|---|---|---|---|
| hdp-100k-100step-validation-run | [QUEUE FRONT, DO NOT START WITHOUT GO-AHEAD] Run the validated HDP fix (K_ctrl) on the real 100k-neuron cortical column, | proposed | scripts/cortical_column_localized_workflow.py; jaxfne/emitters.py::simulate_edge_recurrent_izhikevich_hdp |
| colab-notebook-gpu-tpu-100k-column | [QUEUE FRONT, DO NOT START WITHOUT GO-AHEAD] Colab notebook: clone+GPU/TPU setup + real GPU/TPU speed estimate for the 1 | proposed | tutorials/; scripts/cortical_column_localized_workflow.py |
| P3b | Model.checkpoint() / Model.restore() methods | not_started | jaxfne/core.py |
| P3c | Document the canonical HDP call pattern (AGENTS.md or equivalent) | not_started | AGENTS.md |
| pipeline-configuration_to_tensor | configuration_to_tensor(cfg) -> NeuronalTensor | not_started | jaxfne/_pipeline.py |
| pipeline-tensor_to_graph | tensor_to_graph(tensor) -> internal flattened graph | not_started | jaxfne/_pipeline.py |
| F-019 | rho_passive/H^2 passive-income formula redesign | blocked | jaxfne/emitters.py::simulate_edge_recurrent_izhikevich_hdp; jaxfne/hdp_network.py |
| jaxfneconfig-test-migration | Migrate or delete the 21 JaxFNEConfig-dependent tests | done | tests/test_config_schema_v015.py; tests/test_config_runtime_hardening_v028.py; tests/test_v021_config_runtime_source_fid |
| stale-test-fixture-fix | Fix the 4 call sites of removed default_spectrolaminar_config/default_nuclei_config | done | tests/test_public_builders_complete.py; tests/test_vis_phase5.py; tests/test_etude1_agsdr_convergence.py; etudes/jaxfne_ |
| configs-dir-schema-assumption-fix | Fix test_neuronal_tensor.py's assumption that all of jaxfne/configs/ is NeuronalTensor-schema | done | tests/test_neuronal_tensor.py |
| merge-build_model-apply_drive_correction | Migrate 3 scripts' un-consolidated build_model/apply_drive_correction copies onto hdp_network.py's generic versions | done | scripts/hdp_1000_neuronal_tensor_column.py; scripts/hdp_suite2_visualizations.py; scripts/spectrolaminar_tfne_izhikevich |
| file-by-file-review | Systematic file-by-file review of the 374 score=null placeholder entries | not_started |  |
| test-1000n-fast-laminar-lfp-csd-hdp | 1000-neuron fast float32 laminar default cortex LFP+CSD+HDP smoke test | not_started | tests/ (new file, name TBD e.g. tests/test_laminar_1000n_lfp_csd_hdp.py) |
| stale-fixture-remaining-notebooks | Fix 2 more notebooks still calling the removed default_spectrolaminar_config/default_nuclei_config | done | tutorials/etudes/jaxfne_etude_no_2_spectrolaminar_power.ipynb; tutorials/templates/jaxfne_notebook_template.ipynb |
| optim-bounds-duplicate-functions | Merge enforce_parameter_bounds/apply_parameter_constraints (jaxfne/optim/bounds.py) -- byte-identical logic, different n | done | jaxfne/optim/bounds.py |
| optim-sdr-family-misleading-docstrings | step_sdr_transform/step_gsdr_transform/step_agsdr_transform docstrings claim distinct algorithms but are byte-identical  | done | jaxfne/optim/sdr.py; jaxfne/optim/gsdr.py; jaxfne/optim/agsdr.py |
| vis-dead-stub-plotting-functions | Fix or remove 4 dead-stub plotting functions, publicly exported, zero test coverage | done | jaxfne/vis/fields.py; jaxfne/vis/rasters.py; jaxfne/vis/traces.py; jaxfne/vis/spectra.py |
| connectivity-dense-on2-jax-compiler | compile_connection_rules_jax builds a dense O(n_pre*n_post) grid via jnp.tile, contradicting sparse-first design | not_started | jaxfne/connectivity.py |
| streaming-jit-redefined-in-loop | run_stdp_stream applies @jax.jit to a closure (run_chunk_scan) redefined fresh inside the per-chunk for-loop | done | jaxfne/streaming.py |
| cowork-goal-etude3-1k-replication | Cowork /goal: run+verify etude 3 (V1 1k spectrolaminar) notebook, low-effort Sonnet 5 | proposed | tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb; tests/test_etude3_v1_spectrolaminar_1k.py; local/etude3/; |
| vis-smoke-test-coverage-gap | Add tests/test_vis_smoke_all.py: parametrized smoke coverage for 18 untested vis/vis.plotly modules | done | jaxfne/vis/exporters.py; jaxfne/vis/hdp_diagnostics.py; jaxfne/vis/plasticity_viz.py; jaxfne/vis/plotly/connectivity.py; |
| smart-test-matrix-configs-2-5 | 4 remaining configs of the small-network smart-test matrix (config #1 done) | proposed | tests/test_epv_2neuron_pipeline_smoke.py; tests/test_ei_jaxley_izhikevich_parity.py; tests/test_ei_jaxley_hh_field_reado |
| bf16-quantized-tfne-izhikevich-mode | bf16 quantized-compute mode for TFNE-Izhikevich/HDP networks (not HH) + jaxfne.units.py dtype-aware defaults | in_progress | jaxfne/core.py::RuntimeConfig; jaxfne/core.py::Model.with_hdp_initial_state; jaxfne/emitters.py::simulate_edge_recurrent |
| hdp-universal-default-kernel-consolidation | Make HDP a default property of every TFNE emitter (K_HDP coefficient, not a separate opt-in kernel) | proposed | jaxfne/emitters.py::simulate_edge_recurrent_izhikevich; jaxfne/emitters.py::simulate_edge_recurrent_izhikevich_homeostat |
| localized-distance-limited-connectivity-rule | New connectivity rule: distance/radius-limited sampling with constant target in-degree (not %-of-N^2) | done | jaxfne/connectivity.py |
| cortical-column-scaleup-ladder-100-to-1M | Scale-up ladder for a TFNE-Izhikevich+HDP cortical column: 100 -> 1k -> 10k -> 100k -> 1M neurons | not_started | jaxfne/connectivity.py; jaxfne/emitters.py::simulate_edge_recurrent_izhikevich_hdp; jaxfne/core.py::Configuration.unifor |
| release-0.4.5-code-quality | 0.4.5: code release -- zero placeholders, merged/optimized pipelines, tutorials in jaxfne grammar, etudes/suites all rea | proposed | tutorials/etudes/jaxfne_etude_no_8_continuous_adaptation.ipynb; tutorials/etudes/jaxfne_etude_no_9_local_oddball.ipynb;  |
| release-0.4.6-docs-alignment | 0.4.6: docs release -- code-doc-theory alignment, 95/100 | in_progress | docs/; AGENTS.md; README.md; skills/ |
| release-0.4.7-final-polish | 0.4.7: final polish to 100/100 -- modular vis/util independence, improved paradigm engine for oddball/omission tasks on  | proposed | jaxfne/vis/; jaxfne/util.py; jaxfne/paradigm.py; jaxfne/export.py; jaxfne/tutorial_utils.py; scripts/evidence_figures/ |
| hdp-stability-formula-design-and-validation | [BLOCKING 0.4.5, TARGET 100/100] Design and validate a real, working HDP restoring formula | done | jaxfne/emitters.py::simulate_edge_recurrent_izhikevich_hdp; jaxfne/hdp_network.py::DEFAULT_HDP; scripts/hdp_v2_rho_sweep |
| test-strategy-consolidate-to-etudes-suites | [SCOPING NEEDED, DO NOT START] Redirect test coverage: retire redundant unit tests, grow etude/suite notebooks to cover  | done | tests/; tutorials/etudes/; scripts/*suite*.py |
| release-0.4.7-legacy-code-thinning | [SCOPING NEEDED, DO NOT START] Strip deprecated/legacy/duplicate code before 0.4.7 to shrink repo volume | partially_done |  |
| config-schema-ic-pcb-redesign | [SCOPING NEEDED, DO NOT START] Redesign config schema as an IC/PCB-style declarative description (emitters=elements, syn | done | jaxfne/core.py; jaxfne/builders.py; jaxfne/hdp_network.py |

## brainstorm

- **function_merger_analysis_2026-06-30**: 2026-06-30: repo-wide AST scan for function-merger candidates (pattern: y1=f1(x1), y2=f2(x2), same output identity, inco
- **test_consolidation_and_config_ic_schema_2026-07-02**: 2026-07-02, user directive ahead of 0.4.7: (1) simplify the test suite so real coverage comes from executing suites/etud
