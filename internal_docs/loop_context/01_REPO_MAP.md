# 01_REPO_MAP

Source status: module map generated from uploaded `jaxfne-main.zip` extraction at `repo/jaxfne-main`. The downstream agent must re-check live git before mutation. Live audit summary in `Pasted markdown.md` reports `main == dev == tag v0.3.29 == fab4c9c` at `Pasted markdown.md:L30` and validation evidence at `Pasted markdown.md:L63`.

## Module map

|path|role|JAX-critical?|public API?|risk|tests covering it|note|
|---|---|---|---|---|---|---|
|jaxfne/__init__.py|package module|yes|mixed|medium|tests/test_jaxley_bridge.py, tests/test_multi_area_emitter_runtime.py, tests/test_multi_area_source_projector.py|lines 392|
|jaxfne/bridges.py|optional external backend bridges|yes|yes|medium|tests/test_api_smoke.py, tests/test_jaxley_bridge.py, tests/test_jaxley_optional_dependency.py|lines 482|
|jaxfne/builders.py|package module|yes|yes|medium|tests/test_objectives.py, tests/test_public_builders_complete.py, tests/test_suite_no2_upgrade.py|lines 584|
|jaxfne/core.py|configuration/model/simulation/signals orchestration|yes|yes|low|tests/test_agent_context_hygiene.py, tests/test_computation_basis_v026.py, tests/test_config_schema_v015.py|lines 6080|
|jaxfne/emitters.py|emitter kernels and emitter classes|yes|yes|medium|tests/test_backend_parity_v020.py, tests/test_compact_facade_v034.py, tests/test_emitter_equations_v020.py|lines 1119|
|jaxfne/experimental_hpc/__init__.py|experimental contracts / future surfaces|yes|mixed|low|tests/test_jaxley_bridge.py, tests/test_multi_area_emitter_runtime.py, tests/test_multi_area_source_projector.py|lines 65|
|jaxfne/experimental_hpc/contracts.py|experimental contracts / future surfaces|yes|yes|medium|tests/test_calibration_contracts_v025.py, tests/test_computation_basis_v026.py, tests/test_emitters_generalized.py|lines 539|
|jaxfne/fields/__init__.py|source/field/probe proxy operators|yes|mixed|low|tests/test_jaxley_bridge.py, tests/test_multi_area_emitter_runtime.py, tests/test_multi_area_source_projector.py|lines 89|
|jaxfne/fields/diagnostics.py|source/field/probe proxy operators|yes|yes|low|tests/test_api_smoke.py, tests/test_conservation_proxy_diagnostics_v027.py, tests/test_etude1_agsdr_convergence.py|lines 156|
|jaxfne/fields/probes.py|source/field/probe proxy operators|yes|yes|low|tests/test_agent_context_hygiene.py, tests/test_compact_facade_v034.py, tests/test_config_schema_v015.py|lines 366|
|jaxfne/fields/proxy.py|source/field/probe proxy operators|yes|yes|high-prng|tests/test_api_smoke.py, tests/test_backend_parity_v020.py, tests/test_calibration_contracts_v025.py|lines 1154|
|jaxfne/io.py|JSON-safe IO, manifests, hashes|yes|yes|low|tests/test_agent_context_hygiene.py, tests/test_agsdr_multilaminar_api.py, tests/test_api_smoke.py|lines 194|
|jaxfne/objectives.py|metrics, losses, nulls, objective reports|yes|yes|prng-resolved (B01/PR#22; was high-prng)|tests/test_etude1_agsdr_convergence.py, tests/test_evoked_l4_drive.py, tests/test_objectives_v020.py, tests/test_objective_null_reproducibility_v0330.py|null generators now thread explicit rng/null_seed|
|jaxfne/optim/__init__.py|optimizer specs/search/report helpers|yes|mixed|low|tests/test_jaxley_bridge.py, tests/test_multi_area_emitter_runtime.py, tests/test_multi_area_source_projector.py|lines 68|
|jaxfne/optim/agsdr.py|optimizer specs/search/report helpers|yes|yes|low|tests/test_etude1_agsdr_convergence.py, tests/test_etude1_notebook_thinness.py, tests/test_optim_jax_native_audit.py|lines 36|
|jaxfne/optim/base.py|optimizer specs/search/report helpers|no|yes|low|tests/test_backend_parity_v020.py, tests/test_computation_basis_v026.py, tests/test_config_schema_v015.py|lines 16|
|jaxfne/optim/bounds.py|optimizer specs/search/report helpers|yes|yes|low|tests/test_multi_area_emitter_runtime.py, tests/test_multi_area_spectrolaminar_objective.py, tests/test_multi_area_spectrolaminar_readout.py|lines 31|
|jaxfne/optim/core.py|configuration/model/simulation/signals orchestration|yes|yes|low|tests/test_agent_context_hygiene.py, tests/test_computation_basis_v026.py, tests/test_config_schema_v015.py|lines 1550|
|jaxfne/optim/gsdr.py|optimizer specs/search/report helpers|yes|yes|low|tests/test_etude1_agsdr_convergence.py, tests/test_etude1_notebook_thinness.py, tests/test_optim_jax_native_audit.py|lines 35|
|jaxfne/optim/gsgd.py|optimizer specs/search/report helpers|yes|yes|low|grep-needed|lines 29|
|jaxfne/optim/manifests.py|optimizer specs/search/report helpers|yes|yes|low|tests/test_laminar_geometry_v013.py, tests/test_suite_no1_public_grammar.py, tests/test_suite_no3_low_frequency_scaling_tutorial.py|lines 49|
|jaxfne/optim/sdr.py|optimizer specs/search/report helpers|yes|yes|low|tests/test_etude1_agsdr_convergence.py, tests/test_etude1_notebook_thinness.py, tests/test_optim_jax_native_audit.py|lines 30|
|jaxfne/paradigm.py|package module|no|yes|low|tests/test_config_schema_v015.py, tests/test_evoked_l4_drive.py, tests/test_manifest_v005.py|lines 310|
|jaxfne/presets.py|package module|no|mixed|low|tests/test_spectrolaminar_readiness_v011.py|lines 122|
|jaxfne/runtime.py|JAX backend/dtype/JIT runtime helpers|yes|yes|low|tests/test_agent_context_hygiene.py, tests/test_api_smoke.py, tests/test_backend_parity_v020.py|lines 147|
|jaxfne/sharding_utils.py|package module|yes|yes|low|tests/test_v0318_sharding_stubs.py|lines 154|
|jaxfne/tutorial_utils.py|package module|yes|yes|high-prng|tests/test_etude1_agsdr_convergence.py, tests/test_etude1_notebook_thinness.py, tests/test_suite_no1_public_grammar.py|lines 2011|
|jaxfne/validation.py|invariants, finite checks, claim gates|yes|yes|medium|tests/test_agent_context_hygiene.py, tests/test_backend_parity_v020.py, tests/test_calibration_contracts_v025.py|lines 1291|
|jaxfne/vis/__init__.py|visualization helpers, Plotly/Matplotlib optional paths|yes|mixed|low|tests/test_jaxley_bridge.py, tests/test_multi_area_emitter_runtime.py, tests/test_multi_area_source_projector.py|lines 110|
|jaxfne/vis/core.py|configuration/model/simulation/signals orchestration|yes|yes|low|tests/test_agent_context_hygiene.py, tests/test_computation_basis_v026.py, tests/test_config_schema_v015.py|lines 47|
|jaxfne/vis/fields.py|source/field/probe proxy operators|yes|yes|high-prng|tests/test_calibration_contracts_v025.py, tests/test_compact_facade_v034.py, tests/test_config_schema_v015.py|lines 712|
|jaxfne/vis/network3d.py|visualization helpers, Plotly/Matplotlib optional paths|yes|yes|high-prng|tests/test_network3d_visualize.py, tests/test_vis_network3d_public_api.py|lines 637|
|jaxfne/vis/rasters.py|visualization helpers, Plotly/Matplotlib optional paths|yes|yes|low|grep-needed|lines 92|
|jaxfne/vis/spectra.py|visualization helpers, Plotly/Matplotlib optional paths|yes|yes|low|tests/test_public_builders_complete.py|lines 138|
|jaxfne/vis/traces.py|visualization helpers, Plotly/Matplotlib optional paths|yes|yes|low|tests/test_docs_equations_plotly_v0214.py, tests/test_jaxley_bridge.py, tests/test_spectrolaminar_sources.py|lines 503|
|jaxfne/vis/tutorial_panels.py|visualization helpers, Plotly/Matplotlib optional paths|yes|yes|high-prng|tests/test_tutorial_panels.py|lines 450|

