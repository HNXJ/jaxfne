<!-- auto-generated from review.json by scripts/prp_to_markdown.py — do not hand-edit -->

| path | score | review_status | moved_from_progress_on | review_command |
|---|---|---|---|---|
| local/gen_test_data_100n_1000ms.py | 88 | pending_review | 2026-07-01 | python3 local/gen_test_data_100n_1000ms.py |
| jaxfne/optim/core.py | 88 | pending_review | 2026-07-01 | python3 -m pytest tests/test_optim_jax_native_audit.py tests/test_optim_tune.py tests/test_optim_report_hardening.py tes |
| tutorials/etudes/jaxfne_etude_no_11_omission_local.ipynb | 90 | pending_review | 2026-07-01 | python3 scripts/audit_notebook_grammar.py --check; python3 -m pytest "tests/test_notebook_execution_suite.py::test_relea |
| tutorials/etudes/jaxfne_etude_no_12_omission_global_coop.ipynb | 90 | pending_review | 2026-07-01 | python3 scripts/audit_notebook_grammar.py --check; python3 -m pytest "tests/test_notebook_execution_suite.py::test_relea |
| scripts/hdp_suite2_visualizations.py | 90 | pending_review | 2026-07-01 | grep -n "_hdp_build_model\\|_hdp_apply_drive_correction" scripts/hdp_suite2_visualizations.py |
| examples/02_omission_scaffold.py | 90 | pending_review | 2026-07-01 | python3 -m pytest tests/test_manifest_v005.py::test_examples_02_omission_scaffold_runs tests/test_manifest_v005.py::test |
| examples/03_objective_and_tune_smoke.py | 90 | pending_review | 2026-07-01 | python3 -m pytest tests/test_manifest_v005.py::test_examples_02_omission_scaffold_runs tests/test_manifest_v005.py::test |
| tests/test_jaxley_emitter_bridge_e2e.py | 90 | pending_review | 2026-07-01 | python3 -m pytest tests/test_jaxley_emitter_bridge_e2e.py -v (forward order); then a targeted -n auto run including this |
| tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb | 92 | pending_review | 2026-07-01 | python3 scripts/audit_notebook_grammar.py --check; python3 -m pytest "tests/test_notebook_execution_suite.py::test_relea |
| tests/test_etude3_v1_spectrolaminar_1k.py | 92 | pending_review | 2026-07-01 | python3 -m pytest tests/test_etude3_v1_spectrolaminar_1k.py -v |
| local/verify_plotly_pipeline.py | 92 | pending_review | 2026-07-01 | python3 local/verify_plotly_pipeline.py; (rename data dir away, confirm clean FileNotFoundError, restore, confirm ALL OK |
| jaxfne/optim/manifests.py | 92 | pending_review | 2026-07-01 | python3 -m pytest tests/test_optim_manifests.py -v |
| scripts/audit_notebook_grammar.py | 92 | pending_review | 2026-07-01 | python3 scripts/audit_notebook_grammar.py --check |
| .legacy/internal_docs/scratch_archive/build_notebook.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/scratch_archive/generate_suite_no_2.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/test_v030_docs_audit.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/test_v030_plotly_artifacts.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/test_v030_tutorial_structure.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/test_v031_single_neuron_tutorial.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/test_v032_parameter_sweep_tutorial.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
| .legacy/internal_docs/test_v035_small_recurrent_ei_tutorial.py | 100 | pending_review | 2026-07-01 | grep -n testpaths pyproject.toml; git log -1 --format=%ai -- .legacy/ |
