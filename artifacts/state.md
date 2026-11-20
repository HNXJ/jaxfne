# state

Stamp: jaxfne@dev@10c417ad · 2026-11-20 · sealed. Last matching-host broad gate: 022da7a0 (2026-11-19, Windows: 8337 passed, 0 failed). On 2026-11-20: remote branch consolidation complete; CI (Fast) green on `10c417ad` ([#44817621200](https://github.com/HNXJ/jaxfne/actions/runs/44817621200), 10/10). Checklist 0.5.10: last recorded 43/100 (2026-11-19); planning adds no verified points. Workflows: CI (Fast) + Dashboard enabled; Macroscope, Nightly notebooks, Publish, CI (Release & Scheduled) disabled since 2026-11-14. `budget_files=2813`, `budget_bytes=96400000` (88.77/96.4 MB).

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04 (`v0.5.9` on `main` e0160dfc).
- 2026-11-20: 24 non-core remotes resolved into `dev` or archive tags; PR #127 closed; `scripts/p056_resource_smoke.py` on `dev`; P-056 receipts at `archive/t90-smoke-receipts-20261115`.
- Planning overlay ready on `_temp_solmind_0510_plans` (`a971288d`, also tagged `archive/solmind-0510-plans-20261120`): 55 tasks (`solmind_0510_todo.md`), 15 owner/decision gates (`solmind_0510_open.md`), execution contract (`solmind_0510_execution_contract.md`). Baseline of that plan is this `dev` SHA.

## Broken or in flight

- Execution readiness of the Solmind 0.5.10 plan is **not verified**: independent audit of the 55 tasks / 15 risks / release criteria, and failure-scenario tests (stale SHA, missing host, failed gate, authority conflict), have not been run (O-PLAN, T-PLAN-REVIEW).
- Matching-host broad gate (`T-MATCH`, `CS-10`) still pending on Linux/Windows for post-merge `dev`.
- CAMINOS E1 not equivalent (oracle kept); PCL R5 FAIL + P-055 attributed; P-053 open; P-056 blocked on owner inputs; P-066 units open. Papers are never failed through 0.6 (Hamm 2026-11-03).

## Open decisions for the owner

- Recommended defaults (proceed under these unless Hamm overrides): (1) retain frozen 100-point checklist; report prepublication readiness separately; (2) keep ephaptic experimental unless physical interpretation and numerical correctness are established; (3) frozen evidence and numerical criteria need Hamm before change; (4) publication remains exact-SHA owner authorization.
- Canonical open rows: `open_stack.md` (O-PLAN + O-FIG05, O-NUM, O-73..O-77, Crazy Sailor O-1..O-4, O-6..O-9). Full 15 Solmind decision gates: `origin/_temp_solmind_0510_plans:artifacts/programme/design/plans/solmind_0510_open.md`.

## Next agent — grab remote and continue

1. `git fetch --all --prune`; checkout `dev` at `origin/dev`; read `artifacts/AGENTS.md`, this file, `open_stack.md`, `todo_stack.md`.
2. **Do T-PLAN-REVIEW first** (non-writer reviewer): audit `_temp_solmind_0510_plans` / tag `archive/solmind-0510-plans-20261120` against live `dev` and canonical stacks; test adversarial failure handling; repair on the planning branch if REVISE.
3. **Then T-PLAN-INTEGRATE**: land accepted plan under `artifacts/programme/design/plans/solmind/` (tree-degree), update stack pointers, keep the archive tag, delete `_temp_solmind_0510_plans` only after unique evidence is preserved.
4. Execute ready Solmind packages (max 3 lanes incl. reviewers). Recompute checklist only from exact-SHA receipts. Stop at `READY_FOR_OWNER_GO` for Gate 5 / publication.
5. Do not expand planning further; do not tag/`main`/PyPI without Hamm's exact-SHA go.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/notebooks/*.html`, `index.json` | `scripts/export_dashboard_notebook.py` | `tests/test_export_dashboard_notebook.py` |
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard`) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard`) |
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard`) |
| `docs/_static/atlas*/**.html` (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `docs/reference/errors.md` | `scripts/generate_errors_index.py` | `--check`, `tests/test_generate_errors_index.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows (py 3.14) / WSL `~/venv_jaxfne` (CUDA) / macOS arm64 (py 3.13, jax 0.10.1); warehouse `F:/warehouse/`.
- Remote branches: `main`, `dev`, `gh-pages`, plus planning `_temp_solmind_0510_plans` (retire after T-PLAN-INTEGRATE). Jomission `_temp_solmind` is separate and unaffected.
