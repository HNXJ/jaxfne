# state

Stamp: jaxfne@dev, merge of `_temp_solmind_0510_plans` 5550cf07 onto 3eb6a6a6 · 2026-11-21 · Solmind 0.5.10 plan integrated, execution authorized (S19). Last matching-host broad gate: 022da7a0 (2026-11-19, Windows: 8337 passed, 0 failed). On 2026-11-20: remote branch consolidation complete; CI (Fast) green on `10c417ad` ([#44817621200](https://github.com/HNXJ/jaxfne/actions/runs/44817621200), 10/10); seal commit `48ba03ed` is stacks-only. Checklist 0.5.10: last recorded 43/100 (2026-11-19); planning adds no verified points. Workflows: CI (Fast) + Dashboard enabled; Macroscope, Nightly notebooks, Publish, CI (Release & Scheduled) disabled since 2026-11-14. `budget_files=2813` (tree at 2813), `budget_bytes=96400000`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04 (`v0.5.9` on `main` e0160dfc).
- 2026-11-20: 24 non-core remotes resolved into `dev` or archive tags; PR #127 closed; `scripts/p056_resource_smoke.py` on `dev`; P-056 receipts at `archive/t90-smoke-receipts-20261115`.
- 2026-11-21: Solmind 0.5.10 plan reviewed (audits 1-2 REVISE, repaired; audit 3 ACCEPT by two independent reviewers) and integrated: `artifacts/programme/design/plans/solmind/solmind_0510_plan.md` (contract, R01-R72, gates S01-S23; 2026-10-31 execution plans folded in as Appendix A). Gates on 5550cf07: plan/stack tests 167 passed; tree-size 2813 files, 0 violations; record integrity 255/255; `run_test_gate.py docs` and `report` rc 0. Original planning commit kept at tag `archive/solmind-0510-plans-20261120`. Execution authorized by Hamm 2026-11-21 (S19).
- 2026-11-21: Jaxley x JAX matrix (R71): jaxley 0.13.0/0.14.0/0.14.1 x jax 0.10.1/0.11.2, 98 passed in each cell (Windows, Python 3.14.3; log local to vwin-pc).

## Broken or in flight

- Solmind execution (T-PLAN-EXECUTE): first R01-R04, R70, R72, then R56-R69. Open Hamm gates block other rows (plan §9: S02-S04, S08, S10, S11, S16-S18, S20-S22).
- Matching-host broad gate (`T-MATCH`, `CS-10`) still pending on Linux/Windows for post-merge `dev`.
- CAMINOS E1 not equivalent (oracle kept); PCL R5 FAIL + P-055 attributed; P-053 open; P-056 blocked on owner inputs; P-066 ruled 2026-11-19, repair on `_temp_tfne-manuscript` 98d42ce4 awaiting integration (R17). Papers are never failed through 0.6 (Hamm 2026-11-03).

## Open decisions for the owner

- Recommended defaults (proceed under these unless Hamm overrides): (1) retain frozen 100-point checklist; report prepublication readiness separately; (2) keep ephaptic experimental unless physical interpretation and numerical correctness are established; (3) frozen evidence and numerical criteria need Hamm before change; (4) publication remains exact-SHA owner authorization.
- Canonical open rows: `open_stack.md` (O-FIG05, O-NUM, O-73..O-77, Crazy Sailor O-1..O-4, O-6..O-9). Solmind decision gates S01-S23: plan §9.

## Next agent — grab remote and continue

1. `git fetch --all --prune`; checkout `dev` at `origin/dev`; read `artifacts/AGENTS.md`, this file, `open_stack.md`, `todo_stack.md`, then the plan (`artifacts/programme/design/plans/solmind/solmind_0510_plan.md`, §1-§4 and §8-§9).
2. Work T-PLAN-EXECUTE: R01-R04, R70, R72, then R56-R69 (max 3 lanes incl. reviewers). Recompute the checklist only from exact-SHA receipts. Stop at `READY_FOR_OWNER_GO`.
3. Do not tag/`main`/PyPI without Hamm's exact-SHA go.

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
- Remote branches: `main`, `dev`, `gh-pages`, plus `_temp_tfne-manuscript` (P-066 repair 98d42ce4, R17). `_temp_solmind_0510_plans` retired 2026-11-21 after merge into `dev`. Jomission `_temp_solmind` is separate and unaffected.
