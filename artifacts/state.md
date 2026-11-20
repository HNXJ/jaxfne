# state

Stamp: jaxfne@dev@456cf9ef · 2026-11-20. Last matching-host broad gate: 022da7a0 (2026-11-19, Windows regular clone: 8337 passed, 0 failed, 48 skipped, 6 xpassed; CI (Fast) and Dashboard success; gh-pages 7f9aa2b5). On 2026-11-20 (macOS arm64, jax 0.10.1): 24 remote branches consolidated (13 merged to `dev`, `fd4f3f8c` CI (Fast) 10/10 green; 3 held branches resolved under Hamm's authorization: `_temp_wip-jaxfne-jaxfne-wt-suites-20261107` archived at tag `archive/wip-jaxfne-jaxfne-wt-suites-20261107`, `_temp_t90_smoke_20261115` preserved as `scripts/p056_resource_smoke.py` + tag `archive/t90-smoke-20261115`, PR #127 closed and `audit/pcl-r5-remote-recovery-20261112` deleted); `run_test_gate.py report` rc 0; `check_tree_size.py --check` rc 0 (2813/2813 files, 88.77/96.4 MB); `check_record_integrity.py --check` rc 0 (255/255 pinned); `DEV_PYTEST_TARGETS` (697 passed, 4 skipped, 0 failed) + targeted suite (377 passed, 0 failed). `budget_files` raised 2798 -> 2813 (+15 exact, 0 slack; `budget_bytes` unchanged at 96.4 MB); `budget_amendment_2026-10-23.md` kept at pinned `c52e830b…` (closes `O-10`, `EX-30`, `CS-11`).
GitHub workflows: CI (Fast) and Dashboard enabled; Macroscope, Nightly notebooks, Publish and CI (Release & Scheduled) disabled (`disabled_manually` since 2026-11-14). Checklist 0.5.10: 43/100 (10G G1=3, B4=2, D4=2, E1=7, E2=3, F4=1, G3=1). OCMA-J24.1 at `artifacts/etudes/meta/literature/ocma/`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04 (`v0.5.9` on `main` e0160dfc; `artifacts/release/v0_5_9_release_receipt.json`).
- 2026-11-19/20 on `dev`: P-057..P-065 (`recurrent_backend="auto"`), T-96 retry, T-86 CAMINOS PLoS panels (022da7a0), G-3 delayed rate seam (97289847), AdEx `edge_delivery="event"` (`_temp_ocma_jx1`), Crazy Sailor M0–M2 (`_temp_crazy_sailor_m0m1_b`), AUD-01/02 + T-83/P-053 sparse fixture + T-90/P-056 spec (`_temp_swarm_integ_20261115`), TFNE manuscript stages 0–2 + S1 notation + CSD/units (`_temp_tfne-manuscript`), LTNC & OCMA plans, Dependabot bumps (`ruff==0.17.5`, `mkdocs-material==9.8.1`, `pymdown-extensions==13.1`).
- T-93 closed 2026-11-17 (c387064d; `budget_bytes` kept at 96.4 MB); PCL R5-approx scored FAIL (P-055 attributed; T-85 closed); B1 drive unification (T-79, T-80) landed; Jansen-Rit P3 v1 PASS 5/5.

## Broken or in flight

- Remote branches consolidated to `main`, `dev`, and `gh-pages` only (`archive/wip-jaxfne-jaxfne-wt-suites-20261107` preserves `dashboard_suites.py` + 52 PNGs for `T-O68`; `archive/t90-smoke-20261115` preserves `093483a3`, and `scripts/p056_resource_smoke.py` is tracked on `dev`; draft PR #127 closed as superseded by `4849592e`).
- Matching-host broad gate (`T-MATCH`, `CS-10`) pending on Linux/Windows for post-merge `dev`; Mac host baseline failures attributed in `artifacts/assurance/mac_gate_attribution.md` (`O-FIG05`, `O-NUM`).
- CAMINOS E1 (20/24 not equivalent; oracle kept); PCL R5-exact & R5-approx FAIL (P-055: reported `V = 2.9788` vs `3.1950` mV lack a receipt on disk; replay script missing); P-053 (NOT_REPRODUCED under Amendment 1, open); P-056 (spec v1, blocked on owner inputs); P-066 (z depth-fraction vs x/y mm; probe at `F:/warehouse/jaxfne/P-066_2026-11-19/`). Papers are never failed through 0.6 (Hamm 2026-11-03).

## Open decisions for the owner

- `open_stack.md` carries 15 open rows (`O-FIG05`, `O-NUM`, `O-73`..`O-77`, Crazy Sailor `O-1`..`O-4`, `O-6`..`O-9`).
- Manuscript/prereg calls (2026-11-19): (1) P-066 z scaling vs Euclidean refusal; (2) keeper-signed Jansen-Rit P3 v1 standing. Waiting on Hamm: NeuroAI code mirror (anonymous.4open.science), OCMA round 4 host ruling (todo 28).

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/notebooks/*.html`, `index.json` | `scripts/export_dashboard_notebook.py` (local F:) | `tests/test_export_dashboard_notebook.py` |
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard`) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard`) |
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard`) |
| `docs/_static/atlas*/**.html` (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `docs/reference/errors.md` | `scripts/generate_errors_index.py` | `--check`, `tests/test_generate_errors_index.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment and next action

- Windows (py 3.14) / WSL `~/venv_jaxfne` (CUDA) / macOS arm64 (py 3.13, jax 0.10.1); warehouse `F:/warehouse/`.
- TFNE manuscript (`artifacts/programme/design/plans/arxiv_paper_plan.md`, narrowed core, Hamm 2026-11-19): next `TASK-DMAP`, `TASK-SOURCE-SCOPE`, `TASK-N2-PLANT`, `TASK-EPHAPTIC`, `TASK-HINV`, `TASK-REPL-TFNE`, `TASK-RECEIPTS`, then `TASK-MIXFAM`.
- Older carried items: `T-MATCH` on matching host; `T-73` PCL lane SHA + GSDR DOI in `dashboard/sources.json`; `T-77` `hh-at01`.
