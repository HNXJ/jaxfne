# state

Stamp: jaxfne@dev@0e0d6f85 · 2026-11-22 (O-NUM: five bit-exact identity tests skip on darwin per Hamm 2026-11-22, with exactness kept on Windows, WSL jax 0.10.1 and 0.11.2; R10/O-76: all six bitwise xfails retired into the strict contracts (the last by Hamm 2026-11-22), plus a strict no-drive test and a strict final-state contract; `scripts/internal/onum_measure.py` added; later commits change artifacts/ only) · Solmind execution under S19: R01, R02, R03, R10, R11, R70, R72, P-058 done; S16 ruled and the closed issue batch moved; R17 deferred until O-NUM closes; rest held on Hamm gates. Last broad gate: c0a9021b (2026-11-22, Windows vwin-pc, `run_test_gate.py broad --no-stamp` in a clean clone: 8407 passed, 48 skipped, 0 failed, every lane PASS, rc 0; 8455 tests = 8459 at 99536d99 - 5 retired O-76 xfails + 1 strict replacement); Windows is not the S21 matching host. CI (Fast) green on 007dd80e. Checklist 0.5.10: last recorded 43/100 (2026-11-19); planning adds no verified points. Workflows: CI (Fast) + Dashboard enabled; Macroscope, Nightly notebooks, Publish, CI (Release & Scheduled) disabled since 2026-11-14. `budget_files=2825` (S16, Hamm 2026-11-21; tree at 2813, 82.46 MB), `budget_bytes=96400000`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04 (`v0.5.9` on `main` e0160dfc).
- 2026-11-20 to 2026-11-22: remote consolidation (P-056 receipts at tag `archive/t90-smoke-receipts-20261115`); Solmind plan integrated at 5550cf07 (tag `archive/solmind-0510-plans-20261120`; S19 go); R01, R02, R03, R11, R70, R71, R72, P-058, S16 done; cleanup batch ending 3a4a9fce (no numeric or public-API change). Records: plan §8 rows R01-R72 and §9 S-rows, `issues_closed_2026-11-21.md`, and those commits' messages.

## Broken or in flight

- Solmind execution (T-PLAN-EXECUTE): R04 waits on S21; R17 on S23 (S16 cleared); R57 (P-047) on a Hamm public-API pick; SCIENCE rows R59/R60/R25 on Hamm; R58 (O-85 ruled 2026-11-08) waits on the 0.5.11 held-out training-loss check. Other gates: plan §9 (S02-S04, S06, S08, S10, S11, S15, S17, S18, S20-S23).
- Audit items held for Hamm (2026-11-21): `jit="auto"` always-jit (warm simulate about 25x faster and first call 13-20% faster below the threshold on vwin-pc, but LIF `V_m` moves up to 9.54e-07 and it reverses the opt051_1 receipt); `int(ms / dt)` truncation with float32 dt at 56 sites (readout values move); compact JSON for receipts; atlas `lfp.html` time-axis duplication (future freezes); results_reconstruction drafts vs manuscript. Not taken: merging `benchmark_051/055_matrix.py` (055 is cited by the signed `PREREG_B1_0510.md`, 051 by the opt051_1 receipt).
- Matching-host broad gate (`T-MATCH`) still pending on Linux/Windows for post-merge `dev`.
- CAMINOS E1 not equivalent (oracle kept); PCL R5 FAIL + P-055 attributed; P-053 open; P-056 blocked on owner inputs; P-066 ruled 2026-11-19, repair on `_temp_tfne-manuscript` 98d42ce4 awaiting integration (R17). Papers are never failed through 0.6 (Hamm 2026-11-03).

## Open decisions for the owner

- Recommended defaults (proceed under these unless Hamm overrides): (1) retain frozen 100-point checklist; report prepublication readiness separately; (2) keep ephaptic experimental unless physical interpretation and numerical correctness are established; (3) frozen evidence and numerical criteria need Hamm before change; (4) publication remains exact-SHA owner authorization.
- Canonical open rows: `open_stack.md` (O-FIG05, O-NUM, O-73..O-75, Crazy Sailor O-1..O-4, O-6, O-8, O-9). Solmind decision gates S01-S23: plan §9.

## Next agent — grab remote and continue

1. `git fetch --all --prune`; checkout `dev` at `origin/dev`; read `artifacts/AGENTS.md`, this file, `open_stack.md`, `todo_stack.md`, then the plan (`artifacts/programme/design/plans/solmind/solmind_0510_plan.md`, §1-§4 and §8-§9).
2. Work T-PLAN-EXECUTE rows not held (plan §8 status text names each hold), max 3 lanes incl. reviewers. Recompute the checklist only from exact-SHA receipts. Stop at `READY_FOR_OWNER_GO`.
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
- Remote branches: `main`, `dev`, `gh-pages`, plus `_temp_tfne-manuscript` (P-066 repair 98d42ce4, R17, held on S23). `_temp_solmind_0510_plans`, `_temp_r70` and `_temp_p058` retired 2026-11-21 after merge into `dev`. Jomission `_temp_solmind` is separate and unaffected.
