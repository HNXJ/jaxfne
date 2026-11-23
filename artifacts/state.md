# state

Stamp: jaxfne@dev@fd470064 · 2026-11-23 (P-068 closed: Dashboard run 45224171653 on dev 10e2f85f green, publish and figures; batch `issues_closed_2026-11-23.md`. P-068 repaired: column height passed at 9 etude-script sites, multiscale r90 claims check at rel 1e-6 (Hamm 2026-11-23), `tests/test_rerun_etude_claims_p068.py`; broad gate at fd470064: 8431 passed, 41 skipped, 0 failed, 2 errors = 120 s subprocess timeouts in `tests/test_two_neuron_ei_colab_v029.py` under load, file alone 15 passed in 13 s (flake); broad gate at f719fa95 all 9 lanes PASS; review ACCEPT on f719fa95 and on the delta to d955f3ab, its NaN minor fixed at fd470064. Earlier, at db9ed99a: Hamm rulings of 2026-11-23 on plan §10.3 items 2-7, 9, 10, 13-17 and §10.6 item 6 recorded; P-047, P-052, P-053, P-055, P-056, C1, D3, F6 and F5's `Model.tune` half moved to 0.5.11; O-73 `population_rates` seconds forms removed (breaking, changelog fragment); G2 met (28 known violations, all Hamm-excepted); B1 amendment 2 pinned; B5 CPU-vs-GPU test and receipt; T-P066-FOLLOW closed (`tests/test_p066_follow.py`); nbconvert in the dev extra and the figure lock; Release CI enabled (S18); later commits change artifacts/ only) · Last broad gate: db9ed99a (2026-11-23, Windows vwin-pc, `run_test_gate.py broad --no-stamp` in a clean clone: 8431 passed, 41 skipped, 0 failed, every lane PASS, rc 0); independent review ACCEPT (first pass REVISE: ruling basis per item, S01 not ruled, glossary row; fixed). CI (Fast) green on 4773b4d9. Release CI 45185114841 on eb0adbb8: tree-check, release-test 3.11 and 3.14, release-build pass, publication failed on the figure lock (fixed at 67452861; publication passes on db9ed99a in run 45198260669). Checklist 0.5.10: last recorded 43/100 (2026-11-19); B5 and G2 met at non-candidate SHAs, counted only at the candidate. Workflows: CI (Fast), Dashboard, CI (Release & Scheduled) enabled; Macroscope, Nightly notebooks, Publish disabled. `budget_files=2825` (tree at 2820), `budget_bytes=96400000`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04 (`v0.5.9` on `main` e0160dfc).
- 2026-11-20 to 2026-11-23: Solmind plan rows R01, R02, R03, R10, R11, R17, R70, R71, R72, P-058, S16 done; cleanup swarm wave 1; the 2026-11-23 rulings batch. Records: plan §8-§10, `ISSUE_LOG.md` ("Moved to 0.5.11"), closed batches, and those commits' messages.
- WSL `~/venv_jaxfne` (Python 3.12.3, jax 0.10.1, CUDA): brian2 2.10.1 and NEST 3.10.0 import (B3 baselines, 2026-11-23).

## Broken or in flight

- T-MATCH (S21 host: Linux, Python 3.11, jax 0.10.2): WSL has no Python 3.11; sudo needs Hamm's password and a hook blocks installing uv. Needs Hamm.
- T-FIG05 re-freeze needs the macOS freeze platform; vmac unreachable 2026-11-23 (T-MAC).
- B2 scored run waits on plan §10.3 item 18 (B1 drive scaling, M3 drive) and the `budget_files` raise when it lands.
- Solmind execution: R57 (P-047) moved; SCIENCE rows on Hamm; S06, S17 (R20, R21) open; S01 proceeds under the frozen score.
- Audit items held for Hamm (2026-11-21): `jit="auto"` always-jit; `int(ms / dt)` truncation at 56 sites; compact receipt JSON; atlas `lfp.html` time axis; results_reconstruction drafts.
- P-067 (Pose3D xz/yz depth axis vs proxy) open, unmeasured. CAMINOS E1 not equivalent (oracle kept).

## Open decisions for the owner

- Plan §10.3 item 18 (B1 (ii)-(iii)); item 12 (R12-R16, S06); T-MATCH Python 3.11 route; S01 if a change is wanted.
- Canonical open rows: `open_stack.md`. Gates S01-S23: plan §9.

## Next agent — grab remote and continue

1. `git fetch --all --prune`; checkout `dev`; read `artifacts/AGENTS.md`, this file, `open_stack.md`, `todo_stack.md`, plan §8-§10.
2. Top executable: T-AI-DOC, A1/A2 at the frozen candidate, R42-R50 docs rows. Max 3 lanes incl. reviewers. Stop at `READY_FOR_OWNER_GO`.
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

- Windows (py 3.14) / WSL `~/venv_jaxfne` (py 3.12.3, jax 0.10.1, CUDA, brian2, NEST) / macOS arm64 (py 3.13, jax 0.10.1; unreachable 2026-11-23); warehouse `F:/warehouse/`.
- Remote branches (2026-11-23): `main`, `dev`, `gh-pages`, plus other sessions' branches. `_temp_onum`: deleted on the remote, left deleted (Hamm 2026-11-23); local copy on vwin-pc.
