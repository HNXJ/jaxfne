# state

Stamp: jaxfne@dev@8018e191 · 2026-11-25 (field-operator audit landed by fast-forward to the gated SHA: cleanups C1, C2, C3, C7, C11 (net -51 lines; C6 reverted, +28 net), byte parity 1630 items 0 mismatches vs 7930705b, test-only D1, D4, D5 checks (4 strict xfails, P-070, P-071); reviews ACCEPT on both lanes; broad gate at 8018e191: 8466 passed, 54 skipped, 5 xfailed, 0 failed, 9 lanes pass, rc 0. First gate run at 74aaa75d failed: short SHAs in `artifacts/memory.md` (`check_memory_hygiene.py`), fixed at f7bcfd14. Before: `_temp_noisecoef` merged at 68956a75: jit = eager noise coefficient on GPU; broad gate at a66d91c1: 8462 passed, 54 skipped, 1 xfailed, 0 failed, 9 lanes pass, rc 0; review ACCEPT. Before: CI fails, not skips, without jaxley or kaleido-v1 Chrome; broad gate at 173b672d: 8457 passed, 50 skipped, 1 xfailed, 0 failed, 9 lanes pass, rc 0; review ACCEPT; jarness `main` e46855ed installed on vlix (suite lock cap and group kill). Batch to 8f7f56e8: gate hang guard in `scripts/run_test_gate.py` (pytest defaults `JAX_PLATFORMS=cpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`; per-step cap `JAXFNE_GATE_STEP_TIMEOUT_S`, 3600 s, rc 124, process-group kill on timeout, SIGTERM and SIGHUP), task-gpunondet tolerance and docs, jaxley and Chrome skips; broad gate at 8f7f56e8 on vlix: 8458 passed, 47 skipped, 1 xfailed, 0 failed, all 9 lanes pass, rc 0; reviews ACCEPT on each lane before merge. Previous batch at 1fc1042f · 2026-11-23, da42b095..1fc1042f: R12 route table and refusal tests, R13 ephaptic wording, R14 LIF and Izh dt tests, R15-R16 identity and kernel tests, R42 docs inventory, R49 caption audit and text half, B1 manifest writer and check, P-069 fix for the AdEx seed batch and homeostatic E/I (Izh case on O-P069), CSD sign and Protocol C status fixed in docs (guard `test_csd_formulas_carry_the_code_sign`), the three plan-row records under `programme/design/plans/solmind/`; broad gate at 1fc1042f: 8448 passed, 41 skipped, 1 xfailed, 0 failed, all 9 lanes PASS, rc 0; mkdocs strict rc 0; record integrity 0 violations; tree 2825 files; reviews ACCEPT on 0d0c0582, 34dcb3ff, 219d3ba5, 1fc1042f. Earlier batches: git log and the closed issue batches.) · Last broad gate: 8018e191 (above). Release CI 45185114841 on eb0adbb8: tree-check, release-test 3.11 and 3.14, release-build pass, publication failed on the figure lock (fixed at 67452861; publication passes on db9ed99a in run 45198260669). Checklist 0.5.10: last recorded 43/100 (2026-11-19); B5 and G2 met at non-candidate SHAs, counted only at the candidate. Workflows: CI (Fast), Dashboard, CI (Release & Scheduled) enabled; Macroscope, Nightly notebooks, Publish disabled. `budget_files=2825` (tree at 2825), `budget_bytes=96400000`.

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
- task-gpunondet: GPU broad pytest at 29db4b15: 116 failed, all pass on CPU; 60 scatter-add (flag clears), 18 TF32 matmul, 36 residual diagnosed (matrix, todo row; 4 calls in O-GPU); 1 fixed and merged (68956a75); O-GPU (1) ruled, `_temp_pincontract` 6015b92c holds the CPU-scoped pins (WIP; merges with the live check, bound derivation with Hamm). Gates run on CPU.
- vlix worktree admin entry `_temp_jaxfne_bd37s` is partial (no HEAD); repair from Windows or Hamm decides. Never `git worktree prune` here.

## Open decisions for the owner

- Plan §10.3 item 18 (B1 (ii)-(iii)); item 12 (R12-R16, S06); T-MATCH Python 3.11 route; S01 if a change is wanted.
- O-P069 (Izh `enable_hdp=False` with `hdp_params`: refuse or document); O-AIDOC (OCMA section, direction text).
- Rulings (Hamm 2026-11-25): deterministic GPU ops stay off until the GPU failures are reproduced, classified and quantified (task-gpunondet); jarness `main` fast-forwarded to e46855ed on Hamm's go after review ACCEPT and gates, installed on vlix (install drift 0; cap smoke rc 124 at 2 s); vlix venv alignment accepted (`pip check` clean 2026-11-25), prior freeze kept until verification closes; `_temp_jaxfne_bd37s` preserved, repair or inspect from Windows only.
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
- vlix (Ubuntu side of vwin-pc, checkout `/mnt/e/repos/jaxfne`): `~/.venvs/jaxfne` (py 3.14.4, jax 0.10.2 GPU, jaxfne editable, pytest 9.2.0, mkdocs 1.6.1), built 2026-11-24, aligned 2026-11-25 to lock pins (Jaxley 0.14.0, pynwb 4.2.0, h5py 3.16.0, jnwb 0.2.16, numpy 2.4.6; prior freeze `/mnt/e/repos/_wt/jaxfne/_temp_venv_freeze_before.txt`); no Google Chrome (kaleido export tests skip); run suites with `TMPDIR` off `/tmp`; docs gate and `mkdocs build --strict` rc 0 at 59521ff3. `origin` here is HTTPS with no credential: push via `git@github.com:HNXJ/jaxfne.git`.
- Remote branches (2026-11-23): `main`, `dev`, `gh-pages`, plus other sessions' branches. `_temp_onum`: deleted on the remote, left deleted (Hamm 2026-11-23); local copy on vwin-pc.
