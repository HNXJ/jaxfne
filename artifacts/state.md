# state

Stamp: jaxfne@dev@24a95963 · 2026-11-13. Last broad gate: 9020f50e on 2026-11-09 (8263 passed, 0 failed, all lanes PASS, replication audit 0 unlisted; outside review ACCEPT). No pytest gate has run on 24a95963. Checklist 0.5.10: 40/100 (10B B4 scored 2, 10D D4 scored 2, 10E E1 scored 7, E2 scored 3, 10F F4 scored 1, 10G G3 scored 1).
Working checkout `E:/repos/jaxfne` clean on `dev`, 15 commits ahead of `origin/dev` and identical to `origin/_temp_r5_size_budget`. `origin/dev` stays at 342279db: the report gate fails on `tree_size_audit` alone (O-R5SIZE).
Audit 2026-11-13 on d2507956: `run_test_gate.py report` stops at `tree_size_audit` (96 704 629 against a 96 400 000 budget, 304 629 over); the 14 families after it were run one at a time and all pass, `record_integrity` 251 pinned with 0 violations and `replication_audit` 28 known, 0 unlisted.
OCMA-J24.1 result at `artifacts/etudes/meta/literature/ocma/` (dashboard Reference audit table); tooling and ratings in HNXJ/ocma `cases/jaxfne/`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04: tag v0.5.9 on main e0160dfc; `artifacts/release/v0_5_9_release_receipt.json`; checklist 100/100.
- Scored to 2026-11-02 (details in each lane's RESULT): Potjans v2 off, Vogels v4 off, Destexhe v3 FAIL 0 of 5, GSDR P3 fit FAIL; CAMINOS Fig S1 scored; PCL R3 exact to rtol 1e-12 on files 0-4.
- 2026-11-06..08: HH seed refusal; homeostatic_ei stable bounds; Jansen-Rit P3 v1 PASS 5 of 5; PLoS Fig 5A producer and sustained-cycle test; `tv_form="code"` declared (8a6ccfc4); A3 scored, permutation to 0.5.11; P-BALANCED ported (CV pin 0.0585).
- PCL R5-approximate scored 2026-11-12 under signed R4b: 3168 tasks, oracle arm compared (1246 equal, 1922 differing). Verdict FAIL; P-055 attributed. See `pcl/prereg/P3_fig3_r5_approx_result.md`.
- 10E PCL Fig 3 scored: R5-approximate FAIL under signed R4b; attribution addendum pins the cause to `no-local` inhibition. 10E 10/10.
- 2026-11-12: B4 `build_config` repaired, D4 S16 panel added, F4 GSDR log-scale named, G3 audit §7 updated (checklist 40/100); B1 drive unification and baseline parameter match (T-79, T-80) landed and smoked.

## Broken or in flight

- CAMINOS E1 at n 12: not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`); oracle kept (Hamm 2026-11-03).
- PCL R5-exact FAIL (4/16) and R5-approximate FAIL (see Works). P-055 attributed: long complex `tau_m` and 64x fan-in accumulate sub-mV float differences, delaying near-threshold complex spikes and top-down feedback.
- P-053 (sparse-weight gap, unreproduced); P-056 (kernel-LFP deep-layer power < 0.5% of L2/3).
- Papers are never "failed" through 0.6 (Hamm 2026-11-03).
- Cursor CLI blocked on this host (2026-11-12): Group Policy `ExecutionPolicy=AllSigned` stops `cursor-agent.ps1` and its `.cmd` shim; direct `node.exe index.js` then fails because `--sandbox` needs macOS or Linux; agy produced no run. OCMA round 4 blocked.
- P-055 cites `V = 2.9788` vs `3.1950` mV; no receipt on disk holds those values (`approx_diff.json` has spike times only). The attribution stands on spike times; the replay script is missing.

## Open decisions for the owner

`open_stack.md` has no items. Ruled 2026-11-08: O-85, O-86, O-87; O-89 done; O-88, O-90, O-91, O-92 on standing go. Waiting on Hamm: the NeuroAI code link (anonymous.4open.science mirror). `origin/_temp_kernel-lfp` (ff45364b, 3 patches not in dev) is kept.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/notebooks/*.html`, `index.json` | `scripts/export_dashboard_notebook.py` on an executed suite notebook (data on F:, local only) | `tests/test_export_dashboard_notebook.py` |
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard`) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard`) |
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard`) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `docs/reference/errors.md` raise-site index | `scripts/generate_errors_index.py` | `--check`, `tests/test_generate_errors_index.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: Python 3.14; the broad gate runs through `suite_lock.py`, one suite at a time.
- WSL: `~/venv_jaxfne` (CUDA, numba 0.68), `~/venv_jaxfne_gate` (CPU gate), TeX Live 2026 at `~/texlive/cur`.
- Large data outside git: `F:/warehouse/{caminos,gsdr,pcl,vogels2011,potjans2014,destexhe2009}/`.
- NeuroAI 2027 draft: private thesis repo, branch `_temp_neuroai2027`.

## Next action

O-R5SIZE blocks the merge to `dev`: the 3168-row armcheck record is pinned and puts the tree 304 629 bytes over
budget, so `run_test_gate.py report` fails at `tree_size_audit` while all 14 later families pass. Three routes are
ranked in `open_stack.md`; the smallest is raising `budget_bytes` by 400 KB.

Executable now, no ruling needed: T-86 (CAMINOS PLoS panels), T-F5 (candidate batching), T-83 (P-053 fixture),
T-90 (kernel-LFP). T-D3 (Mejias G6) needs a filled signed P3 first. T-87 (F4 benchmark) follows T-79/T-80, which
landed. T-77 (`hh-at01`) needs a lane folder or an exception. OCMA round 4 waits on a reachable second model family.
