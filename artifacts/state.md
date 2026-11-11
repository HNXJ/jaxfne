# state

Stamp: jaxfne@dev@22adb682 · 2026-11-10. Last broad gate: 9020f50e on 2026-11-09 (8263 passed, 0 failed, all lanes PASS, replication audit 0 unlisted; outside review ACCEPT). No gate run on 22adb682 yet. Checklist 0.5.10: 24/100.
Working checkout `E:/repos/jaxfne` was absent on 2026-11-10; this fresh clone (`E:/repos/jaxfne-r4-clone`, dev) holds the stack commit.
OCMA-J24.1 result at `artifacts/etudes/meta/literature/ocma/` (dashboard Reference audit table); tooling and ratings in HNXJ/ocma `cases/jaxfne/`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04: tag v0.5.9 on main e0160dfc; `artifacts/release/v0_5_9_release_receipt.json`; checklist 100/100.
- P3 scored 2026-11-02: Potjans v2 off, Vogels v4 off (S12), Destexhe v3 FAIL 0 of 5. GSDR P3 fit FAIL 2026-11-01; GSDR stays open until replicated.
- CAMINOS Fig S1 scored (`caminos/RESULT_FIGURES.md`); core route fc0e13df; equivalence criterion declared at f54c022c.
- PCL R3 chain (6b10132a): oracle arm equals the C++ chain on files 0-4, exact at rtol 1e-12.
- P-BALANCED ported for 0.5.10 (e179a2d0..9dd1c111); canonical CV pin 0.0585 (1680659e).
- 2026-11-06..08: HH seed refusal test; homeostatic_ei stable bounds; B1 scripts and M3 fix; Jansen-Rit P3 v1 PASS 5 of 5 (`jansenrit1995/RESULT_P3.md`); PLoS Fig 5A producer and sustained-cycle test; T_v code form declared as variant `tv_form="code"` (8a6ccfc4); A3 scored, permutation moved to 0.5.11.

## Broken or in flight

- CAMINOS E1 at n 12: not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`); oracle kept (Hamm 2026-11-03).
- PCL R5-exact FAIL, 4 of 16 files (P-055, `pcl/prereg/P3_fig3_r5_exact_result.md`). R5-approximate resumed 2026-11-06; 2860 of 3168 done at 2026-11-07 22:18Z (finish about 2026-11-08 10Z, inferred). Its driver logs `rc=$?` after `$(stamp)`, so the logged rc is always 0: judge the run by its output files. Its workers hold the CPU, so broad gates time out 17 example subprocesses until it ends.
- P-053 (PCL scan vs oracle gap on sparse weights, unreproduced); P-056 (kernel-LFP deep-layer power < 0.5% of L2/3).
- Papers are never "failed" through 0.6 (Hamm 2026-11-03).
- Tooling: Cursor write fails at launch (read works); Muse code workers may exit early (resume with `--session`).

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

Re-export the PLoS notebook (Fig 5A cell in, a6f4d867); B1 drive unification, then scored runs (T-B1); E1 at R5 end. Run the broad gate on 22adb682 before any stamp that claims green.
