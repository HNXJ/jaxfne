# state

Stamp: jaxfne@dev@a5da71ad · 2026-11-12. Last broad gate: 9020f50e on 2026-11-09 (8263 passed, 0 failed, all lanes PASS, replication audit 0 unlisted; outside review ACCEPT). No gate run on a5da71ad yet. Checklist 0.5.10: 40/100 (10B B4 scored 2, 10D D4 scored 2, 10E E1 scored 7, E2 scored 3, 10F F4 scored 1, 10G G3 scored 1).
Working checkout `E:/repos/jaxfne` clean on dev.
OCMA-J24.1 result at `artifacts/etudes/meta/literature/ocma/` (dashboard Reference audit table); tooling and ratings in HNXJ/ocma `cases/jaxfne/`.

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04: tag v0.5.9 on main e0160dfc; `artifacts/release/v0_5_9_release_receipt.json`; checklist 100/100.
- P3 scored 2026-11-02: Potjans v2 off, Vogels v4 off (S12), Destexhe v3 FAIL 0 of 5. GSDR P3 fit FAIL 2026-11-01; GSDR stays open until replicated.
- CAMINOS Fig S1 scored (`caminos/RESULT_FIGURES.md`); core route fc0e13df; equivalence criterion declared at f54c022c.
- PCL R3 chain (6b10132a): oracle arm equals the C++ chain on files 0-4, exact at rtol 1e-12.
- P-BALANCED ported for 0.5.10 (e179a2d0..9dd1c111); canonical CV pin 0.0585 (1680659e).
- 2026-11-06..08: HH seed refusal test; homeostatic_ei stable bounds; B1 scripts and M3 fix; Jansen-Rit P3 v1 PASS 5 of 5 (`jansenrit1995/RESULT_P3.md`); PLoS Fig 5A producer and sustained-cycle test; T_v code form declared as variant `tv_form="code"` (8a6ccfc4); A3 scored, permutation moved to 0.5.11.
- PCL R5-approximate scored 2026-11-12 under signed R4b: all 3168 tasks completed on Windows CPU (`pcl/prereg/P3_fig3_r5_approx_result.md`, `scores/fig3_r5_approx.json`); oracle arm compared (1246 equal, 1922 differing; `approx_diff.json`). Verdict: FAIL. P-055 attributed (`ISSUE_LOG.md`).
- 2026-11-12: 10B B4 `build_config` repaired; 10D D4 S16 Destexhe panel added; 10F F4 GSDR log-scale parameters named; 10G G3 reproducibility audit §7 updated (checklist 40/100).
- B1 drive unification and baseline parameter match (T-79, T-80) landed (`b1_harness.py`, `b1_brian2.py`, `b1_nest.py`, `b1_jaxley.py`; `--smoke` verified).

## Broken or in flight

- CAMINOS E1 at n 12: not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`); oracle kept (Hamm 2026-11-03).
- PCL R5-exact FAIL (4/16 files) and R5-approximate FAIL under signed R4b (`pcl/prereg/P3_fig3_r5_approx_result.md`). P-055 attributed: complex cell integration time constant (`tau_m = 50 ms`) and 64x fan-in accumulate sub-millivolt float differences, delaying near-threshold complex spikes by ~1 ms and delaying top-down feedback.
- P-053 (PCL scan vs oracle gap on sparse weights, unreproduced); P-056 (kernel-LFP deep-layer power < 0.5% of L2/3).
- Papers are never "failed" through 0.6 (Hamm 2026-11-03).
- Tooling: Cursor CLI blocked on this host, two causes observed 2026-11-12: Group Policy sets
  `ExecutionPolicy=AllSigned` so `cursor-agent.ps1` and its `.cmd` shim cannot run (`-ExecutionPolicy Bypass` does
  not override a policy), and the direct `node.exe index.js` launch then fails because `--sandbox` needs macOS or
  Linux. agy produced no run directory. OCMA round 4 is blocked on this (todo 28).
- P-055's cited evidence is incomplete: `F:/warehouse/pcl/r5/approx_diff.json` records differing spike times but no
  membrane potentials, so the `V = 2.9788` vs `3.1950` mV values in the issue row have no receipt on disk
  (observed 2026-11-12). The attribution stands on the spike-time differences; a replay script is missing.

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

Tier 1 is closed except the two blocked items. T-73 waits on committing the R5 artifacts and moving the PCL lane
SHA in `dashboard/sources.json`; T-77 waits on an `hh-at01` lane folder or a dated exception. Next executable:
Tier 2 lane on B4 (`build_config`) and F4 (GSDR log-scale names), then T-79/T-80 for the B1 benchmark.
