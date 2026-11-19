# state

Stamp: jaxfne@dev@56ce3a87 · 2026-11-19. Last broad gate: 56ce3a87 (T-96 run_caminos rename retry) on 2026-11-19, regular clone, Windows: exit 0 on the second run (8335 passed, 0 failed, 48 skipped, 6 xpassed, all lanes PASS; the first run's 16 subprocess timeouts under machine load passed alone, 88 of 88); independent review ACCEPT. CI (Fast) run 44569364330 on 56ce3a87: success, 10 of 10 jobs. Earlier: 314f99f2 (P-061) CI 44546980997, 3bfac5d0 (recurrent_backend validated on every Configuration route) CI 44526636527, 4dee0a6d (P-065) CI 44517404267, 32d24add (P-063, P-064) CI 44442354065, 0f1f0d22 (P-062 step 2) CI 44424229013, 06f9bf26 (P-062 step 1) CI 44392030376, 6b7acd62 (P-057) CI 44324199890, 98d5863c (ephaptic kernel) CI 44249549774, each 10 of 10. GitHub workflows: CI (Fast) re-enabled 2026-11-17 and Dashboard re-enabled 2026-11-19 (Hamm; gh-pages showed dev 24a95963 until the next dev push); Macroscope, Nightly notebooks, Publish and CI (Release & Scheduled) disabled (`disabled_manually` since 2026-11-14). Branches `plan/ocma-jaxfne-0515`, `audit/b1-m3-remote-baseline-20261112`, `audit/pcl-r5-remote-recovery-20261112`, `fix/b1-m3-signed-grid-20261112` keep their names while draft PRs #125-#128 are open (Hamm 2026-11-19; a GitHub branch rename closed PR #125, which was restored). Checklist 0.5.10: 40/100 (10B B4 scored 2, 10D D4 scored 2, 10E E1 scored 7, E2 scored 3, 10F F4 scored 1, 10G G3 scored 1).
Checkout `E:/repos/jaxfne` (2026-11-18): clean, at f146932c, two state-only commits of 2026-11-13 ahead of its stale `origin/dev` (24a95963), never pushed; agents work in `D:/tmp` clones.
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
- T-93 closed 2026-11-17 on dev c387064d (Hamm: budget stays 96.4 MB; bundles gzipped): the pinned 413,532-byte
  armcheck record is in the tree, `check_tree_size.py --check` exit 0 (88,411,937 of 96,400,000 bytes) and
  `check_record_integrity.py --check` exit 0 (251 of 251 pinned). `_temp_r5_size_budget`'s 96.8 MB budget raise
  (357bb357) and state stamps were not taken; its P-053/P-056 audit doc was (Hamm 2026-11-17).
- PCL attribution closed 2026-11-12 (`pcl/prereg/P3_fig3_r5_attribution_addendum.md`): complex-cell collapse attributed to no-local inhibition, tree drift excluded (`fig3_v4tree_nolocal.json`, T-85).

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
