# state

Stamp: jaxfne@dev@9dd1c111 · 2026-11-05

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.9 published 2026-11-04: tag v0.5.9 on main e0160dfc, Release, PyPI (hashes equal to the Release CI dist);
  `artifacts/release/v0_5_9_release_receipt.json`. Checklist 100/100 (`etudes/meta/plans/release_plan_0.5.9-0.6.0.md`).
  Main needed two extras fixes after the candidate (PyYAML in dev, Markdown in viz; PRs #121, #122), both picked
  onto dev. Post-candidate dev commits (0.5.10) unpushed until a broad gate passes.
- Signed and scored 2026-11-02, each as its pilots predicted: Potjans P3 v2 off, Vogels P3 v4 off on R-E (in S12),
  Destexhe P3 v3 FAIL 0 of 5 (`RESULT_P3.md` in each lane; Destexhe CALIBRATION step 7). GSDR P3 fit FAIL
  (2026-11-01); GSDR stays open until precisely replicated (Hamm 2026-11-02).
- CAMINOS Cell Reports Fig S1 scored (`caminos/RESULT_FIGURES.md`); core route landed (fc0e13df: `run_cell(route=)`,
  `FIGA_ROUTE`); equivalence criterion declared before the core output was read (f54c022c).
- PCL R3: `run_r2_reference.py chain` (6b10132a). Oracle arm equals the C++ chain on training files 0-4, exact at
  rtol 1e-12, max relative weight difference 7.4e-15 (`F:/warehouse/pcl/r3_oracle_f0_4/`).
- Release CI on d96dc49b failed 4 tests; all fixed: tree degree (079610f4), a test that could not fail (afe8852f),
  two slow regressions from the P-048 default u0 (bisected to b1d9dfe1; Hamm 2026-11-02 split: the atlas test runs
  the pre-ruling u0, the HDP etude re-baselined as `metrics_v059.json`, 5934aac1).
- P-BALANCED ported for 0.5.10 (e179a2d0..9dd1c111; Hamm 2026-11-05): `BackgroundNoise`, `NeuronalTensor.baseline_drive`
  and `background_noise` (preset noise in `cfg.metadata["background_noise"]`), balanced-1000n preset retuned under
  the unchanged bounds (seeds 0-4: CV 0.742-0.755, shift 0.236-0.257); canonical CV pin 0.0585 (b1d9dfe1 moved it).
- 0.5.10 checklist approved (Hamm 2026-11-05, after a review pass that enumerated items, stated gate weights and
  added 10G replication coverage); scored 7/100 (release plan, 0.5.10 section).
- Last pushed: 9dd1c111. Broad gate 8134 passed, 3 failed and 13 errors, all 120 s / 60 s timeouts under the R5
  load; each re-run alone passed (105 tests). Report fail_count 0.

## Broken or in flight

- CAMINOS E1 at n 12 (2026-11-03): not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`); Hamm kept the oracle (2026-11-03).
- PCL R3 done 2026-11-03: core arm equals the C++ after file 0 (`pcl/CALIBRATION.md` R3); R4b signed by Hamm 2026-11-03; R5-exact FAIL, 4 of 16 files (`pcl/prereg/P3_fig3_r5_exact_result.md`, P-055).
- Papers are never "failed" through 0.6 (Hamm 2026-11-03): pending until a failure is proven; criteria may fail.
- P-053 (PCL scan vs oracle gap on sparse weights, reported, not reproduced); P-054 (VIP and I presets: class-1
  a,b,c,d in the standard equation, rheobase 22.56 vs the source's 2.25; Hamm's ruling).

## Open decisions for the owner

`open_stack.md` O-85 to O-87 (P-052, T-62 data request, Lee/Mazzoni figure); NeuroAI code link
(anonymous.4open.science mirror, Hamm's act). `kernel-lfp` copied to origin `_temp_kernel-lfp` (ff45364b, 2026-11-05);
deleting the plain-named `origin/kernel-lfp` is Hamm's call.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/notebooks/*.html`, `index.json` | `scripts/export_dashboard_notebook.py` on an executed suite notebook (data on F:, so local only) | `tests/test_export_dashboard_notebook.py` |
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `docs/reference/errors.md` raise-site index | `scripts/generate_errors_index.py` | `--check`, `tests/test_generate_errors_index.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: Python 3.14 runs the broad gate through `suite_lock.py`, one suite at a time. WSL: `~/venv_jaxfne`
  (CUDA, numba 0.68 added 2026-11-02 for the PCL core arm; each long job in its own `systemd-run` memory cap),
  `~/venv_jaxfne_gate` (CPU gate); TeX Live 2026 at `~/texlive/cur` (pdfLaTeX for the NeuroAI draft).
- Tracked tree: `python scripts/check_tree_size.py --check`; budget 2796 files and 96 MB in `scripts/tree_size_allowlist.json`.
- Large data outside git: `F:/warehouse/{caminos,gsdr,pcl,vogels2011,potjans2014,destexhe2009}/` (receipts named
  in each result file); PCL training set `F:/warehouse/pcl/data/training_natural/events5_re2/` (from Zenodo
  `PCL_datasets_and_networks.zip`).
- NeuroAI 2027 draft: private thesis repo, branch `_temp_neuroai2027`, `artifacts/notes/neuroai2027/`.
- jchat: review session with the jomission keeper open in `main` (#130-137, MIN-07 VIP answered).

## Next action

CAMINOS figure suites (Hamm 2026-11-04) are on the dashboard Notebooks tab (a35aaa9d, live). Hamm 2026-11-05: show
jaxfne usage; each notebook opens with its network built and run live in jaxfne, code shown. The figures stay on
the oracle runs except PLoS Fig 1E (P6 jaxfne runs); Cell Reports core route not equivalent (E1, 20 of 24).
Raster run 497e9bb0 complete: 3 seeds × 9 conditions in `F:/warehouse/caminos/cellrep_rasters/` (panels read seed 1). Not run yet:
Cell Reports 1A, 1D, 2A/2B bottom, 6A top, 6B; PLoS Fig 5, Fig 3 and 4A mean-field rows. R5-approximate running for
0.5.10 (920 of 2562 files, 2026-11-05). Next, in worker lanes: 10D D1 (kernel-LFP picks from
`_temp_kernel-lfp`), 10F F5-F6 (T-32, T-R28), 10G G1 (CAMINOS panels not yet run).
