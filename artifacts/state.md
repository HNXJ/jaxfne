# state

Stamp: jaxfne@dev@fdd29614 · 2026-11-07 (pushed; broad on f290fd70: 8222 passed, 18 R5-load timeouts passed alone, 9 lanes PASS) · checklist 0.5.10: 16/100

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
- P-BALANCED ported for 0.5.10 (e179a2d0..9dd1c111; Hamm 2026-11-05): `BackgroundNoise`, `NeuronalTensor.baseline_drive`
  and `background_noise` (preset noise in `cfg.metadata["background_noise"]`), balanced-1000n preset retuned under
  the unchanged bounds (seeds 0-4: CV 0.742-0.755, shift 0.236-0.257); canonical CV pin 0.0585 (1680659e re-pinned it).
- Landed 2026-11-06 evening (f0489d18..1e7ee1e5): HH seed refusal test; homeostatic_ei stable x bounds from construct
  (±10); B1 baseline scripts; tree budget 96.1 MB. jnwb 16-12 isi_cv on jnwb `_temp_jaxfne-16e` (5fa63dd3,
  accepted); keep `E:/cl-jnwb16e` until jnwb-orchestrator confirms its pick.
- Landed 2026-11-07 (1e7ee1e5..8516c721): B1 M3 harness fix; Jansen-Rit P1 record, P3 v1 draft and driver (O-92);
  PLoS Fig 5A producer, sustained-cycle test and panel. Broad 8238 passed, 2 load timeouts (passed alone); 9 lanes PASS.
- 2026-11-07 (b24f7a07..): match-check fixes; O-93 (Hamm): T_v code form = declared variant `tv_form="code"`
  (8a6ccfc4), printed default; code form gives SOM->SOM 1.6/1.7 nS, 35.0 Hz (`caminos/RESULT_MEANFIELD.md`).

## Broken or in flight

- CAMINOS E1 at n 12 (2026-11-03): not equivalent, 20 of 24 cells (`caminos/RESULT_FIGURES.md`); Hamm kept the oracle (2026-11-03).
- PCL R3 done 2026-11-03: core arm equals the C++ after file 0 (`pcl/CALIBRATION.md` R3); R4b signed by Hamm 2026-11-03; R5-exact FAIL, 4 of 16 files (`pcl/prereg/P3_fig3_r5_exact_result.md`, P-055).
- Papers are never "failed" through 0.6 (Hamm 2026-11-03): pending until a failure is proven; criteria may fail.
- P-053 (PCL scan vs oracle gap on sparse weights, unreproduced); P-056 (kernel-LFP deep-layer power < 0.5% of L2/3).
- R5-approximate resumed detached 2026-11-06 16:54Z: 2860 of 3168 tasks (777 of 1085 since the resume) at
  2026-11-07 22:18Z; finish about 2026-11-08 10Z (inferred). Its driver logs `rc=$?` after `$(stamp)`, so the
  logged rc is always 0: judge the run by its output files. Its 20 workers hold the CPU at 100%: broad gates
  time out 17 example subprocesses (120 s) until it ends; re-run them alone.
- Tooling: Cursor write fails at launch (read works); Muse code workers may exit early (resume with `--session`).

## Open decisions for the owner

`open_stack.md` O-85 to O-92 (P-052, T-62 data request, Lee/Mazzoni figure, G1 Cell Reports panels, stale-clone
delete script, B1 M3 class, A3 scope, Jansen-Rit P3 signature); NeuroAI code link
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
- Large data outside git: `F:/warehouse/{caminos,gsdr,pcl,vogels2011,potjans2014,destexhe2009}/` (receipts named
  in each result file); PCL training set `F:/warehouse/pcl/data/training_natural/events5_re2/` (from Zenodo
  `PCL_datasets_and_networks.zip`).
- NeuroAI 2027 draft: private thesis repo, branch `_temp_neuroai2027`, `artifacts/notes/neuroai2027/`.

## Next action

CAMINOS suites on the dashboard Notebooks tab (a35aaa9d; each opens with its network run live in jaxfne, Hamm
2026-11-05); figures on the oracle runs except PLoS Fig 1E (P6). Rasters: `F:/warehouse/caminos/cellrep_rasters/`
(497e9bb0). Not run: Cell Reports 1A, 1D, 2A/2B bottom, 6A top, 6B; PLoS Fig 3 and 4A mean-field rows. Next:
re-export the PLoS notebook (Fig 5A cell in, a6f4d867); B1 scored runs (T-B1); E1 at R5 end; Jansen-Rit P3 on O-92.
