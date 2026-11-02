# state

Stamp: jaxfne@dev@d96dc49b · 2026-11-02

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.8 published 2026-10-26 (tag v0.5.8 on main 11957afa, Release, PyPI; `artifacts/release/v0_5_8_release_receipt.json`).
  0.5.9 needs Hamm's go. Checklist score 63.3/100 at fd5a9c27 (`etudes/meta/plans/release_plan_0.5.9-0.6.0.md`).
- Signed and scored 2026-11-02, each as its pilots predicted: Potjans P3 v2 off, Vogels P3 v4 off on R-E (in S12),
  Destexhe P3 v3 FAIL 0 of 5 (`RESULT_P3.md` in each lane; receipts under `F:/warehouse/<lane>/`). GSDR P3 fit FAIL
  (2026-11-01); GSDR stays open until precisely replicated (Hamm 2026-11-02).
- CAMINOS Cell Reports Fig S1 scored against criteria declared first (d96dc49b): F pass, S partial, G fail, D pass
  (`caminos/RESULT_FIGURES.md`); supplementary, outside the dashboard's main-text scope.
- T-30 core lanes a-d landed (d 2cda7117: per-step edge gain, no-edge refusal, equivalence tests).
- PCL R3 C++ arm: training files 0-4 chained from the R2 init; two chains bit-identical, weights moved from init
  (`F:/warehouse/pcl/r3_cpp_f0_4/`).
- Last pushed: d96dc49b (report gate fail_count 0; CI success on d095f023); last broad gate 8038 passed on dbcdb2cb.
- Local branches follow the naming rule since 2026-11-02 (30 renamed to `_temp_*`; none on the remote).

## Broken or in flight

- Muse CELLREP-CORE (clone `E:/repos/_wt/jaxfne/crcore`): `cellrep_figures.run_cell(route=...)` on the core
  (launched, not complete); then the full-size equivalence check against the frozen sweeps.
- Muse PCL-R3-HARNESS (clone `E:/repos/_wt/jaxfne/pclr3`): `chain` subcommand, oracle and core arms (launched, not
  complete).
- Release CI run 41465568642 on d96dc49b (dispatched for the 9F rc gate; launched, not complete).
- 9G: 11 commits since v0.5.8 change behavior without a same-commit fragment; coverage elsewhere being checked.

## Open decisions for the owner

Owner queue in `todo_stack.md`: P-052 (waits on the T-R28 beta horizon scan); PCL R4 signature when drafted;
NeuroAI code link (anonymous.4open.science mirror, Hamm's act). Deleting the 22 merged `_temp_*` branches is Hamm's
call. T-62 ruled 2026-11-02: Fig 1D excluded for 0.5.9, moved to 0.5.10.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `docs/reference/errors.md` raise-site index | `scripts/generate_errors_index.py` | `--check`, `tests/test_generate_errors_index.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: Python 3.14 runs the broad gate through `suite_lock.py`, one suite at a time. WSL: `~/venv_jaxfne`
  (CUDA; each long job in its own `systemd-run` memory cap), `~/venv_jaxfne_gate` (CPU gate); TeX Live 2026 at
  `~/texlive/cur` (pdfLaTeX for the NeuroAI draft).
- Tracked tree: `python scripts/check_tree_size.py --check`; budget 2778 files in `scripts/tree_size_allowlist.json`.
- Large data outside git: `F:/warehouse/{caminos,gsdr,pcl,vogels2011,potjans2014,destexhe2009}/` (receipts named
  in each result file); PCL training set `F:/warehouse/pcl/data/training_natural/events5_re2/` (from Zenodo
  `PCL_datasets_and_networks.zip`).
- NeuroAI 2027 draft: private thesis repo, branch `_temp_neuroai2027`, `artifacts/notes/neuroai2027/`.

## Next action

Integrate CELLREP-CORE, then the full-size equivalence run; integrate PCL-R3-HARNESS, run the oracle and core arms on
files 0-4; rc gate after Release CI; close the 9G fragment gaps; update the checklist score.
