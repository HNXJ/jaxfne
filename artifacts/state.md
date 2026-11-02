# state

Stamp: jaxfne@dev@b5137f3a · 2026-11-02

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.8 published 2026-10-26 (tag v0.5.8 on main 11957afa, Release, PyPI; `artifacts/release/v0_5_8_release_receipt.json`).
  0.5.9 needs Hamm's go. Checklist score 63.3/100 at fd5a9c27 (`etudes/meta/plans/release_plan_0.5.9-0.6.0.md`).
- Signed and scored 2026-11-02, each as its pilots predicted: Potjans P3 v2 off (`potjans2014/RESULT_P3.md`), Vogels
  P3 v4 off on R-E (`vogels2011/RESULT_P3.md`, in S12), Destexhe P3 v3 FAIL 0 of 5 with the 6A control holding
  (`destexhe2009/RESULT_P3.md`); receipts under `F:/warehouse/<lane>/`. GSDR P3 fit FAIL (2026-11-01); GSDR stays
  open until precisely replicated (Hamm 2026-11-02).
- AdEx per-step edge gain between two cell types (2cda7117); TFNE `Base[key=value]` refused by name (0971d0f5);
  backend labels record `jax.default_backend()` (9d75412e); Destexhe manifests record status and prereg hash (b5137f3a).
- Last pushed: 59718745, broad gate 8032 passed, 0 failed, 9 paper lanes PASS.

## Broken or in flight

- Broad gate on b5137f3a (launched, not complete); push after its summary line.
- T-59 Muse lane (clone `E:/repos/_wt/jaxfne/t59`): Cell Reports Fig S1 runner with p·NE drive (launched, not complete).
- PLoS rasters seeds 1-3 at paper scale on the WSL GPU from `~/jaxfne_plosr` at 7f7a660e, out `~/plosr_7f7a660e/out`
  (seed 1 written; launched, not complete). Then receipts to `F:/warehouse/caminos/` and the 2C-F, 3C-F figures (T-63b).
- The Vogels P3 v4 manifest says `cpu` from the label bug fixed at 9d75412e; the runs used the GPU (inferred).

## Open decisions for the owner

Owner queue in `todo_stack.md`: P-052 (waits on the T-R28 beta horizon scan); PCL R4 signature when drafted.
T-62 ruled 2026-11-02: Fig 1D excluded for 0.5.9, moved to 0.5.10.

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
  (CUDA; each long job in its own `systemd-run` memory cap), `~/venv_jaxfne_gate` (CPU gate).
- Tracked tree: `python scripts/check_tree_size.py --check`; budget 2774 files in `scripts/tree_size_allowlist.json`.
- Large data outside git: `F:/warehouse/{caminos,gsdr,pcl,vogels2011,potjans2014,destexhe2009}/` (receipts named
  in each result file).

## Next action

Push b5137f3a after its gate; integrate T-59; PLoS receipts and figures; T-30 CAMINOS port equivalence; PCL R3.
