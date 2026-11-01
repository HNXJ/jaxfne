# state

Stamp: jaxfne@dev@dceb49b0 · 2026-11-01

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.8 published 2026-10-26 (tag v0.5.8 on main 11957afa, Release, PyPI; `artifacts/release/v0_5_8_release_receipt.json`).
  0.5.9 needs Hamm's go. Checklist score 52.7/100 at dceb49b0 (`etudes/meta/plans/release_plan_0.5.9-0.6.0.md`).
- GSDR P3 fit signed (Hamm 2026-11-01) and scored FAIL, 0 of 5 seeds (`gsdr2026/RESULT_P3_fit.md`,
  `run/p3_fit_score.json`); S14 (`supplement.py write_s14`), REPLICATIONS and dashboard updated (8f2e40ef).
- CAMINOS PLoS CB Figs 2-4 network panels scored: 6 pass, 2 partial, 3 fail of 11 claims
  (`caminos/RESULT_PLOS_FIGS.md`, b27cd476); receipts `F:/warehouse/caminos/figP_out/`.
- AdEx registered-HDP kernel resumes from its returned state, split == whole bit-for-bit; Vogels pilot runs the
  plastic leg in segments (f3d73cb1).
- PCL R2: oracle equals the C++ reference on weights within rtol 1e-12 (pcl `CALIBRATION.md` R2, fe4231bf).
- Destexhe P3 v3 pinned, unsigned (d1e495d0). Broad gate on 922270fd: 7904 passed, 0 failed.

## Broken or in flight

- Dev is 16 commits ahead of origin/dev; push waits on the broad gate at dceb49b0 (the gates at fddfe62b,
  08a91712 and b27cd476 stopped on ruff F841, a stale `docs/reference/errors.md` and ruff E402, each fixed).
- Vogels convergence pilot: n 10 000, 120 s plastic in 5 s segments, WSL GPU scope (relaunched 2026-11-01T02:44Z), clone
  `~/jaxfne_pilots` at dceb49b0, receipts `~/vog_conv/s0_T120/` (launched, not complete).
- Potjans O-80: LIF continuation landed (78e94621, after a Muse review round); the chunked Potjans pilot is next.
- O-83: the CAMINOS PLoS `default` set leaves gamma intact under SOM silencing; unattributed.

## Open decisions for the owner

Owner queue in `todo_stack.md` (O-82 T-45, T-O73 S1 drive, T-65, T-46, T-62, P-052, T-O74-79), Destexhe v3
decisions, and the signatures as drafts are ready (Vogels and Potjans P3 v2, Destexhe, PCL R4).

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
- Tracked tree: `python scripts/check_tree_size.py --check`; budget 2747 files in `scripts/tree_size_allowlist.json`.
- Large data outside git: `F:/warehouse/{caminos,gsdr,pcl,vogels2011,potjans2014}/` (receipts named in each
  result file).

## Next action

Push dev after the broad gate passes; integrate `poisscan` after its fix round; read the Vogels rate course and
fix T; attribute O-83; PCL oracle spike receipt then R3.
