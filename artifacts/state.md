# state

Stamp: jaxfne@dev@b9e55903 · 2026-10-13 (written 2026-10-13T08:09:54Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- main ff89b24b (PR #103, merge of dev 4dce6512): CI (Release & Scheduled, all tests including
  slow) success, run 38361054020; CI (Fast) success. Read the Docs `latest` build 35110238 of
  ff89b24b succeeded with the atlas `pre_build`; published panels are real (plotly CDN, no
  stand-in text).
- dev b9e55903: experiment_a and hdp_controllability_reachability etude HTML (12 files) is a
  build product (`build_docs_assets.py --build` runs both etude generators, which hash-check the
  frozen bundles, and keeps the tracked figure state). CI (Fast) success.
- Linux probe of the Read the Docs steps at ededf86f (WSL, fresh venv, Python 3.12; RTD uses
  3.13): 25/25 atlases ok, both etudes ok (3.4 s, 14.3 s), `mkdocs build --strict` with
  `READTHEDOCS=True` rc 0, 0 stand-ins, published etude manifests `validated`.
- Tracked tree: 2391 files, 88960885 bytes (budget 89128960, 85 MiB); 0.5.5 tag: 2522 files,
  125982465. Target 2269 files: 122 to go (T-4).
- Replication dashboard live at https://hnxj.github.io/jaxfne/dashboard/ (HTTP 200).
- CAMINOS Fig. 1: R1–R3 PASS (v6), R4 FAIL (v6), R4v7 PASS. PCL Fig. 3: v4 FAIL stands.
  Izhikevich 2003: "in progress", stage P3.

## Broken or in flight

- CI (Fast) on dev bf541dd0 failed once: `test_build_fails_when_a_registry_html_file_is_not_written`
  compared a fake file's mtime with the build start in the same clock tick. Fixed in ededf86f
  (fixtures stamp outputs 2 s after the start); b9e55903 green.
- Hamm's decisions T-1 (etude drift), T-2 (`.lab`), T-3 (PCL Fig. 3). Others: `open_stack.md`.

## Last gate results

- Broad gate in a fresh clone of b9e55903: 5377 passed, 0 failed, 86 skipped;
  `tests/test_mkdocs_atlas_stubs.py` under system Python (mkdocs): 3 passed.
- Second reviews: ededf86f FAIL (budget 1202 bytes under the total) → fixed in b9e55903.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `docs/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `artifacts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11; no mkdocs). System Python
  has mkdocs. WSL: `~/venv_rtd313` (actually Python 3.12, `.[viz]` + docs requirements) for
  Read the Docs probes; `~/venv_jaxfne` for GPU runs, one job at a time after a jchat claim.

## Next action

T-4 waits mostly on T-1 and T-2 (120 of the 122 files). Independent of Hamm: T-6 (F6 "SOM" in
the AdEx table, then the CAMINOS P6 prereg), T-7 (Izhikevich 2003 P3 calibration gate).
