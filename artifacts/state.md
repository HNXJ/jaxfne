# state

Stamp: jaxfne@dev@7426505b · 2026-10-13 (written 2026-10-13T04:16:41Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- dev 7426505b: CI (Fast) green (run 38345593663, head_sha 7426505b); Dashboard green on fea6676e.
- Docs atlas HTML is a build product: `scripts/build_docs_assets.py` (`--build` regenerates the 25
  atlases and checks each against `docs/_static/atlas_registry.json`; `--check` compares tracked
  manifests and resolves every relative `.html` link in `docs/`). `.readthedocs.yaml` runs
  `--build` before mkdocs; `scripts/mkdocs_atlas_stubs.py` adds stand-ins where the HTML is
  unbuilt and fails on Read the Docs. WSL Ubuntu probe of those steps from a tree with no atlas
  HTML: `--build` 123 s, 25/25 match, `mkdocs --strict` with `READTHEDOCS=True` passes, 0
  stand-ins, 47/47 iframes resolve.
- Docs iframes: 34 of 47 resolved to nonexistent paths under directory URLs before 932607e5; 0 since.
- Tracked tree: 2402 files, 90302671 bytes (budget 91226112); 0.5.5 tag: 2522 files, 125982465.
- Replication dashboard live at https://hnxj.github.io/jaxfne/dashboard/; the docs Replications
  page links each paper's page (`scripts/generate_replications_page.py --check` in the
  `Dashboard` workflow).
- CAMINOS Fig. 1: R1–R3 PASS (v6), R4 FAIL (v6), R4v7 PASS (drive p·NE, onset p = 0.04).
  PCL Fig. 3: v4 FAIL (T2, T4) stands. Izhikevich 2003: class "in progress", stage P3.
- AdEx conductance kernel `jaxfne.emitters.simulate_adex` (ADVANCED): matches the CAMINOS
  reference cell and recurrent step within rtol 1e-6 (float64); `emitter(family="adex")`;
  `Simulation(poisson_drive=)` bit for bit with a transcription of the reference.
- `connections(sampling="bernoulli")` (G-2 F1); default path and config hashes unchanged.

## Broken or in flight

In `artifacts/open_stack.md`.

## Last gate results

- Broad gate in a fresh clone of 7426505b (no atlas HTML on disk): 5356 passed, 0 failed, 90
  skipped. CI fast gates: ALL fail=0. Second reviews of R2 steps 2-4 and the hook: findings fixed.
- `tests/test_mkdocs_atlas_stubs.py` (needs mkdocs; skips in the jax venv): 3 passed under the
  system Python.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `docs/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html` (untracked), manifests | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `artifacts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11; no mkdocs). System Python
  has mkdocs. GPU runs: WSL2 CUDA `~/venv_jaxfne`, one job at a time after a jchat claim.

## Next action

0.5.6 release (G-5, R2-R6 in `todo_stack.md`): R2 next asset classes toward -133 files; R3
version-currency test; R4 stage artifacts on the dashboard track; G-2 F6, then the CAMINOS P6
prereg; Izhikevich 2003 P3 calibration gate. RTD first runs the atlas build at the 0.5.6 merge
to main. 0.5.5 is tagged v0.5.5 and on PyPI (2026-10-11).
