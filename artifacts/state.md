# state

Stamp: jaxfne@dev@4dce6512 · 2026-10-13 (written 2026-10-13T05:28:25Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- dev 4dce6512 pushed 2026-10-13: R3 version-currency test, R4 stage artifacts on the dashboard
  track, the sorted todo stack, the `artifacts/README.md` index with its test, the mkdocs
  stand-in test fix.
- Docs atlas HTML is a build product: `scripts/build_docs_assets.py` (`--build` regenerates the 25
  atlases and checks each against `docs/_static/atlas_registry.json`; `--check` resolves every
  relative `.html` link in `docs/`). `.readthedocs.yaml` runs `--build` before mkdocs;
  `scripts/mkdocs_atlas_stubs.py` adds stand-ins where the HTML is unbuilt and fails on Read the
  Docs. WSL probe from a tree with no atlas HTML: `--build` 123 s, 25/25 match, strict build with
  `READTHEDOCS=True` passes, 0 stand-ins, 47/47 iframes resolve.
- Tracked tree: 2402 files, 90302671 bytes (budget 91226112); 0.5.5 tag: 2522 files, 125982465.
- `artifacts/` folders stay in place: publication specs cite protocol, etude, atlas and audit
  receipts by path; `artifacts/README.md` lists every tracked folder with its frozen status.
- Replication dashboard live at https://hnxj.github.io/jaxfne/dashboard/; each finished stage of a
  paper links its files at the lane commit.
- CAMINOS Fig. 1: R1–R3 PASS (v6), R4 FAIL (v6), R4v7 PASS. PCL Fig. 3: v4 FAIL (T2, T4) stands.
  Izhikevich 2003: class "in progress", stage P3.
- AdEx kernel `jaxfne.emitters.simulate_adex` (ADVANCED) matches the CAMINOS reference within rtol
  1e-6 (float64); `connections(sampling="bernoulli")` (G-2 F1).

## Broken or in flight

- Release CI with slow tests on dev 4dce6512: run 38351542234, in flight. The previous run on
  226a7952 (38347551311) failed 2 tests in `tests/test_mkdocs_atlas_stubs.py` (fixtures read atlas
  HTML from disk); fixed in 4dce6512, second review PASS.
- dev → main PR (T-5) waits on that run. Other open items: `artifacts/open_stack.md`.

## Last gate results

- Broad gate in a fresh clone of bb86fe9a: 5368 passed, 0 failed, 86 skipped (the mkdocs module
  skips in the jax venv). `tests/test_mkdocs_atlas_stubs.py` at 4dce6512 under system Python: 3
  passed in a fresh clone and in a tree with the HTML; the old version fails 2 in a fresh clone.
- Docs asset inventory (53 files in `tutorial_figures`, `_static/gallery`, `docs/assets`): 3
  regenerate byte-identical, 36 drift, 6 have no writer, 1 never written, 7 no generator.

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

T-5: release CI green on 4dce6512, then `gh pr create --base main --head dev`, merge, then watch
main CI, the Read the Docs latest build (first `pre_build` atlas build) and the Dashboard
workflow. Then T-4 (narrowing: etude bundles `experiment_a`, `hdp_mcc3`) and Hamm's T-1 to T-3.
