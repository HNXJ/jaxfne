# Narrowing: regenerate docs assets at the docs build (B4; Hamm 2026-10-12)

Docs build: `mkdocs build --strict` in CI (`ci.yml`) and on Read the Docs (`.readthedocs.yaml`,
`docs/requirements.txt` installs mkdocs only, no jaxfne or jax).

## Inventory (tracked bytes, 2026-10-12, dashboard branch at 0758b26c)

| Tree | Files | KB | Generator | Plan |
|---|---|---|---|---|
| docs/_static/atlas | 216 | 9235 | `scripts/generate_doc_page_atlases.py` (`--slug`), `jaxfne/vis/atlas_suite.py`, `promote_atlas_state.py` | regenerate at build (step 3) |
| docs/_static/etudes | 29 | 3076 | `scripts/plot_frozen_etude_panels.py`, `rerun_etude_figures.py` | from frozen receipts: regenerate at build |
| docs/_static/tutorial_figures | 13 | 572 | `scripts/generate_tutorial_figures.py --output-dir` | regenerate at build |
| docs/_static/gallery | 10 | 1221 | `scripts/generate_release_gallery.py` | regenerate at build |
| docs/_static/atlas_three_area | 9 | 322 | `scripts/generate_doc_page_atlases.py --slug three_area` | regenerate at build (step 3) |
| docs/assets | 37 | 21460 | showcases, readme, visuals, interactive (`build_v037_source_column_3d.py`), etudes | step 1, then regenerate |
| artifacts/tutorials | 50 | 15269 | notebooks (sources) | keep; strip outputs if large |
| artifacts/etudes, artifacts/publication, artifacts/figures | 592 | 48385 | evidence, frozen | keep (frozen manifest) |

## Steps

1. Done in bb67ea26: five Plotly pages load plotly.js 3.5.0 from the CDN with an SRI hash; tree
   124.1 → 99.9 MB. `artifacts/publication/atlas/AT-10-N20/*/lfp.html` stay as they are (frozen).
2. A build entry point `scripts/build_docs_assets.py` that runs each generator above with fixed seeds,
   with `--check` comparing against a manifest of input hashes; CPU float32, time budget per generator.
3. RTD: `build.jobs.pre_build` installs jaxfne (`pip install .`) and runs step 2; CI docs job the same.
   Measure the RTD build time first (limit 15 min on the free plan, inferred, to check).
4. Untrack the regenerated trees (`git rm --cached`, `.gitignore`); tests that read them
   (`test_tutorial_figure_contract_v0219.py`, `test_v037_interactive_column_docs.py`, atlas gates)
   generate into tmp or are marked docs-build.
5. Atlas promotion state (`promote_atlas_state.py`) must keep working: memory note "atlas regen resets
   state". Decide whether validated state lives in a tracked manifest only.
