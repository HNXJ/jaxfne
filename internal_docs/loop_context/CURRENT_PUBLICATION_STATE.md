# Current publication state

Last updated: ED8 on `pub/ed08-tutorial-atlas-coverage`.

## Sync target

```bash
git fetch --all --prune
git switch cur
git pull --ff-only origin cur
git rev-parse HEAD
python3 scripts/publication_inventory.py
```

## Inventory (expected after sync)

- 8/8 main figures
- 8/10 Extended Data (after ED8 on branch)

## Completed main figures

fig01–fig08 (see `docs/publication/publication_checklist.json`)

## Completed Extended Data

- ED1 API stability snapshot
- ED2 JSON/config schema validation
- ED3 notebook execution receipts
- ED4 optional dependency laziness
- ED5 manifest hashes (`ed05_manifest_hashes`)
- ED6 benchmark scaling tables (`ed06_benchmark_scaling_tables`)
- ED7 probe/readout operator contracts (`ed07_probe_operator_contracts`)
- ED8 tutorial atlas coverage (`ed08_tutorial_atlas_coverage`)

## Next work

ED9 failure modes and null controls:

```bash
git switch cur
git pull --ff-only origin cur
git switch -c pub/ed09-failure-modes-null-controls
```

## Agent anchor

Full glossary, scoreboard, ED ladder, and Cursor prompts: [`JAXFNE_BIOPHYSICS_GLOSSARY.md`](JAXFNE_BIOPHYSICS_GLOSSARY.md).


# Gamma Project alignment
All agents refer to the Gamma Project board for coordination. See
`internal_docs/loop_context/GAMMA_PROJECT_ALIGNMENT.md`.
