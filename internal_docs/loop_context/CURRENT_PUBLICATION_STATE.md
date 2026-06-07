# Current publication state

Last updated: after ED5 + ED7 merged to `cur`.

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
- 6/10 Extended Data

## Completed main figures

fig01–fig08 (see `docs/publication/publication_checklist.json`)

## Completed Extended Data

- ED1 API stability snapshot
- ED2 JSON/config schema validation
- ED3 notebook execution receipts
- ED4 optional dependency laziness
- ED5 manifest hashes (`ed05_manifest_hashes`)
- ED7 probe/readout operator contracts (`ed07_probe_operator_contracts`)

## Next work

ED6 benchmark scaling tables:

```bash
git switch cur
git pull --ff-only origin cur
git switch -c pub/ed06-benchmark-scaling-tables
```

## Agent anchor

Full glossary, scoreboard, ED ladder, and Cursor prompts: [`JAXFNE_BIOPHYSICS_GLOSSARY.md`](JAXFNE_BIOPHYSICS_GLOSSARY.md).
