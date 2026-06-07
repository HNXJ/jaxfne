# Current publication state

Last updated: after PRs #40–#42 merged.

## Sync target

```bash
git fetch --all --prune
git switch cur
git pull --ff-only origin cur
git rev-parse HEAD
```

Expected:

```text
5c41cd248caf98a4bb5e986eef364d38bc63c79a
```

## Inventory

```bash
python3 scripts/publication_inventory.py
```

Expected:

- 8/8 main figures
- 4/10 Extended Data

## Completed main figures

- fig01 TFNE architecture
- fig02 source-field contracts
- fig03 backend/reproducibility boundary
- fig04 minimal install + 10 s smoke
- fig05 runtime scaling receipt
- fig06 eight proxy readout families
- fig07 reproducibility chain
- fig08 adjacent tools positioning

## Completed Extended Data

- ED1 API stability snapshot
- ED2 JSON/config schema validation
- ED3 notebook execution receipts
- ED4 optional dependency laziness

## Next work

ED5 manifest hashes:

```bash
git switch cur
git pull --ff-only origin cur
git switch -c pub/ed05-manifest-hashes
```

Goal:

- summarize manifest/PNG/receipt hash coverage across main figures and ED1–ED4
- local artifact hash receipt only
- no package API changes
- preserve truth gates
