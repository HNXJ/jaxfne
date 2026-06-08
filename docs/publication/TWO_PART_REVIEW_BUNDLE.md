# Two-part external review bundle

jaxfne alpha review uses **two artifacts** because `outputs/publication/*` is gitignored in the repository.

## Frozen review target

```text
branch: cur
source_head: 9f831bd1d9d6464d4bee8b18f92d01b2e0ad45e5
jaxfne_version: 0.3.29
release_action_taken: false
physical_amplitude_claim_allowed: false
```

## Part A — Repository (code + docs + figure PNGs)

Export committed `cur` at `source_head`:

```bash
git archive --format=zip --prefix=jaxfne/ -o jaxfne-alpha-repo-9f831bd.zip 9f831bd1d9d6464d4bee8b18f92d01b2e0ad45e5
```

Includes: source, tests, docs, `figures/publication/*.png`, alpha handout/scoreboard/report docs.  
Does **not** include `outputs/publication/*` manifests/receipts.

## Part B — Private outputs (manifests + receipts)

Verified provenance-patched bundle:

```text
file: outputs-provenance-patched.zip
sha256: a0292503268616c1c3756ad3e03d426bcda7ac01f6c6bc37575db2fcf87107c4
```

Contains `publication/` with 29 strict JSON files, `inventory.json`, ED receipts, and `alpha_review_zip_receipt.txt`.

**Send Part A + Part B together** to external reviewers. Do not use an older combined repo zip that embeds stale outputs from mixed SHAs.

## Regeneration recipe (provenance-safe)

On clean checkout at `source_head`:

```bash
python3 -m venv .venv-provenance
. .venv-provenance/bin/activate
pip install -e ".[dev]" -r docs/requirements.txt
rm -rf outputs/publication && mkdir -p outputs/publication
# run all 18 scripts/publication/fig*.py and ed*.py (canonical names in publication_checklist.json)
python scripts/publication/ed10_release_archive_receipt.py   # last; do not run publication_inventory.py after ED10
```

All manifests must report:

```text
repo_sha: 9f831bd1d9d6464d4bee8b18f92d01b2e0ad45e5
jaxfne_version: 0.3.29
```

## Notebook execution scope (structural receipts)

ED3 and ED8 are **structural atlas/receipt** panels. They survey notebook paths and metadata; they do **not** claim universal notebook PASS execution unless separate execution logs/receipts exist. Wording must remain: receipt-driven, structural scan, `notebook_execution_completeness_claim_allowed: false`.

## Confirmatory gates (reproduce)

| Gate | Expected |
|---|---|
| outputs zip SHA256 | `a0292503...7107c4` |
| strict JSON | 29/29 pass |
| all `repo_sha` | `9f831bd...` |
| all `jaxfne_version` | `0.3.29` |
| inventory | 8/8 + 10/10 ED |
| manifest→PNG SHA | 18/18 match |
| ED10 inventory hash | matches `inventory.json` |
| ED10 release fields | all false |
