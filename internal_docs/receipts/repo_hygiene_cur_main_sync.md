# Repo hygiene receipt — `cur` / `main` sync pass

## Identity

| Field | Value |
|---|---|
| branch (work) | `cur` |
| pre-sync SHA | `76376bfebf7433dcad167b492f2d1dfa9873965c` |
| post-patch SHA | hygiene commit on `cur` (query: `git log -1 --format=%H -- internal_docs/receipts/repo_hygiene_cur_main_sync.md`) |
| platform | Darwin |
| python | 3.13.7 |

## Files moved / edited

| Action | Path |
|---|---|
| moved | `CLAUDE.md` → `internal_docs/agent_context/claude/CLAUDE.md` |
| moved | `docs/RELEASE_CHECKLIST.md` → `internal_docs/receipts/RELEASE_CHECKLIST.md` |
| edited | `scripts/upload_testpypi.sh` (checklist path) |
| edited | `internal_docs/receipts/RELEASE_CHECKLIST.md` (internal paths) |
| edited | `.gitignore` (ignore private review output zips) |
| added | this receipt |

No package API, solver, Jaxley, PyNWB, or manuscript TBD edits.

## Private artifact handling

| Artifact | Handling |
|---|---|
| `figures/publication/*.png` (14 files) | stashed: `hold local P0 regenerated publication PNG drift` — **not committed** |
| `outputs-provenance-patched.zip` | moved to `/tmp/jaxfne_private_review/` — **not committed** |
| `outputs/publication/*` | gitignored; inventory regen local only — **not committed** |

## Validation

| Command | Result |
|---|---|
| `python3 -m compileall -q jaxfne scripts tests` | exit 0 |
| `pytest tests/test_public_docs_hygiene.py -q` | 218 passed, 5 skipped |
| `pytest test_api_smoke + root_import + signals_get + hygiene -q` | 248 passed, 6 skipped |
| `python3 scripts/publication_inventory.py` | 8/8 main + 10/10 ED |
| `python3 -m json.tool docs/publication/publication_checklist.json` | exit 0 |
| `python3 -m mkdocs build --strict` | exit 0 |

## Release / tag / build / archive

**Not performed.** No tag, wheel build, PyPI upload, GitHub release, archive, or DOI action.

## Branch sync

Recorded after push/ff attempt in this pass.

## P0/P1 alpha preservation

Frozen review outputs pin unchanged: `outputs_zip_sha256=a0292503268616c1c3756ad3e03d426bcda7ac01f6c6bc37575db2fcf87107c4` (generated at `9f831bd` / `0.3.29`). Hygiene pass is docs/process layout only.
